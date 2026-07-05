"""
quill/_tuning.py
----------------
Measured-latency dispatcher state.

Why this exists
===============
Each backend in :mod:`quill._backends` declares a hardcoded ``min_n`` crossover
and a static ``priority``. Those numbers were picked on the maintainers' boxes
and they are routinely wrong on someone else's CPU (a 4-core laptop, a 128-core
EPYC server, an M-series Mac). The honest fix is to *measure* what is actually
fastest on this machine and use that — but only after we have collected enough
samples to trust the measurement.

This module owns the timing database. Backends call :func:`record` after every
real sort with the elapsed wall time and the element count; the dispatcher in
``_backends`` calls :func:`best_for` to ask, for a given ``(dtype_kind, size)``
bucket, which eligible backend has the lowest EWMA latency per element. If we
don't yet have ``K`` observations for *every* candidate, ``best_for`` returns
``None`` and the caller falls back to the priority order — that way we never
"lock in" a winner before we have actually tried the alternatives.

Persistence
~~~~~~~~~~~
The database is stored as JSON at ``~/.quill/timings.json`` (see
:func:`quill._config.config_dir`). Writes are atomic (write to ``.tmp`` then
``os.replace``) and rate-limited to at most one disk write per 30 s, because
``record`` is called on every hot-path sort and fsync'ing the home directory
on every call would dwarf the sort itself.

Failure modes
~~~~~~~~~~~~~
Persistence errors (disk full, permission denied, read-only $HOME) are logged
and swallowed; they NEVER propagate into a caller's sort. If numpy is missing,
the module still imports — :func:`record` becomes a no-op because the
dispatcher won't be calling us anyway.

Env overrides
~~~~~~~~~~~~~
* ``QUILL_TUNING_DISABLED=1`` — :func:`best_for` always returns ``None`` so the
  dispatcher uses its priority order (handy for benchmarking the static path).
* ``QUILL_TUNING_RESET=1`` — on import, the timings file is wiped so the next
  run starts from a clean slate.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np  # noqa: F401  # tuning is a no-op without numpy, but
    _NUMPY = True       # we keep the import test consistent with other modules
except ImportError:
    _NUMPY = False

from ._config import config_dir, load_config

logger = logging.getLogger("quill")

# Smoothing factor for the latency-per-element EWMA. 0.3 follows recent samples
# fast enough to react to a thermal-throttled CPU within a handful of sorts,
# but slow enough that one anomalous GC pause won't flip the dispatcher.
_EWMA_ALPHA = 0.3

# Minimum observations per candidate before we trust the measured ordering.
# Overridable via load_config()['tuning_min_obs']; below this we return None
# and the caller uses static priority.
_DEFAULT_MIN_OBS = 5

# Exploration early-abandonment (see TimingDB.choose). Once a candidate under
# exploration has at least _EXPLORE_PROBE_MIN samples AND its measured latency
# is worse than _DEFAULT_ABANDON_FACTOR x the best already-warmed candidate, we
# stop routing sorts to it — bounding the one-time "explore every newly-installed
# backend" tax to ~1-2 sorts for backends that are clearly dominated on this
# machine, instead of the full min_obs. Overridable via
# load_config()['tuning_abandon_factor'] (set <=0 to disable abandonment).
_EXPLORE_PROBE_MIN = 2
_DEFAULT_ABANDON_FACTOR = 2.0

# Tie-break margin for the EXPLOIT phase (see TimingDB.choose). When the a-priori
# floor (the lowest-priority candidate — the bare np.sort backend) measures within
# this factor of the fastest candidate, prefer the floor. On AVX-512 / Apple-
# silicon hosts a SIMD-delegating backend and bare np.sort run essentially the
# SAME numpy kernel, so which one wins is measurement jitter; pinning to the floor
# makes convergence deterministic and picks the lowest-overhead route to that
# kernel. Overridable via load_config()['tuning_tie_factor'] (<=0 disables it).
_DEFAULT_TIE_FACTOR = 1.1

# At most one save per this many seconds. record() is called on every sort and
# we do not want to fsync the home directory on a microsecond-scale code path.
_SAVE_INTERVAL_S = 30.0


def _timings_path(override: Optional[str] = None) -> str:
    """Resolve the JSON path. Honours an explicit override, else $QUILL_CONFIG_DIR."""
    if override:
        return override
    return os.path.join(config_dir(), "timings.json")


def _truthy_env(name: str) -> bool:
    """Liberal truthy check for env-var flags.

    Accept any of the common "on" spellings — ``1``, ``true``, ``yes``, ``on``
    (case-insensitive) — so users who type ``QUILL_TUNING_DISABLED=true``
    don't silently get the opposite of what they asked for.
    """
    val = os.environ.get(name)
    if val is None:
        return False
    return val.strip().lower() in ("1", "true", "yes", "on")


def _bucket_for(n: int) -> int:
    """Return the log2-floor bucket for ``n`` elements.

    1 -> 0, 2 -> 1, 1024 -> 10, 1_000_000 -> 19, etc. We bucket by power-of-two
    because backend crossovers are roughly log-shaped: the gap between 1k and
    2k matters, the gap between 1.0M and 1.1M does not.
    """
    if n <= 1:
        return 0
    return max(0, n.bit_length() - 1)


class TimingDB:
    """In-memory + on-disk record of measured backend latencies.

    Storage shape (after :func:`export`)::

        {
            "<dtype_kind>": {                # 'i' | 'u' | 'f'
                "<bucket>": {                # log2-floor of n, as a string in JSON
                    "<backend_name>": [count, ewma_seconds_per_element]
                }
            }
        }

    The class is thread-safe: ``record`` is called from sort callers that may
    well be running across a thread pool, and ``save`` may be invoked from any
    of them.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self._path = _timings_path(path)
        self._lock = threading.Lock()
        # data: dtype_kind -> bucket(int) -> backend -> (count, ewma)
        self._data: Dict[str, Dict[int, Dict[str, Tuple[int, float]]]] = {}
        # Start the rate-limit clock NOW, not at the epoch. Initialising to 0.0
        # meant the very first record() would satisfy `now - 0.0 > 30s` and
        # trigger a synchronous disk write inside the first sort's hot path —
        # which is exactly the kind of one-time spike that shows up in
        # benchmarks as a "v7 has higher variance than v6" outlier (~50-200 ms
        # on Windows with antivirus). The first save now happens at least
        # _SAVE_INTERVAL_S after import, off the first user-visible sort.
        self._last_save_ts: float = time.monotonic()
        self._dirty: bool = False
        self.load()

    # ── persistence ──────────────────────────────────────────────────────────

    def load(self) -> None:
        """Read the JSON file into memory. Never raises."""
        with self._lock:
            self._data = {}
            try:
                if not os.path.exists(self._path):
                    return
                with open(self._path, "r", encoding="utf-8") as fh:
                    raw = json.load(fh)
            except (OSError, ValueError, TypeError) as exc:
                # Corrupt file is treated as "no history" — better to relearn
                # than to crash the import of every quill consumer.
                logger.debug("quill tuning: load failed (%s); starting fresh", exc)
                return
            if not isinstance(raw, dict):
                return
            for kind, buckets in raw.items():
                if not isinstance(buckets, dict):
                    continue
                kind_map: Dict[int, Dict[str, Tuple[int, float]]] = {}
                for bucket_s, backends in buckets.items():
                    try:
                        bucket = int(bucket_s)
                    except (TypeError, ValueError):
                        continue
                    if not isinstance(backends, dict):
                        continue
                    inner: Dict[str, Tuple[int, float]] = {}
                    for name, pair in backends.items():
                        # Stored as a 2-list in JSON; tolerate dicts too for
                        # forward-compat with any future richer schema.
                        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                            try:
                                inner[str(name)] = (int(pair[0]), float(pair[1]))
                            except (TypeError, ValueError):
                                continue
                    if inner:
                        kind_map[bucket] = inner
                if kind_map:
                    self._data[kind] = kind_map

    def save(self, force: bool = False) -> None:
        """Atomically write the database to disk. Rate-limited unless ``force``.

        The disk write happens *inside* the lock. That sounds scary on a hot
        path, but ``save`` is rate-limited to once per :data:`_SAVE_INTERVAL_S`
        and the JSON payload is a few KB at most — the alternative (release
        the lock, write, re-acquire to clear ``_dirty``) loses any updates
        recorded between the export and the dirty-flag reset.
        """
        with self._lock:
            now = time.monotonic()
            if not force:
                if not self._dirty:
                    return
                if (now - self._last_save_ts) < _SAVE_INTERVAL_S:
                    return
            payload = self._export_locked()
            try:
                d = os.path.dirname(self._path)
                if d:
                    os.makedirs(d, exist_ok=True)
                tmp = self._path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, sort_keys=True)
                os.replace(tmp, self._path)
                self._last_save_ts = now
                self._dirty = False
            except OSError as exc:
                # Disk full, permission denied, read-only home — none of these
                # are the user's problem; they just lose the next session's
                # warm start. Leave _dirty=True so a later save can retry.
                logger.debug("quill tuning: save failed (%s); continuing", exc)
                return

    # ── mutation ─────────────────────────────────────────────────────────────

    def record(self, backend: str, dtype_kind: str, n: int, elapsed_s: float) -> None:
        """Fold a single measurement into the EWMA for its bucket."""
        # Guard rails: a zero-length sort, a numpy-less build, or a bogus
        # elapsed time must not corrupt the database or crash the caller.
        if not _NUMPY:
            return
        if not backend or not dtype_kind or n <= 0:
            return
        # Reject NaN, +inf, -inf, and anything non-numeric. A finite, non-
        # negative float is the only thing that can safely fold into an EWMA;
        # an inf would poison the average and permanently demote this backend.
        if not (isinstance(elapsed_s, (int, float))
                and not isinstance(elapsed_s, bool)
                and math.isfinite(elapsed_s)
                and elapsed_s >= 0.0):
            return
        per_elem = elapsed_s / float(n)
        bucket = _bucket_for(n)
        with self._lock:
            kind_map = self._data.setdefault(dtype_kind, {})
            inner = kind_map.setdefault(bucket, {})
            prev = inner.get(backend)
            if prev is None:
                inner[backend] = (1, per_elem)
            else:
                count, ewma = prev
                new_ewma = (_EWMA_ALPHA * per_elem) + ((1.0 - _EWMA_ALPHA) * ewma)
                inner[backend] = (count + 1, new_ewma)
            self._dirty = True
        # Best-effort save; rate-limited internally.
        self.save(force=False)

    def reset(self) -> None:
        """Wipe the in-memory database and the on-disk file."""
        with self._lock:
            self._data = {}
            self._dirty = False
            self._last_save_ts = 0.0
        try:
            if os.path.exists(self._path):
                os.remove(self._path)
        except OSError as exc:
            logger.debug("quill tuning: reset unlink failed (%s)", exc)

    # ── query ────────────────────────────────────────────────────────────────

    def best_for(
        self,
        eligible_names: List[str],
        dtype_kind: str,
        n: int,
    ) -> Optional[str]:
        """Return the fastest *warmed-up* backend for this bucket, or ``None``.

        Pure exploit: ranks only the candidates that already have at least
        ``min_obs`` samples in this bucket and returns the one with the lowest
        EWMA latency per element. Under-sampled candidates are ignored (not a
        veto). Returning ``None`` tells the caller to fall back to static
        priority.

        This is retained for callers that want the measured winner *without*
        exploration. :meth:`choose` is the dispatch-facing method — it layers
        exploration on top so a newly installed backend cannot be starved.
        """
        if _truthy_env("QUILL_TUNING_DISABLED"):
            return None
        if not eligible_names:
            return None
        min_obs = int(load_config().get("tuning_min_obs", _DEFAULT_MIN_OBS))
        bucket = _bucket_for(n)
        with self._lock:
            kind_map = self._data.get(dtype_kind)
            if not kind_map:
                return None
            inner = kind_map.get(bucket)
            if not inner:
                return None
            # Rank only the warmed-up backends; let new arrivals keep
            # measuring without yanking the dispatcher's existing answer.
            candidates_with_data: List[Tuple[float, str]] = []
            for name in eligible_names:
                entry = inner.get(name)
                if entry is not None and entry[0] >= min_obs:
                    candidates_with_data.append((entry[1], name))
            if not candidates_with_data:
                return None
            return min(candidates_with_data)[1]

    def sample_count(self, backend: str, dtype_kind: str, n: int) -> int:
        """Number of observations recorded for ``backend`` in this bucket."""
        bucket = _bucket_for(n)
        with self._lock:
            inner = (self._data.get(dtype_kind) or {}).get(bucket)
            if not inner:
                return 0
            entry = inner.get(backend)
            return int(entry[0]) if entry is not None else 0

    def choose(
        self,
        eligible_names: List[str],
        dtype_kind: str,
        n: int,
    ) -> Optional[str]:
        """Explore-then-exploit backend choice for a ``(dtype, size)`` bucket.

        ``eligible_names`` MUST be in the caller's static **priority order**
        (highest priority first). The result is a backend name to run, or
        ``None`` to defer entirely to the caller's priority order (only when
        tuning is disabled or there are no candidates).

        Why this exists — the starvation bug it fixes
        =============================================
        :meth:`best_for` only ever returns a *warmed-up* winner. On a machine
        that already has timing history, installing a new — possibly much
        faster — backend used to be a dead end: the dispatcher kept selecting
        the incumbent (which had ``>= min_obs`` samples), so the newcomer never
        ran, never accumulated samples, and was never reconsidered. A user who
        `pip install`ed a 3x-faster backend saw *no* change until they wiped
        ``~/.quill/timings.json`` by hand.

        The fix is bounded exploration. While *any* eligible candidate is
        under-sampled (``< min_obs`` observations in this bucket), we route to
        the least-sampled one — ties broken toward higher static priority — so
        every backend gets measured. Total exploration cost is bounded by
        ``min_obs * n_candidates`` sorts, after which we switch to pure exploit
        (the measured fastest). This converges deterministically (no RNG, so
        benchmarks stay reproducible) and can never leave a faster backend
        permanently on the bench.
        """
        if _truthy_env("QUILL_TUNING_DISABLED"):
            return None
        if not eligible_names:
            return None
        cfg = load_config()
        min_obs = int(cfg.get("tuning_min_obs", _DEFAULT_MIN_OBS))
        abandon_factor = float(cfg.get("tuning_abandon_factor",
                                       _DEFAULT_ABANDON_FACTOR))
        bucket = _bucket_for(n)
        with self._lock:
            inner = (self._data.get(dtype_kind) or {}).get(bucket) or {}
            counts = {nm: (int(inner[nm][0]) if nm in inner else 0)
                      for nm in eligible_names}
            ewma = {nm: (float(inner[nm][1]) if nm in inner else None)
                    for nm in eligible_names}
            # Yardstick for early-abandonment: the fastest latency among
            # candidates that have at least _EXPLORE_PROBE_MIN samples — the best
            # we have actually *probed*, not only the fully-warmed ones. Using
            # the probed best (rather than requiring min_obs) lets a cheap
            # candidate — crucially the ``numpy`` floor — set the bar after just
            # a couple of samples, so a catastrophically-slow backend (e.g. a
            # parallel radix on a single core, ~5x np.sort) is abandoned after
            # ~2 probes instead of a full min_obs. That is what bounds the
            # exploration tax on low-core machines where every accelerator loses.
            probed = [ewma[nm] for nm in eligible_names
                      if counts[nm] >= _EXPLORE_PROBE_MIN and ewma[nm] is not None]
            best_probed = min(probed) if probed else None
            # EXPLORE: run any candidate that is not yet warmed up so a freshly
            # installed / never-tried backend cannot be starved by an already-
            # warm incumbent — EXCEPT ones we've probed a couple of times and
            # found far slower than the best probed backend (early-abandon), so
            # exploration doesn't tax the user with a full min_obs of slow sorts
            # per dominated backend. Ties break toward higher static priority
            # (index in eligible_names).
            under = []
            for nm in eligible_names:
                if counts[nm] >= min_obs:
                    continue
                if (best_probed is not None
                        and abandon_factor > 0.0
                        and counts[nm] >= _EXPLORE_PROBE_MIN
                        and ewma[nm] is not None
                        and ewma[nm] > abandon_factor * best_probed):
                    continue  # probed enough; clearly dominated on this machine
                under.append(nm)
            if under:
                return min(under, key=lambda nm: (counts[nm],
                                                  eligible_names.index(nm)))
            # EXPLOIT: the measured-fastest candidate — but break statistical ties
            # toward the a-priori floor. eligible_names is in priority order
            # (accelerators first, the numpy floor last), so eligible_names[-1] is
            # the lowest-overhead, most-predictable path. If that floor is within
            # `tie_factor` of the best measured latency, prefer it: a SIMD-
            # delegating backend and bare np.sort run the SAME numpy kernel on
            # AVX-512 / ASIMD hosts, so their ordering is jitter — pinning to the
            # floor is deterministic and picks the cheapest route to that kernel.
            measured = {nm: ewma[nm] for nm in eligible_names
                        if ewma[nm] is not None}
            if not measured:
                return None
            best_nm = min(measured, key=measured.get)
            tie_factor = float(cfg.get("tuning_tie_factor", _DEFAULT_TIE_FACTOR))
            floor_nm = eligible_names[-1]
            if (floor_nm != best_nm
                    and floor_nm in measured
                    and tie_factor > 0.0
                    and measured[floor_nm] <= tie_factor * measured[best_nm]):
                return floor_nm
            return best_nm

    # ── export ───────────────────────────────────────────────────────────────

    def export(self) -> Dict[str, Any]:
        """Return a JSON-friendly snapshot of the database."""
        with self._lock:
            return self._export_locked()

    def _export_locked(self) -> Dict[str, Any]:
        # Buckets become strings because JSON object keys must be strings;
        # load() reverses the cast.
        out: Dict[str, Any] = {}
        for kind, buckets in self._data.items():
            bucket_out: Dict[str, Any] = {}
            for bucket, backends in buckets.items():
                bucket_out[str(bucket)] = {
                    name: [count, ewma] for name, (count, ewma) in backends.items()
                }
            out[kind] = bucket_out
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Module-level default instance + thin wrappers
# ─────────────────────────────────────────────────────────────────────────────

DB: TimingDB = TimingDB()


def record(backend: str, dtype_kind: str, n: int, elapsed_s: float) -> None:
    """Module-level convenience wrapper around :meth:`TimingDB.record`."""
    DB.record(backend, dtype_kind, n, elapsed_s)


def best_for(
    eligible_names: List[str],
    dtype_kind: str,
    n: int,
) -> Optional[str]:
    """Module-level convenience wrapper around :meth:`TimingDB.best_for`."""
    return DB.best_for(eligible_names, dtype_kind, n)


def choose(
    eligible_names: List[str],
    dtype_kind: str,
    n: int,
) -> Optional[str]:
    """Module-level convenience wrapper around :meth:`TimingDB.choose`."""
    return DB.choose(eligible_names, dtype_kind, n)


# Honour the reset env-var *after* the DB is constructed so the import path
# is otherwise pure. We deliberately do not raise if reset fails — the only
# consequence is that stale data persists into this session.
if _truthy_env("QUILL_TUNING_RESET"):
    try:
        DB.reset()
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("quill tuning: env-driven reset failed (%s)", exc)

"""
quill/_parallel.py
------------------
Parallel sort engine.

The honest physics (measured across machines): numpy's single-threaded sort is
an AVX introsort that runs near memory bandwidth, so a general parallel sort can
only *modestly* beat it, and only on large arrays with enough cores. So the
contract here is **never lose**:

  * ``should_parallelize`` gates engagement on n AND core count (from per-machine
    config). Below the threshold we don't even build threads.
  * The kernel uses ``np.partition`` (introselect) to split the array into P
    contiguous value-ordered blocks with a single C call (no gather), then sorts
    the blocks concurrently in a ThreadPool (numpy releases the GIL). No merge.
  * If anything is marginal, the caller still has plain np.sort to fall back to.

Threads (not processes) are used on purpose: numpy releases the GIL inside
``ndarray.sort``, so threads give true parallelism with zero pickling/spawn tax —
which also means identical behaviour on Windows (spawn) and Linux (fork).

Worker count is now adaptive per ``cpu_count`` (see
:func:`_adaptive_partition_workers`); the legacy hardcoded cap can still be
overridden via the ``parallel_partition_workers`` (or legacy
``numpy_partition_workers``) key in :mod:`quill._config`.
"""

from __future__ import annotations

import atexit
import heapq
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False

from ._config import load_config

# ─────────────────────────────────────────────────────────────────────────────
# PERSISTENT THREAD POOL (cheap to create, but reuse avoids churn).
# ─────────────────────────────────────────────────────────────────────────────

_POOL: Optional[ThreadPoolExecutor] = None
_POOL_SIZE = 0
_POOL_LOCK = threading.Lock()


def _get_pool(n_workers: int) -> ThreadPoolExecutor:
    global _POOL, _POOL_SIZE
    with _POOL_LOCK:
        if _POOL is None or _POOL_SIZE < n_workers:
            if _POOL is not None:
                _POOL.shutdown(wait=False)
            _POOL = ThreadPoolExecutor(max_workers=n_workers)
            _POOL_SIZE = n_workers
        return _POOL


def _shutdown_pool() -> None:
    global _POOL
    with _POOL_LOCK:
        if _POOL is not None:
            _POOL.shutdown(wait=False)
            _POOL = None


atexit.register(_shutdown_pool)


def worker_count(ncores: Optional[int] = None) -> int:
    cfg = load_config()
    if ncores is None:
        ncores = os.cpu_count() or 1
    cap = cfg.get("parallel_max_workers", 0) or ncores
    return max(1, min(ncores, cap))


def should_parallelize(n: int, dtype_kind: str = "i") -> bool:
    """Adaptive, per-machine gate. Returns True only when parallelism is
    expected to *win* on this box for this dataset size."""
    if not _NUMPY:
        return False
    cfg = load_config()
    ncores = os.cpu_count() or 1
    if ncores < cfg.get("parallel_min_cores", 4):
        return False
    if n < cfg.get("parallel_min_n", 3_000_000):
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# ARRAY KERNEL — parallel partition sort. Operates on an ndarray in place.
# ─────────────────────────────────────────────────────────────────────────────

# Memory-bandwidth wall: the parallel np.partition sort saturates at a SMALL
# worker count (measured 1.5-1.9x at P=2-3 int64; using all cores was a 0.72x
# *regression*). Cap workers so this path never loses to np.sort.
_PARTITION_WORKERS = 3


def _adaptive_partition_workers(n: int, dtype_kind: str) -> int:
    """Return optimal worker count for parallel partition sort.

    Memory bandwidth wall: real measurements on DDR4/DDR5 show diminishing
    returns past ~3 workers per memory channel. For NUMA Xeon/EPYC boxes
    users can override via config (parallel_partition_workers).
    """
    cfg = load_config()
    override = cfg.get('parallel_partition_workers', 0) or cfg.get('numpy_partition_workers', 0)
    if override > 0:
        return override
    ncores = os.cpu_count() or 1
    # Heuristic: 1 worker per ~8 cores beyond 4, capped at 8
    if ncores <= 4:
        return min(2, ncores)
    if ncores <= 12:
        return 3
    if ncores <= 24:
        return 4
    if ncores <= 48:
        return 6
    return 8


def parallel_sort_array(arr: "np.ndarray", stable: bool = True,
                        ncores: Optional[int] = None) -> "np.ndarray":
    """
    Sort *arr* ascending in place using a parallel partition sort.
    Returns *arr*. Falls back to a plain sort when the array is too small to
    benefit (so this is always safe to call).
    """
    import time as _time
    n = arr.size
    # Adaptive worker count: respects config override but otherwise scales by
    # cpu_count rather than the old hardcoded 3-cap, so NUMA boxes can use
    # more threads where the memory subsystem can actually feed them.
    P = max(2, min(worker_count(ncores),
                   _adaptive_partition_workers(n, arr.dtype.kind)))
    # Bounded integers: counting sort beats any comparison sort, parallel or
    # not — take it before spending threads.
    if arr.dtype.kind in "iu" and n > 1 and arr.dtype.itemsize >= 8:
        from ._strategies import _counting_is_worth_it, counting_sort_array
        mn = int(arr.min()); mx = int(arr.max())
        if mx > mn and _counting_is_worth_it(n, mx - mn):
            return counting_sort_array(arr, mn, mx)
    # Value-only sort: fastest kernel == stable result (see sort_array).
    if P < 2 or n < 2 * P:
        arr.sort()
        return arr

    # Split into P contiguous value-ordered blocks with one introselect pass.
    _t0 = _time.perf_counter()
    kths = [(i * n) // P for i in range(1, P)]
    arr.partition(kths)
    bounds = [0] + kths + [n]
    slices = [(bounds[i], bounds[i + 1]) for i in range(P)
              if bounds[i + 1] > bounds[i]]

    def _sort_block(se):
        s, e = se
        arr[s:e].sort()

    pool = _get_pool(P)
    list(pool.map(_sort_block, slices))
    # Tuning telemetry: let the per-machine dispatcher learn whether the
    # parallel-partition path actually wins on this box. Errors swallowed —
    # never let measurement break a sort.
    try:
        from ._tuning import DB as _TUNING_DB
        _TUNING_DB.record('parallel_partition', arr.dtype.kind, n,
                          _time.perf_counter() - _t0)
    except BaseException:  # noqa: BLE001
        pass
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# LIST ENTRY POINT — used by the core for the parallel=True / auto path.
# ─────────────────────────────────────────────────────────────────────────────

def parallel_sort(data: list, key_fn, profile: dict,
                  identity_key: bool = False, stable: bool = True,
                  reverse: bool = False) -> None:
    """Sort *data* (a Python list) in place using all useful cores.
    Numeric identity-key data goes through the array kernel; everything else
    uses a chunked generic merge.

    When *reverse* is True the result is the ascending sort reversed via
    ``list.reverse()``. Note: this flips the relative order of equal-key
    elements, so a reverse=True parallel sort is **not stable** for ties —
    callers that need stable descending ordering should sort with a negated
    key on the ascending path instead. NaNs (floats) always sort to the end
    of the ascending result, which means they land at the *start* under
    reverse=True (matching the v6 contract in :mod:`quill._strategies`).
    """
    dtype = profile.get("dtype", "")
    numeric = dtype in ("int_pos", "int_neg", "int_mixed", "float")

    if _NUMPY and identity_key and numeric:
        arr = np.asarray(data)
        if arr.dtype.kind in "iuf":
            # NaN handling for floats: strip, sort, reappend.
            if arr.dtype.kind == "f":
                nan_mask = np.isnan(arr)
                nan_count = int(nan_mask.sum())
                if nan_count:
                    arr = arr[~nan_mask]
            else:
                nan_count = 0
            arr = parallel_sort_array(arr, stable=stable)  # may return new array
            out = arr.tolist()
            if reverse:
                out.reverse()
                # NaN-to-end contract under reverse=True means NaNs at start.
                if nan_count:
                    out = [float("nan")] * nan_count + out
            else:
                if nan_count:
                    out.extend([float("nan")] * nan_count)
            data[:] = out
            return

    _parallel_generic(data, key_fn, worker_count())
    if reverse:
        data.reverse()


def _parallel_generic(data: list, key_fn: Callable, ncores: int) -> None:
    """Chunk → sort each chunk in a thread → k-way merge. Numpy-free safe path.
    (Pure-Python comparison work is GIL-bound, so this mainly helps when the
    key function releases the GIL or for moderate sizes; it is always correct.)"""
    n = len(data)
    chunk_n = max(50_000, math.ceil(n / ncores))
    chunks = [data[i:i + chunk_n] for i in range(0, n, chunk_n)]
    if len(chunks) <= 1:
        data.sort(key=key_fn if key_fn else None)
        return

    def _sort_chunk(c):
        c.sort(key=key_fn if key_fn else None)
        return c

    pool = _get_pool(min(ncores, len(chunks)))
    sorted_chunks = list(pool.map(_sort_chunk, chunks))
    data[:] = list(heapq.merge(*sorted_chunks, key=key_fn if key_fn else None))

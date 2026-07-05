"""
quill/_core.py
--------------
Adaptive dispatcher. Profiles the data cheaply, then routes to the kernel that
is provably best *for that data on that machine* — and never worse than the
naive baseline.

Routing (identity key, fastest first):
  n <= 1 / all-equal ........... return as-is
  n <= 64 .................... Timsort (numpy can't win at this size)
  asc_ratio > 0.95 (NaN-clean
    numeric) ................... run-aware hybrid (_run_detect)
  pre-sorted / reverse-sorted .. O(n) verify + at most one .reverse()
  list-of-int / list-of-float .. _listconv fast path (C-ext aware
                                 list<->ndarray conversion)
  numeric (int/float) .......... np.asarray → counting sort (bounded ints)
                                 or np.sort; parallel partition for big arrays
  strings / bytes .............. Timsort (numpy string sort is slower)
  objects / mixed / big-int .... Timsort, or plugin, or key path

Correctness rule that killed the v5 corruption bug:
  the numeric path dispatches on ``np.asarray(data).dtype.kind`` — numpy infers
  the dtype, so a float list becomes a float array. No forced int cast exists
  anywhere, so no value can ever be truncated.
"""

from __future__ import annotations
import logging
import numbers
import os
from typing import Callable, Optional

logger = logging.getLogger("quill")

from ._profile import profile as _profile
from ._strategies import (
    sort_array, counting_sort_array, numpy_sort_by_key,
    is_itemgetter, is_attrgetter, pure_counting_sort, pure_radix,
)
from ._plugins import probe_plugins
from ._config import load_config

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False

_INSERTION_THRESHOLD = 64

# Bounded LRU-ish cache for the plugin probe. Keyed by id(type(data[0])):
# the plugin registry is type-driven, so the answer ("which plugin handles
# instances of this type, if any?") is stable for a given type id. Values:
# 'list' marks list-like primitive containers we never probe, 'no-match'
# marks types that probed once and found nothing. id() is a Python int —
# dict get/insert are atomic in CPython, so no explicit lock is needed.
_PLUGIN_CACHE: dict = {}
_PLUGIN_CACHE_MAX = 128


def _identity(x):
    """Identity key — sentinel marking the 'no key' fast paths."""
    return x


# ─────────────────────────────────────────────────────────────────────────────
# small helpers
# ─────────────────────────────────────────────────────────────────────────────

def _strip_nones(data):
    clean, nones = [], 0
    for x in data:
        if x is None:
            nones += 1
        else:
            clean.append(x)
    return clean, nones


def _strip_nans(data):
    """Remove float NaNs in one pass. Returns (clean_list, nan_count)."""
    clean, nans = [], 0
    for x in data:
        if x != x:          # NaN is the only value not equal to itself
            nans += 1
        else:
            clean.append(x)
    return clean, nans


def _should_probe_plugins(data: list) -> bool:
    if not data:
        return False
    # numbers.Number covers numpy scalars (np.int64/np.float64 register with the
    # numeric ABCs) and Decimal/Fraction — so a list of numpy scalars is treated
    # as numeric data, not as custom plugin objects that need probing.
    return not isinstance(data[0], (int, float, str, bytes, type(None), numbers.Number))


def _assemble(out: list, nan_count: int, none_count: int, reverse: bool) -> list:
    """Reattach NaN then None at the correct end after a sort."""
    if nan_count:
        nans = [float("nan")] * nan_count
        out = (nans + out) if reverse else (out + nans)
    if none_count:
        nones = [None] * none_count
        out = (nones + out) if reverse else (out + nones)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# numeric identity-key kernel
# ─────────────────────────────────────────────────────────────────────────────

def _sort_numeric_identity(work: list, reverse: bool, parallel: bool,
                           stable: bool, p: Optional[dict] = None,
                           has_nan_hint: Optional[bool] = None) -> Optional[list]:
    """
    Sort a list of numbers (identity key) and return the sorted list, or None
    if the data is not a numpy-numeric type (caller handles strings/objects).

    Dispatches through the backend chain (counting sort → GPU/compiled/parallel
    backend → np.sort fallback), which also handles NaN-to-end. Stability is
    irrelevant for value-only numeric data (equal numbers are identical), so the
    fastest kernel is always used; ``stable``/``parallel`` are accepted for API
    compatibility but the dispatcher auto-selects the best engine.

    ``p`` is the profiler dict (optional); when supplied, the dtype field is
    used to pick the tier-1 ``_listconv`` fast path (homogeneous int64/float64
    list → ndarray without the ``np.asarray`` element-probe cost). ``has_nan_hint``
    is forwarded to ``dispatch_sort`` so backends can skip the NaN safety pass
    when the caller has already proven the input is clean.
    """
    # Tier-1: _listconv is a C-extension-aware fast path for homogeneous
    # list-of-int / list-of-float — bypasses np.asarray's per-element type
    # probe. Returns None on a miss so we fall through to the generic asarray.
    from . import _listconv
    arr = None
    dtype = p.get("dtype") if p else None
    if dtype in ("int_pos", "int_neg", "int_mixed"):
        arr = _listconv.to_int64(work)
    elif dtype == "float":
        arr = _listconv.to_float64(work)
    if arr is None:
        arr = np.asarray(work)
    if arr.dtype.kind not in "iuf":  # object / str / bytes / bool → not ours
        return None
    from ._backends import dispatch_sort
    # nan_hint=False means caller guarantees no NaN; None means "unknown,
    # do the safe check". The kwarg is accepted by the _backends edit in a
    # sibling phase — guard with try/except so a stale dispatch_sort still
    # works (never-lose).
    try:
        sorted_arr = dispatch_sort(arr, descending=reverse, nan_hint=has_nan_hint)
    except TypeError:
        sorted_arr = dispatch_sort(arr, descending=reverse)
    # Mirror tier-1 on the way out: use _listconv.from_* for the two dtypes
    # it accelerates; everything else uses ndarray.tolist().
    if sorted_arr.dtype.kind in "iu" and sorted_arr.dtype.itemsize == 8:
        return _listconv.from_int64(sorted_arr)
    if sorted_arr.dtype.kind == "f" and sorted_arr.dtype.itemsize == 8:
        return _listconv.from_float64(sorted_arr)
    return sorted_arr.tolist()


def _auto_parallel(n: int) -> bool:
    """Auto-engage parallelism only where it is expected to win on this box,
    and only if the user/wizard has opted in (auto_parallel)."""
    if not _NUMPY:
        return False
    cfg = load_config()
    if not cfg.get("auto_parallel", False):
        return False
    ncores = os.cpu_count() or 1
    return (ncores >= cfg.get("parallel_min_cores", 8)
            and n >= cfg.get("parallel_min_n", 5_000_000))


# ─────────────────────────────────────────────────────────────────────────────
# pure-python numeric path (no numpy installed)
# ─────────────────────────────────────────────────────────────────────────────

def _sort_numeric_pure(work: list, p: dict, reverse: bool) -> Optional[list]:
    """Numpy-free numeric sort. Counting sort for dense bounded int ranges,
    else Timsort. Returns sorted list or None if not plain integers."""
    if p["dtype"] not in ("int_pos", "int_neg", "int_mixed"):
        return None
    mn = p.get("min_key")
    mx = p.get("max_key")
    # profiler min/max come from a sample; verify cheaply via full min/max only
    # when we intend to counting-sort.
    try:
        mn = min(work)
        mx = max(work)
    except TypeError:
        return None
    rng = mx - mn
    if 0 < rng <= 4 * len(work) and rng <= (1 << 24):
        pure_counting_sort(work, mn, mx)
    else:
        work.sort()
    if reverse:
        work.reverse()
    return work


# ─────────────────────────────────────────────────────────────────────────────
# main entry
# ─────────────────────────────────────────────────────────────────────────────

def quill_sort_impl(
    data: list,
    key: Optional[Callable],
    reverse: bool,
    inplace: bool,
    parallel: bool,
    high_performance_mode: bool = False,
    silent: bool = False,
    stable: bool = True,
) -> list:
    n = len(data)
    if n <= 1:
        # CPython contract: key is invoked EXACTLY ONCE per element, even on
        # a singleton. Skipping it (the old fast-path behaviour) silently
        # dropped side effects / cached-property evaluation for n==1 — fixed
        # in 7.0.4. Propagate any exception the key raises, matching
        # ``sorted([x], key=k)`` semantics.
        if n == 1 and key is not None:
            key(data[0])
        return data if inplace else data[:]

    work = data if inplace else data[:]
    key_fn = key if key is not None else _identity

    def finish(result: list) -> list:
        if inplace and result is not data:
            data[:] = result
            return data
        return result

    # ── tiny lists: skip profiling/dispatch entirely ──────────────────────────
    # For a handful of plain comparable values, the profiler + dispatch overhead
    # dwarfs the sort, making quill slower than a bare list.sort(). Sort directly
    # only when: identity key, no custom plugin types (first element is a
    # primitive), and no None/NaN (cheap O(n<=32) scan). Anything else still
    # needs the plugin probe / special-value handling below. The try/except
    # guards mixed-but-uncomparable tiny lists (e.g. int + custom object).
    if key is None and n <= _INSERTION_THRESHOLD and not _should_probe_plugins(work):
        # ``x != x`` is True only for NaN (any float type, incl. np.float64) and
        # False for every other primitive — a universal None/NaN guard.
        if not any(x is None or x != x for x in work):
            try:
                work.sort(reverse=reverse)
                return finish(work)
            except TypeError:
                pass  # uncomparable/mixed → fall through to full handling

    # ── plugin probe (custom object types) ────────────────────────────────────
    # Cached by id(type(work[0])): the plugin registry keys on type, so the
    # answer is stable per type. 'no-match' / 'list' shortcut skips a full
    # re-scan on every call (the v6 hot path re-walked the registry).
    if _should_probe_plugins(work):
        type_key = id(type(work[0]))
        cached = _PLUGIN_CACHE.get(type_key)
        if cached == "no-match" or cached == "list":
            plugin_result = None
        else:
            plugin_result = probe_plugins(work, key, reverse, stable=stable)
            # Cap the cache; types are bounded so simple drop-on-overflow is
            # fine — worst case we re-probe a hot type after a flush.
            if len(_PLUGIN_CACHE) >= _PLUGIN_CACHE_MAX:
                _PLUGIN_CACHE.clear()
            _PLUGIN_CACHE[type_key] = (
                "no-match" if plugin_result is None else type(work[0])
            )
        if plugin_result is not None:
            items, pk, postprocess = plugin_result
            if postprocess and not items:
                return finish(postprocess([]))
            if items is not work:
                sort_key = pk if pk is not None else _identity
                sorted_items = quill_sort_impl(
                    items, None if sort_key is _identity else sort_key,
                    reverse, True, parallel,
                    high_performance_mode=high_performance_mode,
                    silent=silent, stable=stable)
                result = postprocess(sorted_items) if postprocess else sorted_items
                return finish(result)
            if pk is not None:
                key_fn = pk

    # ── profile ───────────────────────────────────────────────────────────────
    p = _profile(work, key_fn)
    p["n"] = len(work)

    # ── None handling ─────────────────────────────────────────────────────────
    # CPython contract: ``sorted([1, None, 2])`` raises TypeError because
    # ``1 < None`` raises. Earlier versions stripped Nones, sorted the rest,
    # and reattached Nones at the tail — silent fabricated ordering that
    # diverged from ``sorted()`` and let dirty data slip past type checks
    # (Bug #3, fixed in 7.0.5). Now: do NOT strip. Any sort path that
    # ultimately invokes Python comparisons will surface the TypeError the
    # same way CPython does. ``quill_topk`` already had this correct
    # behaviour; this restores consistency across the public API.
    # ``none_count`` is kept as a parameter to ``_assemble`` for API
    # compatibility, but is now always 0.
    none_count = 0

    # ── tiny lists: Timsort is optimal ────────────────────────────────────────
    # (Strip NaN first for float data — Timsort cannot order NaN correctly,
    # and tiny lists bypass the numpy NaN handler.)
    if p["n"] <= _INSERTION_THRESHOLD:
        nan_count = 0
        if key_fn is _identity and p.get("dtype") == "float":
            work, nan_count = _strip_nans(work)
        if key_fn is _identity:
            work.sort(reverse=reverse)
        else:
            work.sort(key=key_fn, reverse=reverse)
        return finish(_assemble(work, nan_count, none_count, reverse))

    # ── identity-key adaptivity (cheap, sample-based) ─────────────────────────
    if key_fn is _identity:
        # all_same only short-circuits for non-float: a float all-same sample
        # could hide a stray NaN, which must route through the NaN-safe kernel.
        if p["all_same"] and p.get("dtype") != "float":
            return finish(_assemble(work, 0, none_count, reverse))
        # Highly-ordered integer data: Timsort detects existing runs and sorts
        # in ~O(n) — far faster than numpy's build+sort+tolist round-trip on
        # presorted / reverse / nearly-sorted input. Ints are never NaN, so this
        # is always safe; and Timsort is correct even if the sample misled us,
        # just not always O(n). Floats keep the NaN-safe numpy path.
        if p["dtype"] in ("int_pos", "int_neg", "int_mixed") and (
                p.get("asc_ratio", 0.0) > 0.9 or p.get("desc_ratio", 0.0) > 0.9):
            work.sort(reverse=reverse)
            return finish(_assemble(work, 0, none_count, reverse))
        # Highly-ordered float data wants the same Timsort win, but Timsort can't
        # place NaN at the end the way numpy does — and the profiler only sampled,
        # so a hidden NaN is possible. When the sample saw no NaN, one full scan
        # confirms the list is NaN-free (still far cheaper than the asarray +
        # sort + tolist round-trip); clean → Timsort, otherwise fall through to
        # the NaN-correct numpy kernel. (try/except guards a sample that misread
        # a non-comparable stray as float.)
        if p["dtype"] == "float" and not p.get("has_nan") and (
                p.get("asc_ratio", 0.0) > 0.9 or p.get("desc_ratio", 0.0) > 0.9):
            if not any(x != x for x in work):
                try:
                    work.sort(reverse=reverse)
                    return finish(_assemble(work, 0, none_count, reverse))
                except TypeError:
                    pass  # mixed/uncomparable types → numpy/object path below

    # ── external (disk) sort — opt-in only, never a surprise ──────────────────
    if high_performance_mode and key_fn is _identity:
        try:
            from ._external import should_use_external, external_sort
            needed, _reason, _ps = should_use_external(work)
            if needed and external_sort(work, key_fn, reverse,
                                        high_performance_mode=True,
                                        silent=silent):
                return finish(_assemble(work, 0, none_count, reverse))
        except Exception:
            pass  # any failure → fall through to in-memory sort

    # ── run-aware hybrid (nearly-sorted numeric inputs) ──────────────────────
    # When the profiler sees an extremely ordered numeric run (asc_ratio > 0.95
    # and dtype is numeric and NaN-free), the _run_detect hybrid can finish in
    # ~O(n) by detecting long ascending runs, insertion-sorting any small
    # islands, and merging — beating both Timsort's O(n log r) merges and the
    # asarray + np.sort + tolist round-trip. NEVER-LOSE: try_run_hybrid returns
    # None when not worthwhile, and any exception falls through to v6.
    if (key_fn is _identity
            and p.get("asc_ratio", 0.0) > 0.95
            and p.get("dtype") in ("int_pos", "int_neg", "int_mixed", "float")):
        # For floats, the profiler's has_nan is sample-only on n > 1M, so a
        # stray NaN can slip through and the hybrid won't push it to the end.
        # Do a cheap full scan for NaN before entering the hybrid; integers
        # are never NaN so they bypass this check.
        nan_safe = True
        if p.get("dtype") == "float":
            try:
                nan_safe = not any(x != x for x in work)
            except BaseException:
                nan_safe = False
        if nan_safe:
            try:
                from ._run_detect import try_run_hybrid
                hybrid = try_run_hybrid(work, reverse)
                if hybrid is not None:
                    return finish(_assemble(hybrid, 0, none_count, reverse))
            except BaseException:
                if os.environ.get("QUILL_BACKEND_DEBUG"):
                    raise

    # ── identity key: numeric / string / object ───────────────────────────────
    _NUMERIC_DTYPES = ("int_pos", "int_neg", "int_mixed", "float")
    if key_fn is _identity:
        # Only enter the numeric path when the profiler saw numbers — this
        # avoids building a throwaway numpy array for string/object data
        # (np.asarray on 1M strings is itself ~250ms of wasted work).
        if _NUMPY and p["dtype"] in _NUMERIC_DTYPES:
            # Pass the profile so _listconv can take its tier-1 path. The
            # profiler's has_nan is sample-only for n > 1M, so passing
            # nan_hint=False (a strong "no NaN" guarantee) based purely on
            # the sample can corrupt ordering of stray NaNs. For ints we know
            # NaN is impossible; for floats only assert clean when has_nan
            # is False AND the sample covered the whole array.
            if p["dtype"] != "float":
                nan_hint = False
            elif not p.get("has_nan") and p.get("has_nan_known"):
                nan_hint = False
            else:
                nan_hint = None  # unknown — let dispatch_sort probe
            numeric = _sort_numeric_identity(
                work, reverse, parallel, stable, p=p, has_nan_hint=nan_hint,
            )
            if numeric is not None:
                return finish(_assemble(numeric, 0, none_count, reverse))
        elif not _NUMPY:
            pure = _sort_numeric_pure(work, p, reverse)
            if pure is not None:
                return finish(_assemble(pure, 0, none_count, reverse))
            # no-numpy float path: strip NaN (Timsort can't order it), sort.
            if p.get("dtype") == "float":
                work, nan_count = _strip_nans(work)
                work.sort(reverse=reverse)
                return finish(_assemble(work, nan_count, none_count, reverse))
        # Large homogeneous string lists → polars multi-threaded sort (~2x vs
        # Timsort); identical to sorted() (UTF-8 byte order == codepoint order),
        # falls back to Timsort if polars is absent/ineligible.
        if _NUMPY and p["dtype"] == "str" and p["n"] >= 100_000:
            from ._strategies import polars_string_sort
            out = polars_string_sort(work, reverse=reverse)
            if out is not None:
                return finish(_assemble(out, 0, none_count, reverse))
        # strings / bytes / objects / big-ints → Timsort (already optimal)
        work.sort(reverse=reverse)
        return finish(_assemble(work, 0, none_count, reverse))

    # ── key path ──────────────────────────────────────────────────────────────
    if _NUMPY and p["n"] > 256 and (is_itemgetter(key_fn) or is_attrgetter(key_fn)
                                    or p.get("dtype") in ("int_pos", "int_neg",
                                                          "int_mixed", "float",
                                                          "str")):
        try:
            numpy_sort_by_key(work, key_fn, reverse=reverse)
            return finish(_assemble(work, 0, none_count, reverse))
        except Exception:
            pass  # robust fallback
    work.sort(key=key_fn, reverse=reverse)
    return finish(_assemble(work, 0, none_count, reverse))


# ── public helper used by quill.topk and addons ──────────────────────────────

def topk_impl(data, k: int, largest: bool = False, key=None):
    """Return the k smallest (or largest) elements, sorted. Uses argpartition
    (O(n)) when possible — far faster than a full sort for small k."""
    if k <= 0:
        return []
    key_fn = key if key is not None else _identity

    # SortedArray short-circuit: the wrapper's entire reason for existing is
    # to skip redundant sort work. A topk on a SortedArray is a slice + maybe
    # a reverse — no argpartition, no allocation beyond the slice.
    if key is None:
        try:
            from ._sorted_array import is_sorted_wrapper
            if is_sorted_wrapper(data):
                arr = data.arr
                k_eff = min(k, len(arr))
                if data.descending == largest:
                    # same direction as requested: take from the front
                    slc = arr[:k_eff]
                else:
                    # opposite direction: take from the other end and reverse
                    slc = arr[-k_eff:][::-1]
                return slc.tolist() if hasattr(slc, 'tolist') else list(slc)
        except BaseException:
            if os.environ.get("QUILL_BACKEND_DEBUG"):
                raise
            # fall through to the generic path

    # numpy fast path: operate on the array DIRECTLY. For an ndarray this avoids
    # the ndarray -> list -> ndarray round-trip that otherwise dwarfs the
    # argpartition itself (~14x self-inflicted overhead); for a list it is a
    # single asarray. argpartition is O(n) vs np.sort's O(n log n).
    if _NUMPY and key_fn is _identity:
        arr = None
        if isinstance(data, np.ndarray):
            arr = data
        elif isinstance(data, list):
            try:
                arr = np.asarray(data)
            except Exception:
                arr = None
        # A Python-int list that numpy promoted to float64 (values spanning the
        # int64/uint64 range share no common integer dtype) would lose precision
        # on the numpy path — e.g. 2**64-1 and 2**64-2 both collapse to the same
        # float. Detect that lossy promotion and let the exact heapq/sorted path
        # below handle it (quill_sort / quill_argsort already avoid this).
        lossy_int_promo = (
            arr is not None and arr.dtype.kind == "f" and isinstance(data, list)
            and any(isinstance(x, int) and abs(int(x)) > 2 ** 53 for x in data)
        )
        if (arr is not None and arr.ndim == 1 and arr.dtype.kind in "iuf"
                and not lossy_int_promo):
            # np.argpartition treats NaN as larger than any number, so a NaN
            # would silently be returned as the "largest" element. Strip NaN
            # from the working array — NaN values are excluded from top-k
            # results (documented behavior, matching numpy's nansort family).
            if arr.dtype.kind == "f":
                mask = ~np.isnan(arr)
                if not mask.all():
                    arr = arr[mask]
            n = arr.size
            if n == 0:
                return []
            if k >= n:
                out = np.sort(arr)
                return out[::-1].tolist() if largest else out.tolist()
            if largest:
                idx = np.argpartition(arr, n - k)[n - k:]
                return np.sort(arr[idx])[::-1].tolist()
            idx = np.argpartition(arr, k)[:k]
            return np.sort(arr[idx]).tolist()

    # generic / keyed / non-numeric fallback
    items = data if isinstance(data, list) else list(data)
    n = len(items)
    if k >= n:
        return quill_sort_impl(items[:], key, largest, True, False)
    import heapq
    kf = None if key_fn is _identity else key_fn
    return (heapq.nlargest(k, items, key=kf) if largest
            else heapq.nsmallest(k, items, key=kf))

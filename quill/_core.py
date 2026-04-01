"""
quill/_core.py
--------------
Main strategy dispatcher.

Key improvements over previous versions:
  - numpy min/max:     14x faster than Python min()/max() on large arrays
  - Single numpy pass: build array once, extract min/max from it
  - np.diff probe:     vectorised presortedness detection in 0.4ms
  - np.bincount:       vectorised counting sort replaces O(range) Python loop
  - np.lexsort:        tuple keys 1.6x faster than Python sort
  - Key negation:      stable descending matches Python's sorted(reverse=True) exactly

Reverse handling contract:
  Identity-key paths: sort ascending, caller applies single .reverse()
  Non-identity paths: pass reverse to numpy_sort_by_key (key negation)
"""

from __future__ import annotations
import os
from typing import Callable, Optional

from ._profile    import profile as _profile
from ._strategies import (
    insertion_sort, counting_sort, numpy_counting_sort,
    numpy_sort_ints, numpy_sort_floats, numpy_sort_by_key,
    pure_radix, is_itemgetter, is_attrgetter,
)
from ._plugins import probe_plugins

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False

_INSERTION_THRESHOLD   = 32
_NUMPY_THRESHOLD       = 5_000   # below this, numpy conversion overhead > benefit
_DUPLICATE_UNIQ_THRESH = 0.05    # < 5% unique values in sample → counting sort


# Narrow-range short-circuit (public-safe defaults; not machine-specific):
# use counting sort before external sort when range is small enough that
# auxiliary memory is predictably bounded.
_NARROW_RANGE_MIN_N           = 100_000
_NARROW_RANGE_MAX_RANGE       = 1_000_000
_NARROW_RANGE_TO_N_RATIO      = 8.0
_NARROW_RANGE_MAX_COUNT_BYTES = 64 * 1024 * 1024
_NARROW_RANGE_MAX_OUTPUT_COPY = 512 * 1024 * 1024


def _identity(x):
    return x


def _strip_nones(data):
    nones = sum(1 for x in data if x is None)
    clean = [x for x in data if x is not None]
    return clean, nones


def _reinsert_nones(data, none_count, reverse):
    nones = [None] * none_count
    if reverse:
        data[:] = nones + data
    else:
        data.extend(nones)


def _numpy_presorted_frac(data: list, sample: int = 2048) -> float:
    """
    Vectorised presortedness probe using np.diff.
    Samples up to `sample` elements and returns the fraction of adjacent
    pairs that are non-decreasing. Cost: ~0.4ms for 2048 elements.
    """
    n    = len(data)
    step = max(1, n // sample)
    pts  = [data[i * step] for i in range(min(sample, n))]
    try:
        arr  = np.array(pts, dtype=np.int64)
        return float(np.mean(np.diff(arr) >= 0))
    except (TypeError, ValueError):
        return 0.5


def _is_non_decreasing(data: list) -> bool:
    try:
        return all(data[i] <= data[i + 1] for i in range(len(data) - 1))
    except TypeError:
        return False


def _is_non_increasing(data: list) -> bool:
    try:
        return all(data[i] >= data[i + 1] for i in range(len(data) - 1))
    except TypeError:
        return False


def _maybe_narrow_range_short_circuit(
    data: list,
    key_fn: Callable,
    reverse: bool,
) -> bool:
    """
    Run an early in-memory counting-sort shortcut for dense integer ranges.
    This deliberately uses fixed dataset-based guards so behavior is stable
    across different user machines.
    """
    if not _NUMPY or key_fn is not _identity:
        return False

    n = len(data)
    if n < _NARROW_RANGE_MIN_N:
        return False

    try:
        arr = np.array(data, dtype=np.int64)
    except (TypeError, ValueError, OverflowError):
        return False

    if arr.size == 0:
        return False

    mn = int(arr.min())
    mx = int(arr.max())
    range_len = mx - mn + 1
    if range_len <= 0:
        return False

    # Only treat truly narrow ranges as counting-sort candidates.
    if range_len > _NARROW_RANGE_MAX_RANGE:
        return False
    if range_len > int(_NARROW_RANGE_TO_N_RATIO * n):
        return False

    count_bytes = range_len * np.dtype(np.intp).itemsize
    if count_bytes > _NARROW_RANGE_MAX_COUNT_BYTES:
        return False

    # numpy counting path currently allocates a full output copy.
    output_copy_bytes = n * np.dtype(np.int64).itemsize
    if output_copy_bytes > _NARROW_RANGE_MAX_OUTPUT_COPY:
        return False

    shifted = arr - mn
    counts = np.bincount(shifted.astype(np.intp), minlength=range_len)
    result = np.repeat(np.arange(mn, mx + 1, dtype=np.int64), counts)
    if reverse:
        result = result[::-1]
    data[:] = result.tolist()
    return True


def sort_sequential(data: list, key_fn: Callable, p: dict,
                    reverse: bool = False) -> None:
    n = p["n"]
    if n <= 1:
        return
    if n <= _INSERTION_THRESHOLD:
        data.sort() if key_fn is _identity else data.sort(key=key_fn, reverse=reverse)
        return

    if p["all_same"]:
        return

    if key_fn is _identity:
        if p["presorted"] and _is_non_decreasing(data):
            return
        if p["reversed"] and _is_non_increasing(data):
            data.reverse()
            return

    dtype = p["dtype"]

    # ── Integer fast paths (identity key) ─────────────────────────────────────
    if dtype in ("int_pos", "int_neg", "int_mixed") and key_fn is _identity:

        if _NUMPY:
            # Single-pass: build numpy array once, extract min/max from it.
            # Avoids two separate Python-level min()/max() traversals.
            a  = np.array(data, dtype=np.int64)
            mn = int(a.min())
            mx = int(a.max())
        else:
            a  = None
            mn = min(data)
            mx = max(data)

        rng = mx - mn
        if rng == 0:
            return

        # Duplicate shortcut — counting sort wins when very few unique values
        n_uniq_est = p.get("n_unique_est")
        use_counting = (
            rng <= 2 * n or
            (n_uniq_est is not None and n_uniq_est / min(n, 512) < _DUPLICATE_UNIQ_THRESH)
        )

        if use_counting:
            if _NUMPY and a is not None:
                # Vectorised np.bincount counting sort (3x faster than Python loop)
                shifted = a - mn
                counts  = np.bincount(shifted.astype(np.intp), minlength=rng + 1)
                result  = np.repeat(np.arange(mn, mx + 1, dtype=np.int64), counts)
                data[:] = result.tolist()
            else:
                counting_sort(data, mn, mx)
            return

        if _NUMPY and a is not None:
            # Reuse already-built array, cast to narrowest safe dtype.
            # Use benchmark-proven optimal sort kind per dtype:
            #   uint8/uint16 → stable (radix, 1-2 passes, 10-17x faster)
            #   int32        → default/quicksort (20x faster than radix for 4-pass)
            #   int64        → heapsort (cache-efficient, 15% faster)
            shift = mn if mn < 0 else 0
            if rng < 256:
                dtype2, kind = np.uint8,  'stable'
            elif rng < 65_536:
                dtype2, kind = np.uint16, 'stable'
            elif rng < 2**31:
                dtype2, kind, shift = np.int32, None, 0   # quicksort (default)
            else:
                dtype2, kind, shift = np.int64, 'heapsort', 0

            if shift:
                b = (a - shift).astype(dtype2)
            else:
                b = a.astype(dtype2)

            if kind:
                b.sort(kind=kind)
            else:
                b.sort()  # default = quicksort/introsort

            if shift:
                data[:] = (b.astype(np.int64) + shift).tolist()
            else:
                data[:] = b.tolist()
            return

        pure_radix(data)
        return

    # ── Float path ────────────────────────────────────────────────────────────
    if dtype == "float" and key_fn is _identity:
        if _NUMPY and n >= _NUMPY_THRESHOLD:
            numpy_sort_floats(data)
        else:
            data.sort()
        return

    # ── String/bytes ──────────────────────────────────────────────────────────
    if dtype in ("str", "bytes") and key_fn is _identity:
        data.sort()
        return

    # ── itemgetter / attrgetter fast path ─────────────────────────────────────
    if _NUMPY and (is_itemgetter(key_fn) or is_attrgetter(key_fn)):
        numpy_sort_by_key(data, key_fn, reverse=reverse)
        return

    # ── Generic object sort ───────────────────────────────────────────────────
    if n <= 256:
        data.sort(key=key_fn, reverse=reverse)
        return
    if _NUMPY:
        numpy_sort_by_key(data, key_fn, reverse=reverse)
        return
    data.sort(key=key_fn, reverse=reverse)


def _should_probe_plugins(data: list) -> bool:
    if not data:
        return False
    return not isinstance(data[0], (int, float, str, bytes, type(None)))


def quill_sort_impl(
    data                 : list,
    key                  : Optional[Callable],
    reverse              : bool,
    inplace              : bool,
    parallel             : bool,
    high_performance_mode: bool = False,
    silent               : bool = False,
) -> list:
    n = len(data)
    if n <= 1:
        return data if inplace else data[:]

    work   = data if inplace else data[:]
    key_fn = key if key is not None else _identity

    # ── Plugin probe ──────────────────────────────────────────────────────────
    if _should_probe_plugins(work):
        plugin_result = probe_plugins(work, key, reverse)
        if plugin_result is not None:
            items, pk, postprocess = plugin_result
            if postprocess and not items:
                return postprocess([])
            if items is not work:
                sort_key = pk if pk is not None else _identity
                p2 = _profile(items, sort_key)
                p2["n"] = len(items)
                sort_sequential(items, sort_key, p2, reverse=False)
                if reverse:
                    items.reverse()
                result = postprocess(items) if postprocess else items
                if inplace:
                    data[:] = result
                return result
            if pk is not None:
                key_fn = pk

    # ── None handling ─────────────────────────────────────────────────────────
    none_count = 0
    if any(x is None for x in work):
        work, none_count = _strip_nones(work)
        if not work:
            result = [None] * none_count
            if inplace: data[:] = result
            return result

    # ── Observer ──────────────────────────────────────────────────────────────
    # Profile once so all routing decisions share the same view.
    p      = _profile(work, key_fn)
    p["n"] = len(work)

    # Exact monotonic short-circuit for identity keys.
    # We only run full O(n) confirmation when the sampled profile suggests it.
    if key_fn is _identity:
        if p["all_same"]:
            if none_count:
                _reinsert_nones(work, none_count, reverse)
            if inplace and work is not data:
                data[:] = work
            return work

        if p["presorted"] and _is_non_decreasing(work):
            if reverse:
                work.reverse()
            if none_count:
                _reinsert_nones(work, none_count, reverse)
            if inplace and work is not data:
                data[:] = work
            return work

        if p["reversed"] and _is_non_increasing(work):
            if not reverse:
                work.reverse()
            if none_count:
                _reinsert_nones(work, none_count, reverse)
            if inplace and work is not data:
                data[:] = work
            return work

    # Narrow-range short-circuit (runs before external-sort decision).
    if _maybe_narrow_range_short_circuit(work, key_fn, reverse):
        if none_count:
            _reinsert_nones(work, none_count, reverse)
        if inplace and work is not data:
            data[:] = work
        return work

    from ._external import should_use_external, external_sort
    needed, reason, page_size = should_use_external(work)
    if needed:
        did_external = external_sort(
            work, key_fn, reverse,
            high_performance_mode=high_performance_mode,
            silent=silent,
        )
        if did_external:
            if none_count: _reinsert_nones(work, none_count, reverse)
            if inplace and work is not data: data[:] = work
            return work

    # ── CPU awareness ─────────────────────────────────────────────────────────
    ncores = os.cpu_count() or 1
    if not parallel and ncores >= 4 and p["n"] >= 5_000_000:
        parallel = True

    if parallel and p["n"] >= 10_000:
        from ._parallel import parallel_sort
        parallel_sort(work, key_fn, p, identity_key=(key_fn is _identity))
        if reverse: work.reverse()
    else:
        if key_fn is _identity:
            sort_sequential(work, key_fn, p, reverse=False)
            if reverse: work.reverse()
        else:
            sort_sequential(work, key_fn, p, reverse=reverse)

    if none_count:
        _reinsert_nones(work, none_count, reverse)

    if inplace and work is not data:
        data[:] = work

    return work
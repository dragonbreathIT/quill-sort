"""
quill/_strategies.py
--------------------
Sort kernels.

Design rule (v6): **dispatch on numpy's *inferred* dtype, never on a guess.**

The v5 corruption bug came from doing ``np.array(data, dtype=np.int64)`` on data
that the *sampler* believed was integer.  For a list of floats that silently
truncated every value to its integer part.  v6 builds ``np.asarray(data)`` with
*no forced dtype* and dispatches on ``arr.dtype.kind`` — numpy decides the type,
so a float list becomes a float array and can never be truncated.

Kernels
  sort_array(arr, stable)            : sort a numeric ndarray, best kernel by kind
  counting_sort_array(arr, mn, mx)   : O(n+k) counting sort for bounded ints
  numpy_sort_by_key(data, key, rev)  : key/object sort accelerated by argsort
  pure_counting_sort / pure_radix    : numpy-free fallbacks

v7 additions
  - Fused C counting sort kernel (optional _counting_ext) — branch-free
    counts+writeback for i32/i64/u32/u64 dense bounded ranges.
  - array.array zero-copy view via ``array_array_to_ndarray`` — skips the
    slow list-of-int conversion when callers already hold an array.array.
  - Bounded-int-key object sort (``counting_sort_objects_by_int_key``) —
    stable bucket sort for dense integer keys (age, priority, etc.), faster
    than argsort+permute when the key range is small relative to n.
  - ``float_to_fixed_for_counting`` helper — narrow-range float → int64
    bridge so future callers can route through the counting kernel.
"""

from __future__ import annotations
import os
from operator import itemgetter, attrgetter
from typing import Optional, Tuple

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False

_ITEMGETTER_TYPE = type(itemgetter(0))
_ATTRGETTER_TYPE = type(attrgetter('x'))

# A bounded integer range is counting-sort territory when the value range is
# not much larger than n and the count array stays cache-friendly. Counting
# sort then beats np.sort by ~2-3x (measured). These guards are value-based,
# not machine-based, so the decision is identical on every user's box.
_COUNTING_MAX_RANGE = 1 << 24            # 16.7M distinct buckets max (~64-134MB)

# Optional fused-C counting sort kernel. When present it replaces the
# bincount+repeat path for the supported dtypes (i32/i64/u32/u64); when not,
# we fall straight back to the v6 numpy implementation below — never lose.
_FUSED_COUNTING = None
try:
    from . import _counting_ext as _fc_ext
    _FUSED_COUNTING = _fc_ext
except ImportError:
    pass


def _counting_is_worth_it(n: int, rng: int) -> bool:
    """True when an integer range is dense enough that counting sort wins.

    The dispatcher pays an extra O(n) min/max to choose this path (np.sort does
    not), so counting must beat np.sort by enough to absorb that. Measured
    end-to-end (sort_array vs np.sort, int64): counting wins comfortably up to
    rng ~= n/4 and starts losing by rng ~= n/2. The bound is therefore n//4,
    which guarantees the never-lose contract (it never picks counting where it
    would be slower than np.sort); denser-but-larger ranges are left to the
    radix/np.sort backends, which are faster there anyway.
    """
    if rng <= 0:
        return True
    if rng > _COUNTING_MAX_RANGE:
        return False
    return rng <= n // 4


# ─────────────────────────────────────────────────────────────────────────────
# CORE NUMPY KERNEL — sort an already-built numeric ndarray (no NaN/None).
# ─────────────────────────────────────────────────────────────────────────────

def counting_sort_array(arr: "np.ndarray", mn: int, mx: int) -> "np.ndarray":
    """Counting sort for integer arrays with a bounded range. Returns new array.
    Avoids an int64 upcast: bincount accepts any non-negative integer array, so
    shift in the array's own dtype where the range allows.

    When the optional fused C extension is built, dispatch i32/i64/u32/u64 to
    its branch-free count+writeback kernel (one pass, no temporary arange/
    repeat). Any failure falls through to the v6 bincount+repeat path so the
    never-lose contract is preserved on every code path."""
    if _FUSED_COUNTING is not None:
        try:
            # Pick the dtype-specific entry point. The C kernel takes the raw
            # array + signed bounds (it normalises internally) and returns a
            # new ndarray of the same dtype.
            if arr.dtype == np.int64:
                return _FUSED_COUNTING.fused_counting_sort_i64(arr, int(mn), int(mx))
            if arr.dtype == np.int32:
                return _FUSED_COUNTING.fused_counting_sort_i32(arr, int(mn), int(mx))
            if arr.dtype == np.uint64:
                return _FUSED_COUNTING.fused_counting_sort_u64(arr, int(mn), int(mx))
            if arr.dtype == np.uint32:
                return _FUSED_COUNTING.fused_counting_sort_u32(arr, int(mn), int(mx))
        except BaseException:
            if os.environ.get('QUILL_BACKEND_DEBUG'):
                raise
            # fall through to v6 path
    # v6 fallback: bincount + repeat
    shifted = arr - mn if mn != 0 else arr
    counts = np.bincount(shifted, minlength=mx - mn + 1)
    return np.repeat(np.arange(mn, mx + 1, dtype=arr.dtype), counts)


def sort_array(arr: "np.ndarray", stable: bool = True) -> "np.ndarray":
    """
    Sort a numeric ndarray ascending using the best kernel for its dtype.
    Assumes no NaN/None (caller strips those). Returns a sorted ndarray
    (may be *arr* sorted in place, or a fresh array for the counting path).

    Note on stability: this sorts raw *values* with no attached payload, so a
    stable and an unstable sort produce a bit-identical result (equal numbers
    are indistinguishable). We therefore always pick the *fastest* kernel and
    ignore ``stable`` here — it only matters for keyed/object sorts elsewhere.
    """
    kind_code = arr.dtype.kind

    # Integer: counting sort for dense bounded ranges. Measured: it beats
    # np.sort by 1.7-2.8x ONLY for 8-byte integers (int64/uint64). For narrower
    # ints numpy's AVX radix is faster, so we leave those to np.sort. The Python
    # list path always builds int64, so it gets the win automatically.
    if kind_code in "iu" and arr.size > 1 and arr.dtype.itemsize >= 8:
        mn = int(arr.min())
        mx = int(arr.max())
        if mx == mn:
            return arr  # all equal — already sorted
        if _counting_is_worth_it(arr.size, mx - mn):
            return counting_sort_array(arr, mn, mx)

    # numpy's default introsort/AVX kernel is the fastest comparison sort and,
    # for value-only data, is output-identical to a stable sort.
    arr.sort()
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# NUMPY KEY / OBJECT SORT — argsort an extracted key column, then permute.
# Stability matches Python's sorted() exactly (stable ascending; negate or
# reverse-permute for descending).
# ─────────────────────────────────────────────────────────────────────────────

def numpy_sort_by_key(data: list, key_fn, reverse: bool = False) -> None:
    """Sort *data* in place by key, using numpy argsort where the key column
    is numeric/string. Falls back to Timsort when the key isn't vectorisable."""
    raw = [key_fn(x) for x in data]
    if not raw:
        return
    first = raw[0]

    # Tuple keys → lexsort (multi-column stable sort).
    if isinstance(first, tuple):
        # For reverse=True the lexsort+[::-1] trick flips tie order, which
        # breaks stability vs sorted(reverse=True). The simplest correct fix
        # is to fall back to Timsort for the descending tuple case — it
        # preserves original tie order natively.
        if reverse:
            data.sort(key=key_fn, reverse=True)
            return
        try:
            width = len(first)
            cols = [np.array([r[i] for r in raw]) for i in range(width)]
            idx = np.lexsort(cols[::-1])  # last col = primary
            data[:] = [data[i] for i in idx.tolist()]
            return
        except (TypeError, ValueError):
            data.sort(key=key_fn, reverse=reverse)
            return

    # Numeric keys → argsort the inferred-dtype array (no forced cast).
    if isinstance(first, (int, float)) and not isinstance(first, bool):
        try:
            keys = np.asarray(raw)
            if keys.dtype.kind in "iuf":
                if reverse:
                    # Negating values overflows (INT64_MIN, uint64 wrap); rank-
                    # encode and negate the BOUNDED ranks instead. Equal values
                    # share a rank so original tie order is preserved — matches
                    # sorted(reverse=True) exactly.
                    inv = np.unique(keys, return_inverse=True)[1].ravel()
                    idx = np.argsort(-inv, kind="stable")
                else:
                    idx = np.argsort(keys, kind="stable")
                data[:] = [data[i] for i in idx.tolist()]
                return
        except (TypeError, ValueError):
            pass

    # String / mixed comparable → rank-encode then argsort.
    try:
        unique_vals = sorted(set(raw))
        rank = {v: i for i, v in enumerate(unique_vals)}
        rank_keys = np.fromiter((rank[k] for k in raw), dtype=np.int64,
                                count=len(raw))
        if reverse:
            # Negate the ranks (not [::-1] the index) so equal-key ties keep
            # ORIGINAL order — matches sorted(reverse=True) stability.
            idx = np.argsort(-rank_keys, kind="stable")
        else:
            idx = np.argsort(rank_keys, kind="stable")
        data[:] = [data[i] for i in idx.tolist()]
        return
    except (TypeError, ValueError):
        pass

    data.sort(key=key_fn, reverse=reverse)


def is_itemgetter(fn) -> bool:
    return isinstance(fn, _ITEMGETTER_TYPE)


def is_attrgetter(fn) -> bool:
    return isinstance(fn, _ATTRGETTER_TYPE)


# ─────────────────────────────────────────────────────────────────────────────
# NUMPY-FREE FALLBACKS — correctness for users without numpy.
# ─────────────────────────────────────────────────────────────────────────────

def pure_counting_sort(arr: list, mn: int, mx: int) -> None:
    """In-place counting sort, pure Python. For bounded integer ranges."""
    counts = [0] * (mx - mn + 1)
    for v in arr:
        counts[v - mn] += 1
    idx = 0
    for offset, cnt in enumerate(counts):
        if cnt:
            val = offset + mn
            arr[idx:idx + cnt] = [val] * cnt
            idx += cnt


def pure_radix(arr: list) -> None:
    """LSD radix sort over bytes, pure Python. Stable. For integers."""
    n = len(arr)
    if n <= 1:
        return
    mn = min(arr)
    shift = -mn if mn < 0 else 0
    work = [v + shift for v in arr] if shift else arr[:]
    max_val = max(work) if work else 0
    if max_val == 0:
        if shift:
            arr[:] = [v - shift for v in work]
        return
    needed = max(1, (max_val.bit_length() + 7) // 8)
    buf = work
    out = [0] * n
    for p in range(needed):
        counts = [0] * 257
        sh = p * 8
        for v in buf:
            counts[((v >> sh) & 0xFF) + 1] += 1
        nonzero = sum(1 for i in range(1, 257) if counts[i])
        if nonzero == 1:
            continue
        for i in range(1, 257):
            counts[i] += counts[i - 1]
        for v in buf:
            d = (v >> sh) & 0xFF
            out[counts[d]] = v
            counts[d] += 1
        buf, out = out, buf
    arr[:] = [v - shift for v in buf] if shift else buf


# ─────────────────────────────────────────────────────────────────────────────
# ZERO-COPY array.array → ndarray VIEW + BOUNDED-INT-KEY OBJECT SORT + FLOAT
# BRIDGE. These are stand-alone helpers callers can opt into — they don't alter
# the sort_array dispatch.
# ─────────────────────────────────────────────────────────────────────────────

def array_array_to_ndarray(arr):
    """If *arr* is an array.array of a numeric type, return a numpy ndarray
    that views the same buffer (zero copy). Returns None otherwise.

    Callers holding an ``array.array`` can use this to skip the slow
    list-of-int conversion: the resulting ndarray shares memory with the
    original array.array and can be passed straight into ``sort_array`` /
    ``counting_sort_array`` without copying."""
    import array as _array
    if not isinstance(arr, _array.array):
        return None
    # array.array typecodes: b/B/h/H/i/I/l/L/q/Q/f/d. l/L are platform-sized
    # (4 bytes on Windows, 8 on most Linux/macOS) so probe itemsize at runtime.
    tc = arr.typecode
    dtype_map = {
        'b': np.int8, 'B': np.uint8, 'h': np.int16, 'H': np.uint16,
        'i': np.int32, 'I': np.uint32,
        'l': np.int32 if _array.array('l').itemsize == 4 else np.int64,
        'L': np.uint32 if _array.array('L').itemsize == 4 else np.uint64,
        'q': np.int64, 'Q': np.uint64,
        'f': np.float32, 'd': np.float64,
    }
    dt = dtype_map.get(tc)
    if dt is None:
        return None
    return np.frombuffer(arr, dtype=dt)


def counting_sort_objects_by_int_key(data: list, key_fn, mn: int, mx: int,
                                      reverse: bool = False) -> list:
    """Sort *data* (a list of objects) by a bounded-integer key.
    Faster than argsort+permute when the key range is small (e.g. age,
    priority): ~3-5x at n>=100k. STABLE.

    Algorithm: counting sort on the keys; for each value v in [mn,mx],
    collect (in order) the data items whose key == v. Skip empty values.

    Caller-side conditions where this beats argsort+permute:
      - n >= 50_000
      - range_size <= n // 4  (dense)
      - range_size <= 1<<24
    """
    n = len(data)
    range_size = mx - mn + 1
    # Build buckets: list of lists. Append preserves input order within each
    # bucket, which is what gives this routine its stability.
    buckets = [[] for _ in range(range_size)]
    for item in data:
        buckets[key_fn(item) - mn].append(item)
    out = []
    rng = range(range_size - 1, -1, -1) if reverse else range(range_size)
    for k in rng:
        out.extend(buckets[k])
    return out


def float_to_fixed_for_counting(arr: "np.ndarray", precision: float
                                ) -> Optional[Tuple["np.ndarray", int, int, float]]:
    """Try to convert a narrow-range float array to int64 for counting sort.
    Returns (int_arr, shift, max_shifted, scale) where
    ``int_arr = round((arr - mn) / precision)``.
    Returns None when the resulting range would exceed counting-sort bounds.

    Reverse mapping: ``float_val = (int_val * precision) + mn``.

    INFORMATIONAL helper — not wired into sort_array's float path yet; left
    here so future callers (e.g. a fixed-point bridge in the dispatcher) have
    a vetted conversion routine to call.
    """
    if arr.dtype.kind != 'f':
        return None
    mn = float(arr.min())
    mx = float(arr.max())
    if not (np.isfinite(mn) and np.isfinite(mx)):
        return None
    rng = mx - mn
    if rng == 0:
        return None
    n_buckets = int(rng / precision) + 1
    if n_buckets > (1 << 22):
        return None    # too sparse — counting sort would lose
    shifted = np.round((arr - mn) / precision).astype(np.int64)
    # Returned shape pairs with counting_sort_array(shifted, 0, n_buckets - 1).
    return shifted, 0, n_buckets - 1, precision


# ─────────────────────────────────────────────────────────────────────────────
# STRING-LIST SORT + GENERAL STABLE ARGSORT  (polars-accelerated, never-lose)
# A large homogeneous list of strings, and the public quill_argsort, delegate to
# polars' multi-threaded Rust sort where it wins (measured ~2x string-list,
# 4-6x argsort) and otherwise fall back to the EXACT sorted()/np.argsort result.
# ─────────────────────────────────────────────────────────────────────────────

_POLARS_STR_MIN_N = 100_000   # below this, Timsort wins (polars setup overhead)


def _polars_available() -> bool:
    try:
        import polars  # noqa: F401
        return True
    except Exception:
        return False


def polars_string_sort(data: list, reverse: bool = False):
    """Sort a Python list of str via polars. Returns a NEW list == sorted(data),
    or None to signal 'not handled' (caller keeps Timsort). polars sorts UTF-8
    bytewise, which equals Python codepoint order for every valid string
    (stress-verified over random unicode incl. astral characters).

    If the list contains ANY None, this returns None (declines) so the Timsort
    fallback raises the same TypeError sorted() raises on a None+str mix — matching
    the 7.0.5 contract, rather than letting polars order the None as a SQL null."""
    if not _polars_available():
        return None
    n = len(data)
    if n < _POLARS_STR_MIN_N:
        return None
    step = max(1, n // 256)            # cheap homogeneity gate on a sample
    saw = False
    for x in data[::step]:
        if x is None:
            continue
        if type(x) is not str:
            return None
        saw = True
    if not saw:
        return None
    try:
        import polars as pl
        s = pl.Series("v", data, dtype=pl.Utf8)
        # The sampled gate above can miss a None outside the sample; check the
        # built series directly. Any null (a None, or a value polars couldn't
        # represent as Utf8) → decline so Timsort/sorted() raises TypeError.
        if s.null_count():
            return None
        return s.sort(descending=reverse, nulls_last=not reverse).to_list()
    except Exception:
        return None


def _is_nan(v) -> bool:
    # ``x != x`` is True only for NaN (any float type). Kept separate from the
    # None check so the two sentinel groups can be reattached in a stable,
    # documented order (values, then NaN, then None — see _argsort_python).
    return type(v) is float and v != v


def argsort_general(data, key=None, reverse: bool = False) -> list:
    """STABLE permutation indices matching ``sorted(range(n), key=...)`` EXACTLY,
    including equal-key tie order; None/NaN sort to END (FRONT if reverse).

    reverse matches ``sorted(reverse=True)``: ties keep ORIGINAL order (the
    comparison is reversed, not the output) — so reverse is a native descending
    stable sort, NEVER ascending[::-1]."""
    if _NUMPY and isinstance(data, np.ndarray) and data.ndim != 1:
        raise ValueError(
            f"quill_argsort requires 1-D input; got ndim={data.ndim}")
    n = len(data)
    if n == 0:
        return []
    if key is None and _NUMPY and isinstance(data, np.ndarray) \
            and data.ndim == 1 and data.dtype.kind in "iuf":
        r = _argsort_ndarray(data, reverse)
        if r is not None:
            return r
    col = ((data if isinstance(data, list) else list(data)) if key is None
           else [key(x) for x in data])
    r = _argsort_polars_col(col, reverse)
    if r is not None:
        return r
    return _argsort_python(col, reverse)


def _argsort_ndarray(arr, reverse):
    has_nan = arr.dtype.kind == "f" and bool(np.isnan(arr).any())
    if has_nan:
        m = np.isnan(arr)
        nn = np.flatnonzero(~m)
        nan = np.flatnonzero(m)
        if reverse:
            order = nn[np.argsort(-arr[nn], kind="stable")]   # float negate: safe
            return np.concatenate([nan, order]).tolist()
        order = nn[np.argsort(arr[nn], kind="stable")]
        return np.concatenate([order, nan]).tolist()
    if reverse:
        # Descending matching sorted(reverse=True): equal values keep ORIGINAL
        # order. Negating the VALUES overflows (INT64_MIN negates to itself;
        # uint64 >= 2^63 wraps), so rank-encode to BOUNDED ints via np.unique's
        # inverse and stable-argsort the negated RANKS — overflow-free, and
        # equal values share a rank so their original order is preserved.
        inv = np.unique(arr, return_inverse=True)[1].ravel()
        return np.argsort(-inv, kind="stable").tolist()
    return np.argsort(arr, kind="stable").tolist()


def _argsort_polars_col(col, reverse):
    if not _polars_available():
        return None
    n = len(col)
    if n < _POLARS_STR_MIN_N:
        return None
    step = max(1, n // 256)
    saw_str = saw_num = False
    for x in col[::step]:
        if x is None:
            continue
        t = type(x)
        if t is str:
            saw_str = True
        elif (t is int or t is float) and t is not bool:
            saw_num = True
        else:
            return None
    if saw_str == saw_num:            # mixed or empty sample → exact python path
        return None
    try:
        import polars as pl
        if saw_num:
            if any((type(v) is float and v != v) for v in col):
                return None           # NaN → exact python path
            ser = pl.Series("v", col)
        else:
            ser = pl.Series("v", col, dtype=pl.Utf8)
        # A None among comparable values makes sorted() raise TypeError; the
        # sampled gate can miss one outside the sample, so check the series' null
        # count and decline (→ _argsort_python raises) rather than ordering the
        # None as a null. Mirrors the polars_string_sort / 7.0.5 contract fix.
        if ser.null_count():
            return None
        df = pl.DataFrame({"v": ser}).with_row_index("__i")
        return df.sort("v", descending=reverse, nulls_last=not reverse,
                       maintain_order=True)["__i"].to_list()
    except Exception:
        return None


def _argsort_python(col, reverse):
    # Three-way partition so None and NaN both sort to the END (to the FRONT on
    # reverse), matching the documented contract AND the list path's _assemble
    # ordering (values, then NaN, then None ascending; the mirror on reverse).
    # Both sentinel groups keep their original relative order → the permutation
    # is stable. Everything else goes through the normal comparison, so a genuinely
    # uncomparable mix (e.g. int + str) still raises TypeError exactly like sorted().
    normal, nans, nones = [], [], []
    for i, x in enumerate(col):
        if x is None:
            nones.append(i)
        elif _is_nan(x):
            nans.append(i)
        else:
            normal.append(i)
    order = sorted(normal, key=lambda i: col[i], reverse=reverse)
    if reverse:
        return nones + nans + order
    return order + nans + nones

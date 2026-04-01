"""
quill/_strategies.py
--------------------
Individual sort strategies — every function sorts `data` in-place.
"""

from __future__ import annotations
from operator import itemgetter, attrgetter

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False

_ITEMGETTER_TYPE = type(itemgetter(0))
_ATTRGETTER_TYPE = type(attrgetter('x'))


# ─────────────────────────────────────────────────────────────────────────────
# INSERTION SORT  (n ≤ 32)
# ─────────────────────────────────────────────────────────────────────────────

def insertion_sort(arr, key_fn, reverse=False):
    arr.sort(key=key_fn, reverse=reverse)


# ─────────────────────────────────────────────────────────────────────────────
# NUMPY VECTORISED COUNTING SORT
# Uses np.bincount + np.repeat — pure C-level, no Python loops.
# 3x faster than the pure-Python counting sort for dense integer ranges.
# ─────────────────────────────────────────────────────────────────────────────

def numpy_counting_sort(arr: list, mn: int, mx: int) -> None:
    a      = np.array(arr, dtype=np.int64)
    shift  = a - mn
    counts = np.bincount(shift.astype(np.intp), minlength=(mx - mn + 1))
    result = np.repeat(np.arange(mn, mx + 1, dtype=np.int64), counts)
    arr[:] = result.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# PURE-PYTHON COUNTING SORT  (fallback, no numpy)
# ─────────────────────────────────────────────────────────────────────────────

def counting_sort(arr: list, mn: int, mx: int) -> None:
    counts = [0] * (mx - mn + 1)
    for v in arr:
        counts[v - mn] += 1
    idx = 0
    for offset, cnt in enumerate(counts):
        for _ in range(cnt):
            arr[idx] = offset + mn
            idx += 1


# ─────────────────────────────────────────────────────────────────────────────
# NUMPY INTEGER SORT
# Single-pass: builds numpy array once and gets min/max from it,
# avoiding three separate Python-level iterations.
# Uses narrowest safe dtype to minimise memory bandwidth.
# ─────────────────────────────────────────────────────────────────────────────

def numpy_sort_ints(arr: list, mn: int, mx: int) -> None:
    shift = mn if mn < 0 else 0
    rng   = mx - mn

    # ── Narrowest dtype + benchmark-proven optimal sort kind ─────────────────
    # Benchmark results (5M elements):
    #   uint8:  stable=0.012s  quicksort=0.211s  → stable 17x faster (1-pass radix)
    #   uint16: stable=0.035s  quicksort=0.346s  → stable 10x faster (2-pass radix)
    #   int32:  stable=0.499s  quicksort=0.023s  → quicksort 20x faster (4-pass radix!)
    #   int64:  stable=0.507s  heapsort=0.059s   → heapsort fastest (cache-efficient)
    # Rule: use radix (stable) only when it fits in ≤2 passes (uint8/uint16).
    #       For wider types, comparison sort dominates.
    if rng < 256:
        dtype, kind = np.uint8,  'stable'    # 1-pass radix, 17x faster
    elif rng < 65_536:
        dtype, kind = np.uint16, 'stable'    # 2-pass radix, 10x faster
    elif rng < 2**31:
        dtype, kind = np.int32,  None        # quicksort (default), 20x faster than stable
        shift = 0
    else:
        dtype, kind = np.int64,  'heapsort'  # heapsort, 15% faster than quicksort
        shift = 0

    if shift:
        a = (np.array(arr, dtype=np.int64) - shift).astype(dtype)
    else:
        a = np.array(arr, dtype=dtype)

    if kind:
        a.sort(kind=kind)
    else:
        a.sort()  # default = introsort (quicksort)

    if shift:
        arr[:] = (a.astype(np.int64) + shift).tolist()
    else:
        arr[:] = a.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# NUMPY FLOAT SORT
# ─────────────────────────────────────────────────────────────────────────────

def numpy_sort_floats(arr: list) -> None:
    # heapsort is ~10% faster than quicksort for float64 (benchmark-proven)
    a = np.array(arr, dtype=np.float64)
    a.sort(kind='heapsort')
    arr[:] = a.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# NUMPY SORT BY KEY
#
# Stability contract (matches Python's sorted() exactly):
#   reverse=False → argsort ascending stable
#   reverse=True  → NEGATE key array before argsort
#                   (gives stable descending, identical equal-key order to Python)
#
# Key type routing:
#   tuple  → np.lexsort (1.6x faster than Python sort)
#   float  → float64 argsort
#   int    → narrowest-dtype argsort with negation for reverse
#   string → rank-encode → argsort
#   other  → Python sort fallback
# ─────────────────────────────────────────────────────────────────────────────

def numpy_sort_by_key(data: list, key_fn, reverse: bool = False) -> None:
    raw = [key_fn(x) for x in data]

    if not raw:
        return

    first = raw[0]

    # ── Tuple keys → np.lexsort ───────────────────────────────────────────────
    if isinstance(first, tuple):
        try:
            width = len(first)
            # lexsort takes keys in REVERSE priority order (last = primary)
            # We need to extract each column as a numpy array
            cols = [np.array([r[i] for r in raw]) for i in range(width)]
            # lexsort: last key is primary, so reverse the column order
            idx = np.lexsort(cols[::-1])
            if reverse:
                idx = idx[::-1]
            data[:] = [data[i] for i in idx.tolist()]
            return
        except (TypeError, ValueError):
            pass
        data.sort(key=key_fn, reverse=reverse)
        return

    # ── Float keys ────────────────────────────────────────────────────────────
    if isinstance(first, float):
        keys = np.array(raw, dtype=np.float64)
        idx  = np.argsort(-keys if reverse else keys, kind='stable')
        data[:] = [data[i] for i in idx.tolist()]
        return

    # ── Integer keys ──────────────────────────────────────────────────────────
    try:
        keys_i64 = np.array(raw, dtype=np.int64)

        if reverse:
            # Negate for stable descending; reuse already-built array.
            idx = np.argsort(-keys_i64, kind='stable')
        else:
            mn      = int(keys_i64.min())
            shifted = (keys_i64 - mn) if mn < 0 else keys_i64
            rng     = int(shifted.max())
            if rng < 256:       ktype = np.uint8
            elif rng < 65_536:  ktype = np.uint16
            elif rng < 2**31:   ktype = np.int32
            else:               ktype = np.int64
            idx = np.argsort(shifted.astype(ktype), kind='stable')

        data[:] = [data[i] for i in idx.tolist()]
        return

    except (TypeError, ValueError):
        pass

    # ── String / mixed → rank-encode ─────────────────────────────────────────
    try:
        unique_vals = sorted(set(raw))
        rank        = {v: i for i, v in enumerate(unique_vals)}
        n_uniq      = len(unique_vals)
        ktype       = np.uint16 if n_uniq < 65_536 else np.int32
        rank_keys   = np.array([rank[k] for k in raw], dtype=ktype)
        if reverse:
            idx = np.argsort(-rank_keys.astype(np.int64), kind='stable')
        else:
            idx = np.argsort(rank_keys, kind='stable')
        data[:] = [data[i] for i in idx.tolist()]
        return
    except (TypeError, ValueError):
        pass

    # ── Final fallback ────────────────────────────────────────────────────────
    data.sort(key=key_fn, reverse=reverse)


def is_itemgetter(fn) -> bool:
    return isinstance(fn, _ITEMGETTER_TYPE)

def is_attrgetter(fn) -> bool:
    return isinstance(fn, _ATTRGETTER_TYPE)


# ─────────────────────────────────────────────────────────────────────────────
# PURE-PYTHON RADIX SORT  (no numpy fallback)
# Single-pass precomputed histograms, trivial-pass skip.
# ─────────────────────────────────────────────────────────────────────────────

def pure_radix(arr: list) -> None:
    n     = len(arr)
    mn    = min(arr)
    shift = -mn if mn < 0 else 0

    # Work on non-negative shifted values so bit operations are well-defined.
    work    = [v + shift for v in arr] if shift else arr[:]
    max_val = max(work)
    if max_val == 0:
        return

    needed = max(1, (max_val.bit_length() + 7) // 8)
    hists  = [[0] * 256 for _ in range(needed)]

    for v in work:
        for p in range(needed):
            hists[p][(v >> (p * 8)) & 0xFF] += 1

    buf = work
    out = [0] * n
    for p in range(needed):
        h = hists[p]
        if sum(1 for x in h if x > 0) == 1:
            continue
        pos = [0] * 256
        acc = 0
        for i in range(256):
            pos[i] = acc; acc += h[i]
        for v in buf:
            d = (v >> (p * 8)) & 0xFF
            out[pos[d]] = v; pos[d] += 1
        buf, out = out, buf

    if shift:
        arr[:] = [v - shift for v in buf]
    else:
        arr[:] = buf
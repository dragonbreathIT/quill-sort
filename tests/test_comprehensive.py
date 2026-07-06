"""
tests/test_comprehensive.py
===========================
Quill's exhaustive correctness gate — the first thing CI runs on every commit,
before anything else is allowed to proceed.

The design is *matrix-driven*: a handful of parametrized test functions expand,
via pytest parametrization, into 1000+ individual cases that sweep every dtype,
size class, data distribution, direction, and API surface Quill exposes. Every
case checks Quill's output against the ground-truth oracle for that input —
``numpy.sort`` for arrays, the built-in ``sorted`` for lists — because Quill's
one non-negotiable contract is *the result is never wrong*.

Coverage:
  * ``sort_array`` vs ``np.sort`` — 10 dtypes × 11 sizes × 6 distributions ×
    {ascending, descending}, plus the in-place variants.
  * float NaN / ±inf handling (NaN to the end; to the start when descending).
  * ``quill_sort`` / ``quill_sorted`` (list API) vs ``sorted`` — ints, floats,
    strings, bytes, mixed int/float, with ``key=`` / ``reverse=`` / ``stable=``.
  * ``None`` to the end (start when reversed); stability of equal keys.
  * ``quill_topk`` vs ``sorted(...)[:k]``; ``quill_argsort`` vs the stable
    permutation.
  * every *available* backend, forced, vs ``np.sort``.
  * structural edge cases: empty / singleton / 0-dim / 2-dim / non-contiguous /
    read-only / object dtype.
  * ``analyze`` / ``available_backends`` API invariants.
  * property-based fuzzing (hypothesis) against ``sorted`` / ``np.sort``.

Run: ``pytest tests/test_comprehensive.py``   (add ``-m slow`` for the big-N tier).
"""
from __future__ import annotations

import math
import sys

import pytest

np = pytest.importorskip("numpy")
import quill
from quill import _backends as _B


# ─────────────────────────────────────────────────────────────────────────────
# Matrices
# ─────────────────────────────────────────────────────────────────────────────

INT_DTYPES = ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]
FLOAT_DTYPES = ["float32", "float64"]
NUM_DTYPES = INT_DTYPES + FLOAT_DTYPES

# Sizes chosen to straddle every internal threshold: the <=8192 super-fast path,
# the 200k small-array floor, and the backend crossovers just past it.
SIZES = [0, 1, 2, 7, 64, 255, 1000, 8192, 8193, 50_000, 200_001]

DISTS = ["uniform", "bounded", "presorted", "reversed", "constant", "dups"]


def _clamp_n_for_dtype(dtype, n, dist):
    """A presorted/reversed run of distinct values can't exceed a dtype's range
    (uint8 tops out at 255). Fall back to a within-range span for those cases."""
    info = np.iinfo(dtype) if np.dtype(dtype).kind in "iu" else None
    if info is not None and dist in ("presorted", "reversed") and n > 0:
        span = int(info.max) - int(info.min)
        if n - 1 > span:
            return "wrap"
    return None


def gen(dtype, n, dist, seed=0):
    """Deterministic data generator for (dtype, n, distribution)."""
    dtype = np.dtype(dtype)
    rng = np.random.default_rng(seed)
    if n == 0:
        return np.empty(0, dtype=dtype)

    if dtype.kind in "iu":
        info = np.iinfo(dtype)
        lo, hi = int(info.min), int(info.max)
        # keep a comfortable value window so bincount/counting paths stay sane
        wlo = max(lo, -(2 ** 40))
        whi = min(hi, 2 ** 40)
        if dist == "uniform":
            return rng.integers(wlo, whi, n, dtype=dtype, endpoint=False) if whi > wlo \
                else np.full(n, wlo, dtype=dtype)
        if dist == "bounded":
            return rng.integers(0, min(50, hi), n, dtype=dtype, endpoint=True)
        if dist == "constant":
            return np.full(n, min(7, hi), dtype=dtype)
        if dist == "dups":
            return rng.integers(0, min(10, hi), n, dtype=dtype, endpoint=True)
        if dist in ("presorted", "reversed"):
            if _clamp_n_for_dtype(dtype, n, dist) == "wrap":
                base = (np.arange(n) % (min(hi, 2 ** 20) + 1)).astype(dtype)
                base = np.sort(base)
            else:
                start = 0 if lo >= 0 else max(lo, -(n // 2))
                base = np.arange(start, start + n, dtype=dtype)
            return base if dist == "presorted" else base[::-1].copy()
    else:  # float
        if dist == "uniform":
            return (rng.random(n) * 2 ** 20 - 2 ** 19).astype(dtype)
        if dist == "bounded":
            return rng.integers(0, 50, n).astype(dtype)
        if dist == "constant":
            return np.full(n, 3.5, dtype=dtype)
        if dist == "dups":
            return (rng.integers(0, 10, n)).astype(dtype)
        if dist in ("presorted", "reversed"):
            base = np.sort((rng.random(n) * 2 ** 20).astype(dtype))
            return base if dist == "presorted" else base[::-1].copy()
    raise AssertionError((dtype, dist))


def eq(got, expected):
    """Exact array equality, NaN-aware for floats."""
    got = np.asarray(got)
    expected = np.asarray(expected)
    if got.shape != expected.shape:
        return False
    if got.dtype.kind == "f" or expected.dtype.kind == "f":
        return np.array_equal(got, expected, equal_nan=True)
    return np.array_equal(got, expected)


def oracle(a, descending=False):
    out = np.sort(a)
    return out[::-1] if descending else out


# ─────────────────────────────────────────────────────────────────────────────
# 1. sort_array vs np.sort — the big matrix (~1300 cases)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dtype", NUM_DTYPES)
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("dist", DISTS)
@pytest.mark.parametrize("descending", [False, True])
def test_sort_array_matches_numpy(dtype, n, dist, descending):
    a = gen(dtype, n, dist)
    expected = oracle(a, descending)
    got = quill.sort_array(a.copy(), descending=descending)
    assert eq(got, expected), f"{dtype} n={n} {dist} desc={descending}"
    assert np.asarray(got).dtype == a.dtype
    # non-mutating default: the input is untouched
    a2 = gen(dtype, n, dist)
    _ = quill.sort_array(a2)
    assert eq(a2, gen(dtype, n, dist)), "sort_array(copy) mutated its input"


@pytest.mark.parametrize("dtype", NUM_DTYPES)
@pytest.mark.parametrize("n", [0, 1, 2, 1000, 8193, 200_001])
@pytest.mark.parametrize("dist", ["uniform", "bounded", "reversed"])
@pytest.mark.parametrize("descending", [False, True])
def test_sort_array_inplace(dtype, n, dist, descending):
    a = gen(dtype, n, dist)
    expected = oracle(a, descending)
    buf = a.copy()
    out = quill.sort_array(buf, inplace=True, descending=descending)
    assert eq(out, expected)
    # in-place: the buffer we passed now holds the sorted result
    assert eq(buf, expected)


# ─────────────────────────────────────────────────────────────────────────────
# 2. float NaN / inf handling
# ─────────────────────────────────────────────────────────────────────────────

def _with_specials(dtype, n, kind, seed=1):
    rng = np.random.default_rng(seed)
    a = (rng.random(n) * 2 ** 20 - 2 ** 19).astype(dtype)
    if kind == "one_nan":
        a[n // 2] = np.nan
    elif kind == "some_nan":
        a[rng.integers(0, n, max(1, n // 20))] = np.nan
    elif kind == "all_nan":
        a[:] = np.nan
    elif kind == "inf":
        a[rng.integers(0, n, max(1, n // 50))] = np.inf
        a[rng.integers(0, n, max(1, n // 50))] = -np.inf
        a[n // 3] = np.nan
    elif kind == "neg_zero":
        a[: n // 2] = -0.0
        a[n // 2:] = 0.0
    return a


@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("n", [1, 2, 1000, 8193, 200_001])
@pytest.mark.parametrize("kind", ["one_nan", "some_nan", "all_nan", "inf", "neg_zero"])
@pytest.mark.parametrize("descending", [False, True])
def test_sort_array_nan_inf(dtype, n, kind, descending):
    a = _with_specials(dtype, n, kind)
    expected = oracle(a, descending)  # np.sort: NaN to end; [::-1] -> NaN to start
    got = quill.sort_array(a.copy(), descending=descending)
    assert eq(got, expected), f"{dtype} n={n} {kind} desc={descending}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. quill_sort / quill_sorted (list API) vs sorted()
# ─────────────────────────────────────────────────────────────────────────────

def _py_list(kind, n, seed=3):
    rng = np.random.default_rng(seed)
    if kind == "int":
        return [int(x) for x in rng.integers(-(10 ** 6), 10 ** 6, n)]
    if kind == "int_neg":
        return [int(x) for x in rng.integers(-(10 ** 6), 0, n)]
    if kind == "float":
        return [float(x) for x in rng.random(n) * 1000 - 500]
    if kind == "mixed":
        out = []
        for x in rng.random(n):
            out.append(int(x * 100) if x < 0.5 else float(x * 100))
        return out
    if kind == "str":
        pool = ["apple", "Banana", "cherry", "date", "", "aa", "Aa", "z", "10", "2"]
        return [pool[int(x)] for x in rng.integers(0, len(pool), n)]
    if kind == "bytes":
        return [bytes([int(x)]) for x in rng.integers(0, 256, n)]
    if kind == "dups":
        return [int(x) for x in rng.integers(0, 5, n)]
    raise AssertionError(kind)


@pytest.mark.parametrize("kind", ["int", "int_neg", "float", "str", "bytes", "dups"])
@pytest.mark.parametrize("n", [0, 1, 2, 5, 100, 1000, 5000])
@pytest.mark.parametrize("reverse", [False, True])
def test_quill_sorted_matches_sorted(kind, n, reverse):
    data = _py_list(kind, n)
    expected = sorted(data, reverse=reverse)
    got = quill.quill_sorted(data, reverse=reverse)
    assert got == expected, f"{kind} n={n} rev={reverse}"
    # quill_sorted must not mutate its input
    assert data == _py_list(kind, n)


@pytest.mark.parametrize("kind", ["int", "float", "str", "dups"])
@pytest.mark.parametrize("n", [0, 1, 10, 1000])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("stable", [True, False])
def test_quill_sort_inplace_matches_sorted(kind, n, reverse, stable):
    data = _py_list(kind, n)
    expected = sorted(data, reverse=reverse)
    work = list(data)
    out = quill.quill_sort(work, reverse=reverse, stable=stable)
    # value-multiset + order both correct (stable=False still yields a valid sort)
    assert out == expected, f"{kind} n={n} rev={reverse} stable={stable}"
    assert work == expected, "quill_sort(inplace default) must mutate in place"


@pytest.mark.parametrize("n", [0, 1, 5, 100, 2000])
@pytest.mark.parametrize("reverse", [False, True])
def test_quill_sort_key_is_stable(n, reverse):
    # tuples (key, original_index); sort by key must preserve index order on ties
    rng = np.random.default_rng(n + reverse)
    data = [{"k": int(x), "i": i} for i, x in enumerate(rng.integers(0, 5, n))]
    expected = sorted(data, key=lambda r: r["k"], reverse=reverse)
    got = quill.quill_sorted(data, key=lambda r: r["k"], reverse=reverse)
    assert got == expected


# ─────────────────────────────────────────────────────────────────────────────
# 4. None handling — the DOCUMENTED Quill contract: None sorts to the END (to the
#    FRONT on reverse), exactly like NaN. This is a deliberate, useful divergence
#    from sorted(), which raises TypeError on a None+value mix. Real-world data has
#    holes; cleanly collecting them at one end beats a crash — the same bargain we
#    already make for NaN. Removed in 7.0.5 (over-correction of a real polars-null
#    bug), RESTORED in 7.5.2 to honour the README contract users rely on.
#    Genuinely-uncomparable data (no None, e.g. int+str) STILL raises like sorted().
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n_none", [0, 1, 3, 10])
@pytest.mark.parametrize("n_val", [0, 1, 50, 500])
@pytest.mark.parametrize("reverse", [False, True])
def test_none_sorts_to_end(n_none, n_val, reverse):
    rng = np.random.default_rng(n_none * 100 + n_val + reverse)
    vals = [int(x) for x in rng.integers(-1000, 1000, n_val)]
    data = vals + [None] * n_none
    rng.shuffle(data)
    clean = sorted(vals, reverse=reverse)
    # None to the end (ascending) / the front (reverse) — like NaN.
    expected = ([None] * n_none + clean) if reverse else (clean + [None] * n_none)
    assert quill.quill_sorted(data, reverse=reverse) == expected, \
        f"none={n_none} val={n_val} rev={reverse}"
    # quill_sort in-place must land on the same order.
    inplace = list(data)
    quill.quill_sort(inplace, reverse=reverse)
    assert inplace == expected
    # applying the argsort permutation reproduces the same values.
    idx = quill.quill_argsort(data, reverse=reverse)
    assert [data[i] for i in idx] == expected


def test_none_plus_uncomparable_still_raises():
    # None strips to the end, but a genuinely uncomparable REMAINDER (int+str)
    # must still raise TypeError exactly like sorted() — None-to-end must never
    # mask dirty, truly-uncomparable data.
    for bad in ([1, "a", None], [None, 1, "a", 2] * 40):
        with pytest.raises(TypeError):
            quill.quill_sorted(bad)
        with pytest.raises(TypeError):
            quill.quill_argsort(bad)


def test_none_with_nan_both_to_end():
    # NaN and None both sort to the end, NaN before None (ascending); mirrored on
    # reverse. NaN is compared with `is`/`!=` since nan != nan.
    import math
    data = [3.0, None, math.nan, 1.0, None, 2.0]
    out = quill.quill_sorted(data)
    assert out[:3] == [1.0, 2.0, 3.0]
    assert math.isnan(out[3]) and out[4:] == [None, None]
    rout = quill.quill_sorted(data, reverse=True)
    assert rout[:2] == [None, None] and math.isnan(rout[2])
    assert rout[3:] == [3.0, 2.0, 1.0]


# ─────────────────────────────────────────────────────────────────────────────
# 5. quill_topk vs sorted(...)[:k]
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [1, 2, 10, 1000, 50_000])
@pytest.mark.parametrize("k", [1, 2, 5, 10, 100])
@pytest.mark.parametrize("largest", [False, True])
def test_topk_matches_sorted_slice(n, k, largest):
    rng = np.random.default_rng(n + k + largest)
    data = [int(x) for x in rng.integers(-(10 ** 6), 10 ** 6, n)]
    kk = min(k, n)
    got = quill.quill_topk(data, kk, largest=largest)
    expected = sorted(data, reverse=largest)[:kk]
    assert got == expected, f"n={n} k={k} largest={largest}"


@pytest.mark.parametrize("largest", [False, True])
def test_topk_excludes_none(largest):
    # None is EXCLUDED from top-k, exactly like NaN — top-k asks for the k
    # smallest/largest REAL values, so a hole in the data is skipped, never
    # returned as an extreme and never crashing heapq.
    data = [3, None, 1, None, 5, 2, None, 4]
    clean = sorted((x for x in data if x is not None), reverse=largest)
    assert quill.quill_topk(data, 3, largest=largest) == clean[:3]
    # k larger than the real-value count → just the real values, sorted.
    assert quill.quill_topk(data, 99, largest=largest) == clean


# ─────────────────────────────────────────────────────────────────────────────
# 6. quill_argsort vs the stable permutation
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [0, 1, 2, 10, 1000, 20_000])
@pytest.mark.parametrize("reverse", [False, True])
def test_argsort_is_the_stable_permutation(n, reverse):
    rng = np.random.default_rng(n * 7 + reverse)
    data = [int(x) for x in rng.integers(0, 20, n)]  # many ties → exercises stability
    idx = quill.quill_argsort(data, reverse=reverse)
    expected = sorted(range(n), key=lambda i: data[i], reverse=reverse)
    # exact tie order must match sorted()'s stable permutation
    assert idx == expected, f"n={n} rev={reverse}"
    # applying the permutation reproduces the sorted values
    assert [data[i] for i in idx] == sorted(data, reverse=reverse)


# ─────────────────────────────────────────────────────────────────────────────
# 7. every AVAILABLE backend, forced, vs np.sort
# ─────────────────────────────────────────────────────────────────────────────

def _forced(dtype, n, dist, backend, descending=False):
    a = gen(dtype, n, dist)
    return quill._backends.dispatch_sort(a.copy(), descending=descending, force=backend), oracle(a, descending)


ALL_BACKEND_NAMES = ["numpy", "numpy_parallel", "x86_simd_sort", "arm_neon_sort",
                     "polars", "rust_voracious", "rust_parallel_radix", "ips4o",
                     "simd_companion", "numa", "spectre", "cupy_gpu"]


@pytest.mark.parametrize("backend", ALL_BACKEND_NAMES)
@pytest.mark.parametrize("dtype", ["int64", "uint64", "int32", "uint32", "float64", "float32"])
@pytest.mark.parametrize("n", [2_000_000])
@pytest.mark.parametrize("dist", ["uniform", "bounded"])
def test_forced_backend_matches_numpy(backend, dtype, n, dist):
    if backend not in quill.available_backends():
        pytest.skip(f"{backend} not available here")
    a = gen(dtype, n, dist)
    got = quill._backends.dispatch_sort(a.copy(), force=backend)
    used = quill._backends._LAST_BACKEND
    if used != backend:
        pytest.skip(f"{backend} declined this input (dtype/size); ran {used}")
    assert eq(got, oracle(a)), f"force={backend} {dtype} n={n} {dist}"


@pytest.mark.parametrize("backend", ALL_BACKEND_NAMES)
def test_forced_backend_float_nan(backend):
    if backend not in quill.available_backends():
        pytest.skip(f"{backend} not available")
    a = _with_specials("float64", 2_000_000, "some_nan")
    got = quill._backends.dispatch_sort(a.copy(), force=backend)
    if quill._backends._LAST_BACKEND != backend:
        pytest.skip("backend declined")
    assert eq(got, oracle(a)), f"force={backend} NaN handling"


# ─────────────────────────────────────────────────────────────────────────────
# 8. structural edge cases
# ─────────────────────────────────────────────────────────────────────────────

def test_empty_and_singleton():
    for dt in NUM_DTYPES:
        assert eq(quill.sort_array(np.empty(0, dt)), np.empty(0, dt))
        assert eq(quill.sort_array(np.array([5], dt)), np.array([5], dt))
    assert quill.quill_sorted([]) == []
    assert quill.quill_sorted([42]) == [42]


def test_zero_dim_array():
    a = np.array(7)
    out = quill.sort_array(a)
    assert int(out) == 7


@pytest.mark.parametrize("descending", [False, True])
def test_two_dim_array_sorts_last_axis(descending):
    rng = np.random.default_rng(0)
    a = rng.integers(0, 1000, (50, 40)).astype("int64")
    got = quill.sort_array(a.copy(), descending=descending)
    expected = np.sort(a, axis=-1)
    if descending:
        expected = expected[:, ::-1]
    assert eq(got, expected)


def test_non_contiguous_input():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 10 ** 9, 400_002).astype("int64")
    view = base[::2]  # non-contiguous
    assert not view.flags["C_CONTIGUOUS"]
    got = quill.sort_array(view)
    assert eq(got, np.sort(view))


def test_read_only_input_not_mutated():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 10 ** 9, 300_000).astype("int64")
    a.setflags(write=False)
    got = quill.sort_array(a)               # inplace=False
    assert eq(got, np.sort(a))
    got2 = quill.sort_array(a, inplace=True)  # read-only + inplace → returns a copy, no raise
    assert eq(got2, np.sort(a))


def test_object_dtype_array():
    a = np.array([3, 1, 2, 10 ** 30, -5], dtype=object)
    got = quill.sort_array(a.copy())
    assert list(got) == sorted(a.tolist())


def test_mixed_int_float_list_promotes():
    data = [1, 2.5, 3, 4.0, -1, 0.5]
    assert quill.quill_sorted(data) == sorted(data)


@pytest.mark.parametrize("src", ["range", "generator", "tuple"])
def test_non_list_iterables(src):
    if src == "range":
        it = range(100, 0, -1)
        expected = sorted(range(100, 0, -1))
    elif src == "generator":
        it = (i * i % 97 for i in range(200))
        expected = sorted(i * i % 97 for i in range(200))
    else:
        it = tuple(range(50, 0, -1))
        expected = sorted(range(50, 0, -1))
    assert quill.quill_sorted(it) == expected


# ─────────────────────────────────────────────────────────────────────────────
# 9. API invariants
# ─────────────────────────────────────────────────────────────────────────────

def test_available_backends_contract():
    bks = quill.available_backends()
    assert isinstance(bks, list) and bks, "must list at least the numpy floor"
    assert bks[-1] == "numpy" or "numpy" in bks
    assert len(bks) == len(set(bks)), "no duplicate backend names"


def test_analyze_shapes():
    for a in ([3, 1, 2], np.arange(100, dtype="int64"), np.array([1.0, np.nan, 2.0])):
        d = quill.analyze(a)
        assert isinstance(d, dict) and "n" in d and "dtype" in d


def test_stats_return():
    out, stats = quill.quill_sort([3, 1, 2], stats=True)
    assert out == [1, 2, 3]
    assert "time_ms" in stats and "n" in stats


def test_version_present():
    assert isinstance(quill.__version__, str) and quill.__version__.count(".") >= 2


# ─────────────────────────────────────────────────────────────────────────────
# 10. property-based fuzzing (hypothesis) — the long tail of weird inputs
# ─────────────────────────────────────────────────────────────────────────────

hyp = pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st  # noqa: E402
from hypothesis.extra import numpy as hnp  # noqa: E402


@settings(max_examples=200, deadline=None)
@given(st.lists(st.integers(min_value=-(10 ** 9), max_value=10 ** 9), max_size=500))
def test_property_int_list_matches_sorted(data):
    assert quill.quill_sorted(data) == sorted(data)
    assert quill.quill_sorted(data, reverse=True) == sorted(data, reverse=True)


@settings(max_examples=150, deadline=None)
@given(st.lists(st.text(max_size=8), max_size=300))
def test_property_str_list_matches_sorted(data):
    assert quill.quill_sorted(data) == sorted(data)


@settings(max_examples=150, deadline=None)
@given(hnp.arrays(dtype=st.sampled_from([np.int32, np.int64, np.float64]),
                  shape=hnp.array_shapes(min_dims=1, max_dims=1, max_side=1000)))
def test_property_array_matches_numpy(a):
    got = quill.sort_array(a.copy())
    assert eq(got, np.sort(a))


@settings(max_examples=100, deadline=None)
@given(st.lists(st.integers(min_value=0, max_value=10), min_size=1, max_size=200),
       st.integers(min_value=1, max_value=50))
def test_property_topk(data, k):
    kk = min(k, len(data))
    assert quill.quill_topk(data, kk) == sorted(data)[:kk]
    assert quill.quill_topk(data, kk, largest=True) == sorted(data, reverse=True)[:kk]


# ─────────────────────────────────────────────────────────────────────────────
# 11. slow tier — large-N (only with `-m slow`)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.parametrize("dtype", ["int64", "uint64", "int32", "float64", "float32"])
@pytest.mark.parametrize("dist", ["uniform", "bounded", "reversed"])
@pytest.mark.parametrize("descending", [False, True])
def test_large_arrays(dtype, dist, descending):
    a = gen(dtype, 5_000_000, dist, seed=99)
    assert eq(quill.sort_array(a.copy(), descending=descending), oracle(a, descending))


@pytest.mark.slow
def test_large_with_nan():
    a = _with_specials("float64", 5_000_000, "some_nan")
    assert eq(quill.sort_array(a.copy()), np.sort(a))


# ─────────────────────────────────────────────────────────────────────────────
# 12. regression guards for the confirmed edge-case bugs (deep hunt, 2026-07-05)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dt", [">i8", ">u8", ">i4", ">u4", ">f8", ">f4"])
def test_bigendian_array_matches_numpy(dt):
    # Non-native byte order: compiled radix kernels (spectre/ips4o/rust) read raw
    # bytes in native order and would silently corrupt a byteswapped buffer.
    # sort_array must route these to np.sort. (n above spectre's 1M crossover.)
    rng = np.random.default_rng(7)
    ndt = np.dtype(dt)
    n = 1_500_000
    if ndt.kind == "f":
        a = (rng.random(n) * 2 ** 30 - 2 ** 29).astype(dt)
    else:
        info = np.iinfo(ndt.newbyteorder("="))
        a = rng.integers(max(info.min, -(2 ** 40)), min(int(info.max), 2 ** 40), n).astype(dt)
    assert not a.dtype.isnative
    assert eq(quill.sort_array(a.copy()), np.sort(a))


def test_bigendian_forced_backends_never_corrupt():
    # Forcing a native-buffer backend on a byteswapped array must NOT corrupt it.
    a = np.random.default_rng(1).integers(-(10 ** 12), 10 ** 12, 1_500_000).astype(">i8")
    for bk in ("spectre", "rust_voracious", "rust_parallel_radix", "ips4o", "numpy"):
        if bk in quill.available_backends():
            assert eq(quill._backends.dispatch_sort(a.copy(), force=bk), np.sort(a)), bk


def test_topk_uint64_boundary_no_float_promotion():
    # A Python-int list spanning int64/uint64 must not be lossily promoted to
    # float64 (2**64-1 and 2**64-2 would collapse to the same float).
    data = [2 ** 64 - 1, 2 ** 64 - 2, 0, 5, 2 ** 63 + 1]
    assert quill.quill_topk(data, 3, largest=True) == sorted(data, reverse=True)[:3]
    assert quill.quill_topk(data, 3) == sorted(data)[:3]
    assert quill.quill_topk(data, 10) == sorted(data)          # k>=n branch
    assert quill.quill_topk(data, 2, largest=True) == [2 ** 64 - 1, 2 ** 64 - 2]


def test_quill_sort_multidim_ndarray_matches_sorted():
    # sorted() raises on a 0-d (TypeError) or multi-dim (ValueError) ndarray;
    # quill_sort must raise the same class, not silently per-row sort / flatten.
    for a in (np.array([[3, 1, 2], [9, 7, 8]], dtype=np.int64),
              np.arange(24).reshape(2, 3, 4).astype(np.int64),
              np.array([[np.nan, 1.0, 2.0], [9.0, np.nan, 8.0]]),
              np.array(5)):                                     # 0-d
        with pytest.raises((ValueError, TypeError)):
            sorted(a)
        with pytest.raises((ValueError, TypeError)):
            quill.quill_sort(a)


@pytest.mark.parametrize("n", [99_999, 120_001])   # straddle the 100k polars threshold
def test_none_in_large_list_sorts_to_end(n):
    # None-to-end must hold ABOVE the polars fast-path threshold too: the polars
    # path declines on nulls (it would order them as SQL nulls) and falls back to
    # the None-aware Timsort path. sorted() itself would raise TypeError here.
    data = ["m"] * (n // 2) + [None] + ["a"] * (n - n // 2 - 1)
    clean = sorted(x for x in data if x is not None)
    assert quill.quill_sorted(data) == clean + [None]
    assert quill.quill_sorted(data, reverse=True) == [None] + clean[::-1]
    idx = quill.quill_argsort(data)
    assert [data[i] for i in idx] == clean + [None]


def test_none_free_large_string_list_still_fast_path():
    # The polars fast path must remain intact for None-free large string lists.
    import random
    big = [f"k{i:06d}" for i in range(120_000)]
    random.Random(0).shuffle(big)
    assert quill.quill_sorted(big) == sorted(big)


@pytest.mark.parametrize("period", [2, 3, 5, 7, 11])
@pytest.mark.parametrize("n", [1000, 100_000, 200_000])
def test_periodic_list_not_falsely_detected_constant(period, n):
    # A periodic list [i % period] can produce an all-identical 512-point profiler
    # sample when the stride n//512 is a multiple of the period — the old code then
    # mistook it for a constant array (all_same) and SKIPPED the sort, returning it
    # unsorted. (ChatGPT-reported against 6.0.18; reproduced in 7.5.x.) Must fully
    # sort, and the same via the mutating quill_sort.
    data = [i % period for i in range(n)]
    assert quill.quill_sorted(data) == sorted(data), (period, n)
    work = list(data)
    quill.quill_sort(work)
    assert work == sorted(data)


def test_mostly_constant_list_with_hidden_outlier_sorts():
    # An all-same sample must be verified against the full data, or a lone
    # non-constant element (e.g. a tail outlier) is silently left unsorted.
    for data in ([7] * 200_000 + [3],
                 [4] * 199_999 + [1],
                 [(i % 3) - 1 for i in range(300_000)],   # negatives
                 ([5] * 50_000 + [1] * 50_000) * 2):
        assert quill.quill_sorted(data) == sorted(data)


@pytest.mark.slow
@pytest.mark.parametrize("period", [3, 7, 13])
def test_periodic_list_large_n(period):
    for n in (1_000_000, 2_000_000):
        data = [i % period for i in range(n)]
        assert quill.quill_sorted(data) == sorted(data), (period, n)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

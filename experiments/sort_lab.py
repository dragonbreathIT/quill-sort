#!/usr/bin/env python3
"""
sort_lab.py — a sandbox for trying "can we beat np.sort" ideas.

HONEST CAVEAT (read this):
  numpy exposes no O(n) stable integer-partition primitive, so a *faithful* radix
  sort can't be written in pure numpy — it collapses to O(n log n) argsort. That's
  exactly why quill's fast backends are C++ (quill_core.hpp). So this lab is NOT
  where you'll clock true radix speed. It is where you:
    (a) confirm the ONE family that genuinely beats np.sort in numpy — counting
        sort (bincount) — and see on which data it wins;
    (b) probe how well a cheap CDF model buckets each distribution ("learned sort"
        feasibility) so you know what's worth building in C++;
    (c) A/B your own new ideas against np.sort with a correctness gate.

Winning idea here -> port to quill_core.hpp (write-combining radix / learned
single-pass scatter) where it runs at C/SIMD speed.
"""
from __future__ import annotations
import time
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Algorithms under test. Each takes an int64 ndarray, returns a sorted ndarray,
# or raises NotImplementedError to be marked N/A for that input.
# ─────────────────────────────────────────────────────────────────────────────

def np_introsort(a):                      # the baseline everything is measured against
    return np.sort(a)

def np_timsort(a):                        # stable mergesort — shines on nearly-sorted
    return np.sort(a, kind="stable")

def counting_sort(a):
    """O(n + range). The one technique that genuinely beats np.sort in numpy — but
    only when the value range is small enough to be worth a histogram."""
    mn = int(a.min()); mx = int(a.max())
    rng = mx - mn
    if rng > 400_000_000:                 # histogram would blow memory / lose
        raise NotImplementedError("range too large for counting sort")
    counts = np.bincount((a - mn).astype(np.int64), minlength=rng + 1)
    return np.repeat(np.arange(mn, mx + 1, dtype=a.dtype), counts)

def counting_then_np(a):
    """Adaptive combo (mirrors quill's dispatch): counting sort when the range is
    dense, else fall back to np.sort. This is the 'never lose' shape."""
    mn = int(a.min()); mx = int(a.max())
    if (mx - mn) <= 2 * a.size:           # dense enough
        return counting_sort(a)
    return np.sort(a)

def learned_sort_demo(a, fanout=8):
    """Behavioral prototype of LEARNED SORT (not a speed demo — Python bucket loop).
    Linear CDF model -> bucket id, stable-partition into buckets, sort each bucket.
    Correct, and its *promise* is quantified separately by cdf_bucket_report()."""
    mn = int(a.min()); mx = int(a.max())
    if mx == mn:
        return a.copy()
    m = max(1, a.size // fanout)
    # one-pass model: where does each element roughly belong?
    b = (((a.astype(np.float64) - mn) * (m - 1)) / (mx - mn)).astype(np.int64)
    order = np.argsort(b, kind="stable")  # <-- the O(n log n) numpy tax (C would be O(n))
    vals = a[order]; bk = b[order]
    counts = np.bincount(bk, minlength=m)
    out = np.empty_like(a)
    s = 0
    for c in counts:                      # sort within each contiguous bucket
        if c > 1:
            out[s:s + c] = np.sort(vals[s:s + c])
        elif c == 1:
            out[s] = vals[s]
        s += c
    return out


ALGOS = {
    "np.sort (introsort)": np_introsort,
    "np.sort stable":      np_timsort,
    "counting":            counting_sort,
    "counting->np (adaptive)": counting_then_np,
    "learned (demo)":      learned_sort_demo,
}

# ─────────────────────────────────────────────────────────────────────────────
# Data distributions — the shape of the data decides the winner.
# ─────────────────────────────────────────────────────────────────────────────
def make(dist, n, rng):
    if dist == "uniform_wide":  return rng.integers(-(2**40), 2**40, n, dtype=np.int64)
    if dist == "uniform_32":    return rng.integers(0, 2**32, n, dtype=np.int64)
    if dist == "gaussian":      return (rng.standard_normal(n) * 1e6).astype(np.int64)
    if dist == "dense_small":   return rng.integers(0, 2 * n, n, dtype=np.int64)  # counting-friendly
    if dist == "few_unique":    return rng.integers(0, 10, n, dtype=np.int64)
    if dist == "nearly_sorted":
        a = np.arange(n, dtype=np.int64); idx = rng.integers(0, n, n // 100)
        a[idx] = rng.integers(0, n, idx.size); return a
    raise ValueError(dist)

DISTS = ["uniform_wide", "uniform_32", "gaussian", "dense_small", "few_unique", "nearly_sorted"]

# ─────────────────────────────────────────────────────────────────────────────
def bench(fn, a, reps=3):
    best = float("inf")
    for _ in range(reps):
        x = a.copy()
        t = time.perf_counter(); out = fn(x); dt = time.perf_counter() - t
        best = min(best, dt)
    return best, out

def cdf_bucket_report(a, fanout=8):
    """LEARNED-SORT FEASIBILITY: with a cheap linear CDF model into n/fanout
    buckets, how balanced are the buckets? Balanced => one model-scatter pass
    nearly sorts the data (C cleanup is tiny) => learned sort is worth building.
    Lopsided => the linear model doesn't fit this distribution."""
    mn = int(a.min()); mx = int(a.max())
    if mx == mn: return (1.0, 1.0)
    m = max(1, a.size // fanout)
    b = (((a.astype(np.float64) - mn) * (m - 1)) / (mx - mn)).astype(np.int64)
    occ = np.bincount(b, minlength=m)
    ideal = a.size / m
    return occ.max() / ideal, float(occ[occ > 0].mean() / ideal)

def main():
    import os
    rng = np.random.default_rng(0)
    n = 5_000_000
    print(f"sort_lab — n={n:,}  cores={os.cpu_count()}  numpy={np.__version__}\n")
    print("Speed vs np.sort (ratio >1.0 = FASTER than np.sort; '—' = N/A/too slow):\n")
    header = f"{'distribution':14} " + " ".join(f"{name[:15]:>15}" for name in ALGOS)
    print(header); print("-" * len(header))
    for dist in DISTS:
        a = make(dist, n, rng)
        base, ref = bench(np_introsort, a)
        cells = []
        for name, fn in ALGOS.items():
            try:
                t, out = bench(fn, a)
                assert np.array_equal(out, ref), f"WRONG: {name}/{dist}"
                cells.append(f"{base / t:>14.2f}x")
            except NotImplementedError:
                cells.append(f"{'—':>15}")
        print(f"{dist:14} " + " ".join(cells))

    print("\nLearned-sort feasibility (linear CDF model, buckets = n/8):")
    print(f"  {'distribution':14} {'max/ideal':>10} {'mean/ideal':>11}   fit?")
    for dist in DISTS:
        a = make(dist, n, rng)
        mx_r, mean_r = cdf_bucket_report(a)
        fit = "GOOD (≈1 pass)" if mx_r < 4 else ("ok" if mx_r < 20 else "POOR (model misfits)")
        print(f"  {dist:14} {mx_r:>10.1f} {mean_r:>11.2f}   {fit}")

    print("""
Read it like this:
  * counting / adaptive winning on dense_small & few_unique = the real principle:
    avoid comparisons AND minimize passes (1 histogram pass) -> big win. Already in quill.
  * np.sort winning everything else in this lab is EXPECTED: numpy can't express an
    O(n) stable partition, so 'radix'/'learned' pay the argsort tax here. In C++
    (quill_core.hpp) those become O(n) and flip the result.
  * The feasibility table is the actionable part: distributions with GOOD fit are
    where a C++ learned single-pass scatter would beat radix. 'POOR' rows need a
    piecewise-linear CDF (sample-and-fit) instead of the naive linear model.

Next real steps (in C++, not here):
  1. Wider-digit radix (11/16-bit) — fewer passes.  [easy, measurable]
  2. Write-combining scatter buffers — ~1.5-2x, the big one.  [cache engineering]
  3. Learned single-pass scatter for GOOD-fit distributions + radix cleanup.
""")

if __name__ == "__main__":
    main()

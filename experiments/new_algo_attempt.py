#!/usr/bin/env python3
"""
new_algo_attempt.py — actually trying to invent something faster than np.sort.

This is an honest research probe, not a claim. Hypotheses under test:

  H1. numpy has a usable fast RADIX hiding behind kind='stable' for small-range
      integer keys — cheap enough to partition with.
  H2. CACHE-BLOCKED sorting beats one giant np.sort on large n: cheaply split
      into globally-ordered buckets that fit in L2, sort each, concatenate.
      (Why it might work: np.sort's introsort has poor locality at 20M+; many
      small in-cache sorts don't. Why it might fail: the partition's argsort +
      gather move ~2n extra int64 = bandwidth that eats the cache savings.)
  H3. A LEARNED partition (splitters from a sorted sample, via searchsorted)
      balances buckets on skewed data (gaussian) better than raw top-bits.

Every candidate is gated against np.sort for correctness. We measure, then read
the result straight — win or lose.
"""
from __future__ import annotations
import time, os
import numpy as np

SIGN = np.uint64(0x8000000000000000)

# ── H1 probe: is numpy's stable sort a fast radix on small-range keys? ────────
def probe_numpy_radix(n=20_000_000):
    rng = np.random.default_rng(1)
    wide = rng.integers(0, 2**63, n, dtype=np.int64)
    small = rng.integers(0, 256, n, dtype=np.int64)   # 1-byte range
    def t(fn, a):
        b=1e9
        for _ in range(3):
            x=a.copy(); s=time.perf_counter(); fn(x); b=min(b,time.perf_counter()-s)
        return b*1e3
    print("H1 — numpy sort kinds (ms, lower=faster):")
    print(f"   wide  key: quicksort {t(lambda x:np.sort(x,kind='quicksort'),wide):7.1f}   "
          f"stable {t(lambda x:np.sort(x,kind='stable'),wide):7.1f}")
    print(f"   1-byte key: quicksort {t(lambda x:np.sort(x,kind='quicksort'),small):7.1f}   "
          f"stable {t(lambda x:np.sort(x,kind='stable'),small):7.1f}")
    print("   -> if 'stable' on the 1-byte key is much faster, numpy has a radix"
          " we can partition with cheaply.\n")

# ── candidates ────────────────────────────────────────────────────────────────
def np_sort(a):
    return np.sort(a)

def blocked_topbits(a, bits=8):
    """H2: partition by top `bits` (order-preserving), np.sort each bucket."""
    u = a.view(np.uint64) ^ SIGN
    key = (u >> np.uint64(64 - bits)).astype(np.int64)      # bucket id in [0, 2^bits)
    order = np.argsort(key, kind='stable')                  # the cheap-ish partition
    vals = a[order]
    counts = np.bincount(key, minlength=1 << bits)
    out = np.empty_like(a); s = 0
    for c in counts:
        if c > 1: out[s:s+c] = np.sort(vals[s:s+c])
        elif c == 1: out[s] = vals[s]
        s += int(c)
    return out

def blocked_partition_only(a, bits=8):
    """H2 isolated: the SAME partition, but skip per-bucket sort — measures how
    much of the time is partition vs the actual sorting. (Not a valid sort;
    used only for timing, so it's excluded from the correctness-gated table.)"""
    u = a.view(np.uint64) ^ SIGN
    key = (u >> np.uint64(64 - bits)).astype(np.int64)
    order = np.argsort(key, kind='stable')
    return a[order]

def learned_flash(a, nbuck=1024):
    """H3: splitters from a sorted subsample (approx CDF) -> balanced buckets."""
    step = max(1, a.size // 8192)
    sample = np.sort(a[::step])
    idx = np.linspace(0, sample.size - 1, nbuck + 1).astype(np.int64)[1:-1]
    splitters = sample[idx]
    buck = np.searchsorted(splitters, a, side='right').astype(np.int64)  # O(n log nbuck)
    order = np.argsort(buck, kind='stable')
    vals = a[order]
    counts = np.bincount(buck, minlength=nbuck)
    out = np.empty_like(a); s = 0
    for c in counts:
        if c > 1: out[s:s+c] = np.sort(vals[s:s+c])
        elif c == 1: out[s] = vals[s]
        s += int(c)
    return out

CANDIDATES = {
    "np.sort": np_sort,
    "blocked_topbits(8)": lambda a: blocked_topbits(a, 8),
    "blocked_topbits(11)": lambda a: blocked_topbits(a, 11),
    "learned_flash(1024)": lambda a: learned_flash(a, 1024),
}

def make(dist, n, rng):
    if dist == "uniform":  return rng.integers(-(2**40), 2**40, n, dtype=np.int64)
    if dist == "gaussian": return (rng.standard_normal(n) * 1e7).astype(np.int64)
    if dist == "lognormal":return (rng.lognormal(0, 3, n) * 1e4).astype(np.int64)  # heavy skew
    raise ValueError

def bench(fn, a, reps=3):
    b = 1e9
    for _ in range(reps):
        x = a.copy(); s = time.perf_counter(); out = fn(x); b = min(b, time.perf_counter() - s)
    return b, out

def main():
    print(f"new_algo_attempt — cores={os.cpu_count()} numpy={np.__version__}\n")
    probe_numpy_radix()
    rng = np.random.default_rng(0)
    for n in (5_000_000, 20_000_000):
        print(f"n={n:,}  — speedup vs np.sort (>1.0 = we WIN):")
        hdr = f"  {'dist':10} " + " ".join(f"{k[:19]:>19}" for k in CANDIDATES)
        print(hdr); print("  " + "-" * (len(hdr) - 2))
        for dist in ("uniform", "gaussian", "lognormal"):
            a = make(dist, n, rng)
            base, ref = bench(np_sort, a)
            cells = []
            for name, fn in CANDIDATES.items():
                t, out = bench(fn, a)
                ok = np.array_equal(out, ref)
                cells.append(f"{(base/t):>17.2f}x" if ok else f"{'WRONG':>18} ")
            # partition-only cost, for context (fraction of np.sort spent just partitioning)
            pt, _ = bench(lambda x: blocked_partition_only(x, 8), a)
            print(f"  {dist:10} " + " ".join(cells) + f"   [partition alone: {pt/base:.2f}x np.sort]")
        print()
    print("Verdict is whatever the numbers say above — read the ratios, not the hope.")

if __name__ == "__main__":
    main()

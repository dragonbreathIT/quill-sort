#!/usr/bin/env python3
"""
bench/run_bench.py — time whichever ``quill`` is importable against ``numpy.sort``
over a fixed, deterministic case set, and emit JSON to stdout (or --out FILE).

CI runs this TWICE — once with the HEAD build installed, once with the previous
stable release from PyPI installed in a separate venv — then ``compare.py`` diffs
the two JSON files to detect a post-merge performance regression.

Fairness rules (so the numbers mean something and don't false-alarm):
  * matched semantics: ``np.sort(a)`` vs ``quill.sort_array(a)`` — both
    non-mutating (an in-place-vs-copy mismatch would hand numpy a free skipped
    allocation and manufacture phantom regressions).
  * warm the self-tuning dispatcher to convergence before timing.
  * min-of-N (the least-contended run) — robust to a noisy CI runner.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import quill


# (id, dtype, n, distribution) — kept modest so the CI job stays a few minutes.
CASES = [
    ("i64_uniform_1M",  "int64",   1_000_000, "uniform"),
    ("i64_uniform_5M",  "int64",   5_000_000, "uniform"),
    ("i64_bounded_5M",  "int64",   5_000_000, "bounded"),
    ("i32_uniform_5M",  "int32",   5_000_000, "uniform"),
    ("u64_uniform_5M",  "uint64",  5_000_000, "uniform"),
    ("f64_uniform_5M",  "float64", 5_000_000, "uniform"),
    ("f32_uniform_5M",  "float32", 5_000_000, "uniform"),
    ("f64_bounded_5M",  "float64", 5_000_000, "bounded"),
    ("i64_uniform_100k","int64",     100_000, "uniform"),  # small-int win (Spectre crossover)
    ("i64_uniform_50k", "int64",      50_000, "uniform"),
    ("i64_uniform_200k","int64",     200_001, "uniform"),  # small: numpy-competitiveness
    ("f64_uniform_200k","float64",   200_001, "uniform"),
]


def gen(dtype, n, dist, seed=1234):
    dtype = np.dtype(dtype)
    rng = np.random.default_rng(seed)
    if dtype.kind in "iu":
        info = np.iinfo(dtype)
        if dist == "uniform":
            return rng.integers(max(info.min, -(2**40)), min(info.max, 2**40), n, dtype=dtype)
        return rng.integers(0, 50, n, dtype=dtype, endpoint=True)  # bounded
    if dist == "uniform":
        return (rng.random(n) * 2**20).astype(dtype)
    return (rng.integers(0, 50, n)).astype(dtype)  # bounded


def best_ms(fn, a, reps):
    b = 1e18
    for _ in range(reps):
        c = a.copy()
        t = time.perf_counter()
        fn(c)
        b = min(b, (time.perf_counter() - t) * 1e3)
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--warmup", type=int, default=8)
    args = ap.parse_args()

    result = {
        "quill_version": quill.__version__,
        "numpy_version": np.__version__,
        "backends": quill.available_backends(),
        "cases": {},
    }
    for cid, dtype, n, dist in CASES:
        a = gen(dtype, n, dist)
        # correctness guard — a "fast" wrong answer is not a datapoint
        assert np.array_equal(np.asarray(quill.sort_array(a.copy())), np.sort(a), equal_nan=True), \
            f"INCORRECT sort on {cid}"
        for _ in range(args.warmup):
            quill.sort_array(a.copy())          # converge the tuner
        npt = best_ms(lambda c: np.sort(c), a, args.reps)
        qt = best_ms(lambda c: quill.sort_array(c), a, args.reps)
        result["cases"][cid] = {"quill_ms": round(qt, 4), "numpy_ms": round(npt, 4),
                                "backend": quill._backends._LAST_BACKEND}
        print(f"  {cid:20} quill={qt:8.3f}ms  numpy={npt:8.3f}ms  ({quill._backends._LAST_BACKEND})",
              file=sys.stderr)

    text = json.dumps(result, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()

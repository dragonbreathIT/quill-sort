"""
bench_hydra.py — HydraSort vs np.sort vs quill on YOUR machine.

Run:
    python bench_hydra.py             # n = 5,000,000
    python bench_hydra.py 20000000    # custom n

Needs: numpy, gcc (or set CC=clang). quill-sort is optional — the quill
column appears if it's installed. Compiles hydra_sort.c on first run.

Read the table like this:
  - "vs np" > 1.0 means hydra beat numpy's SIMD introsort on that case.
  - "hydra path" tells you which strategy fired (presorted/counting/radix...).
  - "k" is the radix pass count hydra chose. The k where "vs np" crosses
    1.0 on YOUR machine is the max_passes_vs_numpy value to give
    hydra_backend.register_with_quill().
"""
from __future__ import annotations

import sys
import time

import numpy as np

from hydra_backend import hydra_sort, last_path, estimated_passes

try:
    import quill
    HAVE_QUILL = True
except ImportError:
    HAVE_QUILL = False

N = int(sys.argv[1]) if len(sys.argv) > 1 else 5_000_000
REPS = 3


def bench(fn, arr, check):
    best = float("inf")
    for _ in range(REPS):
        a = arr.copy()
        t = time.perf_counter()
        out = fn(a)
        best = min(best, time.perf_counter() - t)
    final = out if out is not None else a
    assert np.array_equal(final, check), "MISMATCH — do not trust this row"
    return best * 1000


def main():
    rng = np.random.default_rng(42)
    nearly = np.arange(N, dtype=np.int64)
    idx = rng.integers(0, N, max(1, N // 100))
    nearly[idx] = rng.integers(0, N, idx.size)

    cases = {
        "int64 full-range":     rng.integers(-2**62, 2**62, N, dtype=np.int64),
        "int64 bounded 1e12":   rng.integers(0, 10**12, N, dtype=np.int64),
        "int64 bounded 1e9":    rng.integers(0, 10**9, N, dtype=np.int64),
        "int64 timestamps(us)": rng.integers(1_700_000_000_000_000,
                                             1_700_086_400_000_000, N,
                                             dtype=np.int64),
        "int64 bounded n//10":  rng.integers(0, max(2, N // 10), N,
                                             dtype=np.int64),
        "int64 small-range":    rng.integers(0, 50_000, N, dtype=np.int64),
        "int64 nearly-sorted":  nearly,
        "int64 already-sorted": np.arange(N, dtype=np.int64),
        "int64 reverse-sorted": np.arange(N, 0, -1, dtype=np.int64),
        "int32 full-range":     rng.integers(-2**31, 2**31 - 1, N,
                                             dtype=np.int32),
        "uint32 full-range":    rng.integers(0, 2**32, N, dtype=np.uint32),
    }

    qcol = "  quill" if HAVE_QUILL else ""
    print(f"n = {N:,}   times in ms (min of {REPS})")
    print(f"{'case':<22} {'np.sort':>8}{qcol:>8} {'hydra':>8} "
          f"{'k':>2} {'hydra path':>11} {'vs np':>7}")
    for name, arr in cases.items():
        ref = np.sort(arr)
        t_np = bench(lambda a: np.sort(a), arr, ref)
        t_q = bench(lambda a: quill.sort_array(a), arr, ref) if HAVE_QUILL else None
        t_h = bench(lambda a: hydra_sort(a), arr, ref)
        k = estimated_passes(int(arr.min()), int(arr.max()))
        qtxt = f"{t_q:>8.1f}" if t_q is not None else ""
        print(f"{name:<22} {t_np:>8.1f}{qtxt} {t_h:>8.1f} "
              f"{k:>2} {last_path():>11} {t_np / t_h:>6.2f}x")

    print("\nMachine context:")
    a = np.empty(N, dtype=np.int64)
    b = rng.integers(0, 2**62, N, dtype=np.int64)
    t = time.perf_counter()
    for _ in range(5):
        np.copyto(a, b)
    bw = 5 * 2 * a.nbytes / (time.perf_counter() - t) / 1e9
    print(f"  approx memcpy bandwidth: {bw:.1f} GB/s "
          f"(sandbox where hydra was tuned: 7.8 GB/s)")
    print("  higher bandwidth favors hydra's radix passes over introsort.")


if __name__ == "__main__":
    main()

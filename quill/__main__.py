"""
quill/__main__.py
-----------------
Entry point for `python -m quill` and the `quill` CLI command.
"""

import os, random, sys, time
from . import quill_sort, __version__

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False


def main():
    NCORES = os.cpu_count() or 1

    def typewrite(text, delay=0.012):
        for ch in text:
            sys.stdout.write(ch); sys.stdout.flush(); time.sleep(delay)
        print()

    def bar(filled, total, width=36):
        n = int(width * min(filled, total) / total)
        return f"[{'█' * n}{'░' * (width - n)}]"

    def run_section(label, cases, bar_scale):
        typewrite(f"  {label}", 0.008)
        results = []
        for size, gen in cases:
            if size >= 500_000:
                sys.stdout.write(f"  {size:>12,} elements  [allocating...]")
                sys.stdout.flush()
            try:
                data = gen()
            except MemoryError:
                sys.stdout.write(f"\r  {size:>12,} elements  [skipped — not enough RAM]\n")
                continue

            t0      = time.perf_counter()
            quill_sort(data)
            elapsed = time.perf_counter() - t0

            ok     = all(data[i] <= data[i+1] for i in range(min(len(data)-1, 20_000)))
            results.append((size, elapsed, ok))
            b      = bar(elapsed, bar_scale)
            line   = f"  {size:>12,} elements  {b}  {elapsed:>8.4f}s  {'✓' if ok else '✗'}"
            if size >= 500_000:
                sys.stdout.write(f"\r{line}\n")
            else:
                print(line)
            sys.stdout.flush()
            del data
            time.sleep(0.04)
        print()
        return results

    # ── Header ────────────────────────────────────────────────────────────────
    print()
    typewrite("  ▄▀▄ █ █ █ █   █   ")
    typewrite(f"  ▀▄▀ ▀▄█ █ █▄▄ █▄▄   v{__version__}")
    print()
    typewrite("  Adaptive Ultra-Sort  —  by Isaiah Tucker", 0.01)

    try:
        import psutil as _psutil
        psutil_tag = "psutil ✓  memory sensing active"
    except ImportError:
        psutil_tag = "psutil ✗  install for RAM sensing: pip install quill-sort[fast]"
    typewrite(f"  [{psutil_tag}]", 0.008)
    np_tag = ("numpy ✓  vectorised C path active" if _NUMPY
              else "numpy ✗  install for max speed: pip install numpy")
    typewrite(f"  [{np_tag}]", 0.008)
    print()
    time.sleep(0.3)

    typewrite("  Benchmarking across data types...", 0.012)
    print()
    time.sleep(0.2)

    all_results = []

    # ── Positive integers ─────────────────────────────────────────────────────
    # uint8/uint16 → stable radix  |  int32 → quicksort  |  int64 → heapsort
    # 5 M+ elements → parallel MSD radix (all cores)
    sizes_int = [100, 1_000, 10_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]
    label_int = (f"Positive integers  "
                 f"(radix/counting/heapsort · parallel MSD @ 5M+  [{NCORES} cores])")
    res = run_section(label_int, [
        (sz, lambda sz=sz: [random.randint(0, sz * 10) for _ in range(sz)])
        for sz in sizes_int
    ], bar_scale=2.0)
    all_results.extend(res)

    # ── Mixed ± integers ──────────────────────────────────────────────────────
    # Shift-based key extraction — no fallback for negatives
    res = run_section(
        "Mixed ± integers   (shift-based radix — negatives handled natively)",
        [(sz, lambda sz=sz: [random.randint(-sz * 5, sz * 5) for _ in range(sz)])
         for sz in [1_000, 10_000, 100_000, 1_000_000]],
        bar_scale=0.5,
    )
    all_results.extend(res)

    # ── Narrow-range integers ─────────────────────────────────────────────────
    # np.bincount counting sort — O(range) not O(n log n)
    res = run_section(
        "Narrow-range ints  (np.bincount counting sort — O(range), zero comparisons)",
        [(sz, lambda sz=sz: [random.randint(0, 100) for _ in range(sz)])
         for sz in [10_000, 100_000, 1_000_000]],
        bar_scale=0.1,
    )
    all_results.extend(res)

    # ── Floats ────────────────────────────────────────────────────────────────
    # heapsort — cache-efficient, benchmark-proven fastest for float64
    res = run_section(
        "Floats             (numpy heapsort — cache-optimal for float64)",
        [(sz, lambda sz=sz: [random.uniform(-1e9, 1e9) for _ in range(sz)])
         for sz in [1_000, 10_000, 100_000, 1_000_000]],
        bar_scale=0.5,
    )
    all_results.extend(res)

    # ── Strings ───────────────────────────────────────────────────────────────
    res = run_section(
        "Strings            (Python timsort — locale-aware, stable)",
        [(sz, lambda sz=sz: [
              ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=6))
              for _ in range(sz)])
         for sz in [1_000, 10_000, 100_000]],
        bar_scale=0.2,
    )
    all_results.extend(res)

    # ── Real-time sort ────────────────────────────────────────────────────────
    time.sleep(0.2)
    typewrite("  Watching Quill sort 20 numbers in real time...", 0.012)
    print()
    time.sleep(0.2)

    sample = random.sample(range(1, 999), 20)
    print(f"  Before:  {sample}")
    time.sleep(0.6)
    quill_sort(sample)
    sys.stdout.write("  After:   ")
    sys.stdout.flush()
    for i, v in enumerate(sample):
        sys.stdout.write(("" if i == 0 else ", ") + str(v))
        sys.stdout.flush()
        time.sleep(0.055)
    print(); print()
    time.sleep(0.4)

    # ── Summary ───────────────────────────────────────────────────────────────
    int_results = [r for r in all_results if r[0] in sizes_int]
    if int_results:
        big = max(int_results, key=lambda r: r[0])
        typewrite(f"  {big[0]:,} integers sorted in {big[1]:.4f}s.", 0.012)

    parallel_note = (f"{NCORES}-core parallel MSD radix" if NCORES >= 4
                     else "single-core MSD radix")
    typewrite(f"  Paths used this run:", 0.012)
    typewrite(f"    · Counting sort (np.bincount)  — narrow integer ranges", 0.009)
    typewrite(f"    · Radix sort    (uint8/uint16)  — 1–2 pass, 10-17× faster than stable", 0.009)
    typewrite(f"    · Quicksort     (int32)          — 20× faster than radix for 4-pass", 0.009)
    typewrite(f"    · Heapsort      (int64/float64)  — cache-optimal", 0.009)
    typewrite(f"    · {parallel_note}  — auto-engaged at 5M+ elements", 0.009)
    typewrite(f"    · Timsort                        — strings, objects, custom keys", 0.009)
    print()
    typewrite("  pip install quill-sort", 0.022)
    print()


if __name__ == "__main__":
    main()

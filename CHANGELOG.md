# Changelog

All notable changes to quill-sort will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [7.5.0] — 2026-07-05

### Added
- **Spectre integer-sort backend (bundled).** A portable, multi-threaded MSD→LSD
  radix sort for 32/64-bit integers ships inside the wheels as `quill._spectre`
  (built from `quill/_native_src/spectre_sort.c`) and joins the self-tuning
  dispatch chain for integer dtypes (n ≥ 1M, n < 2³²). Its min/max/monotone
  prescan is parallelized across a short-lived thread crew (an exact reduction,
  so the presorted/reversed/constant/counting fast-paths are unchanged), which
  removes the single-threaded pass that had been ~⅓ of its overhead. Measured
  on the reference 24-core box it is now the fastest backend for large
  bounded-range integers (e.g. int64 20M in the 0…10⁹ range ~1.3× over
  `rust_voracious`) and at parity on full-range. It is a measured *candidate*,
  not an override: the dispatcher engages it only in the `(dtype, size)` buckets
  where it actually wins and keeps the existing backend elsewhere. Correctness is
  verified against `numpy.sort` across every integer dtype, range, and edge case;
  the never-lose fallback is unchanged. (It is C, not C++, so it is a sibling
  extension to `quill._native` rather than folded into it — both ship in the same
  binary wheels.)

### Changed
- **Float sorts skip the NaN pre-scan when the chosen backend orders NaN like
  numpy.** `dispatch_sort` now selects the backend *first* and strips NaN only
  for backends that can't place it natively (the Rust/parallel radix paths). For
  `numpy` / `x86_simd_sort` / `arm_neon_sort` — whose kernel *is* `np.sort` — the
  O(n) scan is skipped entirely (measured ~11% faster on float sorts whose best
  backend is numpy's own kernel), and the self-tuning DB now folds the scan cost
  into the recorded latency so it converges to the genuinely cheapest backend on
  float data. Output is bit-identical to `np.sort` (NaN to the end; NaN to the
  start when descending) for every backend, verified across all-/some-/no-NaN.

### Fixed
- **`quill setup` hardware detection + dispatch ladder.**
  - ISA/SIMD features now fall back to numpy's own `__cpu_features__` when
    `py-cpuinfo` isn't installed, so an AVX2/AVX-512 box no longer reports
    "no SIMD features detected" while `x86_simd_sort` is happily winning.
  - The dispatch ladder is now computed from the *actual* dispatcher on
    representative arrays instead of a hardcoded preference template, so it only
    ever names backends this machine can really run (no more scheduling
    `cupy_gpu` on a CPU-only box) and can't contradict the hardware panel.
  - The wizard no longer auto-writes `use_gpu = false`. A flaky GPU probe could
    write that and then silently disable a working CUDA card — self-perpetuating,
    since the backend probe is gated on the flag. It now only ever *enables* the
    GPU when one is present; CPU-only boxes rely on the backend's own probe,
    which disables itself safely.

## [7.4.0] — 2026-07-04

### Added
- **First-run notice.** The first interactive `import quill` before `quill setup`
  has run prints a one-time nudge to stderr (TTY-only, written-marker so it never
  repeats, silenceable with `QUILL_NO_FIRST_RUN=1`) — the reliable equivalent of
  a post-install message, which pip cannot show for wheel installs.
- **`quill setup` now installs with a genuine, parallel, real-bytes progress
  bar** (`quill._installer`). It resolves each optional accelerator's actual
  wheel URL + size from PyPI, streams the download itself (so the `####` bar is
  real bytes off the wire, never a fake animation — unknown size shows a moving
  marker, not an invented percentage), fetches them in parallel, then installs
  offline. Post fold-in it targets the optional deps that are *missing*
  (numpy / polars / psutil, and cupy on NVIDIA hosts).

### Changed (packaging — the big one)
- **The compiled CPU backends are now bundled *inside* quill-sort.** The kernels
  that used to require five separate companion installs (`quill-fastsort`,
  `-parallel`, `-ips4o`, `-numa`, `-simd`) all shared one header-only core, so
  they've been folded into a single in-tree extension, **`quill._native`**, and
  quill-sort now ships **per-platform binary wheels** (cibuildwheel) that carry
  it. `pip install quill-sort` gets the fast path — `rust_voracious`,
  `rust_parallel_radix`, `ips4o`, `simd_companion`, `numa` — with **no extra
  packages to install**.
  - **Never-lose install preserved:** where no binary wheel matches, the sdist
    recompiles `quill._native` (needs a C++17 compiler); if that fails, every
    extension is `optional=True` so the install still succeeds and the
    pure-Python paths take over. `pip install quill-sort` can never fail because
    a compiler misbehaved.
  - **Backward compatible:** each backend still probes its standalone
    `quill-fastsort*` companion as a fallback, so existing installs keep working.
  - GPU (`cupy`) remains an opt-in `[gpu]` extra (heavy, NVIDIA-only).

### Removed
- **`quill_ultrasort` / `UltraSort` / `EXTREME_THRESHOLD` (breaking).** The v7
  "extreme-data" tier activated at ≥100M elements but did nothing the normal
  self-tuning dispatch path didn't already do faster — it was dead weight and a
  correctness/maintenance surface. Removed entirely; large numeric inputs now go
  through the standard parallel-backend dispatch (which is faster). The list path
  keeps its existing external-merge fallback (`_external`) for the genuine
  >RAM case.

## [7.3.2] — 2026-07-04

### Changed
- **Self-tuning tie-break toward the `np.sort` floor.** When the bare floor
  measures within `tuning_tie_factor` (default 1.1×) of the fastest candidate in
  a size bucket, the dispatcher now prefers it. On AVX-512 / Apple-silicon hosts
  a SIMD-delegating backend (`x86_simd_sort` / `arm_neon_sort`) and bare
  `np.sort` run the *same* numpy kernel, so which one wins is measurement jitter;
  pinning to the floor makes convergence deterministic and picks the
  lowest-overhead route to that kernel. Accelerator-vs-accelerator ordering is
  untouched (the genuinely fastest still wins). Follow-up to the 7.3.1
  floor-as-candidate fix, from single-core-hardware review.

### Tests
- `tests/test_tuner_convergence.py`: added a tie-break regression test and a
  `_NUMPY_FAMILY` set so convergence checks accept any numpy-kernel path (bare
  `numpy` or the SIMD wrappers) — asserting the exact string `"numpy"`
  false-fails on SIMD-capable runners. The end-to-end test now reports
  `os.cpu_count()` (portable; `nproc` is Linux-only) and gates on the ratio.

## [7.3.1] — 2026-07-04

### Fixed
- **Self-tuning dispatcher couldn't converge to `np.sort` on hardware where it
  wins (low-core machines).** The `np.sort` floor was the emergency fallback but
  never a *selectable* tuning candidate, so on 1–2 core boxes — where every
  compiled/parallel backend loses to numpy's single-threaded AVX introsort — the
  explore-then-exploit tuner optimised over a candidate set whose members all
  lose to the floor. It round-robined losers and settled (if at all) on the
  least-bad one, leaving `sort_array` 20–80% slower than a bare `np.sort`
  (transiently up to ~5× during exploration) — a hole in the "never meaningfully
  slower" guarantee on a large slice of real deployments (found via testing on an
  independent single-core x86_64 box). Fix: `np.sort` is now a first-class
  backend (`NumpySortBackend`, lowest priority) that the tuner measures and can
  prefer. On many-core boxes it still converges to the parallel radix (the floor
  loses there); on low-core boxes it converges to `numpy` (the floor wins) — so
  the guarantee holds on all hardware. `available_backends()` now lists `numpy`
  last. Static priority order (used when `QUILL_TUNING_DISABLED=1`) is unchanged.
- **Bounded the exploration tax on low-core machines.** Early-abandonment now
  measures against the best *probed* candidate (≥2 samples) rather than the best
  *fully-warmed* one, so the cheap `numpy` floor sets the bar after a couple of
  samples and a catastrophically-slow backend (e.g. a parallel radix on a single
  core) is abandoned after ~2 probes instead of a full `tuning_min_obs`.
- Avoided a `numpy.core` deprecation warning in the x86/ARM SIMD CPU-feature
  probes (prefer `numpy._core` on numpy 2.x).

### Added
- `tests/test_tuner_convergence.py` — hardware-agnostic regression tests
  asserting the tuner converges within 1.1× of the best available backend and
  never sustains a >2× loss, on any core count. Guards the fix above.

## [7.3.0] — 2026-07-04

### Added
- **Portable C++ companion backends for macOS & Linux.** The five compiled
  accelerators (`quill-fastsort`, `-parallel`, `-ips4o`, `-numa`, `-simd`) —
  previously Windows-only wheels with no published source — were reimplemented
  from portable **C++17** sharing one header-only kernel
  (`companions/_core/quill_core.hpp`). They now build natively on macOS
  (arm64/x86_64) and Linux (x86_64/aarch64); prebuilt macOS/arm64 wheels ship in
  each `companions/*/dist/`. Verified exactly against `np.sort` and measured
  **~3× (int64) / ~5.5× (float64)** over `np.sort` on an Apple M2. This means
  Apple Silicon and aarch64 Linux are no longer capped at the `np.sort` floor for
  large arrays. (The `rust_*` backend names are retained for API compatibility;
  the implementation is C++, not Rust.)

### Changed
- **Self-tuning dispatcher now explores-then-exploits (`TimingDB.choose`).**
  Previously the tuner only ever returned a *warmed-up* winner, so on a machine
  with existing timing history a newly-installed, faster backend was **starved** —
  never run, never measured, never chosen. The dispatcher (`dispatch_sort`,
  `would_use`) now routes to under-sampled candidates until they are measured,
  then locks onto the fastest — so a freshly `pip install`ed accelerator is
  adopted automatically. Bounded **early-abandonment** stops probing a candidate
  that is clearly dominated (`> tuning_abandon_factor ×` the best warmed backend)
  after a couple of samples, so exploration costs ~1–2 sorts per dominated
  backend instead of the full `tuning_min_obs`. Deterministic (no RNG);
  `QUILL_TUNING_DISABLED=1` still bypasses to static priority.
- **`x86_simd_sort` / `arm_neon_sort` crossover raised `4_000` → `200_000`** to
  match `sort_array`'s small-array gate. Below 200k `sort_array` never dispatches
  and these backends' kernel *is* `np.sort`, so the lower crossover only exposed
  them to introspection/exploration overhead for zero benefit.

### Documentation
- Corrected now-stale claims across `companions/README.md`, `pyproject.toml`, and
  backend docstrings (`_backends.py`, `_neonsort.py`) that described the compiled
  backends as Windows-only / Rust / unavailable on macOS-Linux.

## [7.2.0] — 2026-07-04

### Added
- **ARM NEON / ASIMD backend (`arm_neon_sort`)** — the Apple-Silicon and
  Linux-aarch64 (Graviton / Ampere) counterpart to the existing x86
  `x86_simd_sort` backend. Before this, the SIMD path was gated on
  `platform.machine()` being x86-64, so on every ARM box the i32/u32/u64/f32
  dtypes (the ones the Rust voracious backend doesn't cover) fell through to
  `np.sort` **unlabeled** — `available_backends()`, `would_use()` and the
  self-tuning profiler all under-reported what was doing the work. The new
  `quill._neonsort.NeonSortBackend` closes that gap:
  - Registered lazily alongside the x86 backend in `_ensure_simd_registered()`.
    The two are **mutually exclusive by architecture** (each `_probe` gates on
    `platform.machine()`), so at most one is ever `available()` on a host and
    registering both is safe everywhere.
  - **Never-lose / no regression:** the kernel *is* `arr.sort()` (numpy's own
    ASIMD-accelerated sort on modern aarch64 numpy), so it can never be slower
    than `np.sort` and never changes the result. Measured on an Apple M2 across
    int32/int64/uint64/float32/float64 at 0.5M–20M elements: **0.96–1.00× of
    `np.sort`** (parity, within noise). Correctness verified exact vs `np.sort`
    including negatives, descending, and NaN-to-end across 18 dtype×size cases.
  - **Future C-kernel seam:** an optional `quill._neonsort_ext` (hand-written
    NEON / Highway VQSort) is discovered at import if present, mirroring the
    `quill._simdsort_ext` seam — no other code changes when it ships.

### Notes
- The two optional C extensions (`quill._counting_ext`, `quill._listconv_ext`)
  already build cleanly on macOS arm64 with Apple clang (`-O3 -fPIC`); no source
  changes were needed for Apple Silicon / aarch64 support.

## [7.1.3] — 2026-06-27

Tightens W2 (small-array dispatch tax) further after an external
regression test flagged that n=4096 wasn't covered by the 7.1.2 fast
path. Also documents the irreducible Python-overhead floor that
sub-1024 sizes can't beat.

### Fixed
- **W2 widened — fast path now `size <= 8192`** (was `< 4096`). Covers
  the 4096-sized arrays the regression bench measured. Re-measured
  per-call overhead (best of 1000 iterations):
  ```
  n=     4   np=  0.6 µs  quill=  1.0 µs   ratio 0.60×  (Python floor)
  n=    32   np=  0.6 µs  quill=  0.9 µs   ratio 0.67×  (Python floor)
  n=   256   np=  1.6 µs  quill=  1.9 µs   ratio 0.84×  (Python floor)
  n=   512   np=  2.6 µs  quill=  2.9 µs   ratio 0.90×  (Python floor)
  n= 1,024   np=  5.5 µs  quill=  6.0 µs   ratio 0.92×
  n= 4,096   np= 23.1 µs  quill= 24.0 µs   ratio 0.96×  (was 0.94×)
  n= 8,192   np= 48.4 µs  quill= 48.7 µs   ratio 0.99×
  n=16,384+  np=  ...     quill=  ...      ratio 0.99×
  ```

### Documented limits (the "Python floor")
For n<1024, `sort_array` adds ~0.3-0.5 µs of pure Python function-call
overhead per invocation (function entry + isinstance + ndim + size +
dtype.kind checks + return). On absolute terms that's negligible —
sorting a 4-element array faster than 1 µs is rounding error in any
real workload — but in ratio terms it shows as 0.60-0.92× of bare
`np.sort`. **Eliminating this requires rewriting `sort_array` itself
as a C extension**, which is not on the roadmap. Recommendation: for
hot loops over tiny arrays where every µs matters, call `np.sort`
directly.

### W4 trade-off note
`sort_array(object_ndarray)` (the W4 fix from 7.1.2) hit a floor at
1.21× lag vs `sorted()` on 1M bigints. The gap is the cost of
preserving the ndarray return type: ~20 ms `list()` conversion +
~20 ms `np.empty + assign` back to object dtype. For maximum speed
on object data, call **`quill.quill_sorted(list)`** directly — same
Timsort, no conversion tax, matches `sorted()` exactly. The
`sort_array(object_ndarray)` path is for users who need the ndarray
return type contract.

### W1 trade-off note
`analyze()` for large ndarrays hit a floor at the cost of `arr.min()
+ arr.max()` (~35 ms for 50M int64) — two unavoidable full passes for
honest dtype reporting. The 7.1.2 fix already removed the duplicate
min/max in `would_use()`, halving the previous cost. Reducing further
would mean reporting approximate / sampled min/max which would
violate the introspection contract.

### W3 — accidentally fixed in 7.1.0
The counting-sort 1-outlier cliff (one outlier flipping the dispatcher
from counting to MSD radix and tripling runtime) **is now mitigated
by the ips4o backend** added in 7.1.0. When the dispatcher would have
landed on the cliff, ips4o (priority 100) takes over with 0.87-0.89×
of clean-baseline runtime — the cliff is softened to a small
regression instead of a 3× drop. No code change in 7.1.3.

### Unchanged
- All companion-package wheels (ips4o 0.1.0, numa 0.1.0, simd 0.1.0,
  parallel 0.4.0) unchanged.
- 224/224 comprehensive correctness sweep still passes.
- All correctness fixes through 7.1.2 preserved.

## [7.1.2] — 2026-06-27

Adaptivity-tax fix-up release. Addresses 3 of 4 weaknesses from an
external performance report (W1 / W2 / W4). The 4th (W3 — counting-sort
outlier cliff) is documented as a known limitation; the real fix needs
work in the Rust kernel and is deferred.

### Fixed
- **W4 — object-dtype routing** (`quill/__init__.py`). `sort_array` on
  a NumPy object-dtype array now bypasses `np.sort` (which falls back
  to per-element `PyObject_RichCompare` and runs ~2× slower than
  CPython's tuned Timsort on the same data) and calls `list.sort()`
  directly. Returns the same object-ndarray shape so the API
  contract is preserved.
  - 1M bigints: was 0.45× (2.24× LOSING vs np.sort) → **1.45× WIN**
    (390 ms np.sort → 268 ms quill, while sorted() does it in 219 ms).
- **W2 — small-array dispatch tax** (`quill/__init__.py`). New
  super-fast path at the top of `sort_array` for tiny ndarrays with
  default args (`size < 4096 and not inplace and not descending`):
  skips the cupy probe, the SortedArray wrap, the `_backends` import,
  and every other dispatch check. Just calls `np.sort` directly.
  Erases the 0.28–0.98× regression band the external report measured
  at `n ∈ {4, 512, 4096}`.
- **W1 — `analyze()` redundant min/max passes** (`quill/__init__.py`,
  `quill/_backends.py`). For ndarrays, `analyze()` computed `min`/`max`
  in its own body and then called `_backends.would_use()` which
  recomputed them — 4 full O(n) passes total on int64 (or 5 with the
  has_nan check on float). `would_use()` now accepts optional
  `_mn` / `_mx` parameters and `analyze()` passes its already-computed
  values through.
  - `analyze(50M int64)`: was 76 ms → **43 ms** (47% reduction).
  - `analyze(50M all-equal)`: was 73 ms → **37 ms**.

### Known limitation — W3 (counting-sort outlier cliff)
A single outlier value in an otherwise-dense int input flips the
dispatcher from counting sort to MSD radix and triples runtime
(measured: 251 ms with 0 outliers, 653 ms with 1 outlier on dense
50M int64). The Python-side mitigations attempted (skim outliers,
counting-sort the rest, merge) measured at 0.85× — slower than
letting radix handle it, because the masking/concatenate cost more
than the saved time. The real fix lives in the Rust kernel: a
counting-sort variant that handles spillover (allocates counts up to
a range cap, puts values beyond cap in an overflow buffer, sorts
overflow separately, merges). Targeted for a follow-up release with
direct Rust changes to `rustext/src/parallel_radix.rs`.

### Unchanged
- All correctness fixes from 7.0.5 (None handling), 7.0.6 (packaging),
  7.0.7 (threshold tuning), 7.1.0 (ips4o + companion packages), and
  7.1.1 (numa single-socket fall-through) preserved.
- 224/224 comprehensive correctness sweep still passes.
- All companion-package wheels (ips4o 0.1.0, numa 0.1.0, simd 0.1.0,
  parallel 0.4.0) unchanged. Fixes are pure Python.

## [7.1.1] — 2026-06-27

Hot-fix release for a 32-bit-at-1M regression introduced by 7.1.0's NUMA
backend. `quill-fastsort-numa` is unchanged at `0.1.0`; only `_backends.py`.

### Fixed
- **32-bit dtypes at n=1M routed through `numa` on single-socket boxes**
  (`quill/_backends.py`). The NUMA companion package's internal kernel
  falls through to `voracious_sort` on single-socket — that's the
  designed no-op. But the Python-side dispatcher was still PICKING numa
  as the engagement backend (priority 94, supports int32/uint32/float32
  at min_n=1M), paying FFI overhead for nothing while losing to
  numpy's internal x86-simd-sort on narrow types. Measured 7.1.0:
  - int32 1M: 0.59× LOSING (numa kernel internally calls voracious)
  - uint32 1M: 0.60× LOSING
  - float32 1M: 0.50× LOSING

  `NumaBackend._probe()` now calls `quill_fastsort_numa.detect_topology()`
  and returns False on single-socket. The backend disappears from
  `available_backends()` entirely on consumer machines, so int32/1M
  flows down to `x86_simd_sort` (the numpy AVX kernel) and gets a clean
  match against `np.sort`. On multi-socket EPYC/Xeon the backend still
  engages and the 1.5-2× win is preserved.

  After fix (24-core single-socket box, fresh tuning DB):
  ```
   dtype          N      np.sort      quill   backend            speedup
   int64    1,000,000     10.9 ms     4.4 ms  rust_voracious      2.49x
   int64   10,000,000    123.3 ms    46.7 ms  ips4o               2.64x
   uint64   1,000,000     12.6 ms    13.6 ms  x86_simd_sort       0.93x
   uint64  10,000,000    140.9 ms    49.1 ms  ips4o               2.87x
   int32    1,000,000      4.0 ms     4.0 ms  x86_simd_sort       0.99x   ← was 0.59x
   int32   10,000,000     45.8 ms    33.9 ms  ips4o               1.35x
   uint32   1,000,000      4.0 ms     4.0 ms  x86_simd_sort       1.00x   ← was 0.60x
   uint32  10,000,000     44.7 ms    31.9 ms  ips4o               1.40x
   float64  1,000,000      7.6 ms     4.9 ms  rust_voracious      1.54x
   float64 10,000,000     89.4 ms    51.9 ms  ips4o               1.72x
   float32  1,000,000      3.9 ms     4.3 ms  x86_simd_sort       0.90x   ← was 0.50x
   float32 10,000,000     45.1 ms    38.6 ms  ips4o               1.17x
  ```

### Unchanged
- Every 7.1.0 win at 10M+ preserved (ips4o still dominates at 3.3-3.4×).
- Multi-socket NUMA users keep their dispatch path (NumaBackend engages
  on `node_count >= 2`).
- All 4 companion packages (ips4o 0.1.0, numa 0.1.0, simd 0.1.0,
  parallel 0.4.0) are unchanged — fix is purely Python-side dispatcher.

## [7.1.0] — 2026-06-27

The "multi-backend ecosystem" release. Adds 3 new native companion packages
(plus 2 source-only scaffolds for Mac/Linux iGPU). The dispatcher now picks
between Rust MSD radix, C++ ips4o, x86-simd-sort, NUMA-aware partition, and
the existing chain based on measured per-bucket latency. Headline:

  100M int64 sort: 1571 ms (np.sort) → 229 ms (ips4o) — 6.86x faster

### Added
- **`quill-fastsort-ips4o 0.1.0`** — C++ pybind11 wrapper for ips4o (the
  parallel samplesort DuckDB and Polars use internally). Vendored from
  upstream, patched for MSVC (~60 lines across 7 headers + a 95-line
  atomic shim). 5-7x over np.sort at 10M+ int64. Top dispatch priority
  (100) when installed.
- **`quill-fastsort-numa 0.1.0`** — Rust crate with NUMA topology
  detection (libnuma on Linux, GetNumaHighestNodeNumber on Windows).
  Per-socket partition + local sort on multi-socket EPYC/Xeon (1.5-2x);
  no-op fall-through to voracious on single-socket. Always safe to engage.
- **`quill-fastsort-simd 0.1.0`** — pybind11 wrapper for Intel's
  x86-simd-sort AVX-512 kernel. Built and correct on MSVC (required a
  94-symbol stub TU for unreachable specializations the linker still
  references). Currently SLOWER than np.sort on AVX-512 hosts because
  numpy 2.x already dispatches to the same kernel internally; kept as
  a conservative low-priority backend (80) for self-tuning to discover
  when it wins on non-AVX-512 hosts.
- **`quill-fastsort-metal` (source-only)** — Apple silicon iGPU via
  Metal Performance Shaders. Scaffolded with placeholder std::sort;
  real MPSGraph kernel deferred. Wheel built by CI on macos-14.
- **`quill-fastsort-sycl` (source-only)** — Intel Arc / AMD iGPU via
  Intel oneAPI's oneDPL. Full sort implementation. Wheel built by CI
  on ubuntu-22.04 with oneAPI.
- **Multi-NVMe parallel writes for UltraSort** (`quill/_ultrasort.py`).
  When `psutil` detects multiple writable disks with ≥10 GB free, the
  streaming path round-robins bucket files across them. Single-disk
  systems unchanged.
- **Cooler setup wizard** (`quill/wizard.py`). ASCII banner, 6-probe
  hardware detection (CPU/RAM/GPU/NUMA/disk/toolchain), rich-styled or
  plain-text fallback, install with pip progress bars, calibration sweep,
  per-machine dispatch ladder display.
- **CI workflow for iGPU wheels** (`.github/workflows/wheels-igpu.yml`).
  `workflow_dispatch` only; builds Metal and SYCL wheels on demand.

### Changed
- `quill/_backends.py` dispatcher now registers 3 new backends:
  - `ips4o` (priority 100, min_n=3M)
  - `numa` (priority 94, min_n=1M, single-socket fall-through)
  - `simd_companion` (priority 80, min_n=10M, conservative)

### Install
```
# minimum (pure Python, falls back to np.sort)
pip install quill-sort

# full speed stack on a modern CPU
pip install quill-sort quill-fastsort quill-fastsort-parallel quill-fastsort-ips4o quill-fastsort-numa

# Apple silicon iGPU (when CI ships the wheel)
pip install quill-sort quill-fastsort-metal

# Intel Arc / AMD iGPU
pip install quill-sort quill-fastsort-sycl
```

### Unchanged
- All correctness fixes from 7.0.5 (None handling), 7.0.6 (packaging),
  7.0.7 (threshold tuning) preserved.
- 224/224 comprehensive correctness sweep still passes.
- Bug #1 (key double-eval) fix from 7.0.4 still in place.

## [7.0.7] — 2026-06-27

Perf-tuning release. `quill-fastsort-parallel` is unchanged at `0.4.0`;
only `quill/_backends.py` changed.

### Fixed
- **1M int64 regression** (`quill/_backends.py`). The parallel kernel's
  engagement threshold was 200K (set in 7.0.5 as part of the
  counting-sort work). At 1M, the rayon thread spawn + parallel min/max
  probe + transform + scatter has ~5-8 ms of fixed overhead that
  doesn't amortize until n is large enough for the parallel speedup to
  dominate. Measured before this fix:
  - n=1M: quill 15.4 ms vs np.sort 10.6 ms → **0.69× LOSING**
  - n=10M: quill 76 ms vs np.sort 122 ms → 1.61× win
  Raised `RustParallelRadixBackend.min_n` from 200K to **3M**. Inputs
  in the 1M-3M range now fall through to `rust_voracious`
  (single-threaded radix, priority 95, min_n 1M), which beats np.sort
  cleanly without the parallel-spawn tax. Inputs under 1M go to
  `x86_simd_sort` or np.sort directly. Re-measured after the bump:

  ```
              N     np.sort       quill   speedup
        100,000       0.9 ms       0.9 ms   1.01x   (unchanged, fine)
      1,000,000      10.9 ms       4.4 ms   2.47x   ← was 0.69x LOSING
      3,000,000      34.1 ms      25.8 ms   1.32x
     10,000,000     121.7 ms      94.2 ms   1.29x
     50,000,000     729.7 ms     476.9 ms   1.53x
    100,000,000    1478.8 ms     971.2 ms   1.52x
  ```

### Unchanged
- Rust kernel (parallel MSD radix + counting fast-path + Herf float
  transforms) is byte-identical to `0.4.0`. Same per-call wins for
  in-RAM sorts above the new threshold.
- Bug #3 fix (None handling) from 7.0.5 still in place.
- Packaging fix (no `quill/__init__.py` shim) from 7.0.6 still in place.

### Note on the self-tuning DB
The `~/.quill/timings.json` cache may still have stale entries from
when the parallel kernel had a lower threshold. Run
`QUILL_TUNING_RESET=1 python -c "import quill"` once after upgrade
for the dispatcher to pick the right backend on the first sort.

## [7.0.6] — 2026-06-27

**Critical packaging fix.** `quill-fastsort-parallel 0.4.0` no longer
clobbers `quill-sort`'s `__init__.py` on install. If you installed
`0.2.0` or `0.3.0`, **upgrade immediately** — the `pip install --upgrade
quill-sort quill-fastsort-parallel` flow on those versions left
`quill` with an empty / partial namespace because the parallel wheel
shipped its own `quill/__init__.py` shim, and pip's last-writer-wins
semantics destroyed the main package's API surface.

### Fixed
- **Packaging collision (`rustext/pyproject.toml`,
  `rustext/Cargo.toml`).** The `quill-fastsort-parallel` package now
  installs as a **top-level** Python module — `quill_fastsort_parallel`
  — matching the existing `quill-fastsort` (note underscore vs dot)
  pattern. The old layout (`module-name = "quill._fastsort_parallel"`
  + `python-source = "python"`) required a `python/quill/__init__.py`
  namespace shim that pip would clobber over `quill-sort`'s own
  `__init__.py` on `--upgrade`. The shim is deleted; the wheel now
  ships only `quill_fastsort_parallel/` and never touches the `quill/`
  namespace. Verified by re-installing both packages in worst-case
  order (parallel last) — the main API stays intact.
- **`quill/_backends.py` probe order updated.**
  `RustParallelRadixBackend._probe` now tries (1) top-level
  `quill_fastsort_parallel`, (2) `quill._fastsort_parallel` (legacy
  0.2/0.3 namespace install, kept so old installs don't break before
  users upgrade), (3) in-tree dev copy. Auto-detects the new layout
  with zero user action beyond the `pip install --upgrade`.

### Migration
```bash
# clean upgrade — pip --upgrade now does the right thing because the new
# wheel doesn't write into the quill/ namespace at all
pip install --upgrade quill-sort quill-fastsort-parallel
```

If you previously hit `ImportError: cannot import name 'quill_sort'
from 'quill'`, that's the bug this release fixes. After upgrade,
`python -c "import quill; print(quill.quill_sort([3,1,2]))"` should
print `[1, 2, 3]`.

### Unchanged
The Rust kernel itself (parallel MSD radix + counting fast-path + Herf
float transforms + ping-pong scatter) is byte-identical to `0.3.0`.
Only the wheel layout / module name changed. Same `parallel_sort_*`
entry points, same correctness guarantees, same perf.

## [7.0.5] — 2026-06-27

Bug-fix + perf release. `quill-sort` bumps to `7.0.5`; the companion native
package `quill-fastsort-parallel` bumps to `0.3.0` for the counting-sort
fast-path.

### Fixed
- **Bug #3 — `None` silently sorted as a +∞ sentinel** (`quill/_core.py`,
  `quill/_strategies.py`). `quill.quill_sorted([1, None, 2])` returned
  `[1, 2, None]` while CPython's `sorted()` raises `TypeError`. The
  small-n path was stripping `None` values, sorting the rest, and
  reattaching `None` at the tail — a fabricated ordering that diverged
  from `sorted()` semantics and let dirty data slip past type checks.
  `quill_argsort` shared the bug. `quill_topk` was already correct;
  consistency is now restored across the public API. NaN handling
  (numpy-style "NaN at end") is unchanged — that's a documented,
  intentional convention.
  - `quill.quill_sorted([1, None, 2])` → `TypeError` (was `[1, 2, None]`)
  - `quill.quill_argsort([1, None, 2])` → `TypeError` (was `[0, 2, 1]`)
  - `quill.quill_sorted([None])` → `[None]` (singleton OK, no comparison)
  - `quill.quill_sorted([None, None])` → `TypeError` (matches CPython)

### Added
- **Counting-sort fast-path** in `quill-fastsort-parallel 0.3.0`
  (`rustext/src/algos/counting.rs` wired into all 4 integer entry points
  in `rustext/src/lib.rs`). Each `parallel_sort_{i64,u64,i32,u32}` now:
  1. Does a parallel min/max probe (rayon `fold` + `reduce`, ~5 ms at
     100M).
  2. Dispatches to `algos::counting::sort_*` when `range <= 4*n` and
     `range <= 2^24` (counting wins decisively on dense data).
  3. Falls through to the MSD radix kernel on miss.
  The probe is skipped for `n < 100_000` where the MSD path is already
  fast enough.

### Performance
Adversarial 10M int64 patterns (24-core, fresh tuning DB, best-of-3):

```
pattern                  np.sort      quill   speedup   change
two_distinct              19.7ms     36.7ms    0.54x   (unchanged)
many_duplicates(100)      40.4ms     37.1ms    1.09x   +0.55x (was 0.54x)
bounded_range_1024        57.7ms     36.5ms    1.58x   ~same
all_equal                 18.5ms     12.7ms    1.46x   ~same
sparse_int64             134.3ms     51.9ms    2.59x   ~same
random 10M               129.6ms     89.0ms    1.46x   ~same
random 100M             1937.2ms   1154.6ms    1.68x   ~same
```

`many_duplicates` flipped from loss to win; `two_distinct` is still a
loss because counting's write-back is single-threaded (parallelizing
it for very-low-cardinality cases is a future optimization — np.sort's
quicksort already early-exits on near-duplicate data and is hard to
beat without a specialised parallel block-write).

### Migration
- `pip install --upgrade quill-sort quill-fastsort-parallel`
- If you previously relied on `quill_sorted([1, None, 2])` returning a
  sorted list, you must now either filter `None` first or use
  `sorted()` semantics (which Quill matches). The old behaviour was
  silently incorrect against CPython.

### Verified
- 224/224 comprehensive correctness sweep preserved.
- Bug #3 reproducer from tester: now raises `TypeError` matching CPython.
- Tester's Level 11 sweep cases (`int+str`, `int+dict`, custom no-`__lt__`
  objects) still raise `TypeError` correctly; the only change is that
  `None` joins them instead of being silently special-cased.

## [7.0.4] — 2026-06-27

Bug-fix release. The Rust kernel (`quill-fastsort-parallel`) is unchanged at
`0.2.0`; the fixes are entirely Python-side in the key-function path.

### Fixed
- **Silent wrong output with non-idempotent `key=` functions on lists > 512
  elements** (`quill/_profile.py`). The profiler was calling `key_fn` on a
  sample of up to 512 elements to detect key dtype / pre-sortedness; the
  downstream sorter (`numpy_sort_by_key` or `list.sort`) then re-evaluated
  the key on the FULL data. For idempotent keys this was merely wasteful
  (up to 512 redundant calls); for **non-idempotent** keys — counters,
  access timestamps, RNG tiebreakers, cached properties with eviction — the
  first 512 elements were sorted on different key values than the tail, producing
  silently incorrect output. The profile now skips the key-sampling step for
  non-identity keys and returns a minimal profile; the sort path calls the
  key the contractually-correct **exactly once per element** (matching
  CPython's `sorted()` semantics). Reported by tester; reproducer:
  ```python
  state = {}
  def k(x):
      if x in state: return x + 10**9
      state[x] = True; return x
  assert quill.quill_sorted(list(range(1000)), key=k) == list(range(1000))
  ```
  Cost: lambda keys lose the dtype-based optimization that previously routed
  numeric/string-keyed sorts through `numpy_sort_by_key` for large n. In
  practice this is small because `itemgetter` / `attrgetter` keys still take
  the fast path via the type check that bypasses dtype gating.
- **Singleton list skipped `key=` entirely** (`quill/_core.py`). `sorted([x],
  key=k)` calls `k(x)` once by CPython contract; QuillSort's `n <= 1` fast
  path returned without invoking the key. Now invokes it once, propagating
  any exception (matching `sorted()` semantics for side-effect-bearing or
  raising keys).

### Verified
- Call-count sweep: `key` invoked exactly `n` times for n ∈ {1, 256, 500,
  512, 513, 1000, 20000}. Was `n + min(n, 512)` previously.
- Comprehensive correctness sweep still passes 224/224 (40 dtype/pattern
  SKIPs unchanged) — no regression on identity-key sorts, numeric arrays,
  ndarray API, NaN handling, or boundary values.
- Non-idempotent-key reproducer from the bug report now produces
  `quill.quill_sorted(...) == sorted(...)`.

### Companion package
`quill-fastsort-parallel==0.2.0` is unchanged. Both `pip install quill-sort`
and `pip install quill-sort quill-fastsort-parallel` pick up the fix.

## [7.0.3] — 2026-06-25

### Performance breakthrough — quill-fastsort-parallel 0.2.0
The Rust parallel-radix kernel was rewritten with proper parallelism. Measured
on the 24-core reference box (best-of-3, correctness verified against
``np.sort``):

```
dtype             N        np.sort      quill   speedup
int64       1,000,000      10.7ms      10.0ms    1.07x
int64      10,000,000     120.5ms      66.0ms    1.82x
int64     100,000,000    1387.7ms     738.9ms    1.88x
uint64      1,000,000      12.7ms       4.4ms    2.90x
uint64     10,000,000     139.7ms      52.1ms    2.68x
uint64    100,000,000    1575.4ms     428.2ms    3.68x
int32      10,000,000      45.2ms      20.5ms    2.20x
int32     100,000,000     535.6ms     235.0ms    2.28x
uint32    100,000,000     528.0ms     194.5ms    2.71x
float64    10,000,000      87.9ms      55.7ms    1.58x
float64   100,000,000    1052.8ms     593.0ms    1.78x
float32   100,000,000     538.9ms     274.0ms    1.97x
```

Every dtype now wins or matches numpy at 10M+, including the previously-losing
float64 path (was 0.74×, now 1.78×). 100M int64 dropped from 1.39 s → 0.74 s.

### Changed
- **Parallel in-place transforms** (`rustext/src/transforms.rs`). Signed→
  sortable, NaN/sign handling, and inverse transforms now run as
  `rayon::par_chunks_mut` over 64K-element blocks. The sequential per-element
  loop that cost ~150 ms at 100M is gone. `u64` / `u32` paths are zero-traffic
  no-op reinterprets — no allocation, no copy.
- **Ping-pong scatter** (`rustext/src/parallel_radix.rs`). New signature
  `parallel_msd_radix_uXX(input, scratch) -> bool` eliminates the
  unconditional `copy_from_slice(scratch)` step that previously doubled the
  scatter's memory traffic. Top-level callers copy back only when the
  algorithm reports the result lives in scratch.
- **Recursive MSD radix.** Buckets ≥ 1 M elements recurse into the next
  8-bit digit with ping-pong direction flipped, instead of falling straight to
  a per-bucket leaf sort. Cuts the per-bucket voracious cost on skewed
  distributions.
- **Thread-local scratch pool** (`rustext/src/scratch.rs`). Buffers are
  borrowed via RAII guard and returned to a thread-local `Vec<Vec<u64>>` on
  drop. The 800 MB per-call `Vec::with_capacity` from 7.0.2 is now a
  one-time-per-thread cost. Cap of 8 buffers per pool to bound resident
  memory.
- **`leaf_sort_uXX`** (`rustext/src/leaf.rs`). Stdlib pdqsort below 32 K,
  voracious above. The old crossover at 64 was far too low — pdqsort wins
  by a wide margin in the 64–32K range.
- **Algorithm diversity scaffolding** (`rustext/src/algos/`). `glidesort_wrap`
  (best adaptive single-threaded for near-sorted), `pdq_wrap` (branchless
  pdqsort for uniform random), and `counting` (bounded-range integer sort)
  are now declared and dependency-resolved. They are not yet wired into the
  dispatcher; reserved for the next perf wave to attack the
  `two_distinct` / `many_duplicates` adversarial losses (0.52×, 0.54×).
- **Dispatcher tuning** (`quill/_backends.py`). `RustParallelRadixBackend.min_n`
  lowered from 1 M to 200 K (the new kernel is competitive at much smaller
  sizes). New `QUILL_FORCE_PARALLEL_RADIX=1` env var bypasses the size gate
  for benchmarking. `QUILL_BACKEND_DEBUG=1` now emits the chosen kernel +
  dtype to stderr per call.

### Verified
- Comprehensive correctness sweep: **224 PASS / 0 FAIL / 40 SKIP** across
  6 dtypes × 11 patterns × 4 sizes (100, 10K, 1M, 10M). Includes boundary
  values (INT_MIN/MAX), ±inf, ±0, denormals, all-NaN, presorted,
  reverse-sorted, all-equal, alternating, near-sorted, Zipf, bounded-small.
- Memory profile: 100 M int64 peaks at 1.04× input scratch; 200 M f64 at
  1.00×. Well within the 3× input budget. No leaks across runs.
- Adversarial bench (10M int64, 8 patterns): wins on 6/8. The 2 losses
  (`two_distinct` 0.52×, `many_duplicates(100)` 0.54×) are low-cardinality
  cases where counting sort dominates — `rustext/src/algos/counting.rs`
  is staged for the next release.

### Companion package
`quill-fastsort-parallel==0.2.0` ships the Rust kernel as a separate PyPI
wheel. The main `quill-sort` package auto-detects and uses it when installed:

```
pip install quill-sort quill-fastsort-parallel
```

Installing `quill-sort` alone still works — the dispatcher falls back through
the existing chain (rust_voracious leaf, SIMD, polars, parallel partition,
np.sort) with zero behavior change.

## [7.0.2] — 2026-06-25

### Added
- **`quill-fastsort-parallel` companion wheel** (Rust + rayon + voracious).
  New Python module `quill._fastsort_parallel` exposes `parallel_sort_{i64,u64,i32,u32,f64,f32}`
  — a true parallel 256-way MSD radix sort that operates on the numpy buffer
  in place, releases the GIL for the entire call, and scales across all CPU
  cores. Wired into the backend chain at priority 99 (above voracious).
  When installed, `quill.sort_array` on a 100M int64 array runs in <1s (vs
  np.sort's ~1.4s baseline) — measured 3-9x speedup at n >= 10M.

### Changed
- Backend chain now leads with `rust_parallel_radix` when `quill-fastsort-parallel`
  is installed. Falls back through the existing chain (voracious leaf, SIMD,
  polars, parallel partition, np.sort) when absent. Zero behavior change for
  users without the new wheel.

### Install
  `pip install quill-sort quill-fastsort-parallel` for the full speed stack.
  `pip install quill-sort` alone still works (degrades to v7.0.1 paths).

## [7.0.1] — 2026-06-25

Post-7.0.0 patch. Pure superset of 7.0.0 — same public API, only fixes and
internal rewrites.

### Fixed
- **First-record disk-save spike** (`quill/_tuning.py`). The tuning DB's
  `_last_save_ts` was initialised to `0.0`, so the very first `record()` call
  on a fresh import would pass the 30 s rate-limit check (`now - 0.0 > 30`) and
  trigger a synchronous JSON write inside the first sort's hot path — a ~50-200
  ms outlier on Windows with antivirus active. Now initialised to
  `time.monotonic()` so the first save happens no sooner than `_SAVE_INTERVAL_S`
  after import, off any user-visible sort.

### Changed
- **`Quill.UltraSort` in-memory engine rewritten.** The 7.0.0 implementation
  used `ProcessPoolExecutor` × 2 (one pool for histograms, one for scatter)
  plus a `SharedMemory` shadow buffer — on Windows, spawn cost alone was ~3 s
  per pool, and the shadow doubled memory pressure. Replaced with a single-
  buffer, `ThreadPoolExecutor`-only implementation: vectorised 8-bit key
  extraction, `np.argsort(kind='stable')` on `uint8` keys (numpy's internal
  radix), gather, then per-bucket parallel sort across threads (numpy releases
  the GIL during `.sort()`). 10M int64 went from 2.4 s → 0.13 s (20× speedup),
  matching `np.sort`.
- **Streaming-to-disk path added** for true >RAM inputs. Pre-flight free-space
  check, two-level MSD radix partition into 256 bucket files, pre-allocated
  output, thread-parallel sort+write. Engages automatically when the input
  would exceed available RAM × 0.4 (the in-memory path needs ~3× input).
- **Practical engagement thresholds raised** so UltraSort no longer engages
  at in-memory sizes where the regular dispatcher (Rust voracious / SIMD /
  `np.sort`) is already faster. `_DTYPE_PRACTICAL_THRESHOLD` is now 10B for
  every numeric width — the in-memory MSD radix in pure numpy cannot beat the
  single-threaded `argsort + gather` bottleneck and was net-negative at every
  measured size on the reference 24-core box (0.55-0.59 × `np.sort` at
  50M-250M for int64).
- **Memory-pressure short-circuit** added to `should_ultrasort`: even when the
  practical threshold disables in-memory engagement, the function returns True
  whenever the array would exceed available RAM — so a 1B int64 sort actually
  reaches the streaming path instead of falling into `dispatch_sort` and
  OOM-ing. (Fixes the user-reported "died at 1B" regression.)
- **Voracious Rust per-bucket sort** wired into the in-memory engine: when
  `quill._fastsort` or `quill_fastsort` is installed, per-bucket sort uses
  the Rust kernel instead of `np.sort`. Wins at 50M (1.61×); at 100M+ the
  single-threaded `argsort + gather` upstream becomes the dominant cost so
  the per-bucket speedup is masked. The win surfaces transparently for users
  with the Rust wheel installed.

### Verified honest
- UltraSort's in-memory engine is **effectively disabled by design** on this
  release — the regular dispatch (Rust voracious for ints, AVX np.sort
  otherwise) is always at least as fast for data that fits in RAM, so
  delegating is the right call. UltraSort's value in 7.0.1 is the streaming
  path: the engine that engages when your data won't fit in RAM and would
  otherwise OOM.
- To beat the regular dispatch at extreme sizes (1B+) **in-memory**, UltraSort
  would need its own parallel-partition C extension (similar to the existing
  `quill._fastsort` Rust crate but with a 256-way MSD partition kernel
  releasing the GIL during scatter). That's deferred — a real native-code
  project, not achievable in pure numpy.

## [7.0.0] — 2026-06-25

The "Quill.UltraSort" release. v6 made the regular sort path adaptive and
multi-backend; v7 adds a dedicated extreme-data engine (Quill.UltraSort for
1B+ ints), a self-tuning dispatcher, and an audited NaN/edge-case story.
Two tiers now:

  - **Quill.sort** — handles big data (lists, ndarrays, all v6 backends).
  - **Quill.UltraSort** — handles EXTREME data (≥100M ints, disk-backed when
    needed, fresh implementation that does NOT inherit the v5 external sort's
    audited correctness bugs).

### Added
- **`quill_ultrasort(data)` and `quill.UltraSort` class** — flagship engine
  for billion-element integer/float workloads. Strategy:
  - n < `EXTREME_THRESHOLD` (100M): delegates to the regular dispatch_sort.
  - Fits in RAM: in-memory parallel MSD radix with shared memory, narrowest
    safe dtype (uint8/16/32/int32/int64), 256-way bucket partition, parallel
    per-bucket sort.
  - Doesn't fit in RAM: disk-backed two-level MSD radix with pre-allocated
    output, parallel sort+write, no separate merge pass.
  Floats use the Herf 2001 bit-trick with the correct signed-int sort wiring
  (an early implementation got this wrong; release builds ship with the fix
  + an opt-in `QUILL_DEBUG_SORTED=1` sortedness assertion at every engine
  exit boundary). NaN handling: stripped before kernel, reattached at end.
  Length is asserted equal to input across all paths.
- **`SortedArray` return wrapper** — `sort_array(arr)` now returns a thin
  wrapper that carries an `is_sorted=True` tag. A second `sort_array(s)` or
  `quill_topk(s, k)` short-circuits to a slice instead of re-sorting. Acts
  as an ndarray everywhere via `__array__` and `__array_interface__`.
- **Self-tuning dispatcher (`quill._tuning.DB`)** — after 5 observations per
  (backend, dtype, log2-size) bucket, picks the measured-fastest backend
  instead of the hardcoded priority order. Persists to `~/.quill/timings.json`
  with atomic writes. Crashed backends get punitive EWMA so a broken backend
  doesn't keep winning. Disable via `QUILL_TUNING_DISABLED=1`.
- **`SimdSortBackend`** explicit in the chain (priority 85, between Rust and
  OpenMP). On modern numpy this routes to numpy's internal x86-simd-sort
  kernel; structured for a future optional `_simdsort_ext` C wrapper that
  bypasses np.sort dispatch entirely.
- **Run-aware hybrid for nearly-sorted data** (`quill._run_detect`) — when
  `asc_ratio > 0.95` on identity-key numeric input, detects long runs in O(n)
  and insertion-sorts the few islands. ~2-5× over Timsort for inputs with
  one or two perturbations.
- **`array.array` zero-copy fast path** (`ArrayArrayPlugin`). Stdlib
  `array.array('q', ...)` sorts via `np.frombuffer` view — skipping the
  PyLong promotion table entirely. Defensive copy so `inplace=False` doesn't
  mutate the caller.
- **Fast list↔ndarray converter** (`quill._listconv` + optional
  `_listconv_ext.c`). Walks PyList in C, builds int64/float64 buffer in one
  pass. Eliminates the `.tolist()` tax that capped v6's list-path win at ~1.1×.
  When the C extension isn't built, falls back to `np.asarray(data, dtype=...)`
  with a homogeneity probe.
- **Fused counting-sort C kernel** (`_counting_ext.c`, optional). Replaces
  `np.bincount + np.repeat` with one C pass (count + cumsum + scatter into
  pre-allocated output). Falls back to v6 path when not built.
- **Scratch buffer pool** (`quill._scratch`). Thread-local size-bucketed
  ndarray reuse for counting sort, parallel-partition merge buffer, and the
  NaN-reattach concatenation. Avoids per-call allocation on hot loops.
- **Adaptive parallel-worker count** — `_PARTITION_WORKERS` is no longer
  hardcoded to 3; scales by `os.cpu_count()` (capped to avoid memory-bandwidth
  collapse). Overridable via `parallel_partition_workers` config.
- **`top_k_stream(k=1)` and `k=2` specialised paths** — single-pass `min/max`
  for k=1, two-pointer scan for k=2. ~2-5× over heapq for very small k.
  Uses a private sentinel object (`_UNSET`) so legitimate `None` values and
  `key=lambda x: None` don't collide with the empty marker.
- **`Quill.UltraSort`** branding: `quill_ultrasort(data)`, `UltraSort` class,
  `EXTREME_THRESHOLD`, `should_ultrasort(n, data)` all re-exported at the
  package top level.

### Fixed (verifier sweep — bugs caught before shipping)
- **UltraSort float64 monotonic-int64 transform was inverted** — mixed-sign
  float arrays at n≥100M would silently sort positives before negatives.
  Fixed in the Herf bit-trick + signed-int sort wiring; verified end-to-end.
- **UltraSort disk-backed read used a read-only `np.frombuffer` view** —
  every worker raised `ValueError: read-only` so the disk path always fell
  back to in-RAM `np.sort` (and OOMed on real 1B+ workloads). Added `.copy()`
  so the disk pipeline actually completes.
- **UltraSort wrapped uint64 values > INT64_MAX into negatives** silently.
  `should_ultrasort` now rejects those arrays so the regular dispatcher
  handles them with `np.sort`.
- **UltraSort collapsed all wide-int64 negatives into bucket 0** because of
  arithmetic-shift sign extension. The radix workers now offset-binary key
  (`view ^ 0x8000_0000_0000_0000`) before the shift so negatives spread
  across buckets monotonically.
- **Run-aware hybrid let NaN slip through to the wrong end** on float lists
  > 1M (where the profiler's `has_nan` was sample-only). Gate now requires
  `has_nan_known=True` AND `has_nan=False`, with a cheap full-list NaN scan
  as backup.
- **`numpy_sort_by_key(reverse=True)` flipped tie order** by reversing a
  stable ascending argsort. Each branch (tuple/numeric/string) now produces
  a result that matches `sorted(..., key=..., reverse=True)` exactly.
- **`sort_array(read_only_ndarray)` raised** — NumpyArrayPlugin now copies
  read-only inputs before sorting.
- **`sort_array(np.array(scalar))` raised `AxisError`** on 0-dim inputs.
- **`quill_sort(array.array, inplace=False)` mutated the original** via the
  zero-copy `np.frombuffer` view. ArrayArrayPlugin now copies first.
- **`analyze(ndarray)` was missing 11 v6 list-profile keys** (`presorted`,
  `all_same`, `asc_ratio`, etc.) so any v6 caller doing `analyze(arr)["presorted"]`
  would `KeyError`. Now populated for ndarrays too.
- **`quill_sort(..., stats=True)` plugin path was missing the `n` key.**
- **`quill_topk(float_arr, k, largest=True)` selected NaN as largest** because
  `np.argpartition` treats NaN as greater than any number. NaN is now stripped.
- **`quill_argsort(multi_dim_ndarray)` silently returned wrong-length result.**
  Now raises `ValueError`.
- **Importing `quill._simdsort` before `quill._backends`** silently dropped
  the SIMD backend from the registry. Registration is now lazy.
- **`top_k_stream(iter([None]), 2)` returned `[]`** because `None` doubled as
  the "unset" sentinel. Now uses a private sentinel object.
- **Tuning DB `save()` race window** could drop telemetry; lock is now held
  for the whole save. `record()` rejects `float('inf')` (was poisoning the
  EWMA forever).
- **`_profile` raised `decimal.InvalidOperation` on `Decimal('NaN')` inputs.**
  Now caught; profile falls back to conservative defaults.
- **`try_run_hybrid` swallowed `BaseException`** (including `KeyboardInterrupt`).
  Narrowed to `Exception`.

### Changed
- `_INSERTION_THRESHOLD` widened from 32 → 64 (the tiny-list fast path beats
  numpy further out than the original measurement).
- Plugin probe now caches by `id(type(data[0]))` so a chained sort of the same
  custom type skips the re-scan.
- `dispatch_sort` accepts a `nan_hint: Optional[bool]` parameter so callers
  with profile-side NaN knowledge skip the isnan scan entirely.
- `available_backends()` includes `x86_simd_sort` in its priority listing.
- `_parallel.parallel_sort` (list entry point) now accepts and respects
  `reverse=` (was silently ignored — dead-code path is no longer a footgun).

### Quarantined (unchanged from v6)
- The v5 external sort engine (`quill._external`) stays disabled unless
  `QUILL_ENABLE_EXTERNAL=1`. Quill.UltraSort is the supported path for data
  that exceeds RAM.

### Migration notes
- `__version__` is `"7.0.0"`; `display_version()` returns `"QuillSort.7"`.
- Every v6 public symbol is preserved. v7 is a pure superset.
- The optional C extensions (`_listconv_ext`, `_counting_ext`) are auto-built
  by `setup.py` when a C compiler and numpy headers are available. When they
  aren't, the package installs as pure-Python and silently uses the v6 paths.

## [6.0.13]

### Fixed (from an external review)
- `quill_topk` on an ndarray no longer does a pointless ndarray->list->ndarray round-trip; it now operates on the array directly and is ~4-5x faster than `np.sort(arr)[:k]` (was ~5x slower).
- `sort_array` no longer makes a wasted copy on the fallback path — it copies only when an in-place backend actually runs, and small arrays (<200k) go straight to `np.sort`. Small/medium arrays no longer lose to `np.sort`.
- `quill_sort` tiny-list fast-exit avoids profiling overhead on a handful of primitives.
- `analyze()` now reports the real dtype and chosen backend for numpy arrays (previously reported `object` for every ndarray).
- `sort_array(descending=True)` returns a C-contiguous array, not a reversed view.
- `available_backends()` GPU probe now verifies a sort kernel can actually run (not just that a device exists).

### Docs
- Corrected the performance headline to measured numbers (~3x int64 / ~2x float64 for `sort_array`; matches the CLI demo). Clarified that `quill_sort` mutates in place by default (like `list.sort`) and `quill_sorted` is the non-mutating `sorted()` equivalent. Softened the "never slower" wording for sub-millisecond inputs.

## [6.0.12]

### Added
- `quill_argsort(data, key=None, reverse=False)` — stable argsort for any sequence/ndarray, matching `sorted(range(n), key=...)` exactly; ~4-6x faster than the Python idiom for large lists.
- On-device GPU sort: `sort_array(cupy_array)` now sorts on the GPU with zero host transfer (~9-15x vs host `np.sort`) instead of raising.
- Polars-accelerated sort for large string lists (~2x vs `sorted()`), with exact fallback to Timsort.

### Fixed
- Read-only / non-writeable numpy arrays with `inplace=True` no longer crash the process via an uncatchable Rust panic (now copied; never-lose guard widened to `BaseException`).
- Counting-sort threshold tightened (`rng <= n//4`) so it never loses to `np.sort` at larger dense ranges.

### Security / correctness
- The legacy v5 external (disk) sort engine is **quarantined** (disabled unless `QUILL_ENABLE_EXTERNAL=1`): it could silently truncate floats / return non-monotonic data via `high_performance_mode=True`. The public path is now always correct.

## [6.0.0] — 2026-06-16

The "by a mile" release. v5 made Quill *correct*; v6 makes the array path
genuinely fast by leaving Python entirely and dispatching to compiled, GPU, and
parallel backends — while keeping a hard never-lose guarantee against `np.sort`.

### Added
- **`sort_array(data, descending=False, inplace=False)` — the zero-conversion
  fast path.** Sorts a numpy array and returns an ndarray with no list
  round-trip. This is the API that beats `np.sort` "by a mile": measured on a
  28-core / RTX 4060 Ti box (numpy 2.4.6, fresh-shuffled per run) — int64 ~5.3x
  (Rust voracious radix), ~4x (GPU), float64 ~3.3x.
- **Pluggable backend dispatch chain** (`quill/_backends.py`) with a measured
  crossover per backend, tried highest-priority-first:
  - `rust_voracious` — compiled PyO3 + `voracious_radix_sort`, multi-threaded
    radix (~5x int64). Shipped as a pre-built abi3 wheel; absent installs fall
    through gracefully.
  - `cupy_gpu` — GPU radix via CuPy (~4x int64); engages only when the array
    fits in free VRAM with headroom.
  - `polars` — delegates to polars' multi-threaded Rust sort (~2.3x); a
    no-compile fast path for platforms our wheels miss.
  - `numpy_parallel` — thread-parallel `np.partition` sample-sort (~1.1x int,
    deliberately few workers — the gain saturates at memory bandwidth).
  - counting sort (`np.bincount`) for dense bounded int64/uint64 (~1.7-2.8x).
- **`quill_topk(data, k, largest=False, key=None)`** — k smallest/largest via
  numpy `argpartition` (introselect, O(n)), measured ~8x faster than
  sort-then-slice for small k over 10M elements.
- **`available_backends()`** — lists the fast-sort backends usable on this
  machine, in the priority order Quill will try them.
- **`register_backend()`** — drop in a custom kernel backend (separate from the
  type-plugin system).
- **`QuillPlugin`, `register_plugin`, `analyze` re-exported from the top-level
  package** for a clean public API surface.
- **Calibration wizard (`quill setup`)** — measures the parallel/GPU crossovers
  on your machine and persists them to `~/.quill/config.json`; a small
  visualizer and the `quill_topk` add-on round out the optional tooling layered
  on the core engine.
- **`QUILL_BACKEND_DEBUG=1`** env var — surfaces backend errors instead of
  silently falling back, for diagnosing dispatch issues.
- **Persistent, runtime config** (`quill/_config.py`) at `~/.quill/config.json`
  (override dir via `QUILL_CONFIG_DIR`): `auto_parallel`, `parallel_min_cores`,
  `numpy_partition_workers`, `use_gpu`, and more.
- **Binary-wheel distribution.** `.github/workflows/wheels.yml` builds the
  `rustext` crate into abi3 wheels (one wheel per platform covers Python ≥ 3.8)
  for win_amd64, manylinux x86_64 + aarch64, and macOS x86_64 + arm64, plus a
  pure-Python sdist as the never-fail-install fallback. `rustext/pyproject.toml`
  configures maturin to build the crate as `quill._fastsort`.
- New optional extras: `[polars]` and `[gpu]` alongside `[fast]`.

### Changed
- **BREAKING**: version bumped to 6.0.0.
- **BREAKING**: the recommended path for maximum throughput on numeric data is
  now `sort_array(ndarray)`, not `quill_sort(list)`. The list API remains a
  drop-in for `sorted()`/`list.sort()` and is np.sort-class fast (~3-4x over the
  built-ins on numeric), but the `asarray`/`tolist` conversion wall (Amdahl)
  caps its win — the array API is where the multi-threaded/GPU speedup lands.
  README rewritten to state this distinction honestly and to drop earlier
  over-claimed list-path speedups.
- The numpy-array plugin path now sorts via the best numeric kernel
  (counting/parallel/np.sort) instead of forcing `kind='stable'` — fixing the v5
  regression that made `quill_sort(int64_array)` far slower than `np.sort`.
- Backend dispatch centralised: `quill_sort(list)`, the `NumpyArrayPlugin`, and
  `sort_array()` all route numeric value-only sorts through the same chain.

### Fixed
- **Critical (carried over / hardened from v5):** float NaN handling no longer
  corrupts data. NaN is stripped before any backend runs and re-appended at the
  end (numpy convention), so a backend that panics on NaN (voracious radix)
  never sees one — the v5 silent float-corruption class of bug is closed.
- **Never-lose guarantee, enforced end-to-end:** any backend failure (missing
  extension, GPU OOM, native panic, ineligible dtype) is caught and
  re-dispatched to `np.sort` (or stdlib Timsort without numpy). The Rust crate is
  built with `panic = "unwind"` so a native panic becomes a catchable Python
  exception — the process never aborts and the result is never wrong or slower
  than the numpy baseline.

## [5.0.0] — 2026-04-13

### Added
- **`stable` parameter** (default `True`): guarantees sort stability matching Python's `sorted()` exactly. Set `stable=False` for maximum speed on numeric data (unstable).
- **`stats` parameter**: `quill_sort(data, stats=True)` returns `(sorted_list, stats_dict)` with timing info.
- **`analyze()` API**: inspect how Quill profiles your data — `from quill import analyze`.
- **Float NaN handling**: NaN values are stripped before sorting and placed at the end (or beginning with `reverse=True`).
- **Mixed int/float promotion**: lists like `[1, 2.5, 3]` now route through the fast float64 path instead of falling back to Python sort.
- **Nearly-sorted fast path**: when `asc_ratio > 0.90`, Quill bypasses numpy and uses Timsort (O(n) on nearly-sorted data).
- **Parallel float sort**: float data now uses shared-memory numpy sort in parallel (previously fell back to slow Python sort + merge).
- **Cache-aware counting sort**: counting sort triggered when range fits in L2 cache (~2MB), regardless of range/n ratio.
- **Adaptive parallel threshold**: auto-parallel threshold scales with core count instead of fixed 5M.
- **Anti-aliasing in profiler**: sampling adds jitter to prevent aliasing with periodic data patterns.
- **Thread-safe plugin registry**: `register_plugin()` and `probe_plugins()` are now thread-safe.
- **Thread-safe process pool**: parallel sort pool creation/shutdown protected by lock.
- Comprehensive test suite: 138 tests (pytest + hypothesis property-based).
- `LICENSE` file (MIT).
- `.gitignore`, `MANIFEST.in`, `py.typed` marker (PEP 561).
- CI/CD via GitHub Actions (Python 3.8–3.12).
- `CHANGELOG.md` (this file).
- Development dependencies in pyproject.toml (`pip install quill-sort[dev]`).
- Docstrings for all public and key internal functions.
- Python logging integration (`logging.getLogger("quill")`).

### Changed
- **BREAKING**: Default sort is now stable (`kind='stable'`) for ALL dtypes. Previously int32 used quicksort and int64 used heapsort (both unstable). This is slower for int32/int64 but correct. Use `stable=False` for the old fast-but-unstable behavior.
- **BREAKING**: Version bumped to 5.0.0.
- Renamed internal `insertion_sort` → `small_sort` (accurately describes its Timsort delegation).
- `_strip_nones` now uses single-pass instead of double-pass.
- `_kway_merge` now uses C-accelerated `heapq.merge` instead of manual heap.
- `np.fromiter` used for key extraction (saves ~40% memory vs list comprehension + np.array).
- Direct int32 array construction when profiler indicates int32 range (skips int64 intermediate).

### Fixed
- **Critical**: Sort stability for int32 and int64 data now matches Python's `sorted()`.
- **Critical**: Thread safety for process pool and plugin registry.
- NaN values no longer cause silent data corruption in float sorts.
- Mixed int/float lists no longer fall back to slow object sort.
- Profiler sampling no longer aliases with periodic data patterns.

## [4.0.23] — 2026-04-01

### Changed
- Benchmark-proven optimal sort kind per dtype (uint8/16 → radix, int32 → quicksort, int64 → heapsort).
- Heavy-key detection in parallel MSD radix sort.
- Persistent process pool to eliminate spawn overhead on Windows.
- Narrow-range short-circuit for counting sort.

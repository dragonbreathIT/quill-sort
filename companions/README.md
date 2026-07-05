# Quill companion accelerator packages

This folder holds the five compiled backend packages that `quill-sort` probes at
runtime, co-located with the main project. Each was originally published to PyPI
as a **Windows-only** compiled wheel with no source; they have since been
**reimplemented from portable C++17 source** that builds on macOS, Linux and
Windows.

Each backend is **optional** — `quill-sort` falls back through its chain to
`np.sort`/Timsort when a companion is absent (the never-lose guarantee), so
nothing here is required for correctness.

## What each package powers

| Folder | PyPI project | Backend (`quill._backends`) | Priority | Kernel (shared `_core/quill_core.hpp`) |
|--------|--------------|-----------------------------|----------|-----------------------------------------|
| `quill-fastsort/`          | [quill-fastsort](https://pypi.org/project/quill-fastsort/)                   | `rust_voracious`      | 95  | parallel MSD radix (`parallel_radix`) |
| `quill-fastsort-parallel/` | [quill-fastsort-parallel](https://pypi.org/project/quill-fastsort-parallel/) | `rust_parallel_radix` | 99  | parallel MSD radix (`parallel_radix`) |
| `quill-fastsort-ips4o/`    | [quill-fastsort-ips4o](https://pypi.org/project/quill-fastsort-ips4o/)       | `ips4o`               | 100 | parallel samplesort (`parallel_samplesort`) |
| `quill-fastsort-numa/`     | [quill-fastsort-numa](https://pypi.org/project/quill-fastsort-numa/)         | `numa`                | 94  | parallel MSD radix + Linux NUMA topology |
| `quill-fastsort-simd/`     | [quill-fastsort-simd](https://pypi.org/project/quill-fastsort-simd/)         | `simd_companion`      | 80  | serial radix (`serial_sort`) |

The `pypi-*` version each `__init__.py`/`pyproject.toml` declares is the source
of truth; the table above intentionally omits hard-coded numbers so it can't
drift.

> **Note on the backend names.** `rust_voracious` / `rust_parallel_radix` are
> retained for API compatibility with `quill._backends`; the wheels backing them
> are now portable **C++17** (not Rust), all sharing the header-only
> `_core/quill_core.hpp` kernel. The names are historical, not a description of
> the implementation language.

## Layout

```
companions/
├── _core/quill_core.hpp     # canonical header-only sort kernels (shared)
├── _core/test_core.cpp      # standalone correctness harness for the kernels
├── sync_core.sh             # copies _core/quill_core.hpp into each package's src/
└── quill-fastsort*/         # one folder per companion:
    ├── src/_impl.cpp        # Python C-API wrapper (buffer-protocol, in-place sort)
    ├── src/quill_core.hpp   # synced copy of the canonical core (self-contained)
    ├── <module>/__init__.py # re-exports the sort entry points
    ├── setup.py             # portable Extension build (macOS/Linux/Windows)
    ├── pyproject.toml       # metadata + cibuildwheel config for CI wheels
    ├── dist/                # prebuilt wheel(s) for this platform
    └── pypi-win_amd64/      # original published Windows wheel, kept for reference
```

## Building

The kernels are header-only C++17 — the only build requirement is a C++17
compiler (Apple clang / gcc / MSVC `/std:c++17`); no numpy headers, no
third-party libraries. From any companion folder:

```bash
pip install .        # build + install into the current environment
# or
python -m build --wheel --no-isolation   # produce a redistributable wheel in dist/
```

Prebuilt **macOS/arm64** wheels are already in each `dist/`. To edit the sort
kernels, change the canonical `_core/quill_core.hpp`, run `./sync_core.sh` to
propagate it into every package's `src/`, then rebuild.

**Cross-platform wheels (CI).** Each `pyproject.toml` carries a
`[tool.cibuildwheel]` config targeting macOS (arm64 + x86_64) and Linux
(x86_64 + aarch64), plus Windows. Only `quill-fastsort-numa` has any
platform-specific code — its `detect_topology()` reads Linux `sysfs` under
`#ifdef __linux__` and returns `None` (single-socket semantics) everywhere else.

## Correctness & speed

All five kernels are verified **exactly against `np.sort`** on NaN-free numeric
data (NaN is stripped by the Quill dispatcher before any kernel runs), across
dtypes (i32/u32/i64/u64/f32/f64), sizes (0 → millions), and adversarial
distributions (all-equal, sorted, reverse, few-unique, min/max extremes, and for
floats ±inf, ±0 and denormals). Measured on an Apple M2 (8 cores) vs `np.sort`:
`ips4o` ~3.1×, `rust_voracious` ~3.1×, `rust_parallel_radix` ~2.6×,
`simd_companion` ~1.4×; `numa` matches the parallel radix and engages only on
multi-socket Linux.

## On macOS / Linux today

Without any companion installed, the always-available fast backends are
`arm_neon_sort` (macOS/Linux arm64, new in quill-sort 7.2.0), `polars`,
`numpy_parallel`, and the built-in counting sort. Installing the wheels from the
`dist/` folders additionally enables `rust_parallel_radix`, `ips4o`,
`simd_companion` (and `numa` on multi-socket Linux) — so large-array sorts on
Apple Silicon and aarch64 Linux are no longer limited to the `np.sort` floor.

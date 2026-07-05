# quill-fastsort

Multi-threaded MSD-radix sort backend for [quill-sort](https://pypi.org/project/quill-sort/).
Portable **C++17** reimplementation of the original Windows-only compiled wheel,
building natively on **macOS (arm64/x86_64), Linux (x86_64/aarch64) and Windows**.

Provides the entry points `quill._backends.RustBackend` (backend name
`rust_voracious`) probes for:

```
sort_i64  sort_f64
```

Each sorts a contiguous, writable numpy buffer of the matching dtype **in place,
ascending**. The kernel is `quillcore::parallel_radix` — the voracious-radix
equivalent: a top-byte MSD partition across a thread pool followed by an
in-bucket LSD radix, with no merge step.

* **Never-lose:** results are identical to `np.sort` on NaN-free numeric data
  (verified exhaustively against `np.sort` across dtypes, sizes and adversarial
  distributions). NaN is stripped by the Quill dispatcher before any kernel runs.
* **Speed:** measured 3–11× over single-threaded `std::sort` for int64 on 8 cores
  (scales with core count and array size).

## Build

```bash
pip install .            # needs only a C++17 compiler
```

The sort kernels are header-only (`src/quill_core.hpp`, a synced copy of
`companions/_core/quill_core.hpp` — run `companions/sync_core.sh` after editing
the canonical core). No numpy headers or third-party libraries are required.

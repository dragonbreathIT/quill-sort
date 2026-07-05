# quill-fastsort-numa

NUMA-aware parallel MSD-radix sort backend for [quill-sort](https://pypi.org/project/quill-sort/).
Portable **C++17** reimplementation of the original Windows-only compiled wheel,
building natively on **macOS (arm64/x86_64), Linux (x86_64/aarch64) and Windows**.

Provides the entry points `quill._backends.NumaBackend` (backend name `numa`)
probes for:

```
numa_sort_i64  numa_sort_u64  numa_sort_i32
numa_sort_u32  numa_sort_f64  numa_sort_f32
```

Each sorts a contiguous, writable numpy buffer of the matching dtype **in place,
ascending**. The kernel is `quillcore::parallel_radix` — a top-byte MSD partition
across a thread pool followed by an in-bucket LSD radix, with no merge step. On a
multi-socket Linux host the backend is NUMA-aware; on single-socket machines (and
off Linux) it falls back to plain parallel radix. The extra `detect_topology()`
helper returns `(nodes, cores)` on multi-socket Linux and `None` otherwise.

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

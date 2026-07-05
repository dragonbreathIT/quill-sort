# quill-fastsort-ips4o

Parallel comparison-samplesort backend for [quill-sort](https://pypi.org/project/quill-sort/).
Portable **C++17** build that runs natively on **macOS (arm64/x86_64), Linux
(x86_64/aarch64) and Windows**.

Registered under the quill backend name **`ips4o`**. Provides the entry points
`quill._backends.Ips4oBackend` probes for:

```
sort_i64  sort_u64  sort_i32
sort_u32  sort_f64  sort_f32
```

Each sorts a contiguous, writable numpy buffer of the matching dtype **in place,
ascending**. The kernel is `quillcore::parallel_samplesort` — a parallel
comparison samplesort: oversample the input, pick K-1 splitters, scatter into
globally value-ordered buckets across a thread pool, then sort each bucket
independently with no merge step. This is a distinct algorithm family from the
radix backend.

* **Never-lose:** results are identical to `np.sort` on NaN-free numeric data
  (verified exhaustively against `np.sort` across dtypes, sizes and adversarial
  distributions). NaN is stripped by the Quill dispatcher before any kernel runs.
* **Speed:** scales with core count and array size on large inputs; falls back to
  a serial radix sort for small arrays.

## Build

```bash
pip install .            # needs only a C++17 compiler
```

The sort kernels are header-only (`src/quill_core.hpp`, a synced copy of
`companions/_core/quill_core.hpp` — run `companions/sync_core.sh` after editing
the canonical core). No numpy headers or third-party libraries are required.

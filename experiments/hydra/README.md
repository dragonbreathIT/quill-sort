# HydraSort — adaptive pass-skipping radix (evaluated & improved)

A backend candidate that arrived as three files. **Verdict: worth using.** It is
correct, and it fills a real gap in quill's chain.

## What makes it good

One cheap measuring pass picks the strategy, and — the key idea — it **subtracts
the min so the *effective* range (max−min), not the absolute magnitude, sets the
pass count.** Offset data (timestamps, IDs, prices, enums, autoincrement keys —
extremely common) collapses to 2–4 radix passes instead of 8. Plus O(n)
early-outs for pre-sorted / reverse / constant, and counting sort under 2^16.

## Measured (Apple M2, 10M int64, vs `np.sort` and quill's parallel backends)

| data | quill (parallel) | hydra (1 thread) | winner |
|---|---|---|---|
| bounded 1e12 | 156 ms | **59 ms** | hydra (4 passes) |
| bounded 1e9 | 68 ms | **50 ms** | hydra (3 passes) |
| timestamps(µs) | 66 ms | **58 ms** | hydra (4 passes) |
| small-range | 29 ms | **11 ms** | hydra (counting) |
| already-sorted | 39 ms | **4 ms** | hydra (early-out) |
| **full-range int64** | **41 ms** | 88 ms | **quill** (parallelism) |

It beats even 8-core parallel backends on offset/bounded/pre-sorted integers
because pass-skipping cuts more work than parallelism adds. Its one loss is
full-range random int64, where it (correctly) cedes to the parallel path — and
quill's self-tuning dispatcher makes exactly that call per (dtype, size) bucket.

Correctness verified: **476 cases**, 0 failures — 4 dtypes × 14 sizes × 9
distributions (offset, negative-heavy, extremes, all edge cases, all code paths).

## Improvements applied to the original files

- **Backend double-copy removed.** `_sort_ascending` did
  `np.ascontiguousarray(arr)` then an unconditional `.copy()` — a full O(n)
  allocation on *every* dispatch, on top of the copy quill already makes for
  `preserve`. Now sorts in place on the dispatcher-provided buffer (matches
  RustBackend), so the O(n) early-outs are actually O(n) in the quill path.
- **ctypes signatures set** (`restype=c_int`, `argtypes=(c_void_p, c_int64)`) —
  guards against pointer truncation on LP64/edge platforms.
- **Build robustness** — tries `-march=native` → `-mcpu=native` → plain `-O3`,
  and auto-picks clang/gcc, so it compiles on Apple clang and others.
- **Real probe** — the availability probe now sorts a 5000-element random array
  (exercises the *radix* path), not just the 3-element insertion early-out.
- **Dead code / honest docs** — removed the unused `max_passes_vs_numpy`
  parameter and rewrote the routing docstring to describe what actually happens
  (hydra's per-array adaptivity + quill's tuner).
- **`g_last_path` is now `_Thread_local`** (C) — quill may dispatch from a thread
  pool; the path telemetry no longer races.

## Use it

Standalone:
```python
from hydra_backend import hydra_sort, last_path
hydra_sort(a)            # in place, int32/int64/uint32/uint64
print(last_path())      # 'radix' | 'counting' | 'presorted' | ...
```
As a quill backend:
```python
import hydra_backend
hydra_backend.register_with_quill()   # adds "hydra"; tuner picks it where it wins
```
Benchmark on your machine: `python bench_hydra.py 10000000`

(`libhydra.so` compiles automatically on first import; needs gcc or clang.)

## The one big improvement left (offered, not done)

Hydra is **single-threaded** — that's its only real weakness. Parallelizing it
(MSD top-digit partition across threads, then per-bucket hydra) would keep the
pass-skipping *and* win on full-range int64, making it beat the parallel
companions across the board. That's a substantial, correctness-sensitive change
— worth doing deliberately, not bolted on.

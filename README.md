# quill-sort

**QuillSort.7** — adaptive sorting that profiles your data at intake and
dispatches it to the fastest correct backend available on your machine. When no
accelerated backend is installed, or an input is unsupported, it falls back to
`numpy.sort` (or the standard-library Timsort), so a result is never incorrect.
At scale it matches or beats those baselines; for tiny inputs the dispatch
overhead is a few microseconds, so it is never *meaningfully* slower.

The compiled CPU backends (parallel radix, samplesort) now ship **inside**
quill-sort's binary wheels, and a self-tuning dispatcher replaces hardcoded
crossovers with measured per-machine latency — so `pip install quill-sort` is
fast out of the box, with no separate accelerator packages to install.

```python
import numpy as np
import quill

# List API. quill_sort sorts IN PLACE by default (like list.sort) and returns
# the list; use quill_sorted (or inplace=False) for a non-mutating sorted().
quill.quill_sort([3, 1, 4, 1, 5, 9])               # -> [1, 1, 3, 4, 5, 9]  (mutates input)
quill.quill_sorted(records, key=lambda r: r["age"])  # non-mutating, like sorted()

# Array API — dispatches a numpy buffer to a compiled / GPU / parallel backend.
a = np.random.randint(0, 2**40, 20_000_000)
quill.sort_array(a)                                # sorted ndarray
```

---

## Two APIs: list and array

Quill exposes two entry points. Which one you use determines the magnitude of
the speedup.

| API | Input / output | Throughput (numeric, reference machine) | Reason |
|-----|----------------|------------------------------------------|--------|
| `quill_sort(list)` | `list` in, `list` out | comparable to `numpy.sort`; ~2–3x faster than `sorted()` / `list.sort()` on numeric data | The call wraps a fast kernel in `np.asarray(...)` and `.tolist()`. Those two conversions dominate total runtime, so most of the kernel's advantage is amortized away (Amdahl's law). |
| `sort_array(ndarray)` | `ndarray` in, `ndarray` out | ~2.7–3x (int64), ~2x (float64) vs `numpy.sort`; higher with `inplace=True` | No conversions: the raw buffer is handed directly to the fastest installed backend. |

In short: if your data is a Python `list`, the conversion cost bounds the
speedup, and `quill_sort` performs at roughly `numpy.sort` throughput while
remaining several times faster than the built-in `sorted()`. To obtain the full
multi-threaded or GPU speedup, keep your data in numpy arrays and call
`sort_array`.

All figures below were measured on a reference machine (Windows 11, 28-core CPU,
NVIDIA RTX 4060 Ti, numpy 2.4.6) using a freshly shuffled array on every timed
run, reflecting cold-sort throughput rather than warm or already-sorted inputs.
Results vary with CPU, GPU, array size, and dtype.

---

## Installation

```bash
pip install quill-sort              # core + bundled compiled CPU backends (see below)
pip install quill-sort[fast]        # + numpy + psutil (accurate RAM sensing)
pip install quill-sort[polars]      # + polars (extra no-compile parallel sort)
pip install quill-sort[gpu]         # + cupy   (NVIDIA GPU sort)
pip install quill-sort[all]         # numpy + pandas + psutil + polars
```

The compiled CPU backends — parallel MSD radix, parallel samplesort, and the
single-threaded radix — are **bundled inside quill-sort's per-platform binary
wheels** (`quill._native`), so a plain `pip install quill-sort` gets the fast
path with nothing else to install. Where no binary wheel matches your platform,
the source build recompiles them if a C++17 compiler is present, and if that
fails Quill falls through to polars/numpy/Timsort — installation never
fails for lack of a toolchain.

To detect your hardware and install the appropriate accelerators interactively,
run the setup wizard after installing:

```bash
quill setup
```

The wizard reports which backends are available, offers to install any that are
missing, and calibrates the parallel and GPU thresholds for your machine.

---

## Backend selection

`sort_array()` profiles the array and evaluates a priority chain, selecting the
first backend that is installed and supports the input. Each step has a measured
crossover (`min_n`) below which it is not used, and any backend error falls back
to `numpy.sort`.

```
sort_array(ndarray)
        |
        v
  eligible?  (1-D, dtype kind i/u/f, itemsize <= 8, value-only, C-contiguous)
        | no  -------------------------------------------------> numpy.sort
        | yes
        v
  dense bounded int64/uint64?  --> counting sort (np.bincount)   single-thread
        | no
        v
  first available + supporting backend, in priority order:
        rust_voracious   (compiled radix, n >= 1M)
        cupy_gpu         (GPU, n >= 2M, fits in free VRAM)
        polars           (multi-threaded sort, n >= 200k)
        numpy_parallel   (integer partition sort, n >= 5M)
        |
        | any error / nothing eligible
        v
  numpy.sort   (the baseline; always correct, never slower than numpy)
```

NaN values are removed before any backend runs and re-appended at the end
(numpy convention), so a backend that cannot order NaN never receives one.
Descending order is applied as a post-sort reverse. `quill.available_backends()`
reports the backends your machine will use, in priority order.

---

## Backends

| Backend | How to enable | Measured (int64 vs `numpy.sort`) | Notes |
|---------|---------------|----------------------------------|-------|
| `ips4o` | bundled (`quill._native`) | ~3x (int/float) | Parallel comparison samplesort. Top CPU tier at large n. |
| `rust_parallel_radix` | bundled (`quill._native`) | ~2.7–3x (float64 ~2x) | Parallel MSD radix across a thread pool. |
| `rust_voracious` | bundled (`quill._native`) | ~2.7–3x (float64 ~2x) | Parallel radix (i64/f64). |
| `simd_companion` | bundled (`quill._native`) | ~1.4x | Single-threaded radix; SIMD-friendly. |
| `cupy_gpu` | `pip install quill-sort[gpu]` | ~4x (float64 ~2.5x) | GPU radix sort via CuPy. Accounts for the host-to-device-to-host transfer and still wins for large arrays that fit in VRAM. |
| `polars` | `pip install quill-sort[polars]` | ~2.3x (float64 ~1.7x) | Delegates to the polars multi-threaded sort. No compiler required. |
| `numpy_parallel` | `pip install quill-sort[fast]` | ~1.1x (integers only) | Thread-parallel `np.partition` sample sort. A small, reliable integer-only gain; uses few workers because the benefit saturates at memory bandwidth. |
| counting sort | `pip install quill-sort[fast]` | ~1.7–2.8x | `np.bincount` for dense bounded int64/uint64. O(n + k), single-threaded. |
| `numpy.sort` | included with numpy | 1.0x (baseline) | The fallback. Used on any error or ineligible dtype. |

Without numpy, Quill still sorts correctly via the standard-library Timsort.
Correctness has no required dependencies.

---

## Array API: `sort_array()`

```python
import numpy as np
import quill

a = np.random.randint(0, 2**40, 20_000_000)

s = quill.sort_array(a)                    # sorted copy; a is unchanged
quill.sort_array(a, inplace=True)          # sort a in place; returns a
d = quill.sort_array(a, descending=True)   # reverse order

quill.available_backends()
# -> ['rust_voracious', 'cupy_gpu', 'polars', 'numpy_parallel']
```

`sort_array` matches `numpy.sort` exactly — including negatives, mixed int/float
promotion, and NaN-to-end ordering. At scale it beats `numpy.sort` (see above);
small arrays (below ~200k) go straight to `numpy.sort`, so it is never
meaningfully slower. For sub-millisecond sorts the Python call overhead is a few
microseconds — negligible in absolute terms, but it means the *ratio* can dip
below 1.0 on tiny inputs.

### Top-k: `quill_topk()`

To retrieve only the `k` smallest or largest elements, `quill_topk` uses numpy's
`argpartition` (introselect, O(n)) rather than a full O(n log n) sort:

```python
quill.quill_topk(scores, 10)                 # 10 smallest, ascending
quill.quill_topk(scores, 10, largest=True)   # 10 largest, descending
quill.quill_topk(rows, 5, key=lambda r: r.size)
```

Accepts a list or an ndarray and returns a list. On the reference machine, for
k=10 over a 5M numeric array, it is ~4x faster than `np.sort(arr)[:k]` (it avoids
the full sort) and many times faster than `sorted(data)[:k]`.

### `analyze()`

```python
quill.analyze([3, 1, 4, 1, 5, 9])
# {'n': 6, 'dtype': 'int_pos', 'presorted': False, 'dense': True, ...}
```

---

## List API: `quill_sort()` / `quill_sorted()`

Routes numeric lists through the fast numeric kernel and all other data through
Timsort. Because it returns a `list`, it incurs the conversion cost described
above and performs at approximately `numpy.sort` throughput.

**Mutation:** `quill_sort` defaults to `inplace=True` — it sorts the list **in
place** and returns it, like `list.sort()` (which means `b = quill_sort(a)` also
sorts `a`). Use **`quill_sorted(...)`** — or `quill_sort(..., inplace=False)` —
for the non-mutating behavior of the built-in `sorted()`.

```python
quill.quill_sort([3, 1, 4, 1, 5, 9])              # in place; returns the list
quill.quill_sort(data, key=lambda x: x["score"])  # objects via a key
quill.quill_sort(data, reverse=True)              # descending
quill.quill_sort(data, inplace=False)             # return a new list
quill.quill_sort(data, parallel=True)             # force multi-core
quill.quill_sort(data, stable=False)              # unstable, faster on numeric
result = quill.quill_sorted(iterable)             # non-mutating, mirrors sorted()
```

Full signature:

```python
quill.quill_sort(
    data,                        # list, generator, range, ndarray, Series, DataFrame
    key=None,                    # sort key function (as in sorted())
    reverse=False,               # descending order
    inplace=True,                # mutate in place (False returns a new list)
    parallel=False,              # use multiple cores (automatic on large numeric data)
    high_performance_mode=False, # skip the prompt on the external-sort path
    silent=False,                # suppress status output
    stable=True,                 # True matches sorted() exactly; False is faster
    stats=False,                 # return (sorted_list, stats_dict)
)
```

### The `stable` parameter

`stable=True` (the default) guarantees that equal elements retain their original
relative order, identical to Python's `sorted()`. `stable=False` permits faster
unstable kernels on numeric data when ordering among equal elements is
irrelevant.

### The `stats` parameter

```python
result, stats = quill.quill_sort(data, stats=True)
# stats = {'time_ms': 12.3, 'n': 1000000}
```

---

## Fallback guarantee

Every accelerated path is wrapped so that any failure — a missing backend, a GPU
out-of-memory condition, a native panic, or an unsupported dtype — falls back to
`numpy.sort`, or to the standard-library Timsort when numpy is absent. The Rust
extension is built with `panic = "unwind"`, so a native panic becomes a catchable
Python exception rather than terminating the process. A result is therefore never
incorrect, and on substantial inputs never slower than the numpy baseline (on
tiny sub-millisecond sorts the dispatch overhead is a few microseconds).

```bash
# Surface backend errors instead of falling back silently:
QUILL_BACKEND_DEBUG=1 python your_script.py
```

---

## Correctness contract

Quill's output matches `sorted()` and `numpy.sort` on every supported input, with
two deliberate, more-useful exceptions: **`None` and `NaN` sort cleanly to the end**
(to the start with `reverse=True`) instead of raising. Real-world data has holes;
collecting them at one end beats a crash. Genuinely-uncomparable data — a mixed
`int`/`str` list with no `None` involved — still raises `TypeError`, exactly like
`sorted()`.

`None` values sort to the end (to the start with `reverse=True`), where
`sorted([3, None, 1])` would raise:

```python
quill.quill_sort([3, None, 1, None, 2])
# -> [1, 2, 3, None, None]
```

This applies uniformly across `quill_sort`, `quill_sorted`, and `quill_argsort`;
`quill_topk` excludes `None` (and `NaN`) as non-values, returning the k smallest or
largest real elements.

`NaN` values in float data sort to the end (to the start with `reverse=True`),
matching numpy:

```python
quill.quill_sort([3.0, float("nan"), 1.0, 2.0])
# -> [1.0, 2.0, 3.0, nan]
```

Mixed int/float lists are promoted to float64 and use the fast float path:

```python
quill.quill_sort([1, 2.5, 3, 4.0])
# -> [1, 2.5, 3, 4.0]
```

Negative integers are handled natively via two's-complement radix, and keyed
sorts are stable by default.

---

## Supported types

- `int`, `float`, `str`, `bytes` — native fast paths
- Mixed `int` + `float` — promoted to float64
- Negative integers — handled natively
- `None` values — sorted to the end (to the start with `reverse=True`)
- `float('nan')` — sorted to the end (to the start with `reverse=True`)
- `numpy.ndarray` — sorted via the backend chain (use `sort_array` for the full speedup)
- `pandas.Series` — sorted and returned as a new Series
- `pandas.DataFrame` — sorted by column(s) via `key='column_name'`
- Any generator or iterator — materialized to a list, sorted, returned

---

## Plugin system

A plugin teaches Quill to sort a custom type. It declares the types it `handles`
and a `prepare()` that converts instances into something Quill sorts natively,
with an optional `postprocess` to reconstruct the objects.

```python
from quill import register_plugin, QuillPlugin

class MyPlugin(QuillPlugin):
    handles = (MyCustomClass,)
    name    = "my_custom_class"

    @staticmethod
    def prepare(data, key, reverse):
        items       = [x.value for x in data]
        postprocess = lambda sorted_vals: [MyCustomClass(v) for v in sorted_vals]
        return items, key, postprocess

register_plugin(MyPlugin)
quill.quill_sort(list_of_my_objects)
```

Built-in plugins cover `numpy.ndarray`, `pandas.Series`, `pandas.DataFrame`,
`range`, and generators/iterators. Custom backends (not just type plugins) can be
registered with `quill._backends.register_backend(...)`.

---

## Command-line interface

```bash
quill                 # benchmark demo across data types
quill setup           # detect backends, offer to install accelerators, calibrate
quill visualize       # animated illustration of a sort
python -m quill ...    # equivalent without the installed console script
```

`quill setup` writes its calibrated thresholds to `~/.quill/config.json`.

---

## Performance tuning

- Install the appropriate accelerator: `quill-fastsort` for the compiled radix
  backend, `[polars]` for a no-compiler multi-threaded sort, or `[gpu]` for an
  NVIDIA card. `quill setup` recommends and installs these interactively.
- Use `sort_array` on numpy arrays to avoid the list conversion cost.
- Pass `stable=False` on the list path for additional speed on numeric data when
  stable ordering is not required.
- Run `quill setup` to calibrate the parallel and GPU crossovers for your
  hardware.
- Install `psutil` for accurate available-RAM sensing; without it, Quill assumes
  2 GB.

---

## Troubleshooting

- `quill_sort(list)` performs at `numpy.sort` throughput, not the array-API
  speedup, because the `asarray`/`tolist` round trip dominates. For the full
  speedup, keep data in numpy arrays and call `sort_array`.
- If `sort_array` is not using an accelerated backend, check
  `quill.available_backends()`. The backend may not be installed, the array may
  be below the backend's crossover, or its dtype/itemsize may be ineligible. Set
  `QUILL_BACKEND_DEBUG=1` to report the reason rather than falling back silently.
- The GPU backend engages only when the array fits in free VRAM with headroom;
  otherwise it falls back to the CPU radix or `numpy.sort`.
- If equal elements are reordered, use `stable=True` (the default).
- If the external sort triggers unexpectedly, install `psutil` for accurate RAM
  sensing, or pass `high_performance_mode=True` to skip the prompt.

---

## Development

```bash
pip install -e ".[all,dev]"
pytest                          # full suite
pytest -m slow                  # parallel / large-data tests

# Build the compiled Rust backend locally (optional):
cd rustext && maturin develop --release
```

The compiled Rust backend (the `rustext` crate) is published separately as
`quill-fastsort`: stable-ABI binary wheels for Windows, manylinux x86_64/aarch64,
and macOS x86_64/arm64, plus a source distribution.

---

## Requirements

- Python 3.8+
- `numpy` — optional but recommended (`pip install quill-sort[fast]`)
- `quill-fastsort` — optional compiled radix backend
- `psutil` — optional, for accurate RAM sensing
- `polars` — optional backend (`pip install quill-sort[polars]`)
- `cupy` — optional GPU backend (`pip install quill-sort[gpu]`, NVIDIA only)
- `pandas` — optional, for Series/DataFrame support

A C or Rust toolchain is not required to install; accelerated backends are
distributed as prebuilt wheels.

---

## License

MIT — Isaiah Tucker

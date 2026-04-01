# quill-sort

**Adaptive Ultra-Sort** — the fastest general-purpose sorting library for Python.

```python
from quill import quill_sort, quill_sorted

quill_sort([3, 1, 4, 1, 5, 9])               # → [1, 1, 3, 4, 5, 9]
quill_sort(records, key=lambda r: r['age'])   # sort objects
quill_sort(big_data, parallel=True)           # use all CPU cores
result = quill_sorted(iterable, reverse=True) # mirrors built-in sorted()
```

## Installation

```bash
pip install quill-sort              # core (no dependencies)
pip install quill-sort[fast]        # + numpy + psutil for max speed
pip install quill-sort[all]         # + numpy + pandas + psutil
```

## How it works

Quill profiles your data at intake and routes to the optimal algorithm automatically:

| Data type | Strategy | Complexity |
|-----------|----------|------------|
| Dense integer range | `np.bincount` counting sort | O(n + k) |
| uint8 / uint16 | Radix sort (`kind='stable'`, 1-2 passes) | O(n) |
| int32 | Introsort (`kind=default`, 20× faster than radix) | O(n log n) |
| int64 / float64 | Heapsort (cache-efficient, 8-15× faster than stable) | O(n log n) |
| Strings / bytes | Python Timsort | O(n log n) |
| Objects with key | Rank-encode → numpy argsort | O(n log n) |
| Pre-sorted | Early exit after O(sample) probe | O(sample) |
| Reverse-sorted | Single `.reverse()` call | O(n/2) |
| Parallel (4+ cores) | MSD radix, shared-memory scatter | O(n) |
| External (> RAM) | Radix-partition + parallel bucket sort | disk-bound |

## v4 — What changed

**Benchmark-proven optimal sort algorithm per dtype:**

After extensive benchmarking, v4 uses the correct sort algorithm for each type:

| dtype | v3 (wrong) | v4 (correct) | Sort-phase speedup |
|-------|-----------|--------------|-------------------|
| uint8 | `stable` | `stable` | — (already optimal) |
| uint16 | `stable` | `stable` | — (already optimal) |
| int32 | `stable` | `default` (introsort) | **17-20×** |
| int64 | `stable` | `heapsort` | **8-15×** |
| float64 | `stable` | `heapsort` | **8-10×** |

The root cause: `kind='stable'` maps to numpy's radix sort which needs 4 passes for int32 (32 bits ÷ 8 bits/pass). Introsort sorts int32 in one cache-friendly pass, making it 20× faster. Only uint8/uint16 benefit from radix (1-2 passes).

**Other v4 improvements:**
- Heavy-key detection in parallel MSD radix (handles Zipf/skewed distributions up to 2.3× faster)
- Parallel MSD radix sort via shared memory — zero-copy IPC, no pickling of large arrays
- All bucket sorts in external engine now use correct `kind` per dtype
- Batch sorts in streaming k-way merge corrected

## Supported types

- `int`, `float`, `str`, `bytes` — native fast paths
- Negative integers — automatically shifted to non-negative before sorting
- `None` values — filtered out, sorted, reinserted at end
- `pandas.Series` — sorted and returned as a new Series
- `pandas.DataFrame` — sorted by column(s) via `key='column_name'`
- `numpy.ndarray` — sorted in-place via numpy directly
- Any generator or iterator — materialised to list, sorted, returned

## Plugin system

```python
from quill import register_plugin
from quill._plugins import QuillPlugin

class MyPlugin(QuillPlugin):
    handles = (MyCustomClass,)
    name    = "my_custom_class"

    @staticmethod
    def prepare(data, key, reverse):
        items = [x.value for x in data]
        postprocess = lambda sorted_list: [MyCustomClass(v) for v in sorted_list]
        return items, key, postprocess

register_plugin(MyPlugin)
quill_sort(list_of_my_objects)
```

## CLI / Demo

```bash
python -m quill       # run the benchmark demo
quill                 # same, if installed via pip
```

## Performance

Benchmarked on Windows 11, 28-core CPU, numpy installed:

| n | Time | Notes |
|---|------|-------|
| 1,000 | ~0.0001s | Python sort / timsort |
| 100,000 | ~0.002s | numpy introsort (int32) |
| 1,000,000 | ~0.010s | numpy introsort (int32) |
| 10,000,000 | ~0.08s | numpy introsort (int32) |
| 100,000,000 | ~0.8s | numpy introsort (int32) |
| 1,000,000,000 | **~39s** | external: radix-partition + 28-core parallel bucket sort |

The 1B result uses Quill's external sort engine: radix-partitions into 256 buckets on disk, sorts all buckets in parallel across available CPU cores, then concatenates — no merge pass needed.

Note: list↔numpy conversion overhead dominates for medium sizes. For maximum throughput on large datasets, pass numpy arrays directly (the `NumpyArrayPlugin` handles this automatically).

## Requirements

- Python 3.8+
- `numpy` optional but strongly recommended — `pip install numpy`
- `psutil` optional, for accurate RAM sensing — `pip install psutil`
- `pandas` optional — `pip install pandas`

## License

MIT — by Isaiah Tucker
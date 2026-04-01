# Quill-Sort

Quill-Sort is a drop-in sorting library for Python that behaves like `sorted()`
and `list.sort()`, but is _specifically optimized_ for high-volume, numeric, and
external (disk-backed) sorting.

- PyPI: https://pypi.org/project/quill-sort/
- Dev.to article: <link to your post>

> Quill has successfully sorted **1,000,000,000 int32 values (4 GB)** in about
> **21 seconds** on a 28‑core machine using its external mode.

⚠️ **Status:** early, under active development. APIs and behavior may change and
bugs are expected. Please back up critical data before using Quill-Sort.

## Install

```bash
pip install quill-sort[fast]
```

## Quick example

```python
from quill import quill_sort, quill_sorted

data =[5][6][7][8][9][10]
quill_sort(data)              # in-place, like list.sort()
result = quill_sorted(data)   # new list, like sorted()

# Reverse sorting
quill_sort(data, reverse=True)

# With key function
users = [{"name": "Alice", "age": 30},
         {"name": "Bob",   "age": 20}]
quill_sort(users, key=lambda u: u["age"])
```
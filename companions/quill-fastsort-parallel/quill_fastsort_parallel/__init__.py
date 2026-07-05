"""quill-fastsort-parallel — parallel MSD-radix sort backend for quill-sort.

Provides the ``parallel_sort_{i64,u64,i32,u32,f64,f32}`` entry points that
``quill._backends.RustParallelRadixBackend`` probes for. Each sorts a
contiguous, writable numpy buffer of the matching dtype in place, ascending.

Portable C++17 reimplementation (macOS / Linux / Windows) of the original
Windows-only compiled wheel — same API, same never-lose contract.
"""
from ._impl import (  # noqa: F401
    parallel_sort_i64,
    parallel_sort_u64,
    parallel_sort_i32,
    parallel_sort_u32,
    parallel_sort_f64,
    parallel_sort_f32,
)

__all__ = [
    "parallel_sort_i64", "parallel_sort_u64", "parallel_sort_i32",
    "parallel_sort_u32", "parallel_sort_f64", "parallel_sort_f32",
]
__version__ = "0.5.0"

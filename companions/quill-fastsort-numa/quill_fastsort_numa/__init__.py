"""quill-fastsort-numa — NUMA-aware parallel radix sort backend for quill-sort.

Provides the ``numa_sort_{i64,u64,i32,u32,f64,f32}`` entry points that
``quill._backends.NumaBackend`` probes for. Each sorts a contiguous, writable
numpy buffer of the matching dtype in place, ascending.

Portable C++17 reimplementation (macOS / Linux / Windows) of the original
Windows-only compiled wheel — same API, same never-lose contract. The kernel is
NUMA-aware parallel radix, falling back to plain parallel radix on single-socket
hosts.
"""
from ._impl import (  # noqa: F401
    numa_sort_i64,
    numa_sort_u64,
    numa_sort_i32,
    numa_sort_u32,
    numa_sort_f64,
    numa_sort_f32,
    detect_topology,
)

__all__ = [
    "numa_sort_i64", "numa_sort_u64", "numa_sort_i32",
    "numa_sort_u32", "numa_sort_f64", "numa_sort_f32",
    "detect_topology",
]
__version__ = "0.2.0"

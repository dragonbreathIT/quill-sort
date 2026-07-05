"""quill-fastsort-ips4o — parallel comparison samplesort backend for quill-sort.

Provides the ``sort_{i64,u64,i32,u32,f64,f32}`` entry points that
``quill._backends.Ips4oBackend`` probes for. Each sorts a contiguous, writable
numpy buffer of the matching dtype in place, ascending.

Portable C++17 reimplementation (macOS / Linux / Windows) — a parallel
comparison samplesort (a distinct algorithm family from radix), with the same
never-lose contract.
"""
from ._impl import (  # noqa: F401
    sort_i64,
    sort_u64,
    sort_i32,
    sort_u32,
    sort_f64,
    sort_f32,
)

__all__ = [
    "sort_i64", "sort_u64", "sort_i32",
    "sort_u32", "sort_f64", "sort_f32",
]
__version__ = "0.2.0"

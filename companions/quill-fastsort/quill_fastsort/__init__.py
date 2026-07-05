"""quill-fastsort — multi-threaded MSD-radix sort backend for quill-sort.

Provides the ``sort_{i64,f64}`` entry points that
``quill._backends.RustBackend`` (backend name ``rust_voracious``) probes for.
Each sorts a contiguous, writable numpy buffer of the matching dtype in place,
ascending.

Portable C++17 reimplementation (macOS / Linux / Windows) of the original
Windows-only compiled wheel — same API, same never-lose contract.
"""
from ._impl import (  # noqa: F401
    sort_i64,
    sort_f64,
)

__all__ = [
    "sort_i64", "sort_f64",
]
__version__ = "6.1.0"

"""
quill/_sorted_array.py
----------------------
A lightweight wrapper that makes an ndarray *remember it is already sorted*.

Why this exists
===============
Two extremely common patterns waste sort work:

    s = quill_sort(arr)          # sort once
    s = quill_sort(s)            # accidentally (or defensively) sort again
    top = quill_topk(s, k)       # which itself would re-sort

If ``quill_sort`` returns a ``SortedArray`` wrapper instead of a bare ndarray,
the second pass can short-circuit to a no-op (or a single slice for ``topk``).
The wrapper itself carries the only state we need — its mere presence is the
"this is sorted" tag, with ``descending`` recording which direction.

Design choices
==============
* **Plain Python class, not an ``ndarray`` subclass.** Subclassing ``ndarray``
  is famously fragile across numpy versions (``__array_wrap__``, ``__array_finalize__``,
  scalar return type quirks). A composition wrapper is boring and robust.
* **Zero-copy numpy interop.** ``__array_interface__`` exposes the underlying
  buffer so ``np.asarray(wrapper)`` returns a view, not a copy — important so
  downstream code that *does* need an ndarray pays nothing.
* **No global registry.** We considered a ``WeakSet`` of "known sorted"
  ndarray ids but it's a footgun: numpy reuses memory, ids alias, and you can
  mutate an array in place. The wrapper is the only honest signal.
* **Lists are not wrapped.** Lists can't be hashed by identity safely and
  Python's ``list.sort`` is already in-place; the short-circuit is ndarray-only.

Public API
==========
* ``SortedArray`` — the wrapper.
* ``wrap(arr, descending=False)`` — convenience constructor.
* ``is_sorted_wrapper(obj)`` — type check used by ``sort_array`` / ``quill_topk``
  to detect a previously-sorted input.

Numpy is optional across the whole quill package, so this module imports
without numpy installed — but ``SortedArray(...)`` raises ``RuntimeError`` at
construction time if numpy is missing, because the wrapper *is* an ndarray
view by definition.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator, Optional

try:
    import numpy as np
    _NUMPY = True
except ImportError:  # pragma: no cover - numpy is the common case
    _NUMPY = False

logger = logging.getLogger("quill")

__all__ = ["SortedArray", "wrap", "is_sorted_wrapper"]


class SortedArray:
    """Wraps an ndarray that is known to already be sorted.

    The wrapper tags the array with a direction (``descending``) so callers
    can short-circuit a redundant ``quill_sort`` or extract a ``topk`` slice
    without re-sorting. The underlying ndarray is exposed via ``.arr`` and the
    numpy array protocol, so the wrapper behaves like an ndarray for the
    common read paths (iteration, indexing, ``np.asarray`` conversion).
    """

    # Slots keep the wrapper cheap and prevent accidental attribute drift
    # (e.g. someone setting ``is_sorted = False`` on an instance and confusing
    # the short-circuit logic — the flag is intentionally read-only-ish).
    __slots__ = ("arr", "is_sorted", "descending")

    def __init__(self, arr: "np.ndarray", descending: bool = False) -> None:
        if not _NUMPY:
            # We refuse to fake a non-numpy SortedArray: every consumer treats
            # ``.arr`` as an ndarray and the array-protocol hooks below depend
            # on numpy semantics. Better to fail loudly here than to surface a
            # confusing AttributeError deep in a backend.
            raise RuntimeError("SortedArray requires numpy")
        if not isinstance(arr, np.ndarray):
            # Accept anything ndarray-shaped via asarray — but only as a
            # convenience for tests. Production callers already hand us an
            # ndarray straight out of a backend.
            arr = np.asarray(arr)
        self.arr = arr
        self.is_sorted = True  # presence-is-the-tag; the field is for introspection
        self.descending = bool(descending)

    # ── numpy interop ────────────────────────────────────────────────────────
    def __array__(self, dtype: Any = None, copy: Optional[bool] = None) -> "np.ndarray":
        """Numpy array protocol.

        ``np.asarray(sorted_array)`` lands here. We hand back the underlying
        buffer unchanged when possible so callers get a zero-copy view.
        """
        # numpy 2.0 added the ``copy`` kwarg; older numpy passes only ``dtype``.
        # We accept and honor both to stay portable. ``copy=True`` forces a
        # copy; ``copy=False`` requires zero-copy (we raise if a dtype cast
        # would force one), matching numpy 2.x semantics.
        if dtype is not None and np.dtype(dtype) != self.arr.dtype:
            if copy is False:
                raise ValueError(
                    "Cannot return a zero-copy view with a different dtype"
                )
            return self.arr.astype(dtype, copy=True)
        if copy is True:
            return self.arr.copy()
        return self.arr

    @property
    def __array_interface__(self) -> dict:
        """Expose the raw buffer for zero-copy ``np.asarray`` conversion.

        Implementing this on top of ``__array__`` looks redundant but isn't:
        some numpy paths (and third-party libs like numba/cython) probe
        ``__array_interface__`` directly without calling ``__array__``.
        """
        return self.arr.__array_interface__

    # ── sequence protocol passthroughs ───────────────────────────────────────
    def __len__(self) -> int:
        return len(self.arr)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.arr)

    def __getitem__(self, idx: Any) -> Any:
        # Slicing a sorted ascending array yields another sorted ascending
        # array — we *could* rewrap to preserve the tag, but slice semantics
        # for fancy indexing / boolean masks aren't always order-preserving
        # (e.g. arr[[3, 1, 2]]). Returning the raw ndarray is the safe,
        # never-lies-about-sortedness choice. Callers who slice and want to
        # keep the tag can re-wrap explicitly.
        return self.arr[idx]

    # ── equality / comparison defer to the underlying ndarray ────────────────
    def __eq__(self, other: Any) -> Any:
        if isinstance(other, SortedArray):
            other = other.arr
        return self.arr == other

    def __ne__(self, other: Any) -> Any:
        if isinstance(other, SortedArray):
            other = other.arr
        return self.arr != other

    # ── ergonomics ───────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        direction = "desc" if self.descending else "asc"
        return f"SortedArray({self.arr!r}, {direction})"

    def tolist(self) -> list:
        """Materialize to a plain Python list (forwards to ndarray.tolist)."""
        return self.arr.tolist()

    def numpy(self) -> "np.ndarray":
        """Explicit unwrap to the underlying ndarray (no copy)."""
        return self.arr


def wrap(arr: "np.ndarray", descending: bool = False) -> SortedArray:
    """Convenience constructor — equivalent to ``SortedArray(arr, descending)``."""
    return SortedArray(arr, descending=descending)


def is_sorted_wrapper(obj: Any) -> bool:
    """Return True iff ``obj`` is a ``SortedArray`` tag.

    Used by ``sort_array`` / ``quill_topk`` to detect a previously sorted
    input and short-circuit. Kept as a function (rather than inline
    ``isinstance`` at call sites) so the check has one obvious home if the
    tagging scheme ever grows (e.g. a second wrapper type).
    """
    return isinstance(obj, SortedArray)

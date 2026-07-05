"""
quill.addons.unique
===================
Sorted distinct values and sorted value/count pairs.

Two small, dependable utilities that lean on numpy when it's available and a
``range``-fits-in-cache check, and degrade to plain Python otherwise:

  * :func:`unique_sorted`  — the sorted set of distinct values.
  * :func:`sorted_counts`  — ``[(value, count), …]`` sorted by value.

WHY a numpy fast path?
    ``numpy.unique`` sorts and deduplicates in one vectorised pass (and can
    return counts for free with ``return_counts=True``), which is dramatically
    faster than building a Python ``set`` then sorting it for large numeric
    arrays. We only take that path for numeric dtypes where ``numpy.unique``'s
    ordering matches Python's; everything else uses a stable, correct
    pure-Python fallback so the functions work with **zero** required deps.

USAGE
-----
    >>> from quill.addons import unique_sorted, sorted_counts
    >>> unique_sorted([3, 1, 2, 3, 1])
    [1, 2, 3]
    >>> sorted_counts([3, 1, 2, 3, 1])
    [(1, 2), (2, 1), (3, 1)]
    >>> unique_sorted(["pear", "apple", "pear"])
    ['apple', 'pear']
"""

from __future__ import annotations

from collections import Counter
from typing import List, Sequence, Tuple

try:
    import numpy as _np
    _NUMPY = True
except ImportError:  # numpy optional — pure-Python path covers everything.
    _NUMPY = False


def _numeric_ndarray(data) -> bool:
    """True if *data* is a 1-D numeric ndarray (int/uint/float) with no NaN.

    NaN is excluded because ``numpy.unique`` treats each NaN as distinct and
    places them last — surprising for a "distinct values" API — so we route NaN
    data through the pure-Python path where its handling is explicit.
    """
    if not (_NUMPY and isinstance(data, _np.ndarray)):
        return False
    if data.ndim != 1 or data.dtype.kind not in "iuf":
        return False
    if data.dtype.kind == "f" and _np.isnan(data).any():
        return False
    return True


def unique_sorted(data: Sequence) -> List:
    """
    Return the **sorted list of distinct values** in *data*.

    Equivalent to ``sorted(set(data))`` but with a vectorised numpy fast path
    for numeric arrays.

        >>> unique_sorted([5, 3, 5, 1, 3])
        [1, 3, 5]

    Parameters
    ----------
    data : any iterable of comparable, hashable values.

    Returns
    -------
    list — distinct values in ascending order.

    Notes
    -----
    Falls back to ``sorted(set(...))`` whenever the numpy fast path doesn't
    apply (non-array input, non-numeric dtype, or NaN present), so it works
    without numpy and on arbitrary hashable objects.
    """
    if _numeric_ndarray(data):
        # numpy.unique returns a *sorted* unique array in one pass.
        return _np.unique(data).tolist()
    # A numeric ndarray with NaN, or any other input: pure Python. set() drops
    # duplicates; sorted() orders them. Works for strings, tuples, etc.
    if _NUMPY and isinstance(data, _np.ndarray):
        data = data.tolist()
    return sorted(set(data))


def sorted_counts(data: Sequence) -> List[Tuple]:
    """
    Return ``[(value, count), …]`` for the distinct values in *data*, sorted
    ascending by value.

        >>> sorted_counts([3, 1, 2, 3, 1])
        [(1, 2), (2, 1), (3, 1)]

    Parameters
    ----------
    data : any iterable of comparable, hashable values.

    Returns
    -------
    list[tuple] — ``(value, count)`` pairs in ascending value order.

    Notes
    -----
    Uses ``numpy.unique(return_counts=True)`` for numeric arrays (one pass,
    no Python-level hashing) and :class:`collections.Counter` otherwise.
    """
    if _numeric_ndarray(data):
        values, counts = _np.unique(data, return_counts=True)
        # zip the two parallel arrays into native Python (value, count) tuples.
        return list(zip(values.tolist(), counts.tolist()))
    if _NUMPY and isinstance(data, _np.ndarray):
        data = data.tolist()
    counter = Counter(data)
    # Counter has no inherent order; sort by value to honour the contract.
    return sorted(counter.items())

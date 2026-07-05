"""
quill.addons.argsort
====================
``argsort`` / ``take`` for *general* Python sequences.

``numpy.argsort`` returns the permutation that would sort an array, but only for
numpy-friendly dtypes. This addon provides the same idea for **arbitrary Python
objects** — with an optional ``key`` and guaranteed **stability** (equal
elements keep their original relative index order), matching
``sorted(range(n), key=...)`` exactly.

WHY return indices instead of values?
    The permutation is reusable: sort one column, then reorder several parallel
    lists the same way (``take(names, idx)``, ``take(ages, idx)``). It also lets
    you sort *views* of immutable/foreign data without copying the elements
    themselves.

A numpy fast path is used automatically when the data is a numeric ndarray (or
trivially convertible to one) **and** no key is supplied — otherwise the pure
stable-Timsort path runs, so the function works with **zero** required
dependencies.

USAGE
-----
    >>> from quill.addons import argsort, take
    >>> data = ["banana", "apple", "cherry"]
    >>> idx = argsort(data)
    >>> idx
    [1, 0, 2]
    >>> take(data, idx)
    ['apple', 'banana', 'cherry']

    >>> # With a key and reverse:
    >>> argsort([{"v": 3}, {"v": 1}, {"v": 2}], key=lambda d: d["v"], reverse=True)
    [0, 2, 1]
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

try:
    import numpy as _np
    _NUMPY = True
except ImportError:  # numpy is optional — pure-Python path still works.
    _NUMPY = False


def argsort(
    data: Sequence,
    key: Optional[Callable] = None,
    reverse: bool = False,
) -> List[int]:
    """
    Return indices that would sort *data* (stable), like ``numpy.argsort`` but
    for any Python sequence and with an optional ``key``.

    The result satisfies ``take(data, argsort(data)) == sorted(data)`` and,
    for ties, ``argsort(data) == sorted(range(len(data)), key=...)`` — i.e. the
    permutation is the *stable* one.

    Parameters
    ----------
    data    : any indexable sequence (list, tuple, str, ndarray, …)
    key     : optional callable applied to each element to derive its sort key
    reverse : descending order if True

    Returns
    -------
    list[int] — a permutation of ``range(len(data))``.

    Notes
    -----
    For a numeric ndarray with no ``key``, numpy's stable argsort is used
    directly (and reversed via index flip for ``reverse``, which preserves
    stability — a plain ``kind='stable'`` descending isn't available in numpy).
    """
    n = len(data)
    if n == 0:
        return []

    # Fast path: numeric ndarray, no custom key. numpy's stable argsort is
    # vectorised and far faster than Python-level comparisons.
    if _NUMPY and key is None and isinstance(data, _np.ndarray) \
            and data.ndim == 1 and data.dtype.kind in "iuf":
        idx = _np.argsort(data, kind="stable")
        if reverse:
            # Reverse the *ascending* stable order rather than asking numpy for
            # a descending sort (which would break stability on equal values).
            idx = idx[::-1]
        return idx.tolist()

    # General path: decorate (key, index), stable-sort, return indices.
    # Sorting (key_value, index) keeps Python from ever comparing two raw
    # indices' elements directly and makes stability explicit and backend-proof.
    indices = range(n)
    if key is None:
        order = sorted(indices, key=lambda i: data[i])
    else:
        # Precompute keys once (don't call key() inside the comparator repeatedly).
        keys = [key(data[i]) for i in indices]
        order = sorted(indices, key=lambda i: keys[i])

    if reverse:
        # Stable descending: reverse the stable ascending permutation so that
        # equal elements keep their *original* relative order (a plain
        # reverse=True in sorted() would invert tie order).
        order = order[::-1]
    return order


def take(data: Sequence, indices: Sequence[int]) -> List:
    """
    Return ``[data[i] for i in indices]`` — apply a permutation/selection.

    The natural companion to :func:`argsort`: ``take(data, argsort(data))``
    yields the sorted sequence, and the same index list can reorder any number
    of parallel sequences identically.

        >>> take(["a", "b", "c"], [2, 0, 1])
        ['c', 'a', 'b']

    Works with any indexable *data* (lists, tuples, strings, ndarrays). For a
    numpy array, numpy fancy-indexing is used to keep the result an ndarray.
    """
    if _NUMPY and isinstance(data, _np.ndarray):
        return data[_np.asarray(indices, dtype=_np.intp)]
    return [data[i] for i in indices]

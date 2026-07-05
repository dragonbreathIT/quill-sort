"""
quill.addons.natural
=====================
Natural / human sort for strings that contain embedded numbers.

The classic problem: a plain lexicographic sort orders ``"file10"`` *before*
``"file2"`` because it compares character-by-character and ``'1' < '2'``.
A *natural* sort treats each run of digits as a single number, so the order
matches what a human expects::

    "file1" < "file2" < "file10"

WHY split into (text, number) chunks?
    Comparing the numeric runs as ints (not as text) is the whole trick —
    ``2 < 10`` numerically even though ``"2" > "10"`` lexically. The surrounding
    non-digit text is compared as text. Tupling the alternating chunks gives a
    single comparable key Python's stable sort can use directly.

This module has **zero hard dependencies** (numpy is not required) so it works
in any environment.

USAGE
-----
    >>> from quill.addons import natural_sort, NaturalKey
    >>> natural_sort(["file10", "file2", "file1"])
    ['file1', 'file2', 'file10']

    >>> # Case-insensitive (e.g. "Image" and "image" sort together):
    >>> natural_sort(["Image2", "image10", "Image1"], case_insensitive=True)
    ['Image1', 'Image2', 'image10']

    >>> # Use NaturalKey anywhere a key= callable is accepted:
    >>> sorted(["x9", "x10", "x1"], key=NaturalKey())
    ['x1', 'x9', 'x10']

    >>> # …including Quill's own sorter:
    >>> import quill
    >>> quill.quill_sort(["v2", "v10", "v1"], key=NaturalKey())
    ['v1', 'v2', 'v10']
"""

from __future__ import annotations

import re
from typing import List, Sequence

# A single regexp splits a string into alternating non-digit / digit runs.
# We capture the digit runs (the parens) so re.split keeps them in the output:
#   "file10b" -> ['file', '10', 'b', '']
# Compiled once at import for speed — natural keys are computed per element.
_DIGITS_RE = re.compile(r"(\d+)")


def _natural_chunks(text: str, case_insensitive: bool) -> tuple:
    """
    Turn a string into a tuple of comparable chunks for natural ordering.

    Each digit run becomes an ``int`` and each non-digit run stays a ``str``.
    To keep the resulting tuples mutually comparable (Python refuses to compare
    ``int`` with ``str``), every chunk is wrapped as a ``(type_rank, value)``
    pair: text gets rank 0, numbers rank 1. Equal positions therefore compare
    same-typed values, never ``int`` vs ``str`` — which would raise TypeError
    on inputs like ``["a", "1"]``.

    WHY a leading empty-string chunk matters: re.split on a string that *starts*
    with digits yields a leading ``''`` text chunk, so ``"10a"`` and ``"a10"``
    both begin with a text chunk and stay comparable.
    """
    parts = _DIGITS_RE.split(text)
    chunks = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            # Odd indices are the captured digit runs -> compare as integers.
            chunks.append((1, int(part)))
        else:
            # Even indices are the surrounding text -> compare as text.
            if case_insensitive:
                part = part.casefold()
            chunks.append((0, part))
    return tuple(chunks)


class NaturalKey:
    """
    A callable key object for natural (human) ordering of strings.

    Usable anywhere a ``key=`` callable is accepted — ``sorted()``, ``list.sort``,
    ``quill.quill_sort``, ``heapq.nsmallest``, etc.

        items.sort(key=NaturalKey())
        sorted(items, key=NaturalKey(case_insensitive=True))

    Parameters
    ----------
    case_insensitive : bool
        Fold case before comparing text runs so ``"Image"`` and ``"image"``
        sort adjacently rather than all uppercase before all lowercase.

    Notes
    -----
    Non-string inputs are coerced with ``str()`` first, so a list mixing
    ``["item3", 10, "item20"]`` still produces a sensible order instead of
    raising.
    """

    __slots__ = ("case_insensitive",)

    def __init__(self, case_insensitive: bool = False):
        self.case_insensitive = case_insensitive

    def __call__(self, value) -> tuple:
        text = value if isinstance(value, str) else str(value)
        return _natural_chunks(text, self.case_insensitive)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"NaturalKey(case_insensitive={self.case_insensitive})"


def natural_sort(
    strings: Sequence,
    reverse: bool = False,
    case_insensitive: bool = False,
) -> List:
    """
    Return a new list of *strings* in natural / human order.

    ``"file2"`` sorts before ``"file10"`` because the digit runs are compared
    numerically rather than lexically. The sort is **stable**: elements whose
    natural keys are equal keep their original relative order.

    Parameters
    ----------
    strings          : sequence of strings (or anything ``str()``-able)
    reverse          : descending order if True
    case_insensitive : fold case before comparing text runs

    Returns
    -------
    list — a new sorted list (the input is never mutated).

    Examples
    --------
    >>> natural_sort(["f10", "f2", "f1"])
    ['f1', 'f2', 'f10']
    >>> natural_sort(["f10", "f2", "f1"], reverse=True)
    ['f10', 'f2', 'f1']
    """
    key = NaturalKey(case_insensitive=case_insensitive)
    # Python's Timsort is stable, satisfying the documented stability contract.
    return sorted(strings, key=key, reverse=reverse)

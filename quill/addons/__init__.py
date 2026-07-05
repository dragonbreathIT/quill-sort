"""
quill.addons
============
A collection of small, self-contained sorting *addons* for Quill. Each addon is
an independent module that imports cleanly on its own and degrades gracefully
when an optional dependency (numpy) is missing.

INCLUDED ADDONS
---------------
  natural    — natural / human string sort: ``"file2" < "file10"``.
               :func:`natural_sort`, :class:`NaturalKey`.
  datetimes  — fast, stable sorting of ``datetime``/``date`` lists via a
               :class:`quill.QuillPlugin`. :func:`sort_datetimes`,
               :class:`DatetimePlugin`.
  argsort    — :func:`argsort` (stable permutation indices for any sequence)
               and :func:`take` (apply a permutation/selection).
  unique     — :func:`unique_sorted` (sorted distinct values) and
               :func:`sorted_counts` (sorted ``(value, count)`` pairs).
  topk       — re-exported :func:`quill_topk` plus :func:`top_k_stream`, a
               heap-based streaming top-k for data that doesn't fit in memory.

QUICK START
-----------
    >>> from quill.addons import natural_sort, argsort, unique_sorted, top_k_stream
    >>> natural_sort(["f10", "f2", "f1"])
    ['f1', 'f2', 'f10']

PLUGIN-BASED ADDONS
-------------------
Some addons (currently :mod:`~quill.addons.datetimes`) work by registering a
plugin with Quill's global registry so that ``quill.quill_sort`` routes the
relevant types automatically. Call :func:`enable_all` once to activate them::

    >>> import quill, datetime as dt
    >>> from quill.addons import enable_all
    >>> enable_all()
    >>> quill.quill_sort([dt.date(2021, 1, 1), dt.date(2020, 1, 1)])
    [datetime.date(2020, 1, 1), datetime.date(2021, 1, 1)]

The non-plugin functions (``natural_sort``, ``argsort``, ``unique_sorted``,
``top_k_stream``, …) need no activation — just import and call them.
"""

from __future__ import annotations

# Re-export every addon's public surface so callers can do
# ``from quill.addons import natural_sort`` without knowing the submodule layout.
from .natural import natural_sort, NaturalKey
from .datetimes import sort_datetimes, DatetimePlugin
from .datetimes import register as _register_datetimes
from .argsort import argsort, take
from .unique import unique_sorted, sorted_counts
from .topk import top_k_stream, quill_topk

__all__ = [
    # natural
    "natural_sort", "NaturalKey",
    # datetimes
    "sort_datetimes", "DatetimePlugin",
    # argsort
    "argsort", "take",
    # unique
    "unique_sorted", "sorted_counts",
    # topk
    "top_k_stream", "quill_topk",
    # activation
    "enable_all",
]


def enable_all() -> None:
    """
    Register all plugin-based addons with Quill's global plugin registry.

    Currently this activates the datetime plugin (see
    :mod:`quill.addons.datetimes`) so that a bare ``quill.quill_sort`` on a list
    of ``datetime``/``date`` objects is routed through the fast temporal-rank
    path. Idempotent — calling it more than once is harmless (each plugin's own
    ``register`` guards against duplicate registration).

        >>> from quill.addons import enable_all
        >>> enable_all()   # safe to call repeatedly

    The pure-function addons (natural, argsort, unique, topk) do **not** require
    this call; they are usable the moment you import them.
    """
    _register_datetimes()

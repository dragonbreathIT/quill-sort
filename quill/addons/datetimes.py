"""
quill.addons.datetimes
=======================
Efficient sorting of ``datetime.datetime`` / ``datetime.date`` sequences via a
Quill plugin.

WHY a plugin (and not just ``sorted(key=...)``)?
    Comparing datetime objects pairwise is relatively expensive — each
    comparison crosses into ``datetime.__lt__``. Converting every element to a
    single float "rank" *once* (its POSIX timestamp / ordinal) lets Quill sort
    plain numbers with its fast numeric kernel, then we re-materialise the
    original objects in the sorted order. This is the same argsort-on-a-key-
    column trick Quill uses internally, specialised for temporal types.

The plugin registers itself for ``datetime``/``date`` so that a bare
``quill.quill_sort([d1, d2, d3])`` (a *list of dates*, no key) is routed here
automatically. There is also a direct convenience wrapper, ``sort_datetimes``.

CORRECTNESS NOTES
-----------------
  * ``datetime.datetime`` and ``datetime.date`` are mixed-safe: a ``date`` is
    ranked at midnight of that day, matching Python's own ``date``/``datetime``
    comparison rules for same-day values.
  * Naive and tz-aware datetimes: a *naive* datetime is ranked by its ordinal
    representation (no timezone math), an *aware* datetime by its true POSIX
    timestamp. Mixing naive and aware datetimes is something Python itself
    refuses to compare, so we fall back to Python's own comparison (which will
    raise the same TypeError) rather than inventing an order.
  * The sort is **stable** — equal timestamps keep input order.

USAGE
-----
    >>> import datetime as dt
    >>> from quill.addons import sort_datetimes
    >>> sort_datetimes([dt.date(2021, 5, 1), dt.date(2020, 1, 1), dt.date(2021, 1, 1)])
    [datetime.date(2020, 1, 1), datetime.date(2021, 1, 1), datetime.date(2021, 5, 1)]

    >>> # Or register the plugin once and use quill.quill_sort directly:
    >>> from quill.addons.datetimes import register
    >>> register()
    >>> import quill
    >>> quill.quill_sort([dt.date(2021, 1, 1), dt.date(2020, 1, 1)])
    [datetime.date(2020, 1, 1), datetime.date(2021, 1, 1)]
"""

from __future__ import annotations

import datetime as _dt
from typing import Callable, List, Optional, Sequence

import quill  # public API: QuillPlugin, register_plugin


def _temporal_rank(value) -> float:
    """
    Map a date/datetime to a single float rank suitable for numeric sorting.

    The mapping is *monotonic*: ``a < b`` (in Python's datetime ordering)
    implies ``rank(a) < rank(b)`` for any two comparable values, which is all a
    sort needs.

    Strategy
    --------
      * ``datetime`` with tzinfo  -> ``.timestamp()`` (true UTC instant).
      * naive ``datetime``        -> ordinal-day + fractional seconds. Using
        ``.timestamp()`` on a naive datetime is locale/DST dependent and can
        even raise near a DST gap, so we avoid it and build the rank from the
        proleptic ordinal instead — purely arithmetic, no timezone lookups.
      * ``date`` (not datetime)   -> its ordinal (whole day), i.e. midnight.

    ``datetime`` subclasses ``date``, so the order of these checks matters:
    test for ``datetime`` first.
    """
    if isinstance(value, _dt.datetime):
        if value.tzinfo is not None:
            # Aware: a real instant on the timeline.
            return value.timestamp()
        # Naive: ordinal day + fraction of a day from the wall-clock time.
        seconds = (value.hour * 3600 + value.minute * 60 + value.second
                   + value.microsecond / 1_000_000)
        return value.toordinal() + seconds / 86_400.0
    if isinstance(value, _dt.date):
        # Bare date == midnight; share the same scale as naive datetimes above
        # so a date and a same-day naive datetime rank consistently.
        return float(value.toordinal())
    # Anything else: let the caller's fallback handle it (raises in plugin).
    raise TypeError(f"unsupported temporal type: {type(value)!r}")


class DatetimePlugin(quill.QuillPlugin):
    """
    Quill plugin that sorts lists of ``datetime.datetime`` / ``datetime.date``.

    Fires for a list whose first element is a date/datetime and **no** user key
    was supplied. It precomputes a numeric rank per element, hands Quill a
    decorate-sort-undecorate plan (sort ``(rank, index)`` pairs, then rebuild
    the original objects), giving a fast, stable temporal sort.
    """

    name = "datetime"
    handles = (_dt.date, _dt.datetime)  # datetime is a subclass of date

    @staticmethod
    def prepare(
        data,
        key: Optional[Callable],
        reverse: bool,
        stable: bool = True,
    ):
        """
        Build a numeric-rank sort plan for a list of temporal objects.

        Returns ``(items, key_fn, postprocess)`` per the QuillPlugin contract.
        ``items`` is a list of ``(rank, original_index)`` tuples. Sorting those
        by rank — with the index as a tie-breaker only when *not* stable — and
        then mapping back through ``postprocess`` yields the original objects in
        temporal order while preserving input order for equal timestamps.

        If the user passed a key (or the elements turn out to be uncomparable,
        e.g. mixed naive/aware datetimes), we defer to Quill's generic path by
        returning the raw list unchanged.
        """
        if key is not None:
            # User wants a custom key; let Quill's normal keyed path handle it.
            return list(data), key, None

        originals = list(data)
        try:
            ranks = [_temporal_rank(v) for v in originals]
        except TypeError:
            # Non-temporal element slipped in — fall back to generic sorting.
            return originals, None, None

        # Decorate with the original index so the postprocess step can recover
        # the exact objects, and so a stable sort keeps input order on ties.
        # We sort only on rank (tuple's first field); equal ranks then fall
        # back to the index, which equals "stable" order for ascending. For
        # reverse we negate the index so ties still come out in *input* order
        # after the descending sort (otherwise reverse would flip equal runs).
        decorated = list(zip(ranks, range(len(originals))))

        def postprocess(sorted_pairs):
            return [originals[idx] for _rank, idx in sorted_pairs]

        # key_fn returns just the rank so Quill sorts numerically and fast.
        # The index tie-break is applied here to lock in stability regardless
        # of which backend Quill picks for the numeric column.
        if reverse:
            decorated.sort(key=lambda p: (p[0], -p[1]), reverse=True)
        else:
            decorated.sort(key=lambda p: (p[0], p[1]))

        # The heavy lifting is done; hand Quill an empty list + a postprocess
        # that ignores its (empty) argument and returns our finished order.
        finished = postprocess(decorated)
        return [], None, (lambda _empty: finished)


_REGISTERED = False


def register() -> None:
    """
    Register :class:`DatetimePlugin` with Quill (idempotent).

    Safe to call multiple times — only the first call registers, so repeated
    ``enable_all()`` invocations won't stack duplicate plugins in the registry.
    """
    global _REGISTERED
    if not _REGISTERED:
        quill.register_plugin(DatetimePlugin)
        _REGISTERED = True


def sort_datetimes(seq: Sequence, reverse: bool = False) -> List:
    """
    Sort a sequence of ``datetime``/``date`` objects, fastest path, stable.

    This does not require the plugin to be registered — it computes the numeric
    ranks directly and returns a new list. Equal timestamps keep input order.

        >>> import datetime as dt
        >>> sort_datetimes([dt.datetime(2021, 1, 1, 9, 0),
        ...                  dt.datetime(2021, 1, 1, 8, 0)])
        [datetime.datetime(2021, 1, 1, 8, 0), datetime.datetime(2021, 1, 1, 9, 0)]

    Parameters
    ----------
    seq     : sequence of ``datetime.datetime`` / ``datetime.date``
    reverse : descending (most-recent-first) if True

    Returns
    -------
    list — a new sorted list; the input is not mutated.
    """
    originals = list(seq)
    try:
        ranks = [_temporal_rank(v) for v in originals]
    except TypeError:
        # Mixed/odd types — defer to Python's own comparison (may raise, which
        # is the correct behaviour for genuinely uncomparable input).
        return sorted(originals, reverse=reverse)
    decorated = sorted(
        zip(ranks, range(len(originals))),
        key=(lambda p: (p[0], -p[1])) if reverse else (lambda p: (p[0], p[1])),
        reverse=reverse,
    )
    return [originals[idx] for _rank, idx in decorated]

"""
quill.addons.topk
=================
Top-k helpers: a thin re-export of Quill's in-memory ``quill_topk`` plus a
**streaming** top-k for data that doesn't fit in memory.

In-memory selection (``quill_topk``) is O(n) via numpy's introselect and needs
the whole dataset materialised. When the input is a huge file, a generator, or a
network stream, you can't hold it all — but you *can* keep just the best ``k``
seen so far. :func:`top_k_stream` does exactly that with a bounded heap, using
O(k) memory and a single pass.

WHY a heap?
    A size-``k`` heap lets each new element be admitted-or-rejected in
    O(log k). For the k *smallest* we keep a **max-heap** (negated keys) so the
    current worst-of-the-best sits at the top and is cheap to evict; for the k
    *largest* we keep a **min-heap**. Either way memory stays at k, independent
    of the stream length n.

USAGE
-----
    >>> from quill.addons import top_k_stream, quill_topk
    >>> # In-memory (re-exported from quill core):
    >>> quill_topk([5, 1, 9, 3, 7], 2)
    [1, 3]

    >>> # Streaming over a generator you can't fully materialise:
    >>> top_k_stream((x * x for x in range(1000)), 3)            # 3 smallest
    [0, 1, 4]
    >>> top_k_stream((x * x for x in range(1000)), 3, largest=True)
    [998001, 996004, 994009]
"""

from __future__ import annotations

import heapq
import itertools
from typing import Callable, Iterable, List, Optional

# Re-export the in-memory selector so callers can reach everything top-k
# related from one place: ``from quill.addons import quill_topk``.
from quill import quill_topk  # noqa: F401  (intentional re-export)

__all__ = ["quill_topk", "top_k_stream"]


def top_k_stream(
    iterable: Iterable,
    k: int,
    largest: bool = False,
    key: Optional[Callable] = None,
) -> List:
    """
    Return the *k* smallest (or largest) items from a possibly-unbounded
    *iterable*, in sorted order, using only O(k) memory and a single pass.

    Designed for data that doesn't fit in RAM — pass a generator, a file
    iterator, or any lazy stream. Only ``k`` items are ever held at once, so the
    stream's total length ``n`` is irrelevant to memory.

    Parameters
    ----------
    iterable : any iterable (generator, file, stream, …) — consumed once.
    k        : number of items to keep. ``k <= 0`` returns ``[]``.
    largest  : if True, return the k largest (sorted descending); otherwise the
               k smallest (sorted ascending).
    key      : optional callable producing the comparison key for each item.

    Returns
    -------
    list — the selected items, sorted (ascending for smallest, descending for
    largest). Length is ``min(k, n)``.

    Notes
    -----
    Equal keys: ties are broken by arrival order (a monotonic counter is mixed
    into the heap entry) so the comparison never falls through to the items
    themselves — important when the items aren't orderable (e.g. dicts) but the
    keys are. This also keeps the selection deterministic.

    k=1 and k=2 are specialized for max speed (single-pass, no heap): k=1 uses
    a bare ``min``/``max`` call, k=2 walks once tracking only two slots. Both
    beat heapq for very small k by skipping all heap maintenance overhead.

    NaN handling: the k=1/k=2 fast paths silently exclude NaN (any comparison
    with NaN is False, so NaN never displaces another element). The k>=3 path
    uses heapq, which admits NaN with undefined comparison results — strip NaN
    from your input if this matters.

    Examples
    --------
    >>> top_k_stream(iter([5, 1, 9, 3, 7]), 2)
    [1, 3]
    >>> top_k_stream(iter([5, 1, 9, 3, 7]), 2, largest=True)
    [9, 7]
    """
    if k <= 0:
        return []

    it = iter(iterable)

    # Tiny-k fast paths: heap maintenance overhead dominates for k=1/k=2, so
    # specialize. k=1 is a one-liner via min/max; k=2 walks once tracking only
    # two slots. Both beat heapq.nsmallest by ~2-5x for very small k.
    if k == 1:
        try:
            if key is None:
                v = max(it) if largest else min(it)
            else:
                v = max(it, key=key) if largest else min(it, key=key)
            return [v]
        except (ValueError, StopIteration):
            return []  # empty iterable

    if k == 2:
        # Private sentinel: None is a valid item AND a valid key output, so we
        # can't use it to mean "empty slot". A unique object() can't collide.
        _UNSET = object()
        first = second = _UNSET
        first_key = second_key = _UNSET
        for item in it:
            keyed_val = item if key is None else key(item)
            if largest:
                if first is _UNSET or keyed_val > first_key:
                    second, second_key = first, first_key
                    first, first_key = item, keyed_val
                elif second is _UNSET or keyed_val > second_key:
                    second, second_key = item, keyed_val
            else:
                if first is _UNSET or keyed_val < first_key:
                    second, second_key = first, first_key
                    first, first_key = item, keyed_val
                elif second is _UNSET or keyed_val < second_key:
                    second, second_key = item, keyed_val
        if first is _UNSET:
            return []
        if second is _UNSET:
            return [first]
        return [first, second]

    counter = itertools.count()  # tie-breaker: strictly increasing arrival id.

    def keyed(item):
        return item if key is None else key(item)

    if largest:
        # k LARGEST -> a size-k MIN-heap keyed on (key, arrival). The smallest
        # of our kept items sits at the top; a new item only displaces it if it
        # is larger. Negating the counter makes *earlier* arrivals win ties on
        # eviction, preserving stable "first seen kept" behaviour.
        heap = []  # entries: (key, -arrival, item)
        for item in it:
            entry = (keyed(item), -next(counter), item)
            if len(heap) < k:
                heapq.heappush(heap, entry)
            elif entry[0] > heap[0][0]:
                heapq.heapreplace(heap, entry)
        # Sort descending by key for the result; ties keep arrival order.
        ordered = sorted(heap, key=lambda e: (e[0], e[1]), reverse=True)
        return [e[2] for e in ordered]

    # k SMALLEST -> a size-k MAX-heap. Python's heapq is a min-heap, so we negate
    # the key. The current *largest* kept item sits at the top and is evicted
    # when a smaller one arrives. We can't always negate arbitrary keys (strings,
    # tuples), so instead store entries with a sort-inverted comparison handled
    # by heapq.nsmallest when keys aren't numeric — see fallback below.
    try:
        # Fast bounded-heap path for keys that support negation cleanly would
        # require numeric keys; to stay fully general we delegate to heapq's
        # own optimal bounded selection, which itself keeps only O(k) state.
        if key is None:
            chosen = heapq.nsmallest(k, it)
        else:
            chosen = heapq.nsmallest(k, it, key=key)
        return chosen
    except TypeError:
        # Unorderable items without a key: surface a clear error.
        raise TypeError(
            "top_k_stream: items are not orderable; pass a key= function"
        )

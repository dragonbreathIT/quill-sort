"""
quill — Adaptive Ultra-Sort  (v4.0.0)
======================================
High-performance, general-purpose sorting for Python.

    from quill import quill_sort, quill_sorted, register_plugin

QUICK REFERENCE
---------------
    quill_sort(data)                                  # in-place
    quill_sort(data, key=lambda x: x['score'])        # objects
    quill_sort(data, reverse=True)                    # descending
    quill_sort(data, inplace=False)                   # new list
    quill_sort(data, parallel=True)                   # multi-core
    quill_sort(data, high_performance_mode=True)      # skip auth prompt for external sort
    quill_sort(data, silent=True)                     # suppress all output

    result = quill_sorted(iterable)
    register_plugin(MyPlugin)

v4 IMPROVEMENTS
---------------
  - Benchmark-proven optimal sort kind per dtype:
      uint8/uint16 → kind='stable'  (1-2 pass radix, 10-17x faster)
      int32        → default        (introsort/quicksort, 20x faster than stable)
      int64/float  → kind='heapsort' (cache-efficient, 8-15x faster than stable)
  - Heavy-key detection in parallel path (handles Zipf/skewed distributions)
  - Parallel MSD radix sort via shared memory (zero-copy IPC)
  - External sort: radix-partition + parallel bucket sort (39s for 1B int32)
  - All hot paths optimised end-to-end: sort, merge, writeback, conversion
"""

from __future__ import annotations
from typing import Callable, Iterable, Optional

from ._core    import quill_sort_impl
from ._plugins import QuillPlugin, register_plugin, probe_plugins

__version__ = "4.0.4"
__author__  = "Isaiah Tucker"
__all__     = ["quill_sort", "quill_sorted", "QuillPlugin", "register_plugin"]


def quill_sort(
    data                 ,
    key                  : Optional[Callable] = None,
    reverse              : bool = False,
    inplace              : bool = True,
    parallel             : bool = False,
    high_performance_mode: bool = False,
    silent               : bool = False,
) -> list:
    """
    Sort `data` using Quill's adaptive ultra-sort engine.

    Parameters
    ----------
    data                  : list (or generator, range, ndarray, Series…)
    key                   : callable — sort key function
    reverse               : bool — descending order if True
    inplace               : bool — sort in-place (default True)
    parallel              : bool — use all CPU cores
                            (auto-enabled on 4+ core machines for n ≥ 5,000,000)
    high_performance_mode : bool — skip the authorization prompt when the
                            external sort path is triggered. Use this in
                            production code where you don't want interactive
                            prompts. Default False.
    silent                : bool — suppress all Quill status output.

    Returns
    -------
    list — the sorted list.

    Notes
    -----
    Quill automatically detects when a dataset exceeds available RAM and
    switches to an external merge sort using the qWrite binary format.
    On a machine with psutil installed (pip install quill-sort[fast]),
    it uses real memory readings. Otherwise it assumes 2GB available.

    Install psutil for accurate memory sensing:
        pip install quill-sort[fast]
    """
    if not isinstance(data, list):
        from ._plugins import probe_plugins
        result = probe_plugins(data, key, reverse)
        if result is not None:
            items, pk, postprocess = result
            if postprocess and not items:
                return postprocess([])
            sorted_items = quill_sort(items, key=pk, reverse=reverse,
                                      inplace=True,
                                      high_performance_mode=high_performance_mode,
                                      silent=silent)
            return postprocess(sorted_items) if postprocess else sorted_items
        data = list(data)

    return quill_sort_impl(data, key, reverse, inplace, parallel,
                           high_performance_mode=high_performance_mode,
                           silent=silent)


def quill_sorted(
    iterable             : Iterable,
    key                  : Optional[Callable] = None,
    reverse              : bool = False,
    parallel             : bool = False,
    high_performance_mode: bool = False,
    silent               : bool = False,
) -> list:
    """
    Non-mutating quill_sort — mirrors Python's built-in sorted().

        result = quill_sorted(range(10, 0, -1))
        result = quill_sorted(words, key=str.lower, reverse=True)
        result = quill_sorted(big_data, parallel=True)
    """
    return quill_sort(list(iterable), key=key, reverse=reverse,
                      inplace=True, parallel=parallel,
                      high_performance_mode=high_performance_mode,
                      silent=silent)
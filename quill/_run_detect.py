"""
quill/_run_detect.py
--------------------
Run-aware hybrid sort for nearly-sorted inputs.

Timsort is already run-aware, but it pays an O(n log r) merge bill where r is
the number of runs it discovers. For inputs that are *almost* sorted with one
or two small random islands embedded in long ascending runs, we can do better:

  1. Single forward scan -> list of maximal ascending runs.
  2. Insertion-sort each unsorted island in place (small, near its sorted spot).
  3. Linear merge passes joining adjacent runs in O(n) data movement.

When the input is meaningfully perturbed (many islands, or large ones), we bail
to None and let the caller fall back to Timsort, which is excellent at the
general case. This module is conservative on purpose: the NEVER-LOSE contract
means we'd rather decline than ship a slower path.

Public API
  try_run_hybrid(work, reverse, min_run_len, max_island_frac) -> list | None
  detect_runs(work) -> list[(start, end)]                    # exported for tests
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

logger = logging.getLogger("quill")


# A "run" here is a maximal slice work[start:end] (end exclusive) where every
# adjacent pair satisfies work[i] <= work[i+1]. Equality is allowed so the
# hybrid preserves stability the same way Python's sorted() does — equal
# elements keep their original relative order because we never swap them.
def detect_runs(work: list) -> List[Tuple[int, int]]:
    """Return the (start, end) bounds of every maximal ascending run.

    A single element counts as a run of length 1. Equal adjacent values stay
    in the same run (ascending = a <= b), matching Python sort stability.

    Returns [] for an empty list. Raises no exceptions on comparable data;
    callers that may see un-orderable elements should guard with TypeError —
    try_run_hybrid does exactly that.
    """
    n = len(work)
    if n == 0:
        return []
    runs: List[Tuple[int, int]] = []
    start = 0
    for i in range(1, n):
        # Strictly-less test (b < a) instead of "not a <= b" so types that
        # define only __lt__ (the minimum Python sort requires) still work.
        if work[i] < work[i - 1]:
            runs.append((start, i))
            start = i
    runs.append((start, n))
    return runs


def _insertion_sort_slice(work: list, lo: int, hi: int) -> None:
    """Stable in-place insertion sort of work[lo:hi].

    Used for islands (the short unsorted gaps between long runs). Insertion
    sort is the right tool here: islands are by construction small, and it
    is stable, in-place, and cache-friendly. Uses ``<`` only.
    """
    for i in range(lo + 1, hi):
        x = work[i]
        j = i - 1
        # Shift right while strictly greater than x; equal stays put -> stable.
        while j >= lo and x < work[j]:
            work[j + 1] = work[j]
            j -= 1
        work[j + 1] = x


def _merge_adjacent(work: list, lo: int, mid: int, hi: int) -> None:
    """Stable merge of work[lo:mid] and work[mid:hi] back into work[lo:hi].

    Allocates one temp buffer of size (mid - lo) — the left half — and merges
    in place from the left. This is the classic galloping-free Knuth merge;
    we don't bother with galloping because islands have already been absorbed,
    so the remaining run boundaries are well-behaved.
    """
    # Fast skips: if the boundary is already ordered, nothing to do.
    if mid >= hi or lo >= mid:
        return
    if not (work[mid] < work[mid - 1]):
        return  # work[mid-1] <= work[mid], halves are already in order
    left = work[lo:mid]
    i = 0           # index into left buffer
    j = mid         # index into right half (still in work)
    k = lo          # write cursor in work
    left_len = mid - lo
    while i < left_len and j < hi:
        # "<" on right vs left keeps the left element first on ties -> stable.
        if work[j] < left[i]:
            work[k] = work[j]
            j += 1
        else:
            work[k] = left[i]
            i += 1
        k += 1
    # Drain leftovers. Right side is already in place if it's what remains.
    while i < left_len:
        work[k] = left[i]
        i += 1
        k += 1


def try_run_hybrid(
    work: list,
    reverse: bool = False,
    min_run_len: int = 256,
    max_island_frac: float = 0.05,
) -> Optional[list]:
    """Sort *work* in place IF the hybrid is worthwhile, else return None.

    The caller is expected to fall back to ``list.sort`` (Timsort) when this
    returns None. We never raise on comparison errors — returning None lets
    the never-lose contract take over upstream.

    Parameters
    ----------
    work : list
        Mutated in place if we decide to run.
    reverse : bool
        If True, the final result is reversed at the end. We always *detect*
        ascending runs because mixing ascending + descending detection would
        require two merge strategies; reversing at the end is O(n) and keeps
        the algorithm small.
    min_run_len : int
        If the longest ascending run is shorter than this, the input isn't
        meaningfully pre-sorted and Timsort is the better choice.
    max_island_frac : float
        Fraction of elements allowed to sit inside "island" runs (runs other
        than the longest one). If too many elements are perturbed, hand off
        to Timsort.
    """
    n = len(work)
    # Hybrid has fixed overhead (run detection + buffer alloc); below this it
    # is dominated by that overhead. 2x min_run_len is a soft lower bound that
    # also guarantees at least one "main" run plus headroom for an island.
    if n < min_run_len * 2:
        return None

    # detect_runs uses "<" on adjacent elements. If any pair isn't orderable
    # we'd rather decline than raise — the dispatcher's Timsort path will
    # surface a TypeError to the user with a cleaner traceback.
    try:
        runs = detect_runs(work)
    except TypeError:
        return None
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except Exception:  # never-lose: any exotic compare fail bails out
        return None

    if not runs:
        return None

    # Find the longest run and count "island" elements (everything outside it).
    longest_idx = 0
    longest_len = 0
    for idx, (s, e) in enumerate(runs):
        rl = e - s
        if rl > longest_len:
            longest_len = rl
            longest_idx = idx

    if longest_len < min_run_len:
        # No anchor long enough to amortize the merge overhead.
        return None

    island_total = n - longest_len
    if island_total > max_island_frac * n:
        # Too perturbed — Timsort's adaptive merging will beat our naive one.
        return None

    # Sort each "island" run in place. Islands are the runs adjacent to the
    # main run; by construction they are small (sum bounded by max_island_frac
    # * n), so insertion sort is the right kernel — no allocation, stable,
    # excellent on tiny inputs.
    try:
        for idx, (s, e) in enumerate(runs):
            if idx == longest_idx:
                continue
            if e - s > 1:
                _insertion_sort_slice(work, s, e)
    except TypeError:
        return None
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except Exception:
        return None

    # After island sorting, every run in `runs` is internally sorted but the
    # runs aren't merged with each other. Do a bounded series of left-to-right
    # merge passes joining neighbors. The total data movement is O(n * passes)
    # and passes = ceil(log2(len(runs))). With islands capped at a small
    # fraction we expect very few runs (often 2-4), so this is effectively
    # O(n) in practice.
    try:
        current = [list(rng) for rng in runs]  # [(s, e), ...] — mutable copy
        while len(current) > 1:
            merged: List[List[int]] = []
            i = 0
            while i + 1 < len(current):
                lo = current[i][0]
                mid = current[i][1]      # == current[i+1][0]
                hi = current[i + 1][1]
                _merge_adjacent(work, lo, mid, hi)
                merged.append([lo, hi])
                i += 2
            if i < len(current):
                merged.append(current[i])
            current = merged
    except TypeError:
        return None
    except (KeyboardInterrupt, SystemExit, GeneratorExit):
        raise
    except Exception:
        return None

    if reverse:
        # Safe here because the run-aware hybrid is gated to identity-key numeric
        # values where equal elements are indistinguishable; do NOT extend this path
        # to keyed/object sorts without a proper stable reverse merge.
        work.reverse()

    return work

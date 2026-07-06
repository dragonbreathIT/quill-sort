"""
quill — Adaptive Ultra-Sort  (v7)
==================================
Auto-dispatching, correctness-first sorting for Python. Quill profiles your data
and routes to the fastest *correct* algorithm for your machine, with a guaranteed
fallback to np.sort / Timsort — so a result is never wrong, and at scale it
matches or beats those baselines (on tiny sub-ms sorts the dispatch overhead is a
few microseconds).

    from quill import quill_sort, sort_array, quill_topk, available_backends

QUICK REFERENCE
---------------
    quill_sort(data)                  # sorts IN PLACE (like list.sort), returns the list
    quill_sort(data, key=..., reverse=True, inplace=False, silent=True)
    quill_sorted(iterable)            # non-mutating, mirrors built-in sorted()
    sort_array(ndarray)               # zero-conversion fast path — the largest speedups
    sort_array(a, inplace=True)       #   (int64 ~3x via Rust radix / GPU; float64 ~2x)
    sort_array(a, descending=True)
    quill_argsort(data, key=...)      # stable argsort (~4-6x), any sequence/ndarray
    quill_topk(data, k, largest=False)  # k smallest/largest — ~4x faster than np.sort+slice
    analyze(data)                     # inspect how Quill profiles your data
    available_backends()              # backends usable here, in priority order
    register_plugin(MyPlugin)         # custom types

THE KEY DISTINCTION (honest, measured)
--------------------------------------
  * quill_sort(list) -> list is CONVERSION-BOUND: np.asarray(list)+tolist() dominate,
    so it MATCHES np.sort and beats Python's sorted()/list.sort() ~2-3x on numeric data.
    It sorts IN PLACE by default (inplace=True); use quill_sorted for sorted() semantics.
  * sort_array(ndarray) -> ndarray is the fast path — it never round-trips through
    Python objects and dispatches to a compiled/GPU/parallel backend.

BACKEND CHAIN (first available + worthwhile wins; always falls back to np.sort)
  rust_voracious (compiled radix, ~3x) > cupy_gpu (~3x) > polars (~2x)
  > numpy_parallel (int-only) > counting sort (bounded int64) > simd > np.sort
  Install backends via extras: quill-sort[polars], [gpu], [fast].
  Set QUILL_BACKEND_DEBUG=1 to surface (instead of swallow) backend errors.

v7 HIGHLIGHTS
-------------
  - Self-tuning dispatcher: measured latency replaces hardcoded crossovers.
  - Fast list<->ndarray conversion (optional C ext — kills the .tolist() tax).
  - Fused counting sort kernel (optional C ext).
  - SortedArray return wrapper — skip re-sort on already-sorted output.
  - Run-aware hybrid for nearly-sorted data.
  - SIMD backend explicit in the chain (no longer just np.sort fallback).
  - Scratch buffer pool — fewer per-call allocations.
  - array.array zero-copy fast path.

v6 HIGHLIGHTS (preserved)
-------------------------
  - Fixed the v5 data-corruption bug (float lists >=100k were truncated to ints).
  - sort_array() fast path + auto-dispatching backend chain with a never-lose guarantee.
  - quill_topk(); available_backends()/register_backend().
  - Visualizer (quill.visualize / `quill visualize`), setup wizard (`quill setup`),
    and an addons package (natural sort, datetime plugin, argsort, unique, streaming top-k).
  - 300+ tests incl. hypothesis property tests and a v5-corruption regression guard.
"""

from __future__ import annotations
import time
from typing import Callable, Iterable, Optional

from ._core    import quill_sort_impl, topk_impl, _identity, _profile
from ._plugins import QuillPlugin, register_plugin, probe_plugins
from .visualize import visualize, visualize_sort
from ._sorted_array import SortedArray

__version__ = "7.5.2"
__author__  = "Isaiah Tucker"


def display_version() -> str:
    """Human-facing release name, tracking the MAJOR version only — it changes
    on major releases (QuillSort.6 -> .7 -> .8 -> .9), not on patch bumps, and
    the v10 milestone is named 'QuillSort.O'. The numeric ``__version__`` above
    is the PEP 440 string pip / PyPI use.

        6.x.y -> 'QuillSort.6'   9.x.y -> 'QuillSort.9'   10.x.y -> 'QuillSort.O'
    """
    major = __version__.split(".", 1)[0]
    return "QuillSort." + ("O" if major == "10" else major)


#: Cached human-facing release name (see :func:`display_version`).
__display_version__ = display_version()

# Below this size, sort_array goes straight to numpy: no accelerated backend
# engages below its crossover, and the dispatch machinery would cost more than
# the sub-millisecond sort itself. Keeps sort_array from ever being meaningfully
# slower than np.sort on small inputs.
_SMALL_ARRAY_N = 200_000
# Integer dtypes (i32/i64/u32/u64) that Spectre's radix accelerates have a MUCH
# lower crossover: Spectre beats np.sort from ~20-25k elements up, so for those
# dtypes sort_array dispatches (to Spectre) from here instead of deferring to
# np.sort — turning the former 25k-200k *ties* into wins. Non-accelerated dtypes
# (floats, which numpy already SIMD-sorts optimally; narrow ints) keep the higher
# _SMALL_ARRAY_N floor, where np.sort is the best available and the extra dispatch
# overhead would only lose.
_SMALL_INT_N = 25_000
__all__     = [
    "quill_sort", "quill_sorted", "quill_topk", "quill_argsort", "sort_array",
    "available_backends", "QuillPlugin", "register_plugin", "analyze",
    "visualize", "visualize_sort", "run_wizard", "display_version",
    "SortedArray",
]


def quill_argsort(data, key=None, reverse=False) -> list:
    """
    Return the STABLE permutation of indices that would sort *data* — like
    ``numpy.argsort(kind='stable')`` but for any Python sequence or ndarray,
    with an optional ``key=`` and ``reverse=``.

    Matches ``sorted(range(len(data)), key=lambda i: data[i])`` EXACTLY,
    including equal-key tie order. ``None`` and ``NaN`` sort to the end (to the
    start when ``reverse=True``). Acceleration (numpy stable argsort for numeric
    arrays, polars for large string/numeric columns) is used only where it wins
    and is correct; otherwise it falls back to the exact Python result.

        idx = quill.quill_argsort(scores)            # indices of ascending order
        idx = quill.quill_argsort(rows, key=lambda r: r["name"])
        # take(data, idx) == quill.quill_sorted(data)

    Measured ~4-6x faster than ``sorted(range(n), key=...)`` for large lists.
    """
    from ._strategies import argsort_general
    return argsort_general(data, key=key, reverse=reverse)


def run_wizard(*args, **kwargs):
    """Run the interactive setup/calibration wizard (see ``quill.wizard``).
    Imported lazily so ``import quill`` stays cheap."""
    from .wizard import run_wizard as _rw
    return _rw(*args, **kwargs)


def quill_sort(
    data                 ,
    key                  : Optional[Callable] = None,
    reverse              : bool = False,
    inplace              : bool = True,
    parallel             : bool = False,
    high_performance_mode: bool = False,
    silent               : bool = False,
    stable               : bool = True,
    stats                : bool = False,
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
                            (auto-enabled on 4+ core machines for large n)
    high_performance_mode : bool — skip the authorization prompt when the
                            external sort path is triggered. Use this in
                            production code where you don't want interactive
                            prompts. Default False.
    silent                : bool — suppress all Quill status output.
    stable                : bool — True (default) guarantees sort stability,
                            matching Python's sorted() exactly. False allows
                            unstable sorts for maximum speed on numeric data.
    stats                 : bool — if True, return (sorted_list, stats_dict)
                            instead of just sorted_list.

    Returns
    -------
    list — the sorted list.
    (list, dict) — if stats=True, also returns timing and strategy info.

    Notes
    -----
    Quill automatically detects when a dataset exceeds available RAM and
    switches to an external merge sort using the qWrite binary format.
    On a machine with psutil installed (pip install quill-sort[fast]),
    it uses real memory readings. Otherwise it assumes 2GB available.

    NaN values in float data are placed at the end (or beginning if
    reverse=True), consistent with numpy's convention.
    """
    t0 = time.perf_counter() if stats else 0

    if not isinstance(data, list):
        from ._plugins import probe_plugins
        # For inplace=False, copy array-like objects before plugins
        # potentially modify in-place (e.g. NumpyArrayPlugin).
        if not inplace and hasattr(data, 'copy'):
            data = data.copy()
        result = probe_plugins(data, key, reverse, stable=stable)
        if result is not None:
            items, pk, postprocess = result
            if postprocess and not items:
                out = postprocess([])
                if stats:
                    return out, {"time_ms": (time.perf_counter() - t0) * 1000,
                                 "strategy": "plugin_direct",
                                 "n": len(out) if hasattr(out, "__len__") else None}
                return out
            sorted_items = quill_sort(items, key=pk, reverse=reverse,
                                      inplace=True,
                                      high_performance_mode=high_performance_mode,
                                      silent=silent, stable=stable)
            out = postprocess(sorted_items) if postprocess else sorted_items
            if stats:
                return out, {"time_ms": (time.perf_counter() - t0) * 1000,
                             "strategy": "plugin",
                             "n": len(out) if hasattr(out, "__len__") else None}
            return out
        data = list(data)

    result = quill_sort_impl(data, key, reverse, inplace, parallel,
                             high_performance_mode=high_performance_mode,
                             silent=silent, stable=stable)

    if stats:
        elapsed = (time.perf_counter() - t0) * 1000
        return result, {"time_ms": round(elapsed, 3), "n": len(result)}

    return result


def quill_sorted(
    iterable             : Iterable,
    key                  : Optional[Callable] = None,
    reverse              : bool = False,
    inplace              : bool = False,   # accepted for API symmetry; always returns a copy
    parallel             : bool = False,
    high_performance_mode: bool = False,
    silent               : bool = False,
    stable               : bool = True,
    stats                : bool = False,
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
                      silent=silent, stable=stable, stats=stats)


def sort_array(data, descending: bool = False, inplace: bool = False):
    """
    Sort a numpy array and return a sorted ndarray — Quill's **zero-conversion
    fast path**.

    This is the API that can beat ``np.sort`` *by a mile*: it dispatches to the
    fastest backend available on your machine (GPU → compiled parallel radix →
    polars → numpy parallel-partition) and never round-trips through Python
    objects. Pass and keep numpy arrays to capture the full speedup — the
    list-based ``quill_sort`` unavoidably pays the ``asarray``/``tolist`` tax
    that caps its win near 1.1x.

        import numpy as np
        a = np.random.randint(0, 2**31, 10_000_000)
        s = quill.sort_array(a)              # sorted copy, fastest backend
        quill.sort_array(a, inplace=True)    # sort a in place
        quill.sort_array(a, descending=True) # reverse order

    Correctness matches ``np.sort`` exactly, including negatives and NaN-to-end.
    Small arrays (below ~200k) and ineligible dtypes go straight to ``np.sort``,
    so it is never meaningfully slower; at scale it beats it (int64 ~3x, etc.).
    The descending result is a normal C-contiguous array.
    """
    import numpy as np

    # W2 fix (7.1.2, widened in 7.1.3): SUPER-FAST PATH for small ndarrays
    # with default args. The full dispatch (cupy probe + SortedArray wrap +
    # _backends import + ndim/dtype checks) costs ~3-50 μs of Python
    # overhead, which is more than np.sort itself on small arrays. This
    # path skips everything when the call is the common "small bare
    # ndarray, default args, numeric dtype" shape.
    #
    # Threshold raised to 8192 in 7.1.3 to cover the 4096-sized arrays the
    # external regression bench flagged. Sub-1024 sizes still pay ~0.5-0.7
    # µs of Python function-call overhead (irreducible without rewriting
    # sort_array in Cython); n>=1024 hits parity (≥0.99× of np.sort).
    # Object-dtype is excluded so it routes through the W4 list.sort()
    # path below.
    if (isinstance(data, np.ndarray) and data.ndim == 1
            and data.size <= 8192 and not inplace and not descending
            and data.dtype.kind != 'O'):
        return np.sort(data)

    # v7: short-circuit when the caller already handed us a SortedArray with the
    # right direction — re-sorting a known-sorted buffer would just burn cycles.
    # Lazy-imported so plain ``import quill`` users who never touch SortedArray
    # don't pay the (tiny) module-load cost on every sort_array call.
    from ._sorted_array import is_sorted_wrapper, SortedArray
    if is_sorted_wrapper(data) and data.descending == descending:
        # Already sorted in the requested direction — no work to do, return the
        # wrapper as-is regardless of inplace (caller gets back what they put in).
        return data

    from . import _backends

    original_was_ndarray = isinstance(data, np.ndarray)

    def _maybe_wrap(result):
        """Wrap a 1-D ndarray result in SortedArray for the v7 short-circuit,
        but preserve identity semantics when the caller asked for an in-place
        sort on a real ndarray."""
        if inplace and original_was_ndarray:
            return result
        if result is None or getattr(result, 'ndim', 1) != 1:
            return result
        return SortedArray(result, descending=descending)

    # Already-on-GPU data: sort on-device (zero host transfer) and return a
    # cupy array. Must come before np.asarray, which would otherwise reject a
    # cupy array (it refuses the implicit device->host transfer).
    if _backends._is_cupy_array(data):
        return _backends.sort_cupy_device(data, descending=descending, inplace=inplace)

    # 0-dim ndarray (e.g. np.array(5)): there's nothing to sort, and
    # np.sort(arr, axis=-1) would raise AxisError. Return as-is (or a copy
    # when the caller didn't ask for inplace) before the ndim != 1 branch.
    if isinstance(data, np.ndarray) and data.ndim == 0:
        return data if inplace else data.copy()

    arr = data if isinstance(data, np.ndarray) else np.asarray(data)
    if arr.ndim != 1:
        out = np.sort(arr, axis=-1)
        return np.ascontiguousarray(out[..., ::-1]) if descending else out

    # W4 fix (7.1.2): object-dtype ndarrays. np.sort on object dtype falls
    # back to per-element Python comparison via PyObject_RichCompare and runs
    # ~2x slower than CPython's tuned Timsort on the same data (measured at
    # 1M bigints: np.sort 405 ms, sorted() 209 ms). Bypass the Quill
    # dispatcher (which would add its own profile + plugin-probe overhead on
    # object data) and call list.sort() directly — same Timsort that powers
    # sorted(), no extra layers. Returns the same object-ndarray shape the
    # caller passed in so the API contract is preserved.
    if arr.dtype.kind == 'O':
        lst = list(arr)
        lst.sort(reverse=descending)
        out = np.empty(len(lst), dtype=object)
        out[:] = lst
        return out

    # Small arrays: the dispatch machinery (dtype probe + backend selection)
    # costs more than a sub-millisecond sort, and no backend engages below its
    # crossover anyway — so go straight to numpy. Keeps sort_array from ever
    # being meaningfully slower than np.sort on small inputs. The floor is
    # dtype-aware: Spectre-accelerated integers (i32/i64/u32/u64) dispatch from
    # the much lower _SMALL_INT_N (~25k, where Spectre already beats np.sort);
    # every other dtype keeps the higher _SMALL_ARRAY_N floor.
    small_floor = (_SMALL_INT_N
                   if arr.dtype.kind in "iu" and arr.dtype.itemsize in (4, 8)
                   else _SMALL_ARRAY_N)
    if arr.size < small_floor:
        if inplace and isinstance(data, np.ndarray) and arr is data \
                and data.flags["WRITEABLE"]:
            arr.sort()
            if descending:
                arr[:] = arr[::-1]
            return _maybe_wrap(arr)
        out = np.sort(arr)
        out = np.ascontiguousarray(out[::-1]) if descending else out
        return _maybe_wrap(out)

    # Don't pre-copy: np.asarray(list) is already a fresh array, and for an
    # ndarray we ask dispatch_sort to preserve the input — it copies only when a
    # mutating backend actually runs. Pre-copying here doubled the allocation on
    # the np.sort fallback path and made small inplace=False sorts lose to np.sort.
    if _backends.eligible(arr):
        preserve = (not inplace) and isinstance(data, np.ndarray)
        result = _backends.dispatch_sort(arr, descending=descending, preserve=preserve)
    else:
        if inplace and isinstance(data, np.ndarray) and arr is data \
                and data.flags["WRITEABLE"]:
            arr.sort()
            if descending:
                arr[:] = arr[::-1]
            return _maybe_wrap(arr)
        result = np.sort(arr)
        if descending:
            result = np.ascontiguousarray(result[::-1])

    if inplace and isinstance(data, np.ndarray) and result is not data \
            and result.shape == data.shape and data.flags["WRITEABLE"]:
        # Only write back when the caller's array is actually writeable — a
        # read-only inplace=True request returns the sorted copy instead of
        # raising ValueError on the assignment.
        data[:] = result
        return _maybe_wrap(data)
    return _maybe_wrap(result)


def available_backends() -> list:
    """Return the names of the fast-sort backends usable on this machine, in
    the priority order Quill will try them (e.g. ``['cupy_gpu', 'rust_voracious',
    'numpy_parallel']``). Always at least falls back to numpy/Timsort."""
    from ._backends import available_backends as _ab
    return _ab()


def quill_topk(data, k: int, largest: bool = False, key=None) -> list:
    """
    Return the *k* smallest elements of *data*, sorted ascending
    (or the *k* largest, sorted descending, when ``largest=True``).

    For small k this is dramatically faster than a full sort — it uses
    numpy's ``argpartition`` (introselect) which is O(n) instead of
    O(n log n), measured 3-4x faster than sorting then slicing.

        quill_topk(scores, 10)                 # 10 smallest
        quill_topk(scores, 10, largest=True)   # 10 largest
        quill_topk(rows, 5, key=lambda r: r.size)
    """
    return topk_impl(data, k, largest=largest, key=key)


def analyze(data, key=None):
    """
    Profile *data* and return a dict describing its characteristics, including
    which engine Quill would use.

    For a Python list/iterable this reports the list-path profile::

        >>> analyze([3, 1, 4, 1, 5, 9])
        {'n': 6, 'dtype': 'int_pos', 'presorted': False, ...}

    For a numpy ndarray it reports the *array* path that ``sort_array`` actually
    dispatches on (which keys off ``arr.dtype``, not the list profiler), so the
    reported ``dtype``/``backend`` reflect reality for arrays::

        >>> analyze(np.arange(100, dtype=np.int64))
        {'n': 100, 'dtype': 'int_pos', 'backend': 'counting', 'eligible': True, ...}
    """
    try:
        import numpy as _np
    except ImportError:
        _np = None

    if _np is not None and isinstance(data, _np.ndarray):
        from . import _backends
        arr = data
        kind = arr.dtype.kind
        if kind in "iu":
            if arr.size:
                mn, mx = int(arr.min()), int(arr.max())
            else:
                mn = mx = None
            dt = ("int_pos" if (mn is None or mn >= 0)
                  else ("int_neg" if mx < 0 else "int_mixed"))
        elif kind == "f":
            dt = "float"
            mn = float(arr.min()) if arr.size else None
            mx = float(arr.max()) if arr.size else None
        else:
            dt = {"U": "str", "S": "bytes", "b": "bool"}.get(kind, "object")
            mn = mx = None

        # v6 list-profile keys: callers that came in via analyze(list) expect
        # these to exist on the ndarray path too. Compute cheaply (and skip the
        # whole-array passes on huge arrays where they'd dominate).
        if arr.size <= 1:
            presorted   = True
            reversed_   = True
            all_same    = True
            asc_ratio   = 1.0
            desc_ratio  = 1.0
        elif arr.ndim == 1 and arr.size <= 10_000_000:
            le = arr[:-1] <= arr[1:]
            ge = arr[:-1] >= arr[1:]
            presorted  = bool(le.all())
            reversed_  = bool(ge.all())
            all_same   = bool((arr == arr[0]).all())
            asc_ratio  = float(le.mean())
            desc_ratio = float(ge.mean())
        else:
            # Multi-dim or huge: don't pay for it. Defaults match the "unknown"
            # case the v6 list path uses when it can't determine ordering.
            presorted  = False
            reversed_  = False
            all_same   = False
            asc_ratio  = 0.0
            desc_ratio = 0.0

        if kind in "iu" and mn is not None and mx is not None and arr.size > 0:
            dense  = (mx - mn) <= 2 * arr.size
            sparse = (mx - mn) > 100 * arr.size
        else:
            dense  = False
            sparse = False

        return {
            "n": int(arr.size),
            "ndim": int(arr.ndim),
            "dtype": dt,
            "numpy_dtype": arr.dtype.name,
            "min_key": mn,
            "max_key": mx,
            "has_nan": bool(kind == "f" and arr.size and bool(_np.isnan(arr).any())),
            "eligible": _backends.eligible(arr),
            # W1 fix (7.1.2): pass the min/max we already computed so
            # would_use() doesn't redo two full O(n) passes. Saves ~30 ms
            # at 50M int64. has_nan check below also avoided in the int
            # case via the kind gate.
            "backend": _backends.would_use(arr, _mn=mn, _mx=mx)
                       if kind in "iu" else _backends.would_use(arr),
            # v6 list-profile keys (so v6 callers don't KeyError on ndarray input)
            "presorted": presorted,
            "reversed": reversed_,
            "all_same": all_same,
            "asc_ratio": asc_ratio,
            "desc_ratio": desc_ratio,
            "dense": dense,
            "sparse": sparse,
            "has_none": False,
            "has_nan_known": True,
            "tuple_width": None,
            "n_unique_est": None,
        }

    if not isinstance(data, list):
        data = list(data)
    key_fn = key if key is not None else _identity
    p = _profile(data, key_fn)
    p["n"] = len(data)
    return p


def _maybe_first_run_notice() -> None:
    """Once, on the first interactive import before ``quill setup`` has run, nudge
    the user to run it. pip can't show a post-install message for wheels, so this
    is the reliable equivalent: printed to stderr, only on a TTY, only once (a
    marker is written), and silenceable with QUILL_NO_FIRST_RUN=1. Never raises."""
    import os
    import sys
    try:
        if os.environ.get("QUILL_NO_FIRST_RUN") or os.environ.get("QUILL_QUIET"):
            return
        if not sys.stderr.isatty():
            return  # non-interactive (pipes, CI, library use) -> stay silent
        from ._config import config_dir
        cdir = config_dir()
        # Already calibrated, or already nudged? Then don't nag.
        if os.path.exists(os.path.join(cdir, "config.json")):
            return
        marker = os.path.join(cdir, ".setup_hint_shown")
        if os.path.exists(marker):
            return
        sys.stderr.write(
            "\n\033[1mquill-sort\033[0m: run `quill setup` to tune for this machine and\n"
            "            install optional accelerators for maximum performance.\n"
            "            (silence this with QUILL_NO_FIRST_RUN=1)\n\n")
        try:
            os.makedirs(cdir, exist_ok=True)
            open(marker, "w").close()
        except Exception:
            pass
    except Exception:
        pass


_maybe_first_run_notice()

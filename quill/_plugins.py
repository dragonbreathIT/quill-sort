"""
quill/_plugins.py
-----------------
Autonomous support module system.

When Quill encounters a data type it can't handle through its normal integer/
float/object paths, it consults the plugin registry.  Each plugin declares
which types it handles and provides a `prepare` function that converts the
input into something Quill can sort natively.

BUILT-IN PLUGINS
----------------
  PandasSeriesPlugin    — pandas.Series  →  unwrap to list, sort, wrap back
  PandasDataFramePlugin — pandas.DataFrame  →  sort by column(s)
  NumpyArrayPlugin      — numpy.ndarray  →  sort in-place using numpy directly
  ArrayArrayPlugin      — stdlib array.array  →  zero-copy ndarray view, sort,
                          rebuild a new array.array of the same typecode
  GeneratorPlugin       — any generator / iterator  →  materialise to list
  RangePlugin           — range objects  →  already sorted or trivially reversed

REGISTERING A CUSTOM PLUGIN
-----------------------------
    from quill import register_plugin
    from quill._plugins import QuillPlugin

    class MyPlugin(QuillPlugin):
        # Types this plugin handles
        handles = (MyCustomClass,)

        @staticmethod
        def prepare(data, key, reverse):
            # Convert data to a plain list quill_sort can handle
            items = [x.value for x in data]
            return items, lambda x: x, lambda result: [MyCustomClass(v) for v in result]
            # returns: (list_to_sort, key_fn, postprocess_fn)

    register_plugin(MyPlugin)

AUTONOMOUS BEHAVIOUR
--------------------
When Quill's profiler finds an unrecognised type, it calls `probe_plugins()`
which iterates the registry and asks each plugin "can you handle this?".
The first match wins.  If no plugin matches, Quill falls back to Python's
Timsort with the provided key function.
"""

from __future__ import annotations
import inspect
import threading
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Tuple, Type


# ─────────────────────────────────────────────────────────────────────────────
# BASE CLASS
# ─────────────────────────────────────────────────────────────────────────────

class QuillPlugin(ABC):
    """
    Abstract base class for Quill plugins.

    Subclass this, set `handles`, and implement `prepare`.
    """

    #: Tuple of types this plugin handles.
    handles: Tuple[type, ...] = ()

    #: Human-readable name shown in debug output.
    name: str = "unnamed plugin"

    @classmethod
    def can_handle(cls, data: Any) -> bool:
        """Return True if this plugin can process `data`."""
        return isinstance(data, cls.handles)

    @staticmethod
    @abstractmethod
    def prepare(
        data    : Any,
        key     : Optional[Callable],
        reverse : bool,
    ) -> Tuple[list, Optional[Callable], Optional[Callable]]:
        """
        Convert `data` into a form Quill can sort.

        Parameters
        ----------
        data    : the original input
        key     : the user-supplied key function (may be None)
        reverse : whether descending order was requested

        Returns
        -------
        (items, key_fn, postprocess)
          items       — plain Python list to sort
          key_fn      — key function to use (None = identity)
          postprocess — callable(sorted_list) → final result,
                        or None to return the sorted list as-is
        """
        ...


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _reverse_array_inplace(arr) -> None:
    """
    Reverse a numpy array in-place using minimal extra memory.

    For a 1-billion int32 array (~3.7 GB), a naive ``arr[::-1].copy()``
    would allocate another 3.7 GB just for the reversed copy. This
    function swaps symmetric chunks from the outside in, using only
    ~40 MB of temporary memory per chunk.
    """
    n = len(arr)
    if n <= 1:
        return
    half = n // 2
    chunk = min(10_000_000, half)   # ~40 MB for int32
    for i in range(0, half, chunk):
        end  = min(i + chunk, half)
        left  = slice(i, end)
        right = slice(n - end, n - i)
        temp = arr[left].copy()
        arr[left]  = arr[right][::-1]
        arr[right] = temp[::-1]


# ─────────────────────────────────────────────────────────────────────────────
# PLUGIN REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

_registry: List[Type[QuillPlugin]] = []
_registry_lock = threading.Lock()

# Cache: does plugin.prepare accept a `stable` kwarg? Inspecting once per
# plugin class avoids the cost of repeated signature lookups AND avoids the
# previous TypeError-swallowing dispatch, which masked real bugs raised from
# inside plugin.prepare as "no stable kwarg" fallbacks.
_PLUGIN_SIG_CACHE: dict = {}


def _plugin_accepts_stable(plugin_cls: Type[QuillPlugin]) -> bool:
    cls_id = id(plugin_cls)
    v = _PLUGIN_SIG_CACHE.get(cls_id)
    if v is None:
        try:
            sig = inspect.signature(plugin_cls.prepare)
            v = 'stable' in sig.parameters
        except (ValueError, TypeError):
            v = False
        _PLUGIN_SIG_CACHE[cls_id] = v
    return v


def register_plugin(plugin: Type[QuillPlugin]) -> None:
    """Register a plugin class.  Later registrations take higher priority.
    Thread-safe."""
    with _registry_lock:
        _registry.insert(0, plugin)


def probe_plugins(data: Any, key, reverse: bool, stable: bool = True):
    """
    Try every registered plugin in priority order.  Thread-safe.

    When `data` is a list, checks against the first element so that
    user-registered plugins with `handles = (MyClass,)` fire correctly
    for lists of that type.

    Parameters
    ----------
    stable : bool
        Forwarded to plugins that accept it (e.g. NumpyArrayPlugin).
        Backward-compatible: plugins without a ``stable`` parameter
        are called without it.

    Returns
    -------
    (items, key_fn, postprocess) if a plugin matched, else None.
    """
    with _registry_lock:
        snapshot = _registry[:]

    # First pass: exact match on data itself
    for plugin in snapshot:
        if plugin.can_handle(data):
            if _plugin_accepts_stable(plugin):
                return plugin.prepare(data, key, reverse, stable=stable)
            return plugin.prepare(data, key, reverse)
    # Second pass: for lists, probe against the first element
    if isinstance(data, list) and data:
        probe_target = data[0]
        for plugin in snapshot:
            if (plugin.handles
                    and isinstance(probe_target, plugin.handles)
                    and plugin.can_handle(probe_target)):
                if _plugin_accepts_stable(plugin):
                    return plugin.prepare(data, key, reverse, stable=stable)
                return plugin.prepare(data, key, reverse)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# BUILT-IN PLUGINS
# ─────────────────────────────────────────────────────────────────────────────

class GeneratorPlugin(QuillPlugin):
    """Materialise any generator or iterator into a list."""

    name    = "generator"
    handles = ()   # matched by can_handle override

    @classmethod
    def can_handle(cls, data: Any) -> bool:
        import types
        return isinstance(data, (types.GeneratorType,)) or (
            hasattr(data, '__iter__') and
            hasattr(data, '__next__') and
            not isinstance(data, (list, tuple, str, bytes, dict))
        )

    @staticmethod
    def prepare(data, key, reverse):
        return list(data), key, None


class RangePlugin(QuillPlugin):
    """range objects are already sorted (or reverse-sorted)."""

    name    = "range"
    handles = (range,)

    @staticmethod
    def prepare(data, key, reverse):
        lst = list(data)
        return lst, key, None


class NumpyArrayPlugin(QuillPlugin):
    """
    numpy.ndarray — sort in-place via numpy, return the numpy array.

    CRITICAL DESIGN DECISION (v5.0.1):
      A 1-billion int32 numpy array uses ~3.73 GB.
      The same data as a Python list uses ~36 GB.
      The old code copied the array AND called .tolist(), making it
      impossible to sort 1B integers on a 16 GB machine.

      Now: sort in-place, return the numpy array directly. Zero copies
      for keyless sorts. The user gets back a sorted numpy array.
    """

    name    = "numpy.ndarray"
    handles = ()   # dynamic check below

    @classmethod
    def can_handle(cls, data: Any) -> bool:
        try:
            import numpy as np
            return isinstance(data, np.ndarray)
        except ImportError:
            return False

    @staticmethod
    def prepare(data, key, reverse, stable=True):
        import numpy as np
        if key is None and data.dtype.kind in "iuf":
            # Sort the array directly with Quill's best numeric kernel
            # (counting sort for bounded ints, parallel partition for big
            # arrays, else np.sort). No list round-trip — this is the path that
            # makes quill(ndarray) competitive with np.sort instead of 15x
            # slower (the v5 bug was forcing kind='stable' on int64).
            from ._strategies import sort_array
            nan_count = 0
            arr = data
            # Read-only arrays cannot be sorted in place; work on a writable
            # copy and return that copy as the result (we cannot mutate the
            # original buffer either way).
            read_only = not arr.flags["WRITEABLE"]
            if read_only:
                arr = arr.copy()
            if arr.dtype.kind == "f":
                nan_mask = np.isnan(arr)
                if nan_mask.any():
                    nan_count = int(nan_mask.sum())
                    arr = arr[~nan_mask]
            # sort_array picks counting sort for bounded ints (beats np.sort)
            # and np.sort otherwise — both reliably at-or-better than np.sort.
            sorted_arr = sort_array(arr, stable=stable)
            if reverse:
                sorted_arr = sorted_arr[::-1]
            if nan_count:
                nans = np.full(nan_count, np.nan, dtype=sorted_arr.dtype)
                sorted_arr = (np.concatenate([nans, sorted_arr]) if reverse
                              else np.concatenate([sorted_arr, nans]))
            if read_only:
                # Original buffer is read-only — return the sorted copy
                # without attempting to mutate `data`.
                return [], None, lambda _: sorted_arr
            # Write the result back into the original array object so callers
            # holding the reference see the sort (in-place semantics) and
            # ``quill_sort(arr) is arr`` holds.
            if sorted_arr is not data and sorted_arr.shape == data.shape:
                data[:] = sorted_arr
                result = data
            else:
                result = sorted_arr
            return [], None, lambda _: result
        # Key provided, or non-numeric dtype → fall back to list.
        lst = data.tolist()
        return lst, key, None


class PandasSeriesPlugin(QuillPlugin):
    """pandas.Series — extract values, sort, return as new Series."""

    name    = "pandas.Series"
    handles = ()

    @classmethod
    def can_handle(cls, data: Any) -> bool:
        try:
            import pandas as pd
            return isinstance(data, pd.Series)
        except ImportError:
            return False

    @staticmethod
    def prepare(data, key, reverse):
        try:
            import pandas as pd
            index  = data.index
            name   = data.name
            values = data.tolist()

            def postprocess(sorted_list):
                return pd.Series(sorted_list, name=name)

            return values, key, postprocess
        except ImportError:
            return list(data), key, None


class PandasDataFramePlugin(QuillPlugin):
    """
    pandas.DataFrame — sort rows by one or more columns.

    Pass column name(s) as the key:
        quill_sort(df, key='age')
        quill_sort(df, key=['last_name', 'first_name'])
    """

    name    = "pandas.DataFrame"
    handles = ()

    @classmethod
    def can_handle(cls, data: Any) -> bool:
        try:
            import pandas as pd
            return isinstance(data, pd.DataFrame)
        except ImportError:
            return False

    @staticmethod
    def prepare(data, key, reverse):
        try:
            import pandas as pd
            if key is None:
                # Sort by all columns
                by = list(data.columns)
            elif isinstance(key, (str, list)):
                by = key if isinstance(key, list) else [key]
            else:
                # Callable key — avoid slow per-row iloc by converting to dicts,
                # precomputing all keys, sorting by index, then fancy-indexing.
                records   = data.to_dict('records')
                keys_list = [key(row) for row in records]
                order     = sorted(range(len(records)), key=lambda i: keys_list[i],
                                   reverse=reverse)
                sorted_df = data.iloc[order].reset_index(drop=True)
                return [], None, lambda _: sorted_df

            sorted_df = data.sort_values(by=by, ascending=not reverse)
            # Return a sentinel that signals "already done"
            return [], None, lambda _: sorted_df
        except ImportError:
            return list(data.itertuples()), key, None


class ArrayArrayPlugin(QuillPlugin):
    """stdlib array.array of a numeric typecode — sort via a numpy view
    of the same buffer (ZERO copy). Returns a NEW array.array with sorted
    values. Avoids the list-of-PyLong conversion tax."""

    name    = "array.array"
    handles = ()   # dynamic check below

    # array.array typecode → numpy dtype (only numeric typecodes; 'u' = unicode
    # is excluded — it's not numeric in the sortable sense we want).
    _TYPECODE_DTYPE = {
        'b': 'int8',  'B': 'uint8',
        'h': 'int16', 'H': 'uint16',
        'i': 'int32', 'I': 'uint32',
        'l': 'int32', 'L': 'uint32',   # platform-dependent width; numpy view
        'q': 'int64', 'Q': 'uint64',   # handles itemsize correctly below
        'f': 'float32', 'd': 'float64',
    }

    @classmethod
    def can_handle(cls, data: Any) -> bool:
        try:
            import array as _array
            return isinstance(data, _array.array)
        except ImportError:
            return False

    @staticmethod
    def prepare(data, key, reverse, stable=True):
        try:
            import array as _array
            import numpy as np
            from ._strategies import sort_array as _sort_arr

            # Resolve typecode → numpy dtype using actual itemsize so 'l'/'L'
            # work on both 32- and 64-bit platforms without surprises.
            tc = data.typecode
            if tc not in ArrayArrayPlugin._TYPECODE_DTYPE or key is not None:
                # Non-numeric typecode (e.g. 'u') or keyed sort → list path.
                return list(data), key, None
            itemsize = data.itemsize
            if tc in ('b', 'h', 'i', 'l', 'q'):
                np_dtype = np.dtype('int%d' % (itemsize * 8))
            elif tc in ('B', 'H', 'I', 'L', 'Q'):
                np_dtype = np.dtype('uint%d' % (itemsize * 8))
            else:  # 'f' or 'd'
                np_dtype = np.dtype('float%d' % (itemsize * 8))

            # np.frombuffer gives a writable view of the caller's buffer; the
            # sort kernel may mutate it in place. Copy first to decouple from
            # the caller's array.array so we never mutate the original input.
            arr = np.frombuffer(data, dtype=np_dtype).copy()

            sorted_arr = _sort_arr(arr, stable=stable)
            if reverse:
                sorted_arr = sorted_arr[::-1]

            # Build a fresh array.array of the same typecode so the caller
            # gets back the same kind of object they passed in.
            out_arr = _array.array(tc)
            out_arr.frombytes(np.ascontiguousarray(sorted_arr).tobytes())
            return [], None, lambda _: out_arr
        except BaseException:
            import os
            if os.environ.get("QUILL_BACKEND_DEBUG"):
                raise
            # NEVER-LOSE: degrade to list path; quill_sort will Timsort it.
            return list(data), key, None


# ─────────────────────────────────────────────────────────────────────────────
# REGISTER ALL BUILT-INS
# ─────────────────────────────────────────────────────────────────────────────
# Lower priority first (inserted at front so last = highest priority)

register_plugin(GeneratorPlugin)
register_plugin(RangePlugin)
register_plugin(ArrayArrayPlugin)
register_plugin(NumpyArrayPlugin)
register_plugin(PandasSeriesPlugin)
register_plugin(PandasDataFramePlugin)
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
# PLUGIN REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

_registry: List[Type[QuillPlugin]] = []


def register_plugin(plugin: Type[QuillPlugin]) -> None:
    """Register a plugin class.  Later registrations take higher priority."""
    _registry.insert(0, plugin)


def probe_plugins(data: Any, key, reverse: bool):
    """
    Try every registered plugin in priority order.

    When `data` is a list, checks against the first element so that
    user-registered plugins with `handles = (MyClass,)` fire correctly
    for lists of that type.

    Returns
    -------
    (items, key_fn, postprocess) if a plugin matched, else None.
    """
    # For lists, probe against the first element so custom plugins fire correctly
    probe_target = data[0] if isinstance(data, list) and data else data
    for plugin in _registry:
        if plugin.can_handle(probe_target) or plugin.can_handle(data):
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
    """numpy.ndarray — sort in-place via numpy then return."""

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
    def prepare(data, key, reverse):
        import numpy as np
        arr = data.copy()
        if key is None:
            arr.sort()
            if reverse:
                arr = arr[::-1]
            return arr.tolist(), None, None
        # Key provided — fall back to list sort
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
                # callable key — convert to row list and use it
                rows = [data.iloc[i] for i in range(len(data))]
                return rows, key, None

            sorted_df = data.sort_values(by=by, ascending=not reverse)
            # Return a sentinel that signals "already done"
            return [], None, lambda _: sorted_df
        except ImportError:
            return list(data.itertuples()), key, None


# ─────────────────────────────────────────────────────────────────────────────
# REGISTER ALL BUILT-INS
# ─────────────────────────────────────────────────────────────────────────────
# Lower priority first (inserted at front so last = highest priority)

register_plugin(GeneratorPlugin)
register_plugin(RangePlugin)
register_plugin(NumpyArrayPlugin)
register_plugin(PandasSeriesPlugin)
register_plugin(PandasDataFramePlugin)
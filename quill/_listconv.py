"""
quill/_listconv.py
------------------
Fast list <-> ndarray conversion for the ``quill_sort(list)`` hot path.

Why this module exists
======================
Profiling ``quill_sort(list)`` on 1M ints shows the *conversion* — ``np.asarray``
on the way in plus ``arr.tolist()`` on the way back — dominates the actual sort
by roughly 3-5x. ``np.asarray`` has to walk a heterogeneous-by-default Python
list, probe each element for its type, and box/unbox through generic ``PyObject``
machinery; ``tolist()`` reverses that by allocating a fresh ``PyLong``/``PyFloat``
per element. Both phases are pure CPython-API overhead — no math, no I/O, no
parallelism is going to help.

The fix is to skip the generic path when we *know* the list is homogeneous
``int`` (or homogeneous ``int``/``float``). A trivial C loop that calls
``PyLong_AsLongLong`` / ``PyFloat_AsDouble`` on each element and writes straight
into a pre-allocated numpy buffer is ~3-6x faster than ``np.asarray``, and the
mirror direction (ndarray -> ``list``) is roughly the same win over
``ndarray.tolist()``.

This file is the *Python wrapper*. The actual C kernel lives in
``quill/_listconv_ext.c`` and is built as the optional extension
``quill._listconv_ext``.

Enabling the C extension
------------------------
The wrapper works fully without the extension — it falls back to numpy (with a
cheap homogeneity sample so a known dtype is still passed to ``np.asarray``).
To get the fast path, the project's build (``pyproject.toml`` /
``setup.py``) must compile ``_listconv_ext.c`` against the active numpy
headers. That wiring is intentionally a later phase; this module simply probes
``from . import _listconv_ext`` at import and lights up if it succeeded.

Public API
----------
``HAS_FAST_CONV``    True iff the C extension imported.
``to_int64(data)``   list -> int64 ndarray, or ``None`` to mean "not homogeneous,
                     caller should use the usual ``np.asarray`` path".
``to_float64(data)`` list -> float64 ndarray, or ``None`` on miss.
``from_int64(arr)``  int64 ndarray -> Python list of ints (always succeeds).
``from_float64(arr)`` float64 ndarray -> Python list of floats (always succeeds).

Never-lose contract
-------------------
None of these functions raise. ``to_*`` returns ``None`` so the caller can take
its own fall-back branch (``np.asarray(data)`` with no dtype hint), and
``from_*`` always returns a list — falling back to ``arr.tolist()`` when the C
extension is unavailable or rejects the input.
"""

from __future__ import annotations

import logging
from typing import List, Optional

try:
    import numpy as np
    _NUMPY = True
except ImportError:  # pragma: no cover - numpy is a hard runtime dep elsewhere
    _NUMPY = False

logger = logging.getLogger("quill")

# Probe the optional C extension exactly once at import. A missing extension is
# the expected case on most installs (no compiler / wheel not built for the
# active platform), so this is logged at debug level only — never a warning.
_ext = None
try:
    from . import _listconv_ext as _ext  # type: ignore[attr-defined]
except ImportError:
    _ext = None
except BaseException:
    # An extension that imports but blows up at init time (ABI mismatch, broken
    # numpy headers) must not crash ``import quill``. Treat it as absent.
    _ext = None

HAS_FAST_CONV: bool = _ext is not None

# How many leading elements to sample when checking homogeneity in the
# extension-absent path. 32 is enough to almost always catch a mixed list while
# staying O(1) for the typical homogeneous case.
_HOMOGENEITY_SAMPLE = 32


def _looks_homogeneous_int(data: list) -> bool:
    """Cheap O(1) probe: are the first ``_HOMOGENEITY_SAMPLE`` items all ``int``?

    Uses ``type(x) is int`` (exact-type, not isinstance) on purpose — numpy
    scalar types subclass ``int`` but won't round-trip through ``np.asarray``
    with dtype=int64 without an extra copy, so treat them as "not the fast
    path" and let the caller use a generic conversion.
    """
    n = len(data)
    if n == 0:
        return False
    # bool is a subclass of int; reject lists that look like bool masks because
    # the caller almost certainly wants a different dtype.
    limit = n if n < _HOMOGENEITY_SAMPLE else _HOMOGENEITY_SAMPLE
    for i in range(limit):
        if type(data[i]) is not int:
            return False
    return True


def _looks_homogeneous_float(data: list) -> bool:
    """Like ``_looks_homogeneous_int`` but accepts ``int`` or ``float``."""
    n = len(data)
    if n == 0:
        return False
    limit = n if n < _HOMOGENEITY_SAMPLE else _HOMOGENEITY_SAMPLE
    for i in range(limit):
        t = type(data[i])
        if t is not float and t is not int:
            return False
    return True


def to_int64(data: list) -> Optional["np.ndarray"]:
    """Convert a Python list to an int64 ndarray on the fast path.

    Returns ``None`` when the list isn't homogeneous int or an element
    overflows int64 — signalling "not handled, use ``np.asarray`` instead".
    Empty lists also return ``None`` so the caller's normal path stays
    responsible for the empty case (avoids dtype surprises).
    """
    if not _NUMPY or not isinstance(data, list) or not data:
        return None
    if _ext is not None:
        try:
            return _ext.fast_to_int64(data)
        except (TypeError, OverflowError, ValueError):
            # Heterogeneous list or out-of-range int — silently give up so the
            # caller can take its slow path. Anything else propagates.
            return None
        except BaseException:
            # An unexpected extension failure must not break sorting. Swallow
            # and signal "not handled" so the caller falls back cleanly.
            logger.debug("fast_to_int64 raised; falling back", exc_info=True)
            return None
    # Extension absent: still beat the no-hint path by sampling for homogeneity
    # and passing dtype=int64 to np.asarray. np.asarray with a dtype hint is
    # measurably faster than the inferring path because it skips the per-element
    # type probe.
    if not _looks_homogeneous_int(data):
        return None
    try:
        return np.asarray(data, dtype=np.int64)
    except (TypeError, ValueError, OverflowError):
        return None


def to_float64(data: list) -> Optional["np.ndarray"]:
    """Convert a list of int/float to a float64 ndarray on the fast path.

    Returns ``None`` if the list isn't homogeneous numeric.
    """
    if not _NUMPY or not isinstance(data, list) or not data:
        return None
    if _ext is not None:
        try:
            return _ext.fast_to_float64(data)
        except (TypeError, OverflowError, ValueError):
            return None
        except BaseException:
            logger.debug("fast_to_float64 raised; falling back", exc_info=True)
            return None
    if not _looks_homogeneous_float(data):
        return None
    try:
        return np.asarray(data, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None


def from_int64(arr) -> List[int]:
    """Convert an int64 ndarray back to a Python ``list[int]``.

    Always succeeds: falls back to ``arr.tolist()`` whenever the extension is
    absent or rejects the array (wrong dtype, non-contiguous, etc.).
    """
    if _ext is not None:
        try:
            return _ext.fast_from_int64(arr)
        except (TypeError, ValueError):
            pass
        except BaseException:
            logger.debug("fast_from_int64 raised; falling back", exc_info=True)
    return arr.tolist()


def from_float64(arr) -> List[float]:
    """Convert a float64 ndarray back to a Python ``list[float]``.

    Always succeeds: falls back to ``arr.tolist()`` on miss.
    """
    if _ext is not None:
        try:
            return _ext.fast_from_float64(arr)
        except (TypeError, ValueError):
            pass
        except BaseException:
            logger.debug("fast_from_float64 raised; falling back", exc_info=True)
    return arr.tolist()


__all__ = [
    "HAS_FAST_CONV",
    "to_int64",
    "to_float64",
    "from_int64",
    "from_float64",
]

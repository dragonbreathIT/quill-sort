"""
quill/_profile.py
-----------------
Single-pass data profiler.

Samples up to 512 elements uniformly across the dataset and classifies
dtype, pre-sortedness, density, uniqueness, and special values (None, NaN).
"""

from __future__ import annotations
import numbers
import random as _random

try:
    import numpy as _np
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover - numpy is a hard dep elsewhere
    _np = None
    _HAS_NUMPY = False

_SAMPLE = 512
# Full-array NaN scan is cheap (~5ms / 1M floats); above this we keep the
# sample-only hint and let kernels do their own chunked probe.
_FULL_NAN_SCAN_LIMIT = 1_000_000


def profile(data: list, key_fn) -> dict:
    """
    Profile a dataset by sampling up to 512 elements.

    Returns a dict with keys:
      n              : int   -- dataset length
      dtype          : str   -- one of 'int_pos', 'int_neg', 'int_mixed',
                                'float', 'str', 'bytes', 'tuple', 'object'
      presorted      : bool  -- True if sample is non-decreasing
      reversed       : bool  -- True if sample is non-increasing
      all_same       : bool  -- True if all sampled values are equal
      asc_ratio      : float -- fraction of adjacent pairs that are non-decreasing
      desc_ratio     : float -- fraction of adjacent pairs that are non-increasing
      dense          : bool  -- True if range <= 2*n (counting sort candidate)
      sparse         : bool  -- True if range > 100*n
      min_key        : comparable or None
      max_key        : comparable or None
      has_none       : bool  -- True if any sampled value is None
      has_nan        : bool  -- True if any sampled float is NaN (best-effort
                               unless ``has_nan_known`` is True)
      has_nan_known  : bool  -- True iff we performed a full-array NaN scan;
                                when False, ``has_nan`` reflects only the sample
      tuple_width    : int or None -- arity of tuple keys when dtype == 'tuple'
                                      and all sampled tuples share one width
      n_unique_est   : int or None -- estimated unique count from sample (ints only)
    """
    n = len(data)
    p = {
        "n": n, "dtype": "object",
        "presorted": False, "reversed": False, "all_same": False,
        "sparse": False, "dense": False,
        "min_key": None, "max_key": None,
        "has_none": False, "has_nan": False, "has_nan_known": False,
        "tuple_width": None, "n_unique_est": None,
    }
    if n == 0:
        return p

    # Anti-aliasing: add a small random offset to prevent sampling artefacts
    # with periodic data (e.g., alternating [0, 1, 0, 1, ...]).
    step   = max(1, n // _SAMPLE)
    offset = _random.randint(0, max(0, step - 1)) if step > 1 else 0

    # Identity key_fn is the hot path; skip the per-element call to avoid Python
    # overhead. ``_identity`` is the conventional name used by the public API.
    _is_identity = key_fn is None or getattr(key_fn, "__name__", "") == "_identity"
    if _is_identity:
        if step > 1:
            raw = data[offset::step][:_SAMPLE]
        else:
            raw = data[:_SAMPLE]
        keys = [v for v in raw if v is not None]
        p["has_none"] = (len(keys) != len(raw))
    else:
        # CPython's contract: a user-provided key function is called EXACTLY
        # ONCE per element. Sampling here (the old behaviour) would call it a
        # SECOND time on min(n, 512) elements, since the downstream sorter
        # (numpy_sort_by_key / list.sort) always re-evaluates the key over the
        # full data. For an idempotent key that was merely wasteful; for a
        # non-idempotent key (counter, timestamp, RNG tiebreaker, cached
        # property with eviction) it produced silent wrong output — fixed in
        # 7.0.4. Skip the key-sampling step; return a minimal profile and let
        # the sort path call the key the contractually-correct number of times.
        # Cost: no dtype-based routing for lambda keys (itemgetter/attrgetter
        # bypass the dtype check via the OR in _core.py anyway, so the loss
        # is small in practice).
        return p
    if not keys:
        return p

    s = len(keys)

    # Pre-sortedness
    # ArithmeticError catches decimal.InvalidOperation (raised when comparing
    # Decimal('NaN')) — InvalidOperation → DecimalException → ArithmeticError.
    try:
        asc  = sum(1 for i in range(s-1) if keys[i] <= keys[i+1])
        desc = sum(1 for i in range(s-1) if keys[i] >= keys[i+1])
        p["presorted"]  = (asc  == s-1)
        p["reversed"]   = (desc == s-1)
        p["all_same"]   = (asc  == s-1 and desc == s-1)
        p["asc_ratio"]  = asc  / (s - 1) if s > 1 else 1.0
        p["desc_ratio"] = desc / (s - 1) if s > 1 else 1.0
    except (TypeError, ArithmeticError):
        p["asc_ratio"]  = 0.0
        p["desc_ratio"] = 0.0
        p["presorted"]  = False
        p["reversed"]   = False
        p["all_same"]   = False

    # Type detection.
    # The exact-type set checks are the fast common case (pure-Python int/float).
    # The numbers-ABC fallback catches numpy scalars (np.int64, np.float64, …)
    # and other numeric subclasses, which are NOT exact ``int``/``float`` and so
    # would otherwise be misread as 'object' — silently missing the numeric fast
    # path. ``bool`` is excluded from the fallback so its classification (object)
    # is unchanged.
    sample_types = set(type(k) for k in keys)

    is_int = sample_types <= {int} or all(
        isinstance(k, numbers.Integral) and not isinstance(k, bool) for k in keys)
    is_float = (not is_int) and (
        sample_types <= {int, float} or all(
            isinstance(k, numbers.Real) and not isinstance(k, bool) for k in keys))

    if is_int:
        mn, mx = min(keys), max(keys)
        p["min_key"] = mn
        p["max_key"] = mx
        rng = mx - mn
        p["dtype"]  = "int_pos" if mn >= 0 else ("int_neg" if mx < 0 else "int_mixed")
        p["dense"]  = (rng <= 2 * n)
        p["sparse"] = (rng > 100 * n)
        p["n_unique_est"] = len(set(keys))

    elif is_float:
        # Pure floats or mixed int/float (incl. numpy scalars) → float path.
        p["dtype"]   = "float"
        p["min_key"] = min(keys)
        p["max_key"] = max(keys)
        # NaN detection: NaN != NaN (true for python float and numpy float).
        sample_has_nan = any(v != v for v in keys)
        p["has_nan"] = sample_has_nan
        # Strengthen the hint with a full scan when it's cheap. Only worth the
        # numpy conversion when the sample actually saw a float (avoid paying
        # this on int/string data) and only for identity key extraction (so we
        # know the list contents match the keys).
        if (_is_identity and _HAS_NUMPY
                and n <= _FULL_NAN_SCAN_LIMIT
                and not sample_has_nan):
            try:
                p["has_nan"] = bool(
                    _np.isnan(_np.asarray(data, dtype=_np.float64)).any())
                p["has_nan_known"] = True
            except BaseException:
                # Conversion failed (e.g. mixed types) — keep sample hint.
                pass
        elif sample_has_nan:
            # Sample already proved a NaN exists; no need to scan further.
            p["has_nan_known"] = True

    elif sample_types <= {str}:
        p["dtype"] = "str"
    elif sample_types <= {bytes}:
        p["dtype"] = "bytes"
    elif sample_types <= {tuple}:
        # Multi-key sort hint. Routes already handle tuples via
        # numpy_sort_by_key; the width is informational for future tuning.
        p["dtype"] = "tuple"
        widths = {len(k) for k in keys if isinstance(k, tuple)}
        p["tuple_width"] = widths.pop() if len(widths) == 1 else None

    return p

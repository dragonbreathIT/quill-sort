"""
quill/_profile.py
-----------------
Single-pass data profiler.
"""

from __future__ import annotations

_SAMPLE = 512


def profile(data: list, key_fn) -> dict:
    n = len(data)
    p = {
        "n": n, "dtype": "object",
        "presorted": False, "reversed": False, "all_same": False,
        "sparse": False, "dense": False,
        "min_key": None, "max_key": None,
        "has_none": False, "n_unique_est": None,
    }
    if n == 0:
        return p

    step   = max(1, n // _SAMPLE)
    raw    = [data[i * step] for i in range(min(_SAMPLE, n))]
    p["has_none"] = any(v is None for v in raw)
    keys   = [key_fn(v) for v in raw if v is not None]
    if not keys:
        return p

    s = len(keys)

    # Pre-sortedness
    try:
        asc  = sum(1 for i in range(s-1) if keys[i] <= keys[i+1])
        desc = sum(1 for i in range(s-1) if keys[i] >= keys[i+1])
        p["presorted"] = (asc  == s-1)
        p["reversed"]  = (desc == s-1)
        p["all_same"]  = (asc  == s-1 and desc == s-1)
    except TypeError:
        pass

    # Type detection
    sample_types = set(type(k) for k in keys)

    if sample_types <= {int}:
        mn, mx = min(keys), max(keys)
        p["min_key"] = mn
        p["max_key"] = mx
        rng = mx - mn
        p["dtype"]  = "int_pos" if mn >= 0 else ("int_neg" if mx < 0 else "int_mixed")
        p["dense"]  = (rng <= 2 * n)
        p["sparse"] = (rng > 100 * n)

        # Estimate unique count from sample — key for duplicate detection
        p["n_unique_est"] = len(set(keys))

    elif sample_types <= {float}:
        p["dtype"]   = "float"
        p["min_key"] = min(keys)
        p["max_key"] = max(keys)
    elif sample_types <= {str}:
        p["dtype"] = "str"
    elif sample_types <= {bytes}:
        p["dtype"] = "bytes"

    return p
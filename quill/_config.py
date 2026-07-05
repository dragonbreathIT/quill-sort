"""
quill/_config.py
----------------
Per-machine configuration.

Quill must be *good for every user*, not tuned to one developer's CPU.
Thresholds that depend on hardware (when parallelism starts to pay off, how
much RAM is free) therefore live here and can be calibrated at runtime by the
setup wizard (``quill setup``) and persisted to ``~/.quill/config.json``.

If no config file exists, conservative built-in defaults are used.  The
defaults are chosen so Quill *never loses* to the naive baseline on any
machine — parallelism only engages well past the point where it reliably wins,
so a 2-core laptop or a CI runner pays no penalty.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

# Conservative, machine-agnostic defaults. Calibration can lower the parallel
# threshold on big multi-core boxes; until then we stay safe.
DEFAULTS: Dict[str, Any] = {
    # Auto-parallel is OFF by default. Honest reason: numpy's single-threaded
    # sort is an AVX introsort running near memory bandwidth, so a parallel
    # partition sort only *marginally* beats it (~1.1x) and only on large
    # arrays with many cores — and can lose to thread overhead. Rather than
    # risk a regression on an unknown machine, Quill stays sequential unless
    # the user opts in (parallel=True) or the wizard *measures* a real win on
    # this box and flips this to True.
    "auto_parallel": False,
    # Minimum element count before the parallel engine is even considered.
    "parallel_min_n": 5_000_000,
    # Minimum CPU cores before parallelism is considered worthwhile.
    "parallel_min_cores": 8,
    # Worker count for the parallel engine (0 = use all cores). Calibration
    # tends to find a sweet spot well below the core count.
    "parallel_max_workers": 0,
    # numpy conversion only pays off above this list length.
    "numpy_min_n": 4_000,
    # Was this config produced by the calibrating wizard, or just defaults?
    "calibrated": False,
}


def config_dir() -> str:
    """Return the directory Quill stores its config in (created on demand)."""
    base = os.environ.get("QUILL_CONFIG_DIR")
    if not base:
        base = os.path.join(os.path.expanduser("~"), ".quill")
    return base


def config_path() -> str:
    return os.path.join(config_dir(), "config.json")


_cache: Dict[str, Any] | None = None


def load_config(force: bool = False) -> Dict[str, Any]:
    """Load config, merged over defaults. Cached after first read.

    Never raises: a missing or corrupt file falls back to defaults so the
    library always imports and sorts.
    """
    global _cache
    if _cache is not None and not force:
        return _cache
    cfg = dict(DEFAULTS)
    # Environment overrides win over the file (handy for CI / containers).
    path = config_path()
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                cfg.update(json.load(fh))
    except (OSError, ValueError, TypeError):
        pass  # corrupt file → defaults
    _env_override(cfg, "QUILL_PARALLEL_MIN_N", "parallel_min_n", int)
    _env_override(cfg, "QUILL_PARALLEL_MIN_CORES", "parallel_min_cores", int)
    _env_override(cfg, "QUILL_MAX_WORKERS", "parallel_max_workers", int)
    _cache = cfg
    return cfg


def _env_override(cfg: Dict[str, Any], env: str, key: str, cast) -> None:
    val = os.environ.get(env)
    if val is not None:
        try:
            cfg[key] = cast(val)
        except (ValueError, TypeError):
            pass


def save_config(cfg: Dict[str, Any]) -> str:
    """Persist *cfg* to disk and refresh the cache. Returns the path written."""
    global _cache
    d = config_dir()
    os.makedirs(d, exist_ok=True)
    path = config_path()
    merged = dict(DEFAULTS)
    merged.update(cfg)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2, sort_keys=True)
    _cache = merged
    return path

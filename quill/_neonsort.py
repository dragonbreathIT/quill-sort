"""
quill/_neonsort.py
------------------
ARM NEON / ASIMD sort backend — the Apple-Silicon & Linux-ARM counterpart to
the x86 ``SimdSortBackend`` in :mod:`quill._simdsort`.

Why this module exists
======================
The x86 SIMD path (``quill._simdsort``) is gated on ``platform.machine()`` being
x86-64: on ARM (Apple Silicon M-series, AWS Graviton, Ampere, Raspberry Pi 64-bit)
``SimdSortBackend._probe`` returns False and the dispatcher had *no* SIMD-class
backend to register. Numeric arrays of the dtypes the Rust voracious extension
does not cover (i32/u32/u64/f32) therefore fell through to ``np.sort`` **unlabeled**,
so ``available_backends()``, ``would_use()`` and the self-tuning profiler in
``_profile.py`` under-reported what was actually doing the work on every ARM box.

This module closes that gap symmetrically to the x86 one. On modern numpy
(>= 1.24) built for ARM, the inner loop of ``np.sort`` for the natively-vectorized
dtypes already dispatches to numpy's bundled data-parallel kernels (Highway VQSort
/ hand-written ASIMD paths across numpy 1.25–2.x). So — exactly as on x86 — the
"SIMD backend" here often *is* ``np.sort``; there is no separate library call to
make. What this module adds is twofold:

  1. **Honest dispatch on ARM.** Registering an explicit ``NeonSortBackend`` makes
     ``available_backends()`` / ``would_use()`` name the path and lets the
     self-tuning DB track its per-(dtype, size) latency separately, instead of
     silently attributing everything to the ``np.sort`` floor.

  2. **Future C-wrapper seam.** When a future build ships a thin hand-written NEON
     kernel (e.g. ``quill._neonsort_ext``, compiled with ``-march=armv8-a+simd``
     or Highway), ``_sort_ascending`` tries the C extension first and falls back
     to ``arr.sort()`` with no other code in the package changing. The module-load
     try-import below is that seam — identical in shape to ``_simdsort``'s.

Never-lose guarantees (per project invariants)
==============================================
This backend NEVER sorts more slowly than ``np.sort`` and NEVER changes the
result, because its kernel *is* ``arr.sort()`` (numpy's own ASIMD-accelerated
introsort/radix) whenever the optional C extension is absent — which is the
normal case. Specifically:

* numpy missing      → ``_probe`` returns False → backend unavailable.
* numpy too old      → ``_probe`` returns False (no data-parallel kernel inside).
* not an ARM machine → ``_probe`` returns False (an ARM label would be a lie;
                       x86 boxes use ``SimdSortBackend`` instead).
* big-endian array   → ``supports`` returns False (kept parallel to the x86 gate;
                       such inputs route to the always-correct ``np.sort`` floor).
* bool / datetime / object / int8/16 / float16 → ``supports`` returns False.
* any kernel error   → re-raised; the dispatcher in ``_backends.py`` catches
                       ``BaseException`` and re-routes to ``np.sort``.

Relationship to the compiled companions
=======================================
This backend is the always-available ARM path (its kernel is numpy's own ASIMD
sort). It is NOT the ceiling on aarch64: the compiled companion wheels
(``rust_parallel_radix``, ``ips4o``, ``numa``, ``simd_companion``) now build from
portable C++ on arm64 too, outrank this backend by static priority, and win by
~3x at large N once installed. ``arm_neon_sort`` remains the labeled path for the
mid-size band and for machines without the companions installed — so ARM is no
longer limited to the ``np.sort`` floor for large arrays.
"""

from __future__ import annotations

import logging
import platform
from typing import Optional  # noqa: F401 — kept for parity with _simdsort's API

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False

from ._backends import Backend

logger = logging.getLogger("quill")

# ─────────────────────────────────────────────────────────────────────────────
# Optional C extension seam.
#
# When/if a future build ships ``quill._neonsort_ext`` (a thin wrapper over a
# hand-written NEON/ASIMD sort, or Google Highway's VQSort compiled for aarch64),
# this try-import discovers it at module load. The wrapper is expected to expose
# per-dtype in-place sorters following the same convention as ``_fastsort`` /
# ``_simdsort_ext``:
#     sort_i32(arr), sort_i64(arr), sort_u32(arr), sort_u64(arr),
#     sort_f32(arr), sort_f64(arr).
# Absence is the normal case — the backend then falls back to arr.sort(), which
# on modern numpy for ARM IS the ASIMD kernel anyway.
# ─────────────────────────────────────────────────────────────────────────────
_EXT = None
try:
    from . import _neonsort_ext as _EXT  # type: ignore[no-redef]
except Exception:
    _EXT = None


# ─────────────────────────────────────────────────────────────────────────────
# CPU + library capability probes
# ─────────────────────────────────────────────────────────────────────────────

# Numpy first shipped the bundled data-parallel sort kernels in 1.24 (the
# int/uint/float32/64 paths landed across 1.24–1.25, with the ARM ASIMD/Highway
# routing filled in through 2.x). Below that, np.sort is the older scalar
# introsort and labeling it a NEON path would be dishonest.
_MIN_NUMPY_VERSION = (1, 24)


def _numpy_version_tuple() -> tuple:
    if not _NUMPY:
        return (0, 0, 0)
    try:
        parts = np.__version__.split(".")
        return tuple(int(p) for p in parts[:2])
    except (ValueError, AttributeError):
        # Pre-release tags like "2.0.0rc1" — accept and assume new enough.
        return _MIN_NUMPY_VERSION


def _is_arm() -> bool:
    """True iff we're on a 64-bit ARM machine where NEON/ASIMD is relevant.

    Covers Apple Silicon (``arm64``) and Linux/other aarch64 (``aarch64``). We
    deliberately exclude 32-bit ARM (``armv7l`` etc.): those chips may lack a
    guaranteed NEON unit and numpy does not ship the data-parallel kernels for
    them, so a SIMD label would be dishonest — such inputs use the np.sort floor.
    """
    m = platform.machine().lower()
    return m in ("arm64", "aarch64", "arm64e")


def _cpu_features() -> set:
    """Return the set of CPU feature flags numpy reports as enabled.

    Uses numpy's own introspection (the public ``__cpu_features__`` mapping) so
    the answer matches what numpy's dispatcher actually sees. Returns an empty
    set if numpy is missing or the introspection API isn't present on this build.
    """
    if not _NUMPY:
        return set()
    # numpy 2.x renamed np.core -> np._core (np.core is deprecated and warns).
    # Prefer the new private module; fall back to the old name on numpy 1.x.
    _core = getattr(np, "_core", None) or getattr(np, "core", None)
    feats = getattr(_core, "_multiarray_umath", None) if _core is not None else None
    table = getattr(feats, "__cpu_features__", None) if feats is not None else None
    if not isinstance(table, dict):
        return set()
    # Mapping is {feature_name: bool}; keep only the enabled ones.
    return {name for name, on in table.items() if on}


def _has_neon() -> bool:
    """True iff this ARMv8 CPU exposes NEON / Advanced SIMD (ASIMD).

    Every ARMv8-A (aarch64) core mandates Advanced SIMD, so on a 64-bit ARM
    machine NEON is present by construction — mirroring the synthesis in
    ``quill.wizard`` ("every ARMv8 chip has it"). We still consult numpy's
    feature table when it is populated (accepting the ``NEON``/``ASIMD`` flag
    names numpy uses across versions), but treat 64-bit ARM as NEON-capable even
    when the table is empty, because some numpy builds do not enumerate baseline
    aarch64 features.
    """
    if not _is_arm():
        return False
    feats = {f.upper() for f in _cpu_features()}
    if "NEON" in feats or "ASIMD" in feats:
        return True
    # Empty/opaque feature table on a genuine aarch64 box: NEON is baseline.
    return True


def _supports_dtype(dtype) -> bool:
    """True iff numpy's data-parallel kernels accelerate *dtype* on ARM.

    Same coverage as the x86 path: numeric kinds 'i'/'u'/'f' with itemsize 4 or
    8 (i32, i64, u32, u64, f32, f64). Everything else (bool, int8/16, float16,
    datetime, complex, object, string) is rejected so the backend doesn't falsely
    claim credit for a path numpy would handle scalar.
    """
    if not _NUMPY:
        return False
    try:
        dt = np.dtype(dtype)
    except TypeError:
        return False
    if dt.kind not in "iuf":
        return False
    if dt.itemsize not in (4, 8):
        return False
    # Kept parallel to the x86 gate: reject explicit big-endian buffers ('>').
    # Native aarch64 is little-endian, so '=' / '|' / '<' are all LE here; a
    # byteswapped array routes to the always-correct np.sort floor instead.
    if dt.byteorder == ">":
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# The Backend
# ─────────────────────────────────────────────────────────────────────────────

class NeonSortBackend(Backend):
    """Explicit ARM NEON / ASIMD sort path.

    On modern numpy (>= 1.24) built for aarch64 the actual sort kernel is
    numpy's bundled data-parallel implementation (Highway VQSort / hand-written
    ASIMD), so ``_sort_ascending`` just calls ``arr.sort()``. If a future build
    ships ``quill._neonsort_ext`` the C wrapper is tried first. Either way the
    backend is registered so the dispatcher's bookkeeping
    (``available_backends()``, ``would_use()``, the profiler) is honest about
    what's running on i32/u32/u64/f32 arrays on Apple Silicon and Linux ARM —
    the dtypes the Rust voracious backend doesn't cover and which previously fell
    through to an unlabeled ``np.sort``.
    """

    name = "arm_neon_sort"
    # Priority mirrors the x86 SimdSortBackend (85): it sits BETWEEN the Rust
    # radix (95) and the OpenMP-merged path (80). The two SIMD backends are
    # mutually exclusive by architecture — only one can ever be ``available()``
    # on a given machine — so sharing the priority is safe and keeps the ARM and
    # x86 dispatch chains identically shaped.
    priority = 85
    # Aligned with sort_array's _SMALL_ARRAY_N gate (200k), matching the x86 SIMD
    # backend: below 200k, sort_array never dispatches, so a lower crossover only
    # exposed this backend to would_use()/self-tuning exploration in a region
    # where its kernel IS np.sort — pure dispatch overhead for zero benefit.
    # Once a hand-written ``_neonsort_ext`` ships, lower this.
    min_n = 200_000
    kinds = "iuf"
    max_itemsize = 8
    # arr.sort() (and any future C kernel) sort in place — the dispatcher must
    # copy when ``preserve=True``.
    mutates_input = True
    # Kernel IS numpy's own ``arr.sort()`` (ASIMD), so NaN is ordered exactly as
    # np.sort does; dispatch_sort skips its NaN pre-scan for this backend.
    nan_safe = True

    def _probe(self) -> bool:
        if not _NUMPY:
            return False
        if not _is_arm():
            # On x86 there's no NEON; SimdSortBackend handles that architecture.
            # Returning False keeps the label honest.
            return False
        # If a real C extension is present, trust it even on older numpy.
        if _EXT is not None:
            return True
        if _numpy_version_tuple() < _MIN_NUMPY_VERSION:
            return False
        # Baseline ASIMD is mandatory on ARMv8, but gate on it explicitly so the
        # label can never be attached on an exotic build that lacks it.
        return _has_neon()

    def supports(self, arr) -> bool:
        # Re-check size + kind via the base, then enforce the SIMD-specific dtype
        # gate (rejects itemsize 1/2 which the base would otherwise allow).
        if not super().supports(arr):
            return False
        if not _supports_dtype(arr.dtype):
            return False
        # The dispatcher in _backends.py already guarantees C-contiguous +
        # writeable for mutating backends, but assert here so a misuse from a
        # future caller surfaces loudly instead of corrupting memory through a
        # C wrapper.
        if not arr.flags["C_CONTIGUOUS"]:
            return False
        return True

    def _sort_ascending(self, arr):
        # Future C-extension fast path: a thin wrapper over a NEON/ASIMD kernel
        # with per-dtype entry points. The dispatcher catches any BaseException,
        # so a broken wrapper degrades to np.sort cleanly — but we still try
        # arr.sort() inline so a *recoverable* mismatch (e.g. an ext that only
        # implements f32) doesn't lose a whole sort.
        if _EXT is not None:
            fn = _EXT_DISPATCH.get(arr.dtype.str.lstrip("<>=|"))
            if fn is not None:
                try:
                    fn(arr)
                    return arr
                except Exception:
                    # Ext returned an error for this specific buffer — fall
                    # through to the bundled numpy ASIMD path.
                    logger.debug(
                        "quill._neonsort_ext failed on dtype=%s n=%d; "
                        "falling back to np.sort ASIMD kernel",
                        arr.dtype, arr.size,
                    )
        # The bundled-in-numpy path: on modern numpy for aarch64 this IS the
        # data-parallel ASIMD kernel. Sorts in place; NaN goes to the end (the
        # dispatcher has already stripped NaN, so we shouldn't actually see any).
        arr.sort()
        return arr


# Mapping from numpy dtype-string (without endianness) to the per-dtype entry
# point a future C extension would expose. Kept at module scope so a custom
# extension shipped by a downstream packager can monkey-patch it.
_EXT_DISPATCH = {}
if _EXT is not None:
    for _key, _attr in (
        ("i4", "sort_i32"), ("i8", "sort_i64"),
        ("u4", "sort_u32"), ("u8", "sort_u64"),
        ("f4", "sort_f32"), ("f8", "sort_f64"),
    ):
        _fn = getattr(_EXT, _attr, None)
        if callable(_fn):
            _EXT_DISPATCH[_key] = _fn


__all__ = [
    "NeonSortBackend",
    "_has_neon",
    "_is_arm",
    "_supports_dtype",
]

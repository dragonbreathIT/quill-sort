"""
quill/_simdsort.py
------------------
x86 SIMD-sort backend — an honest, documented fast path for the dtypes the
Rust voracious extension does not cover.

Why this module exists
======================
Modern numpy (>= 1.24, on x86-64 with AVX2/AVX-512) already dispatches the
inner loop of ``np.sort`` to Intel's ``x86-simd-sort`` for the natively-vectorized
dtypes: int32/int64, uint32/uint64, float32/float64. So in many cases the
"SIMD backend" really IS ``np.sort`` — there is no separate library call to
make. The win that *this* module provides is twofold:

  1. **Gap filler.** v6's RustBackend only handles int64/float64. For i32/u32/
     u64/f32, the dispatcher otherwise falls through to ``np.sort`` UNLABELED,
     so ``available_backends()`` and ``would_use()`` underreport what is doing
     the work. Registering an explicit ``SimdSortBackend`` makes the dispatch
     decision honest and lets the self-tuning profiler in ``_profile.py``
     track this path's latency separately.

  2. **Future C-wrapper seam.** When a future build of quill ships its own
     thin C wrapper around ``x86-simd-sort`` (e.g. ``quill._simdsort_ext``,
     compiled with ``-mavx2`` / ``-mavx512f``), the ``_sort_ascending`` method
     can try the C extension first and fall back to ``arr.sort()`` without any
     other code in the package changing. The module-load try-import below is
     the seam.

Never-lose guarantees (per project invariants)
==============================================
* numpy missing  → ``_probe`` returns False → backend unavailable.
* numpy too old  → ``_probe`` returns False (no AVX kernel inside np.sort).
* CPU lacks AVX2 → ``_probe`` returns False (a SIMD label would be a lie).
* big-endian array → ``supports`` returns False (the AVX kernel is LE-only).
* bool / datetime / object dtype → ``supports`` returns False.
* any kernel error → re-raised; the dispatcher in ``_backends.py`` catches
  ``BaseException`` and re-routes to ``np.sort``.
"""

from __future__ import annotations

import logging
import platform
from typing import Optional

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
# When/if a future build ships ``quill._simdsort_ext`` (a thin pybind11 / CFFI
# wrapper over Intel's x86-simd-sort, compiled with -mavx2 / -mavx512f), this
# try-import discovers it at module load. The wrapper is expected to expose
# per-dtype in-place sorters following the same convention as ``_fastsort``:
#     sort_i32(arr), sort_i64(arr), sort_u32(arr), sort_u64(arr),
#     sort_f32(arr), sort_f64(arr).
# Absence is the normal case — the backend then falls back to arr.sort(), which
# on modern numpy IS the AVX kernel anyway.
# ─────────────────────────────────────────────────────────────────────────────
_EXT = None
try:
    from . import _simdsort_ext as _EXT  # type: ignore[no-redef]
except Exception:
    _EXT = None


# ─────────────────────────────────────────────────────────────────────────────
# CPU + library capability probes
# ─────────────────────────────────────────────────────────────────────────────

# Numpy first shipped the bundled x86-simd-sort kernels in 1.24 (int/uint/
# float32/64 paths landed across 1.24-1.25). Below that, np.sort is the older
# scalar introsort and labeling it "x86_simd_sort" would be dishonest.
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


def _is_x86() -> bool:
    """True iff we're on a 64-bit x86 machine where x86-simd-sort is relevant."""
    m = platform.machine().lower()
    return m in ("x86_64", "amd64", "i686", "i386")


def _cpu_features() -> set:
    """Return the set of x86 CPU feature flags numpy reports as enabled.

    We use numpy's own introspection (``np.show_config``-derived dict if
    available, else the public ``__cpu_features__`` mapping) so the answer
    matches what numpy's dispatcher actually sees. Returns an empty set if
    numpy is missing or the introspection API isn't present on this build.
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


def _has_avx2() -> bool:
    """True iff this CPU/numpy combo has AVX2 enabled."""
    if not _is_x86():
        return False
    return "AVX2" in _cpu_features()


def _has_avx512() -> bool:
    """True iff AVX-512F (the foundation feature x86-simd-sort uses) is on."""
    if not _is_x86():
        return False
    return "AVX512F" in _cpu_features()


def _supports_dtype(dtype) -> bool:
    """True iff the AVX x86-simd-sort kernels accelerate *dtype*.

    The kernels cover i32, i64, u32, u64, f32, f64 — that is, numeric kinds
    'i'/'u'/'f' with itemsize 4 or 8. Everything else (bool, int8/16, float16,
    datetime, complex, object, string) is rejected so the backend doesn't
    falsely claim credit for a path numpy would handle scalar.
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
    # x86-simd-sort kernels are little-endian. '|' (not-applicable) and '=' are
    # native — on x86 that's LE. '<' is explicit LE. '>' is big-endian: reject.
    if dt.byteorder == ">":
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# The Backend
# ─────────────────────────────────────────────────────────────────────────────

class SimdSortBackend(Backend):
    """Explicit x86 SIMD sort path.

    On modern numpy (>= 1.24) the actual sort kernel is the AVX2/AVX-512
    implementation of ``x86-simd-sort`` that numpy bundles internally — so
    ``_sort_ascending`` just calls ``arr.sort()``. If a future build ships
    ``quill._simdsort_ext`` the C wrapper is called instead. Either way, the
    backend is registered so the dispatcher's bookkeeping
    (``available_backends()``, ``would_use()``, the profiler) is honest about
    what's running on i32/u32/u64/f32 arrays — the dtypes the Rust backend
    doesn't cover.
    """

    name = "x86_simd_sort"
    # Priority sits BETWEEN Rust (95) and OpenMP (80): Rust still wins on the
    # dtypes it supports (i64/f64), but for everything else the AVX kernel
    # should be tried before the OpenMP-merged path.
    priority = 85
    # Aligned with sort_array's _SMALL_ARRAY_N gate (200k): below that, sort_array
    # goes straight to np.sort and never dispatches, so a lower crossover only
    # exposed this backend to would_use()/self-tuning exploration in a region
    # where its kernel IS np.sort (no compiled _ext yet) — pure dispatch overhead
    # for zero benefit. Once a hand-written ``_simdsort_ext`` ships, lower this.
    min_n = 200_000
    kinds = "iuf"
    max_itemsize = 8
    # arr.sort() (and any future C kernel) sort in place — the dispatcher must
    # copy when ``preserve=True``.
    mutates_input = True
    # The kernel IS numpy's own ``arr.sort()``, so NaN lands exactly where
    # np.sort puts it (the end, ascending). The dispatcher can therefore skip its
    # O(n) NaN pre-scan for this backend and let the kernel handle NaN natively.
    nan_safe = True

    def _probe(self) -> bool:
        if not _NUMPY:
            return False
        if not _is_x86():
            # On ARM (Apple Silicon, Graviton) there's no x86-simd-sort path.
            # Returning False here keeps the label honest; an ARM-specific
            # backend would be a separate module.
            return False
        # If a real C extension is present, trust it even on older numpy.
        if _EXT is not None:
            return True
        if _numpy_version_tuple() < _MIN_NUMPY_VERSION:
            return False
        # Need at least AVX2 for the kernel to be meaningfully faster than
        # the scalar fallback. Without AVX2, labeling np.sort as "x86_simd_sort"
        # would mislead the profiler.
        return _has_avx2()

    def supports(self, arr) -> bool:
        # Re-check size + kind via the base, then enforce the SIMD-specific
        # dtype gate (rejects itemsize 1/2 which the base would otherwise allow).
        if not super().supports(arr):
            return False
        if not _supports_dtype(arr.dtype):
            return False
        # The dispatcher in _backends.py already guarantees C-contiguous + writeable
        # for mutating backends, but assert here so a misuse from a future caller
        # surfaces loudly instead of corrupting memory through the C wrapper.
        if not arr.flags["C_CONTIGUOUS"]:
            return False
        return True

    def _sort_ascending(self, arr):
        # Future C-extension fast path: a thin wrapper over x86-simd-sort with
        # per-dtype entry points. The dispatcher catches any BaseException, so a
        # broken wrapper degrades to np.sort cleanly — but we still try
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
                    # through to the bundled numpy AVX path.
                    logger.debug(
                        "quill._simdsort_ext failed on dtype=%s n=%d; "
                        "falling back to np.sort AVX kernel",
                        arr.dtype, arr.size,
                    )
        # The bundled-in-numpy path: this IS the AVX2/AVX-512 kernel on
        # modern numpy. Sorts in place; NaN goes to the end (the dispatcher
        # has already stripped NaN, so we shouldn't actually see any).
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
    "SimdSortBackend",
    "_has_avx2",
    "_has_avx512",
    "_supports_dtype",
]

"""
quill/_backends.py
------------------
Pluggable fast-sort backends + an auto-detecting dispatcher with a strict
**never-lose** guarantee.

Why this exists
===============
numpy's ``np.sort`` is a single-threaded AVX introsort/radix that runs near
memory bandwidth. To beat it "by a mile" you must leave pure Python: a compiled
parallel radix (Rust/voracious or OpenMP-threaded SIMD), a GPU, or a fast
parallel-sort library (polars). Each is optional and may be absent, so the
dispatcher probes what is installed and *always* falls back to numpy/Timsort.

The dispatch chain (highest priority first) for an eligible numeric ndarray:

    CuPy (GPU, large arrays only)  ──┐
    Rust voracious radix           ──┤  first available + supporting wins
    OpenMP-threaded SIMD (C ext)   ──┤
    polars delegation              ──┤
    numpy parallel-partition       ──┘
                  │ any error / not eligible
                  ▼
    np.sort  (always correct, the floor)

Correctness rules enforced *here* so every backend only has to sort ascending
values with no NaN:
  * eligibility gate: C-contiguous numeric ndarray, dtype kind in i/u/f and
    itemsize <= 8, value-only (no key), n >= the backend's crossover.
  * NaN is stripped before the kernel and re-appended at the end (numpy
    convention), so a backend that panics on NaN (voracious) never sees one.
  * descending is a post-sort reverse.
  * any backend exception is caught and re-dispatched to np.sort — the process
    always survives and the result is always correct.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import List, Optional

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False

from ._config import load_config

_ELIGIBLE_KINDS = "iuf"

# Name of the backend used by the most recent dispatch_sort call (for tests and
# introspection). "counting"/"numpy" denote the inline counting sort / np.sort
# fallback rather than a registered Backend.
_LAST_BACKEND: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Bundled native CPU backends (quill._native)
# ─────────────────────────────────────────────────────────────────────────────
# The former quill-fastsort* companion packages (parallel radix, samplesort,
# serial radix, NUMA) shared one header-only kernel, so they are folded into a
# single in-tree extension, quill._native — shipped inside quill-sort's binary
# wheels so the fast path needs no extra installs. Each compiled backend below
# prefers quill._native and falls back to its standalone companion package (kept
# working for existing installs). ``_NativeShim`` re-exposes the native
# functions under the names each backend's ``_sort_ascending`` already calls, so
# those methods stay unchanged.
_NATIVE = "unset"  # sentinel: not yet probed


def _load_native():
    """Return the bundled ``quill._native`` module, or None if it wasn't built."""
    global _NATIVE
    if _NATIVE == "unset":
        try:
            from . import _native as m
            _NATIVE = m
        except Exception:
            _NATIVE = None
    return _NATIVE


class _NativeShim:
    """Adapter presenting a companion package's function names, backed by the
    bundled ``quill._native``. ``name_map`` is {external_name: native_name}."""

    def __init__(self, native, name_map):
        for external_name, native_name in name_map.items():
            fn = getattr(native, native_name, None)
            if fn is None:
                raise AttributeError(f"quill._native missing {native_name}")
            setattr(self, external_name, fn)


# ─────────────────────────────────────────────────────────────────────────────
# Backend base class
# ─────────────────────────────────────────────────────────────────────────────

class Backend:
    """A fast-sort backend. Subclasses set ``name``/``priority`` and implement
    ``_probe`` (is it importable/usable on this machine?) and ``_sort_ascending``
    (sort a clean, NaN-free, contiguous numeric ndarray ascending, in place or
    returning a new array)."""

    name: str = "base"
    priority: int = 0          # higher = preferred
    min_n: int = 1             # crossover: below this, not worth it
    kinds: str = "iuf"         # dtype kinds this backend handles
    max_itemsize: int = 8
    # Does _sort_ascending mutate its input buffer in place? Backends that build
    # a fresh result (GPU copy-to-device, polars, counting, np.sort) set False;
    # in-place radix/partition backends keep True. Used by dispatch_sort to copy
    # the caller's array only when a mutating backend will actually run.
    mutates_input: bool = True
    # Does this backend place NaN exactly like np.sort (NaN to the end,
    # ascending) with no crash? Only backends whose kernel IS numpy's own sort
    # (numpy, x86_simd_sort, arm_neon_sort — all ``arr.sort()``) qualify. For
    # those, dispatch_sort SKIPS the O(n) NaN pre-scan entirely and lets the
    # kernel order NaN natively — a measured ~11% win on float sorts whose best
    # backend is just numpy's kernel. Radix/parallel backends (voracious, ips4o,
    # polars, …) stay False: they panic on or misorder NaN, so the dispatcher
    # must strip NaN before handing them the data. Default False = always strip
    # (the safe, pre-existing behavior).
    nan_safe: bool = False

    def __init__(self) -> None:
        self._available: Optional[bool] = None

    # availability is probed once and cached
    def available(self) -> bool:
        if self._available is None:
            try:
                self._available = bool(self._probe())
            except Exception:
                self._available = False
        return self._available

    def _probe(self) -> bool:
        raise NotImplementedError

    def supports(self, arr) -> bool:
        return (arr.dtype.kind in self.kinds
                and arr.dtype.itemsize <= self.max_itemsize
                and arr.size >= self.min_n)

    def _sort_ascending(self, arr):
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# numpy parallel-partition backend  (zero-dependency tier)
# ─────────────────────────────────────────────────────────────────────────────

class NumpyParallelBackend(Backend):
    """Thread-parallel np.partition sample-sort. Pure numpy + threads, no new
    deps. Measured ~1.5x int64 / ~1.2x float64 on many-core boxes with a SMALL
    worker count — the gain saturates at memory bandwidth around 2-3 workers, so
    we deliberately do NOT use all cores (P=28 was a measured 0.79x regression).
    """

    name = "numpy_parallel"
    priority = 50
    min_n = 5_000_000
    # Integer only: the partition sort gives a small but reliable win on int64
    # (~1.1x at the capped worker count) yet *regresses* on float64 (~0.8x), so
    # floats stay on the np.sort floor. Never-lose by construction.
    kinds = "iu"

    def _probe(self) -> bool:
        if not _NUMPY:
            return False
        cfg = load_config()
        return (os.cpu_count() or 1) >= cfg.get("parallel_min_cores", 8)

    def _sort_ascending(self, arr):
        from ._parallel import parallel_sort_array
        return parallel_sort_array(arr)


# ─────────────────────────────────────────────────────────────────────────────
# numpy baseline backend  (the np.sort floor, as a first-class tuning candidate)
# ─────────────────────────────────────────────────────────────────────────────

class NumpySortBackend(Backend):
    """``np.sort`` itself, registered as a *selectable* backend.

    np.sort is always the correctness fallback (the terminal ``if sorted_arr is
    None`` below). But it must ALSO be a candidate the self-tuning dispatcher can
    actively *measure and prefer* — otherwise on hardware where none of the
    compiled/parallel backends beat numpy's single-threaded AVX introsort (1–2
    vCPU cloud boxes, cheap containers, low-core laptops), explore-then-exploit
    optimises over a candidate set whose members ALL lose to the floor. It then
    round-robins losers and converges (if at all) to the least-bad one — leaving
    the array sort 20–80% slower than a bare np.sort and breaking the
    "never meaningfully slower" guarantee on a large slice of real deployments.

    Making the floor a candidate closes the loop: the tuner records np.sort's
    latency like any other backend, so on a many-core box it still converges to
    the parallel radix (the floor loses there) and on a single-core box it
    converges to ``numpy`` (the floor wins there). Same explore-then-exploit
    mechanism — the baseline is simply allowed on the ballot.

    Priority is the lowest of any backend so the STATIC order (used when tuning
    is disabled) is unchanged: the a-priori guess still favours the accelerators;
    only measured latency promotes the floor.
    """

    name = "numpy"
    priority = 10          # below every real accelerator — the a-priori floor
    min_n = 1
    kinds = "iuf"
    max_itemsize = 8
    mutates_input = False  # np.sort returns a fresh array; never touches input
    nan_safe = True        # IS np.sort — orders NaN to the end natively

    def _probe(self) -> bool:
        return _NUMPY

    def _sort_ascending(self, arr):
        return np.sort(arr)


# ─────────────────────────────────────────────────────────────────────────────
# Rust parallel MSD radix backend  (compiled, tier-A "by a mile" path)
# ─────────────────────────────────────────────────────────────────────────────

class RustParallelRadixBackend(Backend):
    """Quill's parallel MSD radix sort (Rust, rayon). Multi-threaded across
    all cores. Wins by 3-9x vs np.sort at n >= 10M.

    The crossover was lowered from 1M to 200k once the WF1/WF2 kernel
    parallelized the transform/scatter stages — the wider pipeline keeps all
    cores fed at much smaller sizes. Set ``QUILL_FORCE_PARALLEL_RADIX=1`` to
    engage the backend regardless of array size (useful for forcing a measured
    comparison or working around a too-conservative crossover on novel
    hardware)."""

    name = "rust_parallel_radix"
    priority = 99           # highest — beats voracious because it parallelizes the whole pipeline
    # Raised from 200K (7.0.5) → 3M (7.0.7) because the rayon thread spawn +
    # parallel min/max probe + transform + scatter has ~5-8 ms of fixed
    # overhead that doesn't amortize until n is large enough for the
    # parallel speedup to dominate. Measured on a 24-core box:
    #   n=1M:   quill 15.4 ms vs np.sort 10.6 ms → LOSING by 5 ms
    #   n=10M:  quill 76 ms vs np.sort 122 ms    → 1.61× win
    # Below 3M, dispatch falls through to rust_voracious (single-threaded
    # radix at priority 95, min_n 1M), which beats np.sort cleanly in the
    # 1M–3M range without the parallel-spawn tax. 100K and smaller go to
    # x86_simd_sort or np.sort directly.
    min_n = 3_000_000
    kinds = "iuf"
    max_itemsize = 8
    mutates_input = True

    # Supported dtypes shared between supports() and _describe(); kept as a
    # class attribute so diagnostic callers don't have to duplicate the list.
    _SUPPORTED_DTYPES = (
        "int64", "uint64", "int32", "uint32", "float64", "float32",
    )

    def __init__(self) -> None:
        super().__init__()
        self._mod = None

    def _probe(self) -> bool:
        # Prefer the bundled quill._native (the fold-in); fall back to the
        # standalone quill_fastsort_parallel companion for existing installs.
        nat = _load_native()
        if nat is not None:
            try:
                self._mod = _NativeShim(nat, {
                    "parallel_sort_i64": "radix_i64", "parallel_sort_u64": "radix_u64",
                    "parallel_sort_i32": "radix_i32", "parallel_sort_u32": "radix_u32",
                    "parallel_sort_f64": "radix_f64", "parallel_sort_f32": "radix_f32",
                })
                return True
            except Exception:
                pass
        try:
            import quill_fastsort_parallel as m
            self._mod = m
            return True
        except ImportError:
            pass
        try:
            from . import _fastsort_parallel
            self._mod = _fastsort_parallel
            return True
        except ImportError:
            pass
        try:
            import quill._fastsort_parallel as m
            self._mod = m
            return True
        except ImportError:
            return False

    def supports(self, arr) -> bool:
        # QUILL_FORCE_PARALLEL_RADIX=1 lets benchmarks engage this backend at
        # any size; the dtype gate still applies because the kernel only
        # implements the dtypes listed in _SUPPORTED_DTYPES.
        force = bool(os.environ.get("QUILL_FORCE_PARALLEL_RADIX"))
        if not force and arr.size < self.min_n:
            return False
        # The crate exposes specific dtypes:
        return arr.dtype in (np.int64, np.uint64, np.int32, np.uint32,
                             np.float64, np.float32)

    def _describe(self) -> dict:
        """Diagnostic dtype-support map, e.g. {"int64": True, "float16": False}.

        Used by available_backends() callers (and tests) to introspect which
        numeric dtypes this kernel actually accelerates without having to
        construct probe arrays for each."""
        all_kinds = (
            "int8", "int16", "int32", "int64",
            "uint8", "uint16", "uint32", "uint64",
            "float16", "float32", "float64",
        )
        return {k: (k in self._SUPPORTED_DTYPES) for k in all_kinds}

    def _sort_ascending(self, arr):
        m = self._mod
        debug = bool(os.environ.get("QUILL_BACKEND_DEBUG"))
        if arr.dtype == np.int64:
            kernel = "parallel_sort_i64"
            if debug:
                print(f"[quill] rust_parallel_radix → {kernel} (n={arr.size})",
                      file=sys.stderr)
            m.parallel_sort_i64(arr)
        elif arr.dtype == np.uint64:
            kernel = "parallel_sort_u64"
            if debug:
                print(f"[quill] rust_parallel_radix → {kernel} (n={arr.size})",
                      file=sys.stderr)
            m.parallel_sort_u64(arr)
        elif arr.dtype == np.int32:
            kernel = "parallel_sort_i32"
            if debug:
                print(f"[quill] rust_parallel_radix → {kernel} (n={arr.size})",
                      file=sys.stderr)
            m.parallel_sort_i32(arr)
        elif arr.dtype == np.uint32:
            kernel = "parallel_sort_u32"
            if debug:
                print(f"[quill] rust_parallel_radix → {kernel} (n={arr.size})",
                      file=sys.stderr)
            m.parallel_sort_u32(arr)
        elif arr.dtype == np.float64:
            kernel = "parallel_sort_f64"
            if debug:
                print(f"[quill] rust_parallel_radix → {kernel} (n={arr.size})",
                      file=sys.stderr)
            m.parallel_sort_f64(arr)
        elif arr.dtype == np.float32:
            kernel = "parallel_sort_f32"
            if debug:
                print(f"[quill] rust_parallel_radix → {kernel} (n={arr.size})",
                      file=sys.stderr)
            m.parallel_sort_f32(arr)
        return arr


# ─────────────────────────────────────────────────────────────────────────────
# Rust voracious radix backend  (compiled, secondary fast path)
# ─────────────────────────────────────────────────────────────────────────────

class RustBackend(Backend):
    """Multi-threaded radix sort via the compiled ``quill._fastsort`` extension
    (PyO3 + voracious_radix_sort). Measured array-in/out: int64 ~3-5x, float64
    ~1.3-2.9x vs np.sort. AOT native — no warmup. Panics on NaN, but the
    dispatcher strips NaN first and wraps the call, so a panic (converted to a
    catchable exception by panic='unwind') re-dispatches to np.sort."""

    name = "rust_voracious"
    priority = 95          # fastest measured for int64/float64 (beats GPU once
                           # PCIe transfer is counted), so it leads the chain
    min_n = 1_000_000
    kinds = "if"          # the extension exposes i64 and f64 kernels

    def __init__(self) -> None:
        super().__init__()
        self._mod = None

    def _probe(self) -> bool:
        # Two ways the compiled extension can be present:
        #  1. bundled into the quill package as quill._fastsort (dev / platform
        #     wheels that ship it inside quill/);
        #  2. installed as the standalone companion distribution `quill-fastsort`
        #     (top-level module `quill_fastsort`), which `pip install quill-sort
        #     quill-fastsort` provides on platforms with a prebuilt wheel.
        nat = _load_native()
        if nat is not None:
            try:
                self._mod = _NativeShim(nat, {"sort_i64": "radix_i64",
                                              "sort_f64": "radix_f64"})
                return True
            except Exception:
                pass
        try:
            from . import _fastsort
            self._mod = _fastsort
            return True
        except Exception:
            pass
        try:
            import quill_fastsort
            self._mod = quill_fastsort
            return True
        except Exception:
            return False

    def supports(self, arr) -> bool:
        if arr.size < self.min_n:
            return False
        # exact dtypes the extension provides
        return arr.dtype == np.int64 or arr.dtype == np.float64

    def _sort_ascending(self, arr):
        # extension sorts int64 / float64 contiguous arrays in place
        if arr.dtype == np.int64:
            self._mod.sort_i64(arr)
        else:
            self._mod.sort_f64(arr)
        return arr


# ─────────────────────────────────────────────────────────────────────────────
# OpenMP-threaded SIMD backend  (C extension, secondary)
# ─────────────────────────────────────────────────────────────────────────────

class OpenMPBackend(Backend):
    """Sorts array chunks with numpy's own AVX kernel across a thread pool
    (GIL released), then merges them with a parallel C k-way merge
    (``quill._kmerge``). Measured int64 ~2.5-4.5x. Floats route to fallback
    until the NaN-aware merge lands."""

    name = "openmp_simd"
    priority = 80
    min_n = 1_000_000
    kinds = "i"           # integer only for now (float merge pending)

    def __init__(self) -> None:
        super().__init__()
        self._mod = None

    def _probe(self) -> bool:
        try:
            from . import _kmerge
            self._mod = _kmerge
            return hasattr(self._mod, "pmerge_i64")
        except Exception:
            return False

    def supports(self, arr) -> bool:
        return arr.dtype == np.int64 and arr.size >= self.min_n

    def _sort_ascending(self, arr):
        import math
        from concurrent.futures import ThreadPoolExecutor
        n = arr.size
        cfg = load_config()
        workers = cfg.get("openmp_workers", 0) or min(14, os.cpu_count() or 4)
        workers = max(2, workers)
        step = math.ceil(n / workers)
        bounds = [(i * step, min((i + 1) * step, n)) for i in range(workers)
                  if i * step < n]

        def _sort_chunk(se):
            s, e = se
            arr[s:e].sort()

        with ThreadPoolExecutor(max_workers=len(bounds)) as pool:
            list(pool.map(_sort_chunk, bounds))
        # parallel C merge of the sorted runs
        offsets = np.array([b[0] for b in bounds] + [n], dtype=np.int64)
        out = np.empty_like(arr)
        self._mod.pmerge_i64(arr, offsets, out)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# polars delegation backend  (opt-in [fast] extra)
# ─────────────────────────────────────────────────────────────────────────────

class PolarsBackend(Backend):
    """Delegate to polars' multi-threaded Rust sort. Measured int64 ~3.5-3.8x,
    float64 ~1.8-2.9x. A no-compile fast path on platforms our wheels miss."""

    name = "polars"
    priority = 70
    min_n = 200_000
    mutates_input = False        # builds a fresh array via to_numpy().copy()
    # A no-compile, cross-platform fast path: useful when the compiled companion
    # wheels aren't installed (or no C++ toolchain is available), regardless of
    # platform.

    def __init__(self) -> None:
        super().__init__()
        self._pl = None

    def _probe(self) -> bool:
        try:
            import polars as pl
            self._pl = pl
            return True
        except Exception:
            return False

    def _sort_ascending(self, arr):
        pl = self._pl
        s = pl.Series(arr)
        out = s.sort().to_numpy()
        # to_numpy may return a read-only view into polars memory; copy so the
        # result is an owning, writable array (verified: ascontiguousarray is
        # NOT enough — it returns the same non-owning buffer).
        return np.array(out, copy=True)


# ─────────────────────────────────────────────────────────────────────────────
# CuPy GPU backend  (opt-in [gpu] extra)
# ─────────────────────────────────────────────────────────────────────────────

class CuPyBackend(Backend):
    """GPU radix sort via CuPy. For large host arrays the host->device->sort->
    host round trip still beats np.sort 3-10x (measured on an RTX 4060 Ti). Only
    engages for large arrays that fit in free VRAM; tiny arrays lose to PCIe
    transfer, so the crossover is high."""

    name = "cupy_gpu"
    priority = 90          # below the Rust radix (which wins once PCIe transfer
                           # is counted); still the best path for dtypes Rust
                           # doesn't cover (uint64/int32/float32) and when no
                           # compiled CPU wheel is installed
    min_n = 2_000_000
    mutates_input = False        # cp.asnumpy(...) returns a fresh host array

    def __init__(self) -> None:
        super().__init__()
        self._cp = None

    def _probe(self) -> bool:
        cfg = load_config()
        if not cfg.get("use_gpu", True):
            return False
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() < 1:
                return False
            # Prove a sort KERNEL can actually run — not just that a device
            # exists. CuPy can be importable with a live device yet unable to
            # JIT a kernel (missing CUDA toolkit headers), in which case the
            # first real sort would raise. Verifying here keeps
            # available_backends() honest ("usable", not merely "installed").
            cp.asarray([1, 0]).sort()
            self._cp = cp
            return True
        except Exception:
            return False

    def supports(self, arr) -> bool:
        if not super().supports(arr):
            return False
        # only if it comfortably fits in free VRAM (input + output + headroom)
        try:
            free, _total = self._cp.cuda.runtime.memGetInfo()
            need = arr.nbytes * 3
            return need < free
        except Exception:
            return False

    def _sort_ascending(self, arr):
        cp = self._cp
        # Pinned host memory makes the host->device DMA copy asynchronous and
        # bypasses the kernel's pageable-buffer staging. The win is only worth
        # it past ~50 MB; below that the allocator overhead dwarfs the savings.
        # The pool is created lazily and cached on the backend so repeated
        # large sorts share allocations. Any failure is best-effort.
        try:
            pinned_pool = getattr(self, "_pinned_pool", None)
            if pinned_pool is None and arr.nbytes >= 50_000_000:
                pinned_pool = cp.cuda.PinnedMemoryPool()
                cp.cuda.set_pinned_memory_allocator(pinned_pool.malloc)
                self._pinned_pool = pinned_pool
        except Exception:
            pass    # any pinned-memory failure is best-effort
        try:
            d = cp.asarray(arr)
            d.sort()
            out = cp.asnumpy(d)
            del d
            # release pooled VRAM so back-to-back large sorts don't fragment
            cp.get_default_memory_pool().free_all_blocks()
            return out
        except Exception:
            # OOM or driver hiccup → let dispatcher fall back
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass
            raise


# ─────────────────────────────────────────────────────────────────────────────
# ips4o companion backend  (compiled C++ samplesort, top-tier large-N path)
# ─────────────────────────────────────────────────────────────────────────────

class Ips4oBackend(Backend):
    """ips4o (DuckDB/Polars parallel samplesort) via the quill-fastsort-ips4o
    companion wheel. Beats every other CPU backend at n >= 10M for int/float
    (5-7x over np.sort measured). Top priority when available."""

    name = "ips4o"
    priority = 100
    min_n = 3_000_000     # measured: rust_voracious wins at 1M-3M, ips4o at 3M+
    kinds = "iuf"
    max_itemsize = 8
    mutates_input = True

    _SUPPORTED_DTYPES = ("int64", "uint64", "int32", "uint32",
                         "float64", "float32")

    def __init__(self) -> None:
        super().__init__()
        self._mod = None

    def _probe(self) -> bool:
        nat = _load_native()
        if nat is not None:
            try:
                self._mod = _NativeShim(nat, {
                    "sort_i64": "sample_i64", "sort_u64": "sample_u64",
                    "sort_i32": "sample_i32", "sort_u32": "sample_u32",
                    "sort_f64": "sample_f64", "sort_f32": "sample_f32",
                })
                return True
            except Exception:
                pass
        try:
            import quill_fastsort_ips4o as m
            self._mod = m
            return True
        except ImportError:
            return False

    def supports(self, arr) -> bool:
        if arr.size < self.min_n:
            return False
        return arr.dtype.name in self._SUPPORTED_DTYPES

    def _sort_ascending(self, arr):
        m = self._mod
        n = arr.dtype.name
        if n == "int64":
            m.sort_i64(arr)
        elif n == "uint64":
            m.sort_u64(arr)
        elif n == "int32":
            m.sort_i32(arr)
        elif n == "uint32":
            m.sort_u32(arr)
        elif n == "float64":
            m.sort_f64(arr)
        elif n == "float32":
            m.sort_f32(arr)
        return arr


# ─────────────────────────────────────────────────────────────────────────────
# NUMA-aware companion backend  (multi-socket NUMA radix, single-socket safe)
# ─────────────────────────────────────────────────────────────────────────────

class NumaBackend(Backend):
    """NUMA-aware parallel sort via the quill-fastsort-numa companion wheel.
    On multi-socket boxes it shards across NUMA nodes; on single-socket the C
    extension falls through to voracious internally, so the backend is always
    safe to engage above min_n."""

    name = "numa"
    priority = 94
    min_n = 1_000_000
    kinds = "iuf"
    max_itemsize = 8
    mutates_input = True

    _SUPPORTED_DTYPES = ("int64", "uint64", "int32", "uint32",
                         "float64", "float32")

    def __init__(self) -> None:
        super().__init__()
        self._mod = None

    def _probe(self) -> bool:
        # Only declare available on multi-socket boxes. On single-socket
        # systems the NUMA kernel falls through to a plain voracious sort
        # internally, which adds FFI overhead vs simply letting the next
        # backend in the chain handle it. Refusing here keeps single-socket
        # users from paying that overhead — measured regression in 7.1.0:
        # int32 1M on single-socket went to numa instead of polars/np.sort
        # and ran 0.59× of np.sort. (Fixed in 7.1.1.)
        nat = _load_native()
        if nat is not None and hasattr(nat, "numa_topology"):
            try:
                topo = nat.numa_topology()  # (nodes, cores) on multi-socket Linux, else None
                if isinstance(topo, tuple) and topo and topo[0] >= 2:
                    self._mod = _NativeShim(nat, {
                        "numa_sort_i64": "radix_i64", "numa_sort_u64": "radix_u64",
                        "numa_sort_i32": "radix_i32", "numa_sort_u32": "radix_u32",
                        "numa_sort_f64": "radix_f64", "numa_sort_f32": "radix_f32",
                    })
                    return True
                # topo None / single-socket -> fall through to external, else unavailable
            except Exception:
                pass
        try:
            import quill_fastsort_numa as m
            topo = m.detect_topology()  # Some((nodes, cores)) or None
            if topo is None:
                return False    # single-socket — let the next backend win
            node_count = topo[0] if isinstance(topo, tuple) else None
            if node_count is None or node_count < 2:
                return False
            self._mod = m
            return True
        except ImportError:
            return False
        except Exception:
            return False

    def supports(self, arr) -> bool:
        if arr.size < self.min_n:
            return False
        return arr.dtype.name in self._SUPPORTED_DTYPES

    def _sort_ascending(self, arr):
        m = self._mod
        n = arr.dtype.name
        if n == "int64":
            m.numa_sort_i64(arr)
        elif n == "uint64":
            m.numa_sort_u64(arr)
        elif n == "int32":
            m.numa_sort_i32(arr)
        elif n == "uint32":
            m.numa_sort_u32(arr)
        elif n == "float64":
            m.numa_sort_f64(arr)
        elif n == "float32":
            m.numa_sort_f32(arr)
        return arr


# ─────────────────────────────────────────────────────────────────────────────
# SIMD companion backend  (conservative — let self-tuning measure)
# ─────────────────────────────────────────────────────────────────────────────

class SimdCompanionBackend(Backend):
    """quill-fastsort-simd companion wheel. numpy 2.x already ships
    x86-simd-sort internally on AVX-512 boxes, so this companion is often a
    wash or a slight loss there. min_n is intentionally conservative; the
    self-tuning DB will discover where it actually wins on a given host."""

    name = "simd_companion"
    priority = 80
    min_n = 10_000_000
    kinds = "iuf"
    max_itemsize = 8
    mutates_input = True

    _SUPPORTED_DTYPES = ("int64", "uint64", "int32", "uint32",
                         "float64", "float32")

    def __init__(self) -> None:
        super().__init__()
        self._mod = None

    def _probe(self) -> bool:
        nat = _load_native()
        if nat is not None:
            try:
                self._mod = _NativeShim(nat, {
                    "sort_i64": "serial_i64", "sort_u64": "serial_u64",
                    "sort_i32": "serial_i32", "sort_u32": "serial_u32",
                    "sort_f64": "serial_f64", "sort_f32": "serial_f32",
                })
                return True
            except Exception:
                pass
        try:
            import quill_fastsort_simd as m
            self._mod = m
            return True
        except ImportError:
            return False

    def supports(self, arr) -> bool:
        if arr.size < self.min_n:
            return False
        return arr.dtype.name in self._SUPPORTED_DTYPES

    def _sort_ascending(self, arr):
        m = self._mod
        n = arr.dtype.name
        if n == "int64":
            m.sort_i64(arr)
        elif n == "uint64":
            m.sort_u64(arr)
        elif n == "int32":
            m.sort_i32(arr)
        elif n == "uint32":
            m.sort_u32(arr)
        elif n == "float64":
            m.sort_f64(arr)
        elif n == "float32":
            m.sort_f32(arr)
        return arr


# ─────────────────────────────────────────────────────────────────────────────
# Spectre backend  (bundled parallel integer radix — quill._spectre)
# ─────────────────────────────────────────────────────────────────────────────

class SpectreBackend(Backend):
    """Spectre: a bundled, portable, multi-threaded MSD→LSD radix sort for
    32/64-bit integers (``quill._spectre``, built from _native_src/spectre_sort.c).

    Integer-only by design (no float kernel). On the reference 24-core box it is
    the fastest CPU backend for int64/uint64 at every measured size — ~1.3–1.9×
    over the next-best compiled backend (ips4o / rust_parallel_radix) and up to
    ~5× over np.sort — and usually wins for int32/uint32 too. It does NOT win
    everywhere (e.g. int32 around 5M, where the Rust radix is faster), which is
    exactly why it is a self-tuning *candidate* rather than a hard override: the
    dispatcher measures it per (dtype, size) bucket and keeps whichever backend
    actually wins there. Its top static priority only sets the a-priori guess used
    when tuning is disabled.

    Spectre supports n < 2^32; larger arrays return SPECTRE_TOO_BIG, so
    ``supports`` refuses them and the chain handles them with another backend.
    Any Spectre error is raised out of the C wrapper and caught by dispatch_sort's
    never-lose fallback to np.sort.
    """

    name = "spectre"
    priority = 101          # a-priori top integer candidate (see class docstring)
    # Spectre's serial radix beats np.sort from ~20-25k integer elements up (1.07x
    # at 25k rising to 2.7x by 500k), and its parallel path takes over past ~260k.
    # The crossover was lowered from 1M to 25k so sort_array uses Spectre — not
    # np.sort — across the whole mid range, turning former ties into wins. Below
    # ~20k the Python dispatch overhead (~25us) exceeds the kernel's edge, so
    # sort_array's small-array floor keeps those on np.sort (see quill/__init__.py).
    min_n = 25_000
    kinds = "iu"            # integers only — Spectre has no float kernel
    max_itemsize = 8
    mutates_input = True

    # Spectre's 2^32 element ceiling (SPECTRE_TOO_BIG above this).
    _MAX_N = 1 << 32
    _SUPPORTED_DTYPES = ("int64", "uint64", "int32", "uint32")

    def __init__(self) -> None:
        super().__init__()
        self._mod = None

    def _probe(self) -> bool:
        # QUILL_NO_SPECTRE=1 disables the backend without a rebuild (parity with
        # the other opt-out env switches; handy for A/B measurement).
        if os.environ.get("QUILL_NO_SPECTRE") == "1":
            return False
        try:
            from . import _spectre as m
        except Exception:
            return False
        # Require every kernel we dispatch to, so a partial/older build can't
        # half-register and then AttributeError inside _sort_ascending.
        if all(hasattr(m, fn) for fn in
               ("spectre_i64", "spectre_u64", "spectre_i32", "spectre_u32")):
            self._mod = m
            return True
        return False

    def supports(self, arr) -> bool:
        if arr.size < self.min_n or arr.size >= self._MAX_N:
            return False
        return arr.dtype.name in self._SUPPORTED_DTYPES

    def _sort_ascending(self, arr):
        m = self._mod
        n = arr.dtype.name
        debug = bool(os.environ.get("QUILL_BACKEND_DEBUG"))
        if debug:
            print(f"[quill] spectre → {n} (n={arr.size})", file=sys.stderr)
        if n == "int64":
            m.spectre_i64(arr)
        elif n == "uint64":
            m.spectre_u64(arr)
        elif n == "int32":
            m.spectre_i32(arr)
        elif n == "uint32":
            m.spectre_u32(arr)
        return arr


# ─────────────────────────────────────────────────────────────────────────────
# Registry + dispatcher
# ─────────────────────────────────────────────────────────────────────────────

_BACKENDS: List[Backend] = [
    SpectreBackend(),
    Ips4oBackend(),
    RustParallelRadixBackend(),
    RustBackend(),
    NumaBackend(),
    CuPyBackend(),
    OpenMPBackend(),
    SimdCompanionBackend(),
    PolarsBackend(),
    NumpyParallelBackend(),
    # The np.sort floor, LAST — lowest priority. Present so the self-tuning
    # dispatcher can measure it and converge to it on hardware where the
    # accelerators can't beat it (see NumpySortBackend).
    NumpySortBackend(),
]
_REGISTRY_LOCK = threading.Lock()

# The architecture-specific SIMD backends are registered LAZILY (not at module
# import) to avoid a circular-import hazard: _simdsort.py / _neonsort.py each do
# ``from ._backends import Backend``, so eager import here would silently drop the
# SIMD backend if any downstream caller imports quill._simdsort (or _neonsort)
# before quill._backends — the partial import would make the
# ``from ._simdsort import SimdSortBackend`` line raise ImportError, which our
# try/except would swallow. Registering on first dispatch / probe guarantees both
# modules are fully defined before instantiation.
#
# Both the x86 (``SimdSortBackend`` / "x86_simd_sort") and ARM
# (``NeonSortBackend`` / "arm_neon_sort") backends are registered here. They are
# mutually exclusive by architecture — each ``_probe`` gates on
# ``platform.machine()`` — so at most one is ever ``available()`` on a given host,
# and registering both is safe on any platform.
_SIMD_REGISTERED = False


def _ensure_simd_registered() -> None:
    global _SIMD_REGISTERED
    if _SIMD_REGISTERED:
        return
    _SIMD_REGISTERED = True
    changed = False
    # x86 SSE/AVX SIMD path (Intel/AMD).
    try:
        from ._simdsort import SimdSortBackend
        with _REGISTRY_LOCK:
            if not any(b.name == "x86_simd_sort" for b in _BACKENDS):
                _BACKENDS.append(SimdSortBackend())
                changed = True
    except Exception:
        pass
    # ARM NEON/ASIMD path (Apple Silicon, Linux aarch64 / Graviton / Ampere).
    try:
        from ._neonsort import NeonSortBackend
        with _REGISTRY_LOCK:
            if not any(b.name == "arm_neon_sort" for b in _BACKENDS):
                _BACKENDS.append(NeonSortBackend())
                changed = True
    except Exception:
        pass
    if changed:
        with _REGISTRY_LOCK:
            _BACKENDS.sort(key=lambda b: b.priority, reverse=True)


def register_backend(backend: Backend) -> None:
    """Add a custom backend. Higher ``priority`` is tried first."""
    with _REGISTRY_LOCK:
        _BACKENDS.append(backend)
        _BACKENDS.sort(key=lambda b: b.priority, reverse=True)


def available_backends() -> List[str]:
    """Names of backends usable on this machine, in priority order.

    The list may include an explicit SIMD path for the i32/u32/u64/f32 dtypes the
    Rust voracious backend doesn't cover — ``"x86_simd_sort"`` on Intel/AMD, or
    ``"arm_neon_sort"`` on Apple Silicon / Linux aarch64. On modern numpy that
    path delegates to numpy's own bundled data-parallel kernel (x86-simd-sort on
    x86, ASIMD/Highway on ARM); the entry exists so introspection and the
    self-tuning profiler can name it. The two are mutually exclusive by
    architecture, so at most one ever appears.

    The list also ends with ``"numpy"`` whenever numpy is importable: the
    ``np.sort`` floor is a first-class *tuning candidate* (see NumpySortBackend),
    so the self-tuning dispatcher can measure it and prefer it on hardware where
    the accelerators can't beat it. It sits last because its static priority is
    the lowest; only measured latency promotes it.
    """
    _ensure_simd_registered()
    with _REGISTRY_LOCK:
        ordered = sorted(_BACKENDS, key=lambda b: b.priority, reverse=True)
    return [b.name for b in ordered if b.available()]


def would_use(arr, _mn: Optional[int] = None, _mx: Optional[int] = None) -> str:
    """Name of the backend ``dispatch_sort`` WOULD use for *arr*, without
    sorting it (for analyze()/introspection). Mirrors the dispatch decision:
    counting for dense bounded int64/uint64, else the first available backend,
    else the np.sort floor.

    Optional ``_mn`` / ``_mx``: precomputed min/max (used by ``analyze()`` to
    avoid two full passes — analyze() already computed them. ~30 ms saved
    on 50M int64 in 7.1.2.)"""
    _ensure_simd_registered()
    if not (_NUMPY and eligible(arr)) or arr.size <= 1:
        return "numpy"
    if arr.dtype.kind == "f":
        m = np.isnan(arr)
        if m.any():
            arr = arr[~m]
            if arr.size <= 1:
                return "numpy"
    if arr.dtype.kind in "iu" and arr.dtype.itemsize >= 8:
        from ._strategies import _counting_is_worth_it
        if _mn is None or _mx is None:
            mn = int(arr.min()); mx = int(arr.max())
        else:
            mn, mx = int(_mn), int(_mx)
        if mx == mn or _counting_is_worth_it(arr.size, mx - mn):
            return "counting"
    # Consult self-tuning so the reported choice matches what dispatch_sort
    # would actually run on this machine — including the explore-then-exploit
    # phase, so a freshly installed backend that dispatch is currently probing
    # is reported honestly rather than hidden behind the old incumbent.
    try:
        from ._tuning import DB
        with _REGISTRY_LOCK:
            ordered = sorted(_BACKENDS, key=lambda b: b.priority, reverse=True)
            candidates = [b for b in ordered
                          if b.available() and b.supports(arr)]
        if candidates:
            best_name = DB.choose([b.name for b in candidates],
                                  arr.dtype.kind, arr.size)
            if best_name is not None:
                return best_name
    except Exception:
        pass
    b = _pick_backend(arr)
    return b.name if b is not None else "numpy"


def reset_availability() -> None:
    """Clear every backend's cached availability so the next ``available()``
    re-probes. Used after the setup wizard pip-installs a backend in the running
    process, so a freshly installed extension is picked up without a restart."""
    with _REGISTRY_LOCK:
        for b in _BACKENDS:
            b._available = None


def _pick_backend(arr) -> Optional[Backend]:
    _ensure_simd_registered()
    with _REGISTRY_LOCK:
        ordered = sorted(_BACKENDS, key=lambda b: b.priority, reverse=True)
    for b in ordered:
        try:
            if b.available() and b.supports(arr):
                return b
        except Exception:
            continue
    return None


def eligible(arr) -> bool:
    """Numeric, contiguous, NATIVE-byte-order, value-only ndarray that a fast
    backend may handle.

    Non-native (byteswapped, e.g. ``'>i8'`` on a little-endian host) buffers are
    excluded on purpose: the compiled radix kernels (spectre / ips4o / rust /
    the counting C-ext) read the raw bytes in native order, so a big-endian array
    would be silently sorted by byte-reversed values — a wrong result that never
    raises, so the never-lose wrapper wouldn't catch it. ``np.sort`` is
    byte-order-safe, so non-native arrays fall through to it (see ``dispatch_sort``
    and ``sort_array``)."""
    return (_NUMPY and isinstance(arr, np.ndarray)
            and arr.ndim == 1
            and arr.dtype.kind in _ELIGIBLE_KINDS
            and arr.dtype.itemsize <= 8
            and arr.dtype.isnative
            and arr.flags["C_CONTIGUOUS"])


# ─────────────────────────────────────────────────────────────────────────────
# On-device GPU sort. When a caller already holds a cupy.ndarray, sort_array
# routes here and sorts ON DEVICE with zero host<->device transfer (cupy's own
# sort). For data already on the GPU this is ~9-15x vs np.sort-on-host and ~3x
# vs the host round-trip the CPU CuPyBackend would otherwise pay.
# ─────────────────────────────────────────────────────────────────────────────

_DEVICE_KINDS = "iuf"


def _is_cupy_array(x) -> bool:
    """True iff *x* is a cupy.ndarray — WITHOUT importing cupy if it isn't
    already loaded (so ``import quill`` stays cupy-free on boxes that have cupy
    installed but aren't using it)."""
    cp = sys.modules.get("cupy")
    if cp is None:
        return False
    try:
        return isinstance(x, cp.ndarray)
    except Exception:
        return False


def _device_eligible(d) -> bool:
    """eligible() for a device array: 1-D, numeric kind, itemsize<=8, contiguous."""
    return (d.ndim == 1 and d.dtype.kind in _DEVICE_KINDS
            and d.dtype.itemsize <= 8 and d.flags.c_contiguous)


def sort_cupy_device(d, descending: bool = False, inplace: bool = False):
    """Sort a cupy.ndarray ON DEVICE and return a sorted cupy.ndarray (zero host
    transfer). Matches np.sort exactly, including negatives and NaN-to-end
    (NaN-to-start when descending).

    Never-lose for *device* arrays: the path IS cupy's own radix/merge sort (no
    riskier kernel to retreat from), so on any error (OOM, driver) this RAISES a
    clean exception rather than silently transferring to the host or returning a
    host array for a device input. The result never leaves the GPU.
    """
    import cupy as cp
    work = d if inplace else d.copy()
    if not _device_eligible(work):
        out = cp.sort(work, axis=-1)
        return out[..., ::-1] if descending else out
    work.sort()  # in-place; NaN to end (matches np.sort); value-only => stable-equiv
    if descending:
        work = cp.ascontiguousarray(work[::-1])  # NaN moves to start; re-materialize
    return work


def _has_any_nan(arr) -> bool:
    """Chunked early-exit NaN probe.

    ``np.isnan(arr).any()`` doesn't short-circuit — it materialises the full
    boolean mask first. The usual case (no NaN) returns False after a single
    4K-element chunk, saving the mask allocation and a full pass. We use this
    in dispatch_sort when the caller hasn't given a definite ``nan_hint``.
    """
    for i in range(0, arr.size, 4096):
        if np.isnan(arr[i:i + 4096]).any():
            return True
    return False


def _choose_backend(work, force: Optional[str]):
    """Pick the backend ``dispatch_sort`` will run for *work* (or None → np.sort
    floor). The choice is VALUE-INDEPENDENT — it keys only off dtype/size — so it
    can be made BEFORE NaN stripping, letting the dispatcher decide whether the
    NaN pre-scan is needed at all. Honors ``force`` (exact backend name), else the
    self-tuning explore-then-exploit pick, degrading to the static priority order
    if the tuning DB is unavailable."""
    if force is not None:
        with _REGISTRY_LOCK:
            for b in _BACKENDS:
                if b.name == force and b.available() and b.supports(work):
                    return b
        return None
    try:
        from ._tuning import DB
        with _REGISTRY_LOCK:
            ordered = sorted(_BACKENDS, key=lambda b: b.priority, reverse=True)
        candidates = [b for b in ordered if b.available() and b.supports(work)]
        best_name = DB.choose([b.name for b in candidates],
                              work.dtype.kind, work.size)
        if best_name is not None:
            # measured winner first, original priority for the rest
            chosen = ([b for b in candidates if b.name == best_name]
                      + [b for b in candidates if b.name != best_name])
            return chosen[0] if chosen else None
        return candidates[0] if candidates else None
    except Exception:
        # Tuning lookup must NEVER block dispatch — degrade to static priority.
        return _pick_backend(work)


def dispatch_sort(arr, descending: bool = False, force: Optional[str] = None,
                  preserve: bool = False,
                  nan_hint: Optional[bool] = None):
    """
    Sort a 1-D numeric ndarray ascending (or descending) using the best
    available backend, with NaN handled and a guaranteed np.sort fallback.
    Returns a sorted ndarray.

    ``force`` selects a backend by name (for testing/benchmarking); if it is
    unavailable or errors, the normal fallback still applies.

    ``preserve``: when True, *arr* is guaranteed not to be mutated — a private
    copy is made ONLY if the chosen backend sorts in place (Rust/partition).
    The counting and np.sort paths already return fresh arrays, so this avoids
    the double allocation that previously made small inplace=False sorts slower
    than a bare np.sort.

    ``nan_hint``: caller-supplied knowledge about NaN presence to skip the scan.
      * ``False`` — caller guarantees no NaN; we skip the isnan probe entirely.
      * ``True``  — caller knows NaNs exist; we go straight to mask construction.
      * ``None``  — unknown; we run the chunked early-exit probe (cheap on the
        usual NaN-free case) before paying for a full mask.
    """
    _ensure_simd_registered()
    global _LAST_BACKEND
    debug = bool(os.environ.get("QUILL_BACKEND_DEBUG"))
    is_float = arr.dtype.kind == "f"
    # Non-native byte order guard. The compiled radix kernels read raw bytes in
    # native order, so a byteswapped ('>i8', '<f8' on a big-endian host, …) buffer
    # would be silently sorted by byte-reversed values — wrong, with no exception,
    # so the never-lose fallback never fires. np.sort orders by logical value
    # regardless of byte order. eligible() already routes the public sort_array
    # path here-avoiding; this also covers force= and any direct dispatch_sort
    # caller. (Rare path — non-native arrays are uncommon — so the extra np.sort
    # cost is irrelevant.)
    if not arr.dtype.isnative:
        _LAST_BACKEND = "numpy"
        out = np.sort(arr)
        return np.ascontiguousarray(out[::-1]) if descending else out
    # NaN is stripped LAZILY, and only if the chosen backend can't order it like
    # np.sort (see the deferred block after backend selection). numpy /
    # x86_simd_sort / arm_neon_sort handle NaN natively, so most float sorts skip
    # the O(n) pre-scan entirely — a measured ~11% win when the winning backend
    # is just numpy's own kernel.
    nan_count = 0
    work = arr
    sorted_arr = None
    _LAST_BACKEND = None
    if work.size > 1:
        # Top priority: counting sort for dense bounded int64/uint64 — beats
        # every comparison backend (and np.sort) by 1.7-2.8x, single-threaded.
        if force is None and work.dtype.kind in "iu" and work.dtype.itemsize >= 8:
            from ._strategies import _counting_is_worth_it, counting_sort_array
            mn = int(work.min()); mx = int(work.max())
            if mx == mn:
                sorted_arr = work
                _LAST_BACKEND = "counting"
            elif _counting_is_worth_it(work.size, mx - mn):
                sorted_arr = counting_sort_array(work, mn, mx)
                _LAST_BACKEND = "counting"

        if sorted_arr is None:
            # Select the backend FIRST (value-independent — keys off dtype/size),
            # so we can decide whether the NaN pre-scan is even required.
            backend = _choose_backend(work, force)

            # Deferred NaN handling: strip ONLY for a backend that can't order
            # NaN like np.sort. nan_safe backends (numpy / x86_simd_sort /
            # arm_neon_sort — whose kernel IS np.sort) place NaN at the end
            # natively, so skipping the scan saves a measured ~11% on float sorts
            # whose winner is just numpy's kernel. Radix/parallel backends stay
            # nan-unsafe and still receive stripped input. (int arrays have no NaN.)
            #
            # We TIME the scan and fold it into the recorded latency below, so the
            # self-tuning DB sees the true cost of choosing a nan-unsafe backend
            # on float data (sort + scan) vs a nan-safe one (sort only) and
            # converges to whichever is genuinely cheaper end-to-end.
            nan_scan_s = 0.0
            if is_float and nan_hint is not False and backend is not None \
                    and not getattr(backend, "nan_safe", False):
                _ts = time.perf_counter()
                needs_mask = True if nan_hint is True else _has_any_nan(arr)
                if needs_mask:
                    nan_mask = np.isnan(arr)
                    if nan_mask.any():
                        nan_count = int(nan_mask.sum())
                        work = arr[~nan_mask]
                        # Stripping can shrink the array below a heavy backend's
                        # crossover (or to <=1 element); re-pick on the stripped
                        # data so we don't run a parallel radix on a handful of
                        # values — or at all.
                        backend = (_choose_backend(work, force)
                                   if work.size > 1 else None)
                nan_scan_s = time.perf_counter() - _ts
            if backend is not None:
                try:
                    # The compiled backends sort the buffer in place, so it must
                    # be writeable AND contiguous. np.ascontiguousarray on a
                    # READ-ONLY buffer returns the SAME non-writeable array, so a
                    # mutating backend (the Rust radix) would otherwise raise an
                    # uncatchable pyo3 PanicException and crash the process. Force
                    # a real writeable copy with np.array(). Also copy here (and
                    # only here) when the caller asked to preserve the input and
                    # the backend mutates — non-mutating backends never touch it.
                    if (preserve and getattr(backend, "mutates_input", True)) \
                            or (not work.flags["C_CONTIGUOUS"]) \
                            or (not work.flags["WRITEABLE"]):
                        work = np.array(work)
                    t0 = time.perf_counter()
                    sorted_arr = backend._sort_ascending(work)
                    # Attribute the NaN-scan cost to this backend so the tuner
                    # accounts for it (0 for nan_safe backends and all ints).
                    elapsed = time.perf_counter() - t0 + nan_scan_s
                    _LAST_BACKEND = backend.name
                    # Record the measurement for the self-tuning dispatcher.
                    # Telemetry must never raise into the sort path.
                    try:
                        from ._tuning import DB
                        DB.record(backend.name, work.dtype.kind,
                                  work.size, elapsed)
                    except Exception:
                        pass
                except (KeyboardInterrupt, SystemExit, GeneratorExit):
                    raise
                except BaseException:
                    # never-lose → fall through to np.sort. Catch BaseException
                    # (not just Exception) so a native pyo3 PanicException —
                    # which subclasses BaseException — is also contained. In
                    # debug mode, surface the error so wiring bugs aren't hidden.
                    if debug:
                        raise
                    # Penalize the crashed backend in the self-tuning DB so a
                    # transient panic doesn't leave a stale "fastest" EWMA that
                    # keeps re-selecting the broken kernel forever. A large
                    # punitive elapsed (10 s) decays the EWMA toward "avoid".
                    try:
                        from ._tuning import DB
                        DB.record(backend.name, work.dtype.kind,
                                  work.size, 10.0)
                    except Exception:
                        pass
                    sorted_arr = None
    if sorted_arr is None:
        sorted_arr = np.sort(work) if work.size else work
        if _LAST_BACKEND is None:
            _LAST_BACKEND = "numpy"

    if descending:
        sorted_arr = sorted_arr[::-1]
    if nan_count:
        # Reattach NaN via the scratch pool: one pre-allocated buffer takes
        # the place of np.concatenate's intermediate-plus-output pair, and
        # the backing memory is reused across hot-loop sorts. We copy out of
        # the pooled buffer so the caller gets an owning array — the pool
        # immediately recycles the slab.
        try:
            from ._scratch import POOL
            total = sorted_arr.size + nan_count
            with POOL.borrow(sorted_arr.dtype, total) as h:
                out = h.buf
                if descending:
                    out[:nan_count] = np.nan
                    out[nan_count:] = sorted_arr
                else:
                    out[:sorted_arr.size] = sorted_arr
                    out[sorted_arr.size:] = np.nan
                # Copy OUT of the pool — `out` is a view into the recycled
                # backing buffer that will be handed to the next borrower.
                sorted_arr = np.ascontiguousarray(out.copy())
        except Exception:
            # Scratch failure must not lose the result; fall back to the
            # historical np.concatenate path.
            if debug:
                raise
            nans = np.full(nan_count, np.nan, dtype=sorted_arr.dtype)
            sorted_arr = (np.concatenate([nans, sorted_arr]) if descending
                          else np.concatenate([sorted_arr, nans]))
    elif descending and not sorted_arr.flags["C_CONTIGUOUS"]:
        # Return a normal C-contiguous array, not a reversed view — callers
        # feeding the result to C/GPU code shouldn't hit a surprise copy.
        sorted_arr = np.ascontiguousarray(sorted_arr)
    return sorted_arr

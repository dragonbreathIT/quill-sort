"""
hydra_backend.py — HydraSort as a quill-sort backend (and standalone sorter).

HydraSort is an adaptive pass-skipping LSD radix sort for int32/int64/
uint32/uint64. One measuring pass picks a strategy:

    presorted / reversed / constant  ->  O(n) early-out
    range < 2^16                     ->  counting sort
    otherwise                        ->  k = ceil(bits(max-min)/~11) radix
                                         passes ONLY (offset data like
                                         timestamps collapses to few passes)

Usage (standalone):
    from hydra_backend import hydra_sort, last_path
    hydra_sort(arr)              # sorts a numpy int array IN PLACE
    print(last_path())           # which strategy ran

Usage (as a quill backend):
    import hydra_backend
    hydra_backend.register_with_quill()   # adds "hydra" to the chain

Build: needs gcc or clang on PATH the first time; compiles hydra_sort.c
sitting next to this file into libhydra.so and caches it.
"""
from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "hydra_sort.c")
_LIB = os.path.join(_HERE, "libhydra.so")

_lib = None
_FNS = {}


def _build_if_needed() -> None:
    if os.path.exists(_LIB) and os.path.getmtime(_LIB) >= os.path.getmtime(_SRC):
        return
    cc = os.environ.get("CC") or ("clang" if shutil.which("clang") else "gcc")
    # -march=native squeezes the histogram/scatter loops; some toolchains want
    # -mcpu=native (Apple clang) or neither. Try in order, fall back gracefully.
    for extra in (["-march=native"], ["-mcpu=native"], []):
        cmd = [cc, "-O3", *extra, "-shared", "-fPIC", _SRC, "-o", _LIB]
        try:
            subprocess.run(cmd, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    raise RuntimeError("hydra: failed to compile hydra_sort.c (need gcc or clang)")


def _load() -> None:
    global _lib, _FNS
    if _lib is not None:
        return
    _build_if_needed()
    _lib = ctypes.CDLL(_LIB)
    _lib.hydra_last_path.restype = ctypes.c_char_p
    for fn in (_lib.hydra_sort_i64, _lib.hydra_sort_u64,
               _lib.hydra_sort_i32, _lib.hydra_sort_u32):
        fn.restype = ctypes.c_int
        fn.argtypes = (ctypes.c_void_p, ctypes.c_int64)
    _FNS = {
        np.dtype("int64"): _lib.hydra_sort_i64,
        np.dtype("uint64"): _lib.hydra_sort_u64,
        np.dtype("int32"): _lib.hydra_sort_i32,
        np.dtype("uint32"): _lib.hydra_sort_u32,
    }


SUPPORTED_DTYPES = ("int64", "uint64", "int32", "uint32")


def hydra_sort(arr: np.ndarray) -> np.ndarray:
    """Sort a contiguous 1-D integer ndarray in place. Returns the array."""
    _load()
    fn = _FNS.get(arr.dtype)
    if fn is None:
        raise TypeError(f"hydra supports {SUPPORTED_DTYPES}, got {arr.dtype}")
    if not arr.flags.c_contiguous:
        raise ValueError("hydra needs a C-contiguous array")
    if not arr.flags.writeable:
        raise ValueError("hydra sorts in place; array is read-only")
    rc = fn(arr.ctypes.data_as(ctypes.c_void_p), ctypes.c_int64(arr.size))
    if rc == 1:
        raise MemoryError("hydra: scratch allocation failed")
    if rc == 2:
        raise ValueError("hydra: n must be < 2**32")
    return arr


def last_path() -> str:
    """Which strategy the previous hydra_sort call used (this thread)."""
    _load()
    return _lib.hydra_last_path().decode()


def estimated_passes(mn: int, mx: int) -> int:
    """Radix passes hydra would need for data in [mn, mx] (0 = early-out)."""
    rng = int(mx) - int(mn)
    if rng <= 0:
        return 0
    if rng < 65536:
        return 1  # counting
    return max(1, (rng.bit_length() + 10) // 11)


# ─────────────────────────────────────────────────────────────────────────────
# quill-sort integration
# ─────────────────────────────────────────────────────────────────────────────

def register_with_quill() -> bool:
    """Register HydraSort in quill's backend chain as the ``"hydra"`` backend.

    Routing is left to (a) hydra's own per-array adaptivity — it measures each
    array's *effective* range in one pass and picks counting / few-pass radix /
    early-out accordingly — and (b) quill's self-tuning dispatcher, which learns
    the (dtype, size) buckets where hydra actually wins on this machine. Quill's
    never-lose contract is preserved: any error re-dispatches to np.sort, and on
    full-range data where hydra can't skip passes the tuner will favour the
    parallel backends instead.

    Returns True if quill is importable and the backend was registered.
    """
    try:
        from quill._backends import Backend, register_backend
    except ImportError:
        return False

    class HydraBackend(Backend):
        name = "hydra"
        priority = 55            # above the numpy floor, below the parallel/GPU tiers
        min_n = 200_000          # below this, dispatch overhead dominates
        kinds = "iu"
        max_itemsize = 8
        mutates_input = True     # sorts the (dispatcher-provided) buffer in place

        def _probe(self) -> bool:
            # Exercise the real RADIX path (n>96, range>2^16), not just the
            # insertion early-out, so a broken kernel can't pass the probe.
            try:
                _load()
                r = np.random.default_rng(0).integers(0, 10**9, 5000, dtype=np.int64)
                got = r.copy()
                hydra_sort(got)
                return bool(np.array_equal(got, np.sort(r)))
            except Exception:
                return False

        def supports(self, arr) -> bool:  # type: ignore[override]
            return arr.dtype.name in SUPPORTED_DTYPES and arr.size < 2**32

        def _sort_ascending(self, arr):
            # quill's dispatcher guarantees a C-contiguous, writeable buffer for
            # mutating backends (and already makes a private copy when the caller
            # asked to preserve the input), so sort IN PLACE — the previous
            # version's extra .copy() double-allocated on every hydra sort.
            hydra_sort(arr)
            return arr

    register_backend(HydraBackend())
    return True


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    a = rng.integers(0, 10**9, 1_000_000, dtype=np.int64)
    ref = np.sort(a)
    hydra_sort(a)
    assert np.array_equal(a, ref)
    print(f"hydra OK  (path: {last_path()})")

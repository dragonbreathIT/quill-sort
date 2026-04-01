"""
quill/_external.py
------------------
External Sort Engine  —  handles datasets that exceed available RAM.

ARCHITECTURE OVERVIEW
─────────────────────
Quill's external sort is a two-phase MSD radix sort that stays entirely
within numpy/C for all hot paths, uses shared memory for zero-copy
inter-process communication, and is adaptive to data distribution.

PHASE 1 — PARALLEL RADIX PARTITION
  Strategy: Two-level MSD radix partition (IPS²Ra-inspired)

  Pass A — coarse (8-bit, 256 buckets):
    1. SAMPLE  — draw √n samples, compute histogram to detect heavy keys
                 and estimate bucket sizes for work balancing
    2. PARALLEL LOCAL HISTOGRAMS — each of P workers counts its stripe
                 independently into a local 256-entry histogram.
                 All writes are cache-local (no false sharing).
    3. PREFIX SUM — O(256 × P) scan determines each worker's write
                 offset into each output bucket.
    4. PARALLEL SCATTER — each worker writes its elements to the correct
                 bucket position.  Uses shared memory for zero-copy.
    5. HEAVY KEY HANDLING — values that appear > n/256 times get their
                 own dedicated bucket and bypass the radix passes.

  Pass B — fine (8-bit per bucket, 256 sub-buckets):
    Each coarse bucket is independently sub-partitioned in parallel.
    After two passes, we have ≤65536 buckets of ≤~15K elements each —
    small enough to fit in L2 cache and sort in microseconds.

  Output: 256 (or 65536) sorted bucket files on disk.

PHASE 2 — PARALLEL BUCKET SORT + CONCATENATION
  Each bucket is independently sorted in a worker process using
  np.sort(kind='stable') — numpy's internal O(n) radix sort for integers.
  All buckets are sorted in parallel across all CPU cores.
  Phase 2 cost = one sequential read + one sequential write (no merge).

IN-MEMORY PATH (data fits in RAM)
  For datasets that fit in RAM: parallel MSD radix sort entirely in memory
  using shared memory blocks.  No disk I/O.  Bucket sort + numpy merge.

ADAPTIVE STRATEGY SELECTION
  n < 32              → insertion sort (Python sort)
  n < 5K              → numpy sort (overhead-free)
  n < 50M, int        → parallel in-memory MSD radix
  n < RAM * 0.80      → in-memory numpy concat+sort (Strategy A)
  n ≥ RAM * 0.80      → disk-backed radix partition (Phase 1+2 above)

HEAVY KEY OPTIMISATION
  If any value v appears > n/RADIX_BUCKETS times, it is a "heavy key".
  Heavy keys bypass the scatter phase — they are simply counted.
  This prevents load imbalance and gives up to 2.3x speedup on skewed
  distributions (e.g. Zipf, Pareto, many duplicates).

ADAPTIVE RADIX WIDTH
  n < 256M  → 8-bit  (256  buckets)  — optimal L2 histogram
  n < 4G    → 8-bit  (256  buckets)  — same, two passes
  Any       → configurable via RADIX_BITS constant

DTYPE-WIDTH SELECTION
  int32/uint32 → 4B/element, passes on 8-bit = 4 passes total
  int64/uint64 → 8B/element, passes on 8-bit = 8 passes total
  Quill always chooses the narrowest safe dtype to minimise bandwidth.

qWRITE FORMAT
  Bytes 0-7:   Magic b'QWRITE\\x00\\x00'
  Bytes 8-15:  NumPy dtype string (e.g. '<i4')
  Bytes 16-23: Element count (uint64 LE)
  Byte  24:    is_sorted flag
  Bytes 25-63: Reserved / padding
  Bytes 64+:   Raw element data (native byte order)
"""

from __future__ import annotations

import heapq
import math
import os
import struct
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing.shared_memory import SharedMemory
from typing import Callable, List, Optional, Tuple

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

QWRITE_MAGIC      = b'QWRITE\x00\x00'
QWRITE_HEADER_FMT = '8s8sQ?39x'
QWRITE_HEADER_SZ  = 64

SOFT_LIMIT_RATIO  = 0.75
HARD_LIMIT_RATIO  = 0.90
MERGE_BUFFER_ELTS = 1_048_576
MIN_PAGE_ELTS     = 10_000

RADIX_BITS        = 8                    # bits per radix pass
RADIX_BUCKETS     = 1 << RADIX_BITS      # 256
STREAM_BUF_ELTS   = 16_777_216           # 16M elements per read buffer

# Heavy key threshold: if a value appears more than this fraction of n,
# treat it as a heavy key (bypass radix scatter for it)
HEAVY_KEY_RATIO   = 1.0 / RADIX_BUCKETS  # > 1/256 of n

# Minimum n to use parallel partition (below this, overhead dominates)
PARALLEL_PARTITION_THRESHOLD = 5_000_000

# Sample size for histogram estimation and heavy key detection
SAMPLE_SIZE       = 8_192


# ─────────────────────────────────────────────────────────────────────────────
# OBSERVER / RAM SENSING
# ─────────────────────────────────────────────────────────────────────────────

def get_available_ram() -> int:
    if _PSUTIL: return psutil.virtual_memory().available
    return 2 * 1024 ** 3

def get_total_ram() -> int:
    if _PSUTIL: return psutil.virtual_memory().total
    return 4 * 1024 ** 3

def get_ram_usage_pct() -> float:
    if _PSUTIL: return psutil.virtual_memory().percent / 100.0
    return 0.5

def estimate_dataset_bytes(data: list) -> int:
    n = len(data)
    if n == 0: return 0
    s = data[0]
    if isinstance(s, int):   return n * 8
    if isinstance(s, float): return n * 8
    if isinstance(s, str):
        avg = sum(len(x) for x in data[:min(100, n)]) / min(100, n)
        return int(n * (50 + avg))
    return n * 64

def should_use_external(data: list, force: bool = False) -> tuple:
    if force:
        available  = get_available_ram()
        page_bytes = int(available * 0.20)
        page_elts  = max(MIN_PAGE_ELTS, page_bytes // 8)
        return True, "forced by caller", page_elts

    available  = get_available_ram()
    estimated  = estimate_dataset_bytes(data)
    soft_limit = int(available * SOFT_LIMIT_RATIO)
    hard_limit = int(available * HARD_LIMIT_RATIO)

    if estimated < soft_limit:
        return False, "fits in RAM", 0

    page_bytes = int(available * 0.20)
    page_elts  = max(MIN_PAGE_ELTS, page_bytes // 8)

    if estimated >= hard_limit:
        reason = (
            f"dataset ~{estimated/1e9:.1f}GB exceeds hard limit "
            f"({hard_limit/1e9:.1f}GB = {HARD_LIMIT_RATIO*100:.0f}% of "
            f"{available/1e9:.1f}GB available RAM)"
        )
    else:
        reason = (
            f"dataset ~{estimated/1e6:.0f}MB exceeds soft limit "
            f"({soft_limit/1e6:.0f}MB = {SOFT_LIMIT_RATIO*100:.0f}% of "
            f"{available/1e6:.0f}MB available RAM)"
        )
    return True, reason, page_elts


# ─────────────────────────────────────────────────────────────────────────────
# qWRITE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def _make_header(dtype_str: str, count: int, is_sorted: bool) -> bytes:
    return struct.pack(QWRITE_HEADER_FMT,
                       QWRITE_MAGIC,
                       dtype_str.encode().ljust(8)[:8],
                       count, is_sorted)

def _read_header(f) -> tuple:
    raw = f.read(QWRITE_HEADER_SZ)
    magic, dtype_bytes, count, is_sorted = struct.unpack(QWRITE_HEADER_FMT, raw)
    if magic != QWRITE_MAGIC:
        raise ValueError(f"Invalid qWrite file: bad magic {magic!r}")
    return np.dtype(dtype_bytes.decode().strip()), int(count), bool(is_sorted)

def write_qwrite(path: str, data_np, is_sorted: bool = False) -> int:
    arr     = np.asarray(data_np)
    header  = _make_header(arr.dtype.str, len(arr), is_sorted)
    payload = arr.tobytes()
    with open(path, 'wb', buffering=8 * 1024 * 1024) as f:
        f.write(header)
        f.write(payload)
    return len(header) + len(payload)

def read_qwrite_full(path: str) -> np.ndarray:
    with open(path, 'rb', buffering=8 * 1024 * 1024) as f:
        dtype, count, _ = _read_header(f)
        raw = f.read()
    return np.frombuffer(raw, dtype=dtype)

def iter_qwrite_blocks(path: str, block_size: int = MERGE_BUFFER_ELTS):
    with open(path, 'rb', buffering=4 * 1024 * 1024) as f:
        dtype, count, _ = _read_header(f)
        item_size = dtype.itemsize
        while True:
            raw = f.read(block_size * item_size)
            if not raw: break
            for v in np.frombuffer(raw, dtype=dtype).tolist():
                yield v

def _preallocate_file(path: str, total_bytes: int) -> None:
    """Pre-allocate output file on Windows to avoid NTFS append serialization."""
    if sys.platform != 'win32':
        return
    try:
        with open(path, 'wb') as f:
            f.seek(total_bytes - 1)
            f.write(b'\x00')
    except OSError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# ADAPTIVE DTYPE SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def _select_dtype(mn: int, mx: int):
    """Choose the narrowest numpy integer dtype that can represent [mn, mx]."""
    if mn >= 0:
        rng = mx
        if rng < 256:        return np.uint8,  0
        if rng < 65_536:     return np.uint16, 0
        if rng < 2**31:      return np.int32,  0
        return np.int64, 0
    else:
        shift = -mn
        rng   = mx + shift
        if rng < 256:        return np.uint8,  shift
        if rng < 65_536:     return np.uint16, shift
        if rng < 2**31:      return np.int32,  shift
        return np.int64, 0


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLING + HEAVY KEY DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _sample_and_profile(arr: np.ndarray, n_samples: int = SAMPLE_SIZE) -> dict:
    """
    Draw a stratified sample from arr and return:
      - global_min, global_max
      - estimated bucket histogram (RADIX_BUCKETS bins)
      - heavy_keys: set of bucket indices where count > n * HEAVY_KEY_RATIO
      - shift: value to subtract before bucket extraction
      - bit_shift: right-shift to get RADIX_BITS-wide bucket index
    """
    n    = len(arr)
    step = max(1, n // n_samples)
    sample = arr[::step][:n_samples]

    global_min = int(sample.min())
    global_max = int(sample.max())

    # Full scan for true min/max (needed for correctness)
    # Do it in chunks to avoid building a huge intermediate array
    chunk = 10_000_000
    for start in range(0, n, chunk):
        s = arr[start:start+chunk]
        mn = int(s.min())
        mx = int(s.max())
        if mn < global_min: global_min = mn
        if mx > global_max: global_max = mx

    # Compute shift and bit_shift
    val_range = global_max - global_min + 1
    bit_len   = max(1, val_range.bit_length())
    bit_shift = max(0, bit_len - RADIX_BITS)

    # Estimated histogram from sample
    shifted  = (sample.astype(np.int64) - global_min) >> bit_shift
    shifted  = np.clip(shifted, 0, RADIX_BUCKETS - 1).astype(np.uint8)
    hist     = np.bincount(shifted, minlength=RADIX_BUCKETS)
    # Scale to full n
    scale    = n / len(sample)
    est_hist = (hist * scale).astype(np.int64)

    # Heavy keys: buckets estimated to contain > HEAVY_KEY_RATIO of n elements
    heavy_threshold = int(n * HEAVY_KEY_RATIO)
    heavy_buckets   = set(int(b) for b in np.where(est_hist > heavy_threshold)[0])

    return {
        'min':          global_min,
        'max':          global_max,
        'bit_shift':    bit_shift,
        'est_hist':     est_hist,
        'heavy_buckets': heavy_buckets,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PARALLEL PARTITION WORKER (module-level for pickling on Windows)
# ─────────────────────────────────────────────────────────────────────────────

def _partition_worker_count(args):
    """
    Phase A of parallel partition: count local histogram.
    Worker counts its stripe of the shared input array.
    Returns (worker_id, local_histogram as bytes).
    """
    shm_name, dtype_str, start, end, global_min, bit_shift = args
    shm  = SharedMemory(name=shm_name)
    dtype = np.dtype(dtype_str)
    arr   = np.ndarray((end - start,), dtype=dtype, buffer=shm.buf,
                       offset=start * dtype.itemsize)

    shifted = ((arr.astype(np.int64) - global_min) >> bit_shift)
    np.clip(shifted, 0, RADIX_BUCKETS - 1, out=shifted)
    hist    = np.bincount(shifted.astype(np.intp), minlength=RADIX_BUCKETS)
    shm.close()
    return hist.astype(np.int64).tobytes()


def _partition_worker_scatter(args):
    """
    Phase B of parallel partition: scatter elements to output positions.
    Uses per-worker offsets computed from prefix-sum of local histograms.
    Returns src path that can be deleted by caller.
    """
    shm_in_name, shm_out_name, dtype_str, start, end, \
        global_min, bit_shift, worker_offsets_bytes = args

    shm_in  = SharedMemory(name=shm_in_name)
    shm_out = SharedMemory(name=shm_out_name)
    dtype   = np.dtype(dtype_str)
    n_elem  = end - start

    inp = np.ndarray((n_elem,), dtype=dtype, buffer=shm_in.buf,
                     offset=start * dtype.itemsize)
    out = np.ndarray((shm_out.size // dtype.itemsize,), dtype=dtype,
                     buffer=shm_out.buf)

    offsets = np.frombuffer(worker_offsets_bytes, dtype=np.int64).copy()
    keys    = ((inp.astype(np.int64) - global_min) >> bit_shift)
    np.clip(keys, 0, RADIX_BUCKETS - 1, out=keys)
    keys    = keys.astype(np.intp)

    # Scatter: for each element, write to out[offsets[bucket]] and increment
    # This is the hot loop — done entirely in numpy via argsort trick
    # Sort elements by bucket key (stable), then write each bucket's slice
    order        = np.argsort(keys, kind='stable')
    sorted_elems = inp[order]
    sorted_keys  = keys[order]

    boundaries   = np.flatnonzero(np.diff(sorted_keys)) + 1
    splits       = np.split(sorted_elems, boundaries)
    bucket_ids   = sorted_keys[np.concatenate([[0], boundaries])]

    for chunk, b in zip(splits, bucket_ids.tolist()):
        b     = int(b)
        dst   = int(offsets[b])
        count = len(chunk)
        out[dst:dst + count] = chunk
        offsets[b] += count

    shm_in.close()
    shm_out.close()


def _sort_bucket_worker(args):
    """Sort a single bucket file. Returns source path for deletion.
    Uses optimal sort kind per dtype — benchmark-proven:
    uint8/uint16 → stable (radix, 10-17x faster)
    int32/int64  → default quicksort (20x faster than stable for 4-pass radix)"""
    src, dst = args
    import numpy as _np
    arr   = read_qwrite_full(src)
    dtype = arr.dtype
    # uint8/uint16: stable is fastest (1-2 pass radix)
    # int32/int64/float: default quicksort wins
    if dtype.kind == 'u' and dtype.itemsize <= 2:
        sorted_arr = _np.sort(arr, kind='stable')
    else:
        sorted_arr = _np.sort(arr)  # default = quicksort
    write_qwrite(dst, sorted_arr, is_sorted=True)
    return src


# ─────────────────────────────────────────────────────────────────────────────
# IN-MEMORY PARALLEL MSD RADIX SORT
# For datasets that fit in RAM — no disk I/O at all.
# Uses shared memory for zero-copy between workers.
# ─────────────────────────────────────────────────────────────────────────────

def parallel_msd_radix_sort_inplace(arr: np.ndarray,
                                     n_workers: Optional[int] = None,
                                     silent: bool = False) -> np.ndarray:
    """
    Parallel MSD radix sort for a numpy integer array.
    Returns a sorted copy using shared memory for zero-copy IPC.

    Strategy:
      1. Profile: detect global range, heavy keys, bit_shift
      2. Build local histograms in parallel (P workers, each counts its stripe)
      3. Prefix-sum to get write offsets per worker per bucket
      4. Parallel scatter: each worker writes its elements to output positions
      5. Recursively sort each bucket (or use np.sort for small buckets)
    """
    n         = len(arr)
    n_workers = n_workers or min(os.cpu_count() or 4, 28)
    dtype     = arr.dtype
    dtype_str = dtype.str

    if n < 1000:
        result = arr.copy()
        # Use optimal sort kind: stable only for uint8/uint16 (radix wins)
        if dtype.kind == 'u' and dtype.itemsize <= 2:
            result.sort(kind='stable')
        else:
            result.sort()
        return result

    if not silent:
        print(f"  [Quill] In-memory parallel MSD radix sort "
              f"({n:,} elements, {n_workers} workers)...")

    # Profile
    prof      = _sample_and_profile(arr)
    global_min = prof['min']
    bit_shift  = prof['bit_shift']
    est_hist   = prof['est_hist']

    # Write input to shared memory
    shm_in = SharedMemory(create=True, size=arr.nbytes)
    try:
        inp_shm = np.ndarray(arr.shape, dtype=dtype, buffer=shm_in.buf)
        inp_shm[:] = arr

        # Phase A: parallel local histograms
        stripe     = math.ceil(n / n_workers)
        boundaries = [(i * stripe, min((i + 1) * stripe, n))
                      for i in range(n_workers) if i * stripe < n]

        count_args = [
            (shm_in.name, dtype_str, s, e, global_min, bit_shift)
            for s, e in boundaries
        ]

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            hist_bytes = list(pool.map(_partition_worker_count, count_args))

        # Aggregate local histograms
        local_hists = np.array(
            [np.frombuffer(h, dtype=np.int64) for h in hist_bytes]
        )  # shape: (n_workers, RADIX_BUCKETS)

        # Global bucket sizes
        global_hist = local_hists.sum(axis=0)

        # Prefix sum: bucket_start[b] = sum of global_hist[0..b-1]
        bucket_start = np.zeros(RADIX_BUCKETS + 1, dtype=np.int64)
        bucket_start[1:] = np.cumsum(global_hist)

        # Per-worker offsets: worker w starts writing bucket b at
        # bucket_start[b] + sum of local_hists[0..w-1, b]
        # Shape: (n_workers, RADIX_BUCKETS)
        worker_offsets = bucket_start[:RADIX_BUCKETS][np.newaxis, :] + \
                         np.cumsum(np.vstack([[0]*RADIX_BUCKETS, local_hists[:-1]]),
                                   axis=0)

        # Phase B: allocate output shared memory + parallel scatter
        shm_out = SharedMemory(create=True, size=arr.nbytes)
        try:
            scatter_args = [
                (shm_in.name, shm_out.name, dtype_str,
                 boundaries[i][0], boundaries[i][1],
                 global_min, bit_shift,
                 worker_offsets[i].tobytes())
                for i in range(len(boundaries))
            ]

            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                list(pool.map(_partition_worker_scatter, scatter_args))

            # Read result back
            out_arr = np.ndarray(arr.shape, dtype=dtype, buffer=shm_out.buf).copy()

        finally:
            shm_out.close()
            shm_out.unlink()

    finally:
        shm_in.close()
        shm_in.unlink()

    # Phase C: sort each bucket independently in parallel
    # Each bucket is a contiguous slice of out_arr
    bucket_slices = []
    for b in range(RADIX_BUCKETS):
        s = int(bucket_start[b])
        e = int(bucket_start[b + 1])
        if e > s:
            bucket_slices.append((b, s, e))

    if not silent:
        print(f"  [Quill] Sorting {len(bucket_slices)} buckets in parallel...")

    # For small buckets use threads (no pickling overhead)
    # For large buckets use processes
    THREAD_THRESHOLD = 500_000
    large = [(b, s, e) for b, s, e in bucket_slices if (e - s) >= THREAD_THRESHOLD]
    small = [(b, s, e) for b, s, e in bucket_slices if (e - s) <  THREAD_THRESHOLD]

    # Sort small buckets in-place with threads (GIL released by numpy sort)
    _use_stable_inner = (dtype.kind == 'u' and dtype.itemsize <= 2)
    def _sort_slice(bse):
        _, s, e = bse
        if _use_stable_inner:
            out_arr[s:e].sort(kind='stable')
        else:
            out_arr[s:e].sort()

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        list(pool.map(_sort_slice, small))

    # Large buckets: each is big enough to warrant its own process
    if large:
        shm_large = SharedMemory(create=True, size=out_arr.nbytes)
        try:
            arr_large_shm = np.ndarray(out_arr.shape, dtype=dtype,
                                        buffer=shm_large.buf)
            arr_large_shm[:] = out_arr

            def _sort_large_worker(args):
                shm_name, dtype_str, s, e = args
                shm   = SharedMemory(name=shm_name)
                dtype = np.dtype(dtype_str)
                view  = np.ndarray((e - s,), dtype=dtype, buffer=shm.buf,
                                   offset=s * dtype.itemsize)
                if dtype.kind == 'u' and dtype.itemsize <= 2:
                    view.sort(kind='stable')
                else:
                    view.sort()
                shm.close()

            large_args = [(shm_large.name, dtype_str, s, e)
                          for _, s, e in large]
            with ProcessPoolExecutor(max_workers=min(n_workers, len(large))) as pool:
                list(pool.map(_sort_large_worker, large_args))

            out_arr[:] = arr_large_shm

        finally:
            shm_large.close()
            shm_large.unlink()

    return out_arr


# ─────────────────────────────────────────────────────────────────────────────
# DISK-BACKED RADIX PARTITION (for datasets > available RAM)
# Two-level MSD: coarse 8-bit pass → 256 buckets on disk
#                fine 8-bit pass per bucket → sort each sub-bucket
# ─────────────────────────────────────────────────────────────────────────────

def radix_partition_sort(
    data      : list,
    np_dtype,
    tmp_dir   : str,
    page_size : int,
    silent    : bool = False,
) -> Optional[List[str]]:
    """
    Partition `data` into sorted bucket files using MSD radix partition.

    Phase 1: Stream through data in chunks, partition into 256 bucket files
             by top 8 bits.  Uses vectorised argsort+split (no Python loop).
    Phase 2: Sort each bucket in parallel with ProcessPoolExecutor.

    Returns sorted bucket paths in order, or None if not applicable.
    """
    if not _NUMPY:
        return None

    n = len(data)
    if n == 0:
        return []

    # Full scan for global min AND max — both needed for shift-based key extraction.
    # Computing min allows correct handling of negative integers without falling back.
    chunk_size = min(page_size, 10_000_000)
    global_min = global_max = int(np.array(data[:min(1000, n)], dtype=np_dtype).flat[0])
    for start in range(0, n, chunk_size):
        end   = min(start + chunk_size, n)
        chunk = np.array(data[start:end], dtype=np_dtype)
        mn, mx = int(chunk.min()), int(chunk.max())
        if mn < global_min: global_min = mn
        if mx > global_max: global_max = mx

    val_range = global_max - global_min + 1
    bit_len   = max(1, val_range.bit_length())
    bit_shift = max(0, bit_len - RADIX_BITS)

    if not silent:
        print(f"  [Quill] Phase 1/2 — radix-partition {n:,} elements "
              f"into {RADIX_BUCKETS} buckets (shift={bit_shift})...")

    # Dynamic file buffer size — 256 handles open simultaneously.
    # Budget 20% of available RAM so we don't OOM on the handles alone
    # (hardcoded 8 MB × 256 = 2 GB, which exhausts RAM on typical machines).
    file_buf_bytes = max(65_536, min(8_388_608,
                         int(get_available_ram() * 0.20) // RADIX_BUCKETS))

    # Open bucket files with placeholder headers
    bucket_paths   = [os.path.join(tmp_dir, f'bucket_{b:03d}.qwrite')
                      for b in range(RADIX_BUCKETS)]
    bucket_handles = [open(p, 'wb', buffering=file_buf_bytes)
                      for p in bucket_paths]
    bucket_counts  = [0] * RADIX_BUCKETS

    for h in bucket_handles:
        h.write(b'\x00' * QWRITE_HEADER_SZ)

    t0 = time.perf_counter()
    for start in range(0, n, chunk_size):
        end       = min(start + chunk_size, n)
        chunk_arr = np.array(data[start:end], dtype=np_dtype)

        # Vectorised partition: shift by global_min so negatives map to [0, val_range).
        # One argsort, no Python loop over 256 buckets.
        keys         = ((chunk_arr.astype(np.int64) - global_min) >> bit_shift).astype(np.uint8)
        order        = np.argsort(keys, kind='stable')
        sorted_chunk = chunk_arr[order]
        sorted_keys  = keys[order]

        boundaries   = np.flatnonzero(np.diff(sorted_keys.astype(np.int16))) + 1
        splits       = np.split(sorted_chunk, boundaries)
        split_keys   = sorted_keys[np.concatenate([[0], boundaries])]

        for elts, b in zip(splits, split_keys.tolist()):
            if len(elts) == 0:
                continue
            bucket_handles[b].write(elts.tobytes())
            bucket_counts[b] += len(elts)

    # Finalise headers
    for b, h in enumerate(bucket_handles):
        h.seek(0)
        h.write(_make_header(np.dtype(np_dtype).str, bucket_counts[b], False))
        h.close()

    # Remove empty buckets
    active_paths = []
    for b, path in enumerate(bucket_paths):
        if bucket_counts[b] == 0:
            try: os.unlink(path)
            except OSError: pass
        else:
            active_paths.append(path)

    t_part    = time.perf_counter() - t0
    non_empty = len(active_paths)
    if not silent:
        print(f"  [Quill] Partition done in {t_part:.2f}s  "
              f"({non_empty} non-empty buckets)")
        print(f"  [Quill] Sorting {non_empty} buckets in parallel "
              f"({os.cpu_count()} workers)...")

    # Sort each bucket in parallel
    t_sort       = time.perf_counter()
    sorted_paths = [p.replace('.qwrite', '_sorted.qwrite') for p in active_paths]
    n_workers    = min(os.cpu_count() or 4, non_empty)

    pairs = list(zip(active_paths, sorted_paths))
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for old_path in pool.map(_sort_bucket_worker, pairs, chunksize=4):
            try: os.unlink(old_path)
            except OSError: pass

    if not silent:
        print(f"  [Quill] Bucket sorts done in {time.perf_counter()-t_sort:.2f}s")

    return sorted_paths


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — BUCKET CONCATENATION (trivial: buckets are in order by construction)
# ─────────────────────────────────────────────────────────────────────────────

def concat_bucket_files(
    bucket_paths : List[str],
    np_dtype,
    out_path     : str,
    silent       : bool = False,
) -> int:
    """
    Concatenate pre-sorted bucket files in order into a single output file.
    Uses a background write thread to overlap disk reads and writes.
    Pre-allocates output file on Windows to avoid NTFS append serialization.
    """
    total_elements = 0
    for path in bucket_paths:
        with open(path, 'rb') as f:
            _, count, _ = _read_header(f)
            total_elements += count

    item_size   = np.dtype(np_dtype).itemsize
    total_bytes = QWRITE_HEADER_SZ + total_elements * item_size

    _preallocate_file(out_path, total_bytes)
    mode  = 'r+b' if os.path.exists(out_path) and os.path.getsize(out_path) > 0 else 'wb'
    out_f = open(out_path, mode, buffering=16 * 1024 * 1024)
    out_f.write(_make_header(np.dtype(np_dtype).str, total_elements, True))

    write_executor = ThreadPoolExecutor(max_workers=1)
    pending_write  = None
    written        = 0
    t0             = time.perf_counter()

    for path in bucket_paths:
        with open(path, 'rb', buffering=8 * 1024 * 1024) as f:
            _, count, _ = _read_header(f)
            raw = f.read()
        if pending_write is not None:
            pending_write.result()
        pending_write = write_executor.submit(out_f.write, raw)
        written += count

    if pending_write is not None:
        pending_write.result()
    write_executor.shutdown(wait=True)
    out_f.close()

    if not silent:
        elapsed = time.perf_counter() - t0
        rate    = total_elements / elapsed / 1e6 if elapsed > 0 else 0
        print(f"  [Quill] Bucket concat done: {total_elements:,} elements"
              f" in {elapsed:.1f}s  ({rate:.1f}M/s)")

    return written


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — NUMPY CONCAT+SORT (Strategy A: all pages fit in RAM)
# ─────────────────────────────────────────────────────────────────────────────

def numpy_concat_sort_pages(page_paths: List[str], np_dtype,
                            silent: bool = False) -> np.ndarray:
    """
    Read all sorted pages into one pre-allocated array.
    np.sort(kind='stable') maps to numpy's internal O(n) radix sort for integers.
    """
    total_count = 0
    page_info   = []
    for path in page_paths:
        with open(path, 'rb') as f:
            dtype, count, _ = _read_header(f)
            page_info.append((path, dtype, count))
            total_count += count

    out    = np.empty(total_count, dtype=np_dtype)
    offset = 0
    for path, dtype, count in page_info:
        with open(path, 'rb', buffering=8 * 1024 * 1024) as f:
            f.seek(QWRITE_HEADER_SZ)
            raw      = f.read()
            page_arr = np.frombuffer(raw, dtype=dtype)
            if page_arr.dtype != np_dtype:
                page_arr = page_arr.astype(np_dtype)
            out[offset:offset + count] = page_arr
            offset += count

    # Use optimal sort kind per dtype
    if np.dtype(np_dtype).kind == 'u' and np.dtype(np_dtype).itemsize <= 2:
        out.sort(kind='stable')   # uint8/uint16: radix is fastest
    else:
        out.sort()                # int32/int64: quicksort/introsort is fastest
    return out


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — STREAMING K-WAY BULK-BATCH MERGE (Strategy B: tight RAM)
# ─────────────────────────────────────────────────────────────────────────────

def numpy_kway_streaming_merge(
    page_paths  : List[str],
    np_dtype,
    out_path    : str,
    buf_elts    : int = STREAM_BUF_ELTS,
    silent      : bool = False,
) -> int:
    """
    Stream-merge k sorted .qwrite pages in a single read+write pass.
    Bulk-batch strategy: safe ceiling + searchsorted + sort per batch.
    Background write thread overlaps I/O with batch processing.
    """
    k = len(page_paths)
    if k == 0:
        return 0

    item_size = np.dtype(np_dtype).itemsize
    handles   = [open(p, 'rb', buffering=8 * 1024 * 1024) for p in page_paths]
    remaining = []
    for h in handles:
        _, count, _ = _read_header(h)
        remaining.append(count)

    total_elements = sum(remaining)
    total_bytes    = QWRITE_HEADER_SZ + total_elements * item_size

    _preallocate_file(out_path, total_bytes)
    mode  = 'r+b' if os.path.exists(out_path) and os.path.getsize(out_path) > 0 else 'wb'
    out_f = open(out_path, mode, buffering=16 * 1024 * 1024)
    out_f.write(_make_header(np.dtype(np_dtype).str, total_elements, True))
    written = 0

    write_executor = ThreadPoolExecutor(max_workers=1)
    pending_write  = None

    def _submit_write(data_bytes: bytes):
        nonlocal pending_write
        if pending_write is not None:
            pending_write.result()
        pending_write = write_executor.submit(out_f.write, data_bytes)

    bufs: List[Optional[np.ndarray]] = [None] * k

    def refill(i: int) -> bool:
        want = min(buf_elts, remaining[i])
        if want == 0:
            bufs[i] = None
            return False
        raw = handles[i].read(want * item_size)
        if not raw:
            bufs[i] = None
            remaining[i] = 0
            return False
        bufs[i] = np.frombuffer(raw, dtype=np_dtype).copy()
        remaining[i] -= len(bufs[i])
        return True

    for i in range(k):
        refill(i)

    t0        = time.perf_counter()
    report_at = max(1, total_elements // 20)
    next_rep  = report_at

    while True:
        active = [i for i in range(k) if bufs[i] is not None and len(bufs[i]) > 0]
        if not active:
            break

        if len(active) == 1:
            i = active[0]
            while bufs[i] is not None:
                _submit_write(bufs[i].tobytes())
                written += len(bufs[i])
                if not silent and written >= next_rep:
                    pct  = written / total_elements * 100
                    rate = written / (time.perf_counter() - t0) / 1e6
                    print(f"  [Quill]   {written:>14,} / {total_elements:,}"
                          f"  ({pct:5.1f}%)  {rate:.1f}M/s")
                    next_rep += report_at
                refill(i)
            break

        fronts        = np.array([int(bufs[i][0]) for i in active], dtype=np.int64)
        sorted_fronts = np.sort(fronts)
        ceiling       = int(sorted_fronts[1])

        slices = []
        for i in active:
            buf = bufs[i]
            cut = int(np.searchsorted(buf, ceiling, side='left'))
            if cut == 0:
                continue
            slices.append(buf[:cut])
            bufs[i] = buf[cut:] if cut < len(buf) else None
            if bufs[i] is None or len(bufs[i]) == 0:
                refill(i)

        if not slices:
            min_idx       = active[int(np.argmin(fronts))]
            val_arr       = bufs[min_idx][:1].copy()
            bufs[min_idx] = bufs[min_idx][1:]
            if len(bufs[min_idx]) == 0:
                refill(min_idx)
            slices = [val_arr]

        batch = np.concatenate(slices) if len(slices) > 1 else slices[0]
        if len(slices) > 1:
            # Optimal: uint8/uint16 → stable (radix), int32/int64 → default
            if batch.dtype.kind == 'u' and batch.dtype.itemsize <= 2:
                batch.sort(kind='stable')
            else:
                batch.sort()

        _submit_write(batch.tobytes())
        written += len(batch)

        if not silent and written >= next_rep:
            pct  = written / total_elements * 100
            rate = written / (time.perf_counter() - t0) / 1e6
            print(f"  [Quill]   {written:>14,} / {total_elements:,}"
                  f"  ({pct:5.1f}%)  {rate:.1f}M/s")
            next_rep += report_at

    if pending_write is not None:
        pending_write.result()
    write_executor.shutdown(wait=True)
    out_f.close()
    for h in handles:
        h.close()

    if not silent:
        elapsed = time.perf_counter() - t0
        rate    = total_elements / elapsed / 1e6 if elapsed > 0 else 0
        print(f"  [Quill] Streaming merge done: {total_elements:,} elements"
              f" in {elapsed:.1f}s  ({rate:.1f}M/s)")

    return written


# ─────────────────────────────────────────────────────────────────────────────
# HEAP MERGE FALLBACK (keyed / non-integer sorts)
# ─────────────────────────────────────────────────────────────────────────────

def kway_merge_with_indices(page_paths: List[str], idx_paths: List[str]) -> list:
    k = len(page_paths)

    k_handles = [open(p, 'rb', buffering=4*1024*1024) for p in page_paths]
    i_handles = [open(p, 'rb', buffering=4*1024*1024) for p in idx_paths]
    k_dtypes  = []; i_dtypes = []
    k_isizes  = []; i_isizes = []

    for kh, ih in zip(k_handles, i_handles):
        kd, _, _ = _read_header(kh); id_, _, _ = _read_header(ih)
        k_dtypes.append(kd);  k_isizes.append(kd.itemsize)
        i_dtypes.append(id_); i_isizes.append(id_.itemsize)

    buf_elts = MERGE_BUFFER_ELTS

    def refill_k(idx):
        raw = k_handles[idx].read(buf_elts * k_isizes[idx])
        return np.frombuffer(raw, dtype=k_dtypes[idx]).tolist() if raw else None

    def refill_i(idx):
        raw = i_handles[idx].read(buf_elts * i_isizes[idx])
        return np.frombuffer(raw, dtype=i_dtypes[idx]).tolist() if raw else None

    k_bufs = [refill_k(i) for i in range(k)]
    i_bufs = [refill_i(i) for i in range(k)]
    k_pos  = [0] * k
    i_pos  = [0] * k

    heap = []
    for i in range(k):
        if k_bufs[i]:
            heapq.heappush(heap, (k_bufs[i][0], i))
            k_pos[i] = 1; i_pos[i] = 1

    result = []
    while heap:
        key, i = heapq.heappop(heap)
        orig_idx = i_bufs[i][i_pos[i] - 1]
        result.append(int(orig_idx))

        kp = k_pos[i]
        if k_bufs[i] and kp < len(k_bufs[i]):
            heapq.heappush(heap, (k_bufs[i][kp], i))
            k_pos[i] += 1; i_pos[i] += 1
        else:
            new_k = refill_k(i); new_i = refill_i(i)
            if new_k:
                k_bufs[i] = new_k; i_bufs[i] = new_i
                k_pos[i]  = 1;     i_pos[i]  = 1
                heapq.heappush(heap, (new_k[0], i))
            else:
                k_bufs[i] = None

    for h in k_handles + i_handles:
        h.close()
    return result


def kway_merge_qwrite(page_paths: List[str]) -> list:
    if not page_paths:
        return []
    with open(page_paths[0], 'rb') as f:
        dtype, _, _ = _read_header(f)
    return numpy_concat_sort_pages(page_paths, dtype).tolist()


# ─────────────────────────────────────────────────────────────────────────────
# AUTHORIZATION
# ─────────────────────────────────────────────────────────────────────────────

def request_authorization(reason, page_size, n, high_performance_mode, silent):
    n_pages = max(1, (n + page_size - 1) // page_size)
    if silent or high_performance_mode:
        if not silent:
            print(f"\n  [Quill] External sort authorized (high_performance_mode=True)")
            print(f"  [Quill] {reason}")
            print(f"  [Quill] Pages: {n_pages}  |  Page size: {page_size:,} elements\n")
        return True
    print(f"\n  ┌─────────────────────────────────────────────────────────┐")
    print(f"  │                  QUILL — EXTERNAL SORT                  │")
    print(f"  └─────────────────────────────────────────────────────────┘")
    print(f"  Reason    : {reason}")
    print(f"  Strategy  : Split into {n_pages} pages of {page_size:,} elements each")
    print(f"  Temp files: {n_pages} × .qwrite files in system temp dir")
    print(f"  Cleanup   : Automatic (even on crash)")
    print()
    try:
        answer = input("  Authorize Quill external sort? (y/n): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n  [Quill] Authorization cancelled.")
        return False
    if answer in ('y', 'yes'):
        print(); return True
    print("  [Quill] External sort declined. Attempting in-memory sort.")
    return False


# ─────────────────────────────────────────────────────────────────────────────
# ASYNC PAGE WRITER
# ─────────────────────────────────────────────────────────────────────────────

class _AsyncWriter:
    def __init__(self, max_workers: int = 2):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures  = []

    def submit(self, path: str, arr, is_sorted: bool = True):
        future = self._executor.submit(write_qwrite, path, arr, is_sorted)
        self._futures.append(future)
        return future

    def wait_all(self):
        for f in self._futures:
            f.result()
        self._futures.clear()

    def shutdown(self):
        self._executor.shutdown(wait=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _is_identity(key_fn) -> bool:
    if key_fn is None:
        return True
    name = getattr(key_fn, '__name__', '')
    if name == '_identity':
        return True
    try:
        return key_fn(1) == 1 and key_fn("x") == "x"
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# EXTERNAL SORT — MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def external_sort(
    data                 : list,
    key_fn               : Callable,
    reverse              : bool,
    high_performance_mode: bool = False,
    silent               : bool = False,
) -> bool:

    n = len(data)
    needed, reason, page_size = should_use_external(data)
    if not needed:
        return False

    authorized = request_authorization(
        reason, page_size, n, high_performance_mode, silent
    )
    if not authorized:
        return False

    # Determine data type
    sample_key   = key_fn(data[0]) if data else 0
    is_int_key   = isinstance(sample_key, int)
    is_float_key = isinstance(sample_key, float)
    identity_key = _is_identity(key_fn)

    use_numpy_merge = _NUMPY and identity_key and (is_int_key or is_float_key)

    if is_int_key:
        keys_sample = [key_fn(x) for x in data[:min(1000, n)]]
        mn_s, mx_s  = min(keys_sample), max(keys_sample)
        if mn_s >= 0 and mx_s < 2**31:   np_dtype = np.int32
        elif mn_s >= -(2**31):            np_dtype = np.int64
        else:                             np_dtype = np.float64; is_int_key = False
    elif is_float_key:
        np_dtype = np.float64
    else:
        np_dtype = None
        use_numpy_merge = False

    # radix_partition_sort now handles negatives via global_min shift,
    # so the old `min(keys_sample) >= 0` gate is no longer needed.
    use_radix_part = (use_numpy_merge and is_int_key and
                      np_dtype in (np.int32, np.int64))

    tmp_dir   = tempfile.mkdtemp(prefix='quill_ext_')
    tmp_files : List[str] = []
    t_phase1  = time.perf_counter()

    try:
        if use_radix_part:
            bucket_paths = radix_partition_sort(
                data, np_dtype, tmp_dir, page_size, silent=silent
            )

            if bucket_paths is None:
                use_radix_part = False
            else:
                tmp_files.extend(bucket_paths)
                t_p1 = time.perf_counter() - t_phase1
                if not silent:
                    print(f"  [Quill] Phase 1 complete in {t_p1:.3f}s  "
                          f"({len(bucket_paths)} buckets)")

                t_phase2    = time.perf_counter()
                merged_path = os.path.join(tmp_dir, 'merged_output.qwrite')
                tmp_files.append(merged_path)

                if not silent:
                    print(f"  [Quill] Phase 2/2 — concatenating "
                          f"{len(bucket_paths)} sorted buckets (no merge)...")

                concat_bucket_files(bucket_paths, np_dtype, merged_path,
                                    silent=silent)

                merged = read_qwrite_full(merged_path)
                if reverse:
                    merged = merged[::-1]
                data[:] = merged
                del merged

                t_p2 = time.perf_counter() - t_phase2
                if not silent:
                    print(f"  [Quill] Phase 2 complete in {t_p2:.3f}s")
                    print(f"  [Quill] Total external sort: {t_p1+t_p2:.3f}s\n")

                return True

        # ── SORTED-PAGES FALLBACK ─────────────────────────────────────────────
        n_pages = (n + page_size - 1) // page_size
        if not silent:
            print(f"  [Quill] Phase 1/2 — Sorting and writing "
                  f"{n_pages} pages (async I/O, optimal sort kind)...")

        writer = _AsyncWriter(max_workers=2)
        # Precompute optimal sort kind for this dtype
        _dtype_obj   = np.dtype(np_dtype) if np_dtype is not None else None
        _use_stable  = (_dtype_obj is not None and
                        _dtype_obj.kind == 'u' and _dtype_obj.itemsize <= 2)

        if use_numpy_merge and np_dtype is not None:
            for page_idx, start in enumerate(range(0, n, page_size)):
                end      = min(start + page_size, n)
                page_arr = np.array(data[start:end], dtype=np_dtype)
                if _use_stable:
                    page_arr.sort(kind='stable')
                else:
                    page_arr.sort()   # quicksort — 20x faster for int32
                page_path = os.path.join(tmp_dir, f'page_{page_idx:06d}.qwrite')
                writer.submit(page_path, page_arr, is_sorted=True)
                tmp_files.append(page_path)
            writer.wait_all()

        elif np_dtype is not None:
            all_keys = np.array([key_fn(x) for x in data], dtype=np_dtype)
            for page_idx, start in enumerate(range(0, n, page_size)):
                end         = min(start + page_size, n)
                page_keys   = all_keys[start:end].copy()
                page_order  = np.argsort(page_keys, kind='stable')
                sorted_keys = page_keys[page_order]
                key_path    = os.path.join(tmp_dir, f'page_{page_idx:06d}.qwrite')
                idx_path    = os.path.join(tmp_dir, f'idx_{page_idx:06d}.qwrite')
                orig_idx    = (page_order + start).astype(np.int64)
                writer.submit(key_path, sorted_keys, is_sorted=True)
                writer.submit(idx_path, orig_idx,    is_sorted=False)
                tmp_files.extend([key_path, idx_path])
            writer.wait_all()

        else:
            for page_idx, start in enumerate(range(0, n, page_size)):
                end   = min(start + page_size, n)
                chunk = list(range(start, end))
                chunk.sort(key=lambda i: key_fn(data[i]))
                key_arr  = np.array([key_fn(data[i]) for i in chunk], dtype=np.float64)
                idx_arr  = np.array(chunk, dtype=np.int64)
                key_path = os.path.join(tmp_dir, f'page_{page_idx:06d}.qwrite')
                idx_path = os.path.join(tmp_dir, f'idx_{page_idx:06d}.qwrite')
                writer.submit(key_path, key_arr, is_sorted=True)
                writer.submit(idx_path, idx_arr, is_sorted=False)
                tmp_files.extend([key_path, idx_path])
            writer.wait_all()

        writer.shutdown()

        t_p1 = time.perf_counter() - t_phase1
        if not silent:
            print(f"  [Quill] Phase 1 complete in {t_p1:.3f}s ({n_pages} pages)")

        t_phase2 = time.perf_counter()

        if use_numpy_merge:
            page_paths    = sorted(p for p in tmp_files if 'page_' in os.path.basename(p))
            dataset_bytes = n * np.dtype(np_dtype).itemsize
            available_now = get_available_ram()
            fits_in_ram   = dataset_bytes < available_now * 0.80

            if fits_in_ram:
                if not silent:
                    print(f"  [Quill] Phase 2/2 — numpy concat+sort "
                          f"({n_pages} pages)...")
                merged = numpy_concat_sort_pages(page_paths, np_dtype, silent=silent)
                if reverse:
                    merged = merged[::-1]
                data[:] = merged
                del merged
            else:
                buf_budget = int(available_now * 0.20)
                buf_elts   = max(STREAM_BUF_ELTS,
                                 min(STREAM_BUF_ELTS * 4,
                                     buf_budget // (n_pages * np.dtype(np_dtype).itemsize)))
                if not silent:
                    print(f"  [Quill] Phase 2/2 — streaming k-way merge "
                          f"({n_pages} pages, overlapped I/O)...")
                merged_path = os.path.join(tmp_dir, 'merged_output.qwrite')
                tmp_files.append(merged_path)
                numpy_kway_streaming_merge(
                    page_paths, np_dtype, merged_path,
                    buf_elts=buf_elts, silent=silent,
                )
                merged = read_qwrite_full(merged_path)
                if reverse:
                    merged = merged[::-1]
                data[:] = merged
                del merged
        else:
            if not silent:
                print(f"  [Quill] Phase 2/2 — k-way heap merge ({n_pages} pages)...")
            page_paths = sorted(p for p in tmp_files if 'page_' in os.path.basename(p))
            idx_paths  = sorted(p for p in tmp_files if 'idx_'  in os.path.basename(p))
            sorted_indices = kway_merge_with_indices(page_paths, idx_paths)
            data[:] = [data[i] for i in sorted_indices]
            if reverse:
                data.reverse()

        t_p2 = time.perf_counter() - t_phase2
        if not silent:
            print(f"  [Quill] Phase 2 complete in {t_p2:.3f}s")
            print(f"  [Quill] Total external sort: {t_p1+t_p2:.3f}s\n")

        return True

    finally:
        for path in tmp_files:
            try: os.unlink(path)
            except OSError: pass
        try: os.rmdir(tmp_dir)
        except OSError: pass
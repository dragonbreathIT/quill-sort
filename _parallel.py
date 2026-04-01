"""
quill/_parallel.py
------------------
Parallel sort engine — shared-memory MSD radix sort.

ARCHITECTURE
────────────
For integer data (the common case), Quill uses a parallel MSD radix sort
implemented entirely via shared memory + numpy.  No pickling of large arrays.

  Phase 1 — PARALLEL LOCAL HISTOGRAMS
    P workers each count their stripe of the input independently.
    All writes hit private stack arrays → zero false sharing.

  Phase 2 — PREFIX SUM
    O(256 × P) scan in the main process. Determines each worker's write
    offset into each output bucket.

  Phase 3 — PARALLEL SCATTER
    Each worker writes its elements to the correct bucket position using
    argsort + split (vectorised, no Python loop over 256 buckets).

  Phase 4 — PARALLEL BUCKET SORT
    Each bucket is sorted independently. Small buckets → ThreadPoolExecutor
    (numpy releases GIL). Large buckets → ProcessPoolExecutor.

For non-integer/non-numpy data, falls back to pool.map with keyed sort
and k-way heap merge.

PERFORMANCE vs OLD APPROACH
    Old: pool.map over P chunks → sort each → heap merge
         Bottleneck: heap merge is O(n log P), heap pops in Python
    New: parallel histogram → prefix sum → parallel scatter → parallel sort
         All hot loops stay in C/numpy. Merge = O(1) (buckets in order).
"""

from __future__ import annotations

import math
import heapq
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing.shared_memory import SharedMemory
from typing import Callable, Optional

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

RADIX_BITS    = 8
RADIX_BUCKETS = 1 << RADIX_BITS   # 256

# Below this size per bucket, use threads instead of processes
THREAD_SORT_THRESHOLD = 500_000

# Heavy-key fast path: detect via sample, then bypass worker partition/sort.
HEAVY_KEY_SAMPLE_SIZE  = 1_000
HEAVY_KEY_SAMPLE_RATIO = 0.20


def _detect_heavy_keys(values_i64: np.ndarray) -> np.ndarray:
    """
    Scout phase: sample up to 1,000 random values and flag any key whose
    sample frequency exceeds 20%.
    """
    n = int(values_i64.size)
    if n == 0:
        return np.empty(0, dtype=np.int64)

    sample_n = min(HEAVY_KEY_SAMPLE_SIZE, n)
    if sample_n == n:
        sample = values_i64
    else:
        rng = np.random.default_rng()
        idx = rng.choice(n, size=sample_n, replace=False)
        sample = values_i64[idx]

    uniq, counts = np.unique(sample, return_counts=True)
    heavy = uniq[counts > (sample_n * HEAVY_KEY_SAMPLE_RATIO)]
    return heavy.astype(np.int64, copy=False)


def _reinsert_heavy_keys(
    sorted_light_i64: np.ndarray,
    heavy_counts: dict[int, int],
    total_n: int,
) -> np.ndarray:
    """
    Merge pre-counted heavy keys back into the fully sorted non-heavy stream.
    """
    if not heavy_counts:
        return sorted_light_i64

    merged = np.empty(total_n, dtype=np.int64)
    src = 0
    dst = 0

    for key, count in sorted(heavy_counts.items()):
        insert_at = int(np.searchsorted(sorted_light_i64, key, side='left'))
        run = insert_at - src
        if run > 0:
            merged[dst:dst + run] = sorted_light_i64[src:insert_at]
            dst += run
            src = insert_at
        merged[dst:dst + count] = key
        dst += count

    if src < len(sorted_light_i64):
        merged[dst:] = sorted_light_i64[src:]

    return merged


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL WORKER FUNCTIONS (must be picklable on Windows/spawn)
# ─────────────────────────────────────────────────────────────────────────────

def _worker_count_histogram(args):
    """
    Count local histogram for a stripe of shared input array.
    Returns histogram as bytes (int64 × 256).
    """
    shm_name, dtype_str, start, end, global_min, bit_shift = args
    shm   = SharedMemory(name=shm_name)
    dtype = np.dtype(dtype_str)
    arr   = np.ndarray((end - start,), dtype=dtype, buffer=shm.buf,
                       offset=start * dtype.itemsize)

    shifted = ((arr.astype(np.int64) - global_min) >> bit_shift)
    np.clip(shifted, 0, RADIX_BUCKETS - 1, out=shifted)
    hist    = np.bincount(shifted.astype(np.intp), minlength=RADIX_BUCKETS)
    shm.close()
    return hist.astype(np.int64).tobytes()


def _worker_scatter(args):
    """
    Scatter elements from input stripe to output positions.
    Uses vectorised argsort+split to avoid per-element Python work.
    """
    shm_in_name, shm_out_name, dtype_str, start, end, \
        global_min, bit_shift, worker_offsets_bytes = args

    shm_in  = SharedMemory(name=shm_in_name)
    shm_out = SharedMemory(name=shm_out_name)
    dtype   = np.dtype(dtype_str)

    inp  = np.ndarray((end - start,), dtype=dtype, buffer=shm_in.buf,
                      offset=start * dtype.itemsize)
    out  = np.ndarray((shm_out.size // dtype.itemsize,), dtype=dtype,
                      buffer=shm_out.buf)

    offsets = np.frombuffer(worker_offsets_bytes, dtype=np.int64).copy()
    keys    = ((inp.astype(np.int64) - global_min) >> bit_shift)
    np.clip(keys, 0, RADIX_BUCKETS - 1, out=keys)

    # Vectorised scatter: sort by bucket key, then copy each contiguous run
    order        = np.argsort(keys.astype(np.intp), kind='stable')
    sorted_elems = inp[order]
    sorted_keys  = keys[order]

    boundaries = np.flatnonzero(np.diff(sorted_keys)) + 1
    splits     = np.split(sorted_elems, boundaries)
    bucket_ids = sorted_keys[np.concatenate([[0], boundaries])]

    for chunk, b in zip(splits, bucket_ids.tolist()):
        b   = int(b)
        dst = int(offsets[b])
        cnt = len(chunk)
        out[dst:dst + cnt] = chunk
        offsets[b] += cnt

    shm_in.close()
    shm_out.close()


def _worker_sort_shm_slice(args):
    """Sort a slice of shared memory in-place (large bucket).
    Uses optimal sort kind per dtype — benchmark-proven:
    uint8/uint16 → stable (radix), int32/int64 → default (quicksort)."""
    shm_name, dtype_str, start, end = args
    shm   = SharedMemory(name=shm_name)
    dtype = np.dtype(dtype_str)
    view  = np.ndarray((end - start,), dtype=dtype, buffer=shm.buf,
                       offset=start * dtype.itemsize)
    # uint8/uint16: stable is 10-17x faster (1-2 pass radix)
    # int32/int64: default quicksort is 20x faster than stable (4+ pass radix)
    if dtype.kind == 'u' and dtype.itemsize <= 2:
        view.sort(kind='stable')
    else:
        view.sort()  # default = quicksort/introsort
    shm.close()


def _worker_keyed_sort(args):
    chunk, keys = args
    return [x for _, __, x in sorted(zip(keys, range(len(chunk)), chunk))]


# ─────────────────────────────────────────────────────────────────────────────
# K-WAY HEAP MERGE (generic fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _kway_merge(chunks: list, key_fn) -> list:
    heap  = []
    iters = [iter(c) for c in chunks]
    for i, it in enumerate(iters):
        v = next(it, None)
        if v is not None:
            heapq.heappush(heap, (key_fn(v), i, v, it))
    result = []
    while heap:
        k, i, v, it = heapq.heappop(heap)
        result.append(v)
        nxt = next(it, None)
        if nxt is not None:
            heapq.heappush(heap, (key_fn(nxt), i, nxt, it))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

def parallel_sort(
    data: list, key_fn, profile: dict, identity_key: bool = False
) -> None:
    """Sort `data` in-place using all available CPU cores."""
    ncores = mp.cpu_count() or 4

    is_int = profile.get("dtype", "") in ("int_pos", "int_neg", "int_mixed")

    if is_int and _NUMPY and identity_key:
        _parallel_msd_radix(data, profile, ncores)
        return

    _parallel_generic(data, key_fn, ncores)


# ─────────────────────────────────────────────────────────────────────────────
# PARALLEL MSD RADIX SORT (integer, shared memory)
# ─────────────────────────────────────────────────────────────────────────────

def _parallel_msd_radix(data: list, profile: dict, ncores: int) -> None:
    """
    Parallel MSD radix sort via shared memory.

    Steps:
      1. Convert list → narrowest-dtype numpy array → shared memory
      2. P workers each count their stripe → local histograms
      3. Prefix sum → per-worker write offsets
      4. P workers scatter elements to output shared memory
      5. Sort each bucket in parallel (threads for small, processes for large)
      6. Write result back to data list
    """
    n_total = len(data)
    arr_i64 = np.array(data, dtype=np.int64)

    # Scout phase: detect heavy keys from a tiny random sample.
    sampled_heavy = _detect_heavy_keys(arr_i64)
    heavy_counts: dict[int, int] = {}
    light_i64 = arr_i64
    if sampled_heavy.size:
        heavy_mask = np.isin(arr_i64, sampled_heavy)
        if heavy_mask.any():
            heavy_vals = arr_i64[heavy_mask]
            uniq, counts = np.unique(heavy_vals, return_counts=True)
            heavy_counts = {
                int(k): int(c)
                for k, c in zip(uniq.tolist(), counts.tolist())
                if c > 0
            }
            light_i64 = arr_i64[~heavy_mask]

    n = int(light_i64.size)
    if n == 0:
        # All elements were heavy keys; no worker pass needed.
        result = _reinsert_heavy_keys(np.empty(0, dtype=np.int64), heavy_counts, n_total)
        data[:] = result.tolist()
        return

    mn = int(light_i64.min())
    mx = int(light_i64.max())

    # Select narrowest dtype for non-heavy values only.
    rng = mx - mn
    if mn >= 0:
        if rng < 256:        dtype = np.uint8;  shift = 0
        elif rng < 65_536:   dtype = np.uint16; shift = 0
        elif rng < 2**31:    dtype = np.int32;  shift = 0
        else:                dtype = np.int64;  shift = 0
    else:
        shift = -mn
        rng   = mx + shift
        if rng < 256:        dtype = np.uint8
        elif rng < 65_536:   dtype = np.uint16
        elif rng < 2**31:    dtype = np.int32
        else:                dtype = np.int64; shift = 0

    dtype = np.dtype(dtype)
    dtype_str = dtype.str
    global_min = mn
    val_range  = mx - mn + 1
    bit_len    = max(1, val_range.bit_length())
    bit_shift  = max(0, bit_len - RADIX_BITS)

    # Build compact worker input (heavy keys excluded).
    if shift:
        arr = (light_i64 + shift).astype(dtype)
    else:
        arr = light_i64.astype(dtype)

    shm_in = SharedMemory(create=True, size=arr.nbytes)
    try:
        inp_shm = np.ndarray(arr.shape, dtype=dtype, buffer=shm_in.buf)
        inp_shm[:] = arr
        del arr

        # Phase A: parallel local histograms
        stripe     = math.ceil(n / ncores)
        boundaries = [(i * stripe, min((i + 1) * stripe, n))
                      for i in range(ncores) if i * stripe < n]
        n_workers  = len(boundaries)

        count_args = [
            (shm_in.name, dtype_str, s, e, global_min if shift == 0 else 0, bit_shift)
            for s, e in boundaries
        ]

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            hist_bytes = list(pool.map(_worker_count_histogram, count_args))

        local_hists = np.array(
            [np.frombuffer(h, dtype=np.int64) for h in hist_bytes]
        )  # (n_workers, RADIX_BUCKETS)

        global_hist  = local_hists.sum(axis=0)
        bucket_start = np.zeros(RADIX_BUCKETS + 1, dtype=np.int64)
        bucket_start[1:] = np.cumsum(global_hist)

        # Per-worker write offsets: worker w starts at bucket_start[b] +
        # sum of local_hists[0..w-1, b]
        cum_local      = np.vstack([[0] * RADIX_BUCKETS, local_hists[:-1]])
        worker_offsets = bucket_start[:RADIX_BUCKETS] + np.cumsum(cum_local, axis=0)

        # Phase B: parallel scatter to output shared memory
        shm_out = SharedMemory(create=True, size=inp_shm.nbytes)
        try:
            scatter_args = [
                (shm_in.name, shm_out.name, dtype_str,
                 boundaries[i][0], boundaries[i][1],
                 global_min if shift == 0 else 0, bit_shift,
                 worker_offsets[i].tobytes())
                for i in range(n_workers)
            ]
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                list(pool.map(_worker_scatter, scatter_args))

            out_arr = np.ndarray(inp_shm.shape, dtype=dtype,
                                 buffer=shm_out.buf).copy()

        finally:
            shm_out.close()
            shm_out.unlink()

    finally:
        shm_in.close()
        shm_in.unlink()

    # Phase C: sort each bucket in parallel
    bucket_slices = [
        (int(bucket_start[b]), int(bucket_start[b + 1]))
        for b in range(RADIX_BUCKETS)
        if bucket_start[b + 1] > bucket_start[b]
    ]

    large = [(s, e) for s, e in bucket_slices if (e - s) >= THREAD_SORT_THRESHOLD]
    small = [(s, e) for s, e in bucket_slices if (e - s) <  THREAD_SORT_THRESHOLD]

    # Small buckets: sort in threads (numpy releases GIL).
    # Use optimal kind per dtype.
    _use_stable = (dtype.kind == 'u' and dtype.itemsize <= 2)

    def _sort_small(se):
        s, e = se
        if _use_stable:
            out_arr[s:e].sort(kind='stable')
        else:
            out_arr[s:e].sort()  # default = quicksort

    with ThreadPoolExecutor(max_workers=ncores) as pool:
        list(pool.map(_sort_small, small))

    # Large buckets: sort in processes via shared memory
    if large:
        shm_sort = SharedMemory(create=True, size=out_arr.nbytes)
        try:
            sort_shm = np.ndarray(out_arr.shape, dtype=dtype, buffer=shm_sort.buf)
            sort_shm[:] = out_arr

            large_args = [(shm_sort.name, dtype_str, s, e) for s, e in large]
            with ProcessPoolExecutor(max_workers=min(ncores, len(large))) as pool:
                list(pool.map(_worker_sort_shm_slice, large_args))

            out_arr[:] = sort_shm

        finally:
            shm_sort.close()
            shm_sort.unlink()

    # Write back to data list — convert dtype back, then reinsert heavy keys.
    if shift:
        sorted_light = out_arr.astype(np.int64) - shift
    else:
        sorted_light = out_arr.astype(np.int64)

    result = _reinsert_heavy_keys(sorted_light, heavy_counts, n_total)
    data[:] = result.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# GENERIC PARALLEL SORT (non-integer or no numpy)
# ─────────────────────────────────────────────────────────────────────────────

def _parallel_generic(data: list, key_fn: Callable, ncores: int) -> None:
    n       = len(data)
    chunk_n = max(10_000, math.ceil(n / ncores))
    chunks  = [data[i:i + chunk_n] for i in range(0, n, chunk_n)]
    nw      = min(ncores, len(chunks))

    key_chunks = [[key_fn(x) for x in c] for c in chunks]
    args       = list(zip(chunks, key_chunks))

    with mp.Pool(nw) as pool:
        sorted_chunks = pool.map(_worker_keyed_sort, args)

    data[:] = _kway_merge(sorted_chunks, key_fn)
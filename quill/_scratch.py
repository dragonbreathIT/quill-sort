"""
quill/_scratch.py
-----------------
Reusable numpy scratch buffers.

A surprising fraction of small-array sort latency is spent in ``np.empty`` /
``np.empty_like``: the counting-sort histogram, the parallel-partition output
buffer, and the k-way merge output are all allocated, written once, and then
freed. In a hot loop (e.g. a benchmark that calls ``quill_sort`` 10k times on
arrays of size 5_000) that allocator churn is measurably 10-15% of wall time.

This module hands out *recycled* numpy buffers via a context manager:

    with borrow(np.int64, n) as h:
        h.buf[:] = ...   # h.buf is an ndarray of dtype, .size == n

On context exit, the buffer goes back into the pool. The next caller of the
same dtype and a "compatible" size gets the same memory back — no allocation,
no zero-fill, no page faults.

Design notes
~~~~~~~~~~~~
* Thread-local storage. Sort hot paths shouldn't take a mutex on every borrow;
  using ``threading.local`` lets each worker thread keep its own pool. Worst
  case (a thread that sorts once and dies) we leak a few buffers, which Python
  will collect when the ``local`` object dies with the thread.
* Power-of-two size buckets. 1.2M-element and 1.7M-element borrows both land
  in the 2M bucket, so the same buffer can be reused without an exact size
  match. The bucket cap (``2 * min_size`` when ``min_size`` isn't a power of
  two) keeps a 1.0001M borrow from grabbing a 2M slab.
* Per-bucket FIFO with a hard cap (``max_buffers_per_bucket``, default 4;
  internal hard ceiling of 8). When a bucket is full we drop the *oldest*
  buffer so it can be GC'd — bounded memory.
* Idempotent release. Releasing a handle twice (or after a ``reset()``) is a
  no-op so callers can defensively call ``release()`` and still use ``with``.

Numpy-free fallback
~~~~~~~~~~~~~~~~~~~
If numpy isn't installed, ``ScratchPool`` becomes a degenerate no-op pool:
``borrow`` returns a handle whose ``.buf`` is a fresh ``array.array``-shaped
stand-in. Callers in a numpy-free environment shouldn't be calling us anyway
(the kernels that use scratch all require numpy), but graceful degradation
keeps imports from blowing up on numpy-free installs.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False
    np = None  # type: ignore[assignment]

logger = logging.getLogger("quill")

# Hard ceiling on per-bucket retention. Even if a caller passes
# max_buffers_per_bucket=999, we cap to this to bound memory: a runaway pool
# defeats the point of pooling.
_BUCKET_HARD_CAP = 8


def _next_power_of_2(n: int) -> int:
    """Smallest power of 2 >= n. ``_next_power_of_2(0) == 1``."""
    if n <= 1:
        return 1
    # bit_length: 1 → 1, 2 → 2, 3 → 2 (so we need -1 first), 4 → 3, ...
    return 1 << (n - 1).bit_length()


def _bucket_size(min_size: int) -> int:
    """Choose the actual allocation size for a ``min_size`` request.

    Round up to a power of two — but never more than ``2 * min_size`` for
    requests that aren't already powers of two. This caps the "wasted slack"
    at <2x while still letting nearby request sizes share a bucket.
    """
    if min_size <= 1:
        return 1
    pow2 = _next_power_of_2(min_size)
    if pow2 == min_size:
        return min_size  # exact power of two — no slack needed
    cap = 2 * min_size
    return min(pow2, cap)


# ─────────────────────────────────────────────────────────────────────────────
# HANDLE — what borrow() returns; context-managed.
# ─────────────────────────────────────────────────────────────────────────────


class _ScratchHandle:
    """Context manager wrapping a borrowed buffer.

    ``handle.buf`` is the user-facing view (truncated to the requested size).
    ``handle._backing`` is the underlying full-bucket array we return to the
    pool on release. Keeping them separate means the user can ``handle.buf[:]
    = source`` without worrying about the bucket's extra slack tail.
    """

    __slots__ = ("buf", "_backing", "_pool", "_bucket_key", "_released")

    def __init__(self, buf: Any, backing: Any, pool: Optional["ScratchPool"],
                 bucket_key: Optional[tuple]) -> None:
        self.buf = buf
        self._backing = backing
        self._pool = pool
        self._bucket_key = bucket_key
        self._released = False

    def __enter__(self) -> "_ScratchHandle":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def release(self) -> None:
        """Return the buffer to the pool. Safe to call multiple times."""
        if self._released:
            return
        self._released = True
        pool = self._pool
        if pool is None or self._backing is None or self._bucket_key is None:
            return
        # Drop the user-facing view so callers can't keep writing into a
        # buffer they no longer own — accidental sharing is a vicious bug.
        self.buf = None  # type: ignore[assignment]
        pool._return(self._bucket_key, self._backing)
        self._backing = None
        self._pool = None
        self._bucket_key = None


# ─────────────────────────────────────────────────────────────────────────────
# POOL
# ─────────────────────────────────────────────────────────────────────────────


class ScratchPool:
    """Thread-local pool of reusable numpy buffers.

    Per-thread state means borrow/release never need a lock. The trade-off is
    that buffers don't migrate between threads — fine for our use because each
    sort call typically borrows from the calling thread.
    """

    def __init__(self, max_buffers_per_bucket: int = 4) -> None:
        self.max_per_bucket = min(max(1, max_buffers_per_bucket),
                                  _BUCKET_HARD_CAP)
        # Per-thread storage. Each thread gets its own dict-of-buckets, hit
        # count, and miss count. ``threading.local()`` initialises attributes
        # lazily per-thread; we set them up in ``_local_state``.
        self._tls = threading.local()

    # ── per-thread accessors ────────────────────────────────────────────────

    def _local_state(self) -> Dict[str, Any]:
        st = getattr(self._tls, "state", None)
        if st is None:
            st = {"buckets": {}, "hits": 0, "misses": 0}
            self._tls.state = st
        return st

    # ── public API ──────────────────────────────────────────────────────────

    def borrow(self, dtype: Any, min_size: int) -> _ScratchHandle:
        """Borrow a buffer of ``dtype`` with at least ``min_size`` elements.

        Returns a context manager. Inside the ``with`` block, ``handle.buf``
        is an ndarray of exactly ``min_size`` elements (a view of a possibly-
        larger backing buffer); on exit, the backing buffer is returned to
        the pool for reuse.
        """
        if min_size < 0:
            min_size = 0
        if not _NUMPY:
            # Degenerate fallback: no pooling, but a valid handle so callers
            # don't need to branch. Bare ``object()`` would surprise them.
            return _ScratchHandle(buf=_FakeBuffer(min_size),
                                  backing=None, pool=None, bucket_key=None)

        dt = np.dtype(dtype)
        bucket = _bucket_size(min_size)
        key = (dt.str, bucket)  # dtype.str is hashable & stable
        state = self._local_state()
        buckets: Dict[tuple, List[Any]] = state["buckets"]
        stack = buckets.get(key)
        if stack:
            backing = stack.pop()  # LIFO inside a bucket: hottest cache line
            state["hits"] += 1
        else:
            backing = np.empty(bucket, dtype=dt)
            state["misses"] += 1
        # View truncated to the user's requested length. ``backing[:min_size]``
        # shares memory; writes via ``view`` reach ``backing``.
        view = backing[:min_size] if min_size < bucket else backing
        return _ScratchHandle(buf=view, backing=backing,
                              pool=self, bucket_key=key)

    def stats(self) -> Dict[str, Any]:
        """Pool statistics for the *current thread*.

        Per-thread by design — aggregating across threads would require a
        lock, defeating the point. Call from each worker if you need a
        cross-thread view.
        """
        st = self._local_state()
        current = sum(len(v) for v in st["buckets"].values())
        return {
            "hits": st["hits"],
            "misses": st["misses"],
            "current_buffers": current,
        }

    def reset(self) -> None:
        """Drop all pooled buffers on the *current thread*.

        Other threads' caches are untouched (we don't hold references to
        them). The GC will reclaim the dropped arrays when nothing else
        references them.
        """
        st = self._local_state()
        st["buckets"].clear()
        st["hits"] = 0
        st["misses"] = 0

    # ── internal: called from _ScratchHandle.release() ─────────────────────

    def _return(self, bucket_key: tuple, backing: Any) -> None:
        if backing is None:
            return
        try:
            state = self._local_state()
            buckets: Dict[tuple, List[Any]] = state["buckets"]
            stack = buckets.get(bucket_key)
            if stack is None:
                stack = []
                buckets[bucket_key] = stack
            if len(stack) >= self.max_per_bucket:
                # Drop the oldest (FIFO eviction on overflow). pop(0) is O(n)
                # in list size but n <= _BUCKET_HARD_CAP == 8, so trivial.
                stack.pop(0)
            stack.append(backing)
        except BaseException:  # noqa: BLE001
            # Pool maintenance must NEVER raise into a sort path — losing a
            # buffer is fine, breaking the caller is not. Swallow and move on.
            logger.debug("ScratchPool._return failed", exc_info=True)


# ─────────────────────────────────────────────────────────────────────────────
# NUMPY-FREE FALLBACK BUFFER
# ─────────────────────────────────────────────────────────────────────────────


class _FakeBuffer:
    """Minimal stand-in for an ndarray when numpy is unavailable.

    Real callers of ``borrow`` all live behind ``if _NUMPY:`` guards, so this
    only exists so that ``borrow()`` returns something usable enough not to
    crash on attribute access in a pure-Python codepath.
    """

    __slots__ = ("_data", "size")

    def __init__(self, size: int) -> None:
        self._data = [0] * size
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx):
        return self._data[idx]

    def __setitem__(self, idx, val) -> None:
        self._data[idx] = val


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL DEFAULT POOL
# ─────────────────────────────────────────────────────────────────────────────


POOL = ScratchPool()


def borrow(dtype: Any, min_size: int) -> _ScratchHandle:
    """Borrow from the module-level default pool. See ``ScratchPool.borrow``."""
    return POOL.borrow(dtype, min_size)


__all__ = ["ScratchPool", "POOL", "borrow"]

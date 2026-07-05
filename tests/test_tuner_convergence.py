"""
Regression tests for the self-tuning dispatcher's *convergence* guarantee.

Motivation
==========
The explore-then-exploit tuner (``quill._tuning.TimingDB.choose``) must, on ANY
hardware, converge to a backend within a small factor of the best available and
never *sustain* a large loss. The failure this guards against was real: on
low-core machines every compiled/parallel backend loses to numpy's
single-threaded AVX introsort, and before ``np.sort`` was made a first-class
tuning candidate the chooser optimised over a set whose members all lose to the
floor — so it round-robined losers and settled (if at all) on a backend running
20–80% (transiently up to ~5x) slower than a bare ``np.sort``, silently breaking
the "never meaningfully slower" guarantee on a large slice of real deployments.

These tests are deterministic (they drive the tuner with synthetic per-backend
latencies, so they pass identically on a 1-core CI box or a 128-core server) and
they exercise the *real* backend registry, so removing the numpy floor from the
candidate set — the exact bug — makes them fail.
"""
from __future__ import annotations

import os

import pytest

np = pytest.importorskip("numpy")

from quill import _backends as B
from quill._tuning import TimingDB, _DEFAULT_MIN_OBS

KIND = "i"
N = 10_000_000

# Backends that all run numpy's OWN sort kernel: the bare ``numpy`` floor, and
# the SIMD backends that delegate into numpy's bundled x86-simd-sort (AVX-512) /
# ASIMD kernel. On a SIMD-capable host these are performance-equivalent, so a
# convergence check must accept the whole set — asserting the exact string
# ``"numpy"`` false-fails on every AVX-512 / Apple-silicon runner (which is most
# of them). Gate on the *ratio* to np.sort, or on membership in this set.
_NUMPY_FAMILY = frozenset({"numpy", "x86_simd_sort", "arm_neon_sort"})
# Upper bound on the exploration window: every candidate may be probed up to
# min_obs times before the tuner is allowed to have converged.
def _warmup_bound(n_candidates: int) -> int:
    return 2 * n_candidates * _DEFAULT_MIN_OBS + 5


def _drive(tmp_path, latency, iters):
    """Run ``choose`` + ``record`` for *iters* rounds against a fixed per-backend
    latency oracle. Returns (picks, per_round_latency)."""
    db = TimingDB(path=str(tmp_path / "timings.json"))
    names = list(latency)  # already in "priority order"
    picks, lats = [], []
    for _ in range(iters):
        p = db.choose(names, KIND, N)
        assert p in latency, f"chooser returned unknown backend {p!r}"
        db.record(p, KIND, N, latency[p])
        picks.append(p)
        lats.append(latency[p])
    return picks, lats


# Synthetic "machines": name -> seconds for one sort. The numpy floor is always
# present and always last, mirroring the real registry order.
FLOOR = 0.100  # 100 ms baseline
MACHINES = {
    # single-core: every accelerator loses badly to the floor (the bug's habitat)
    "one_core": {
        "ips4o": 5.0 * FLOOR, "rust_parallel_radix": 5.0 * FLOOR,
        "rust_voracious": 4.0 * FLOOR, "simd_companion": 1.4 * FLOOR,
        "x86_simd_sort": 1.3 * FLOOR, "numpy": FLOOR,
    },
    # many-core: a parallel backend wins by ~3x; the floor loses
    "many_core": {
        "ips4o": FLOOR / 3.1, "rust_parallel_radix": FLOOR / 2.9,
        "rust_voracious": FLOOR / 3.0, "simd_companion": FLOOR / 1.4,
        "x86_simd_sort": FLOOR / 1.3, "numpy": FLOOR,
    },
    # mid: a modest backend wins, floor and heavy-parallel both mediocre
    "mid_range": {
        "ips4o": 1.2 * FLOOR, "rust_parallel_radix": 1.5 * FLOOR,
        "simd_companion": 0.8 * FLOOR, "x86_simd_sort": 0.85 * FLOOR,
        "numpy": FLOOR,
    },
    # near-tie: a SIMD-delegating backend measures marginally faster than the
    # bare floor (jitter — they run the same kernel). The tie-break must pin to
    # the floor rather than the wrapper.
    "avx512_tie": {
        "ips4o": 4.0 * FLOOR, "rust_parallel_radix": 4.0 * FLOOR,
        "x86_simd_sort": 0.95 * FLOOR, "numpy": FLOOR,
    },
}


@pytest.mark.parametrize("machine", list(MACHINES))
def test_converges_within_1_1x_of_best(tmp_path, machine):
    latency = MACHINES[machine]
    best = min(latency.values())
    iters = _warmup_bound(len(latency)) + 40
    picks, lats = _drive(tmp_path, latency, iters)

    # Steady state = the tail after the exploration window.
    tail = lats[_warmup_bound(len(latency)):]
    assert tail, "no steady-state samples collected"
    # Converged to within 1.1x of the genuinely-best backend.
    assert max(tail) <= 1.1 * best, (
        f"[{machine}] did not converge within 1.1x of best "
        f"({max(tail)/best:.2f}x); tail picks={picks[-10:]}"
    )


@pytest.mark.parametrize("machine,expected", [
    ("one_core", "numpy"),          # floor MUST win here — this is the bug's home
    ("many_core", "ips4o"),         # fastest accelerator (FLOOR/3.1) must win
])
def test_converges_to_expected_backend(tmp_path, machine, expected):
    latency = MACHINES[machine]
    true_best = min(latency, key=latency.get)
    iters = _warmup_bound(len(latency)) + 40
    picks, _ = _drive(tmp_path, latency, iters)
    tail = picks[-15:]
    # every steady-state pick is the fastest backend
    assert set(tail) == {true_best}, f"[{machine}] converged to {set(tail)}, want {true_best}"
    assert true_best == expected


@pytest.mark.parametrize("machine", list(MACHINES))
def test_never_sustains_2x_loss_after_warmup(tmp_path, machine):
    latency = MACHINES[machine]
    best = min(latency.values())
    iters = _warmup_bound(len(latency)) + 60
    _, lats = _drive(tmp_path, latency, iters)
    tail = lats[_warmup_bound(len(latency)):]
    # No sustained (2+ consecutive) >2x-of-best sorts once warmed up.
    run = 0
    for x in tail:
        run = run + 1 if x > 2.0 * best else 0
        assert run < 2, f"[{machine}] sustained >2x loss in steady state"


def test_tiebreak_prefers_floor_over_simd_wrapper(tmp_path):
    """When the bare np.sort floor is within the tie margin of a marginally-
    faster candidate, the tuner must pin to the floor — the SIMD-delegating
    backend runs the same numpy kernel, so preferring the lower-overhead floor
    is the deterministic, cheapest choice (and stops noise-driven oscillation)."""
    latency = MACHINES["avx512_tie"]  # x86_simd_sort 0.95x, numpy 1.0x
    iters = _warmup_bound(len(latency)) + 40
    picks, _ = _drive(tmp_path, latency, iters)
    tail = picks[-15:]
    assert set(tail) == {"numpy"}, (
        f"tie-break did not pin to the floor; converged to {set(tail)}"
    )


def test_exploration_tax_is_bounded(tmp_path):
    """Even in the worst case (everything loses to the floor), the number of
    catastrophically-slow sorts is bounded by early-abandonment, not min_obs per
    backend."""
    latency = MACHINES["one_core"]
    best = min(latency.values())
    _, lats = _drive(tmp_path, latency, 200)
    catastrophic = sum(1 for x in lats if x > 2.0 * best)
    # Without abandonment this would be ~min_obs per losing backend and growing;
    # bounded exploration keeps it to a small one-time cost.
    assert catastrophic <= 3 * len(latency), (
        f"exploration tax unbounded: {catastrophic} catastrophic sorts"
    )


def test_numpy_floor_is_a_registered_candidate():
    """The np.sort floor must be a *selectable* backend, not just the emergency
    fallback — otherwise the tuner can never converge to it where it wins."""
    names = {b.name for b in B._BACKENDS}
    assert "numpy" in names, "np.sort floor is not a registered tuning candidate"
    ab = B.available_backends()
    assert "numpy" in ab and ab[-1] == "numpy", (
        "numpy floor should be listed last (lowest priority)"
    )


def test_tuning_disabled_preserves_static_priority(monkeypatch):
    """With tuning off, the a-priori priority order must stand: an accelerator,
    never the low-priority numpy floor, is chosen when one is available."""
    monkeypatch.setenv("QUILL_TUNING_DISABLED", "1")
    db = TimingDB(path=os.devnull + "-nonexistent")
    # floor listed last, as in the real registry
    names = ["ips4o", "rust_parallel_radix", "numpy"]
    assert db.choose(names, KIND, N) is None  # defers to caller's priority order


@pytest.mark.slow
def test_end_to_end_never_meaningfully_slower_than_npsort(monkeypatch, tmp_path):
    """The load-bearing guarantee, on whatever hardware actually runs this: after
    the tuner warms up, quill.sort_array must not be meaningfully slower than a
    bare np.sort — because the floor is always a candidate the tuner can pick."""
    import time
    import quill
    monkeypatch.setenv("QUILL_CONFIG_DIR", str(tmp_path))  # fresh, isolated DB

    rng = np.random.default_rng(0)
    n = 3_000_000
    base = rng.integers(-(2**40), 2**40, n).astype(np.int64)

    def med(fn, reps=5):
        ts = []
        for _ in range(reps):
            a = base.copy()
            t = time.perf_counter(); fn(a); ts.append(time.perf_counter() - t)
        ts.sort()
        return ts[len(ts) // 2]

    np_t = med(lambda a: np.sort(a))
    # Warm the tuner up (let it explore + converge) before measuring steady state.
    for _ in range(40):
        out = quill.sort_array(base.copy())
        assert np.array_equal(np.asarray(out), np.sort(base))
    quill_t = med(lambda a: quill.sort_array(a))
    converged = B._LAST_BACKEND

    # Gate on the RATIO, not the backend name. On a low-core box this converges to
    # a numpy-family path (bare "numpy" OR the SIMD wrapper "x86_simd_sort" /
    # "arm_neon_sort" — all the same kernel); on a many-core box it converges to a
    # parallel backend that is genuinely FASTER than np.sort. Both are correct, so
    # only "never meaningfully slower than np.sort" is a universal guarantee.
    cores = os.cpu_count()  # portable (unlike `nproc`, which is Linux-only)
    assert quill_t <= 1.35 * np_t, (
        f"quill.sort_array steady-state {quill_t*1e3:.1f}ms vs np.sort "
        f"{np_t*1e3:.1f}ms = {quill_t/np_t:.2f}x on {cores} core(s); "
        f"converged backend={converged!r} — tuner failed to fall back to the floor"
    )
    # On a single core, additionally assert it landed on a numpy-family path (not
    # a parallel backend that only loses here). Skipped on multi-core, where an
    # accelerator legitimately wins.
    if cores == 1:
        assert converged in _NUMPY_FAMILY, (
            f"single core: converged to {converged!r}, expected a numpy-family "
            f"path {sorted(_NUMPY_FAMILY)}"
        )

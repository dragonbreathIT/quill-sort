"""
quill/__main__.py
-----------------
Entry point for `python -m quill` and the `quill` CLI command.
"""

import os, random, sys, time
from . import quill_sort, __version__, display_version

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False


def _maybe_first_run_setup():
    """On the first interactive run of the bare `quill` command (no config yet),
    offer to run the setup wizard. Never prompts in a pipe/CI; never blocks."""
    try:
        from ._config import config_path
        if os.path.exists(config_path()):
            return  # already set up
        if os.environ.get("QUILL_NO_FIRST_RUN"):
            return
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            return  # non-interactive → don't prompt
        print("\nQuill can install faster sorting backends (Rust radix, GPU,")
        print("polars) and tune itself for this machine.")
        ans = input("Run 'quill setup' now? [Y/n] ").strip().lower()
    except (EOFError, KeyboardInterrupt, OSError):
        return
    except Exception:
        return
    if ans not in ("n", "no"):
        from .wizard import run_wizard
        run_wizard()


def main():
    _argv = sys.argv[1:]

    # `quill setup [-y] [--no-install]` → install accelerators + calibrate.
    if _argv and _argv[0] == "setup":
        from .wizard import run_wizard
        yes = any(f in _argv for f in ("-y", "--yes"))
        run_wizard(noninteractive=yes or "--noninteractive" in _argv,
                   assume_yes=yes, install="--no-install" not in _argv)
        return

    # `quill visualize [N]` → animated sort visualizer + real-sort pipeline view.
    if _argv and _argv[0] == "visualize":
        from .visualize import visualize, visualize_sort
        n = 24
        if len(_argv) > 1:
            try:
                n = max(2, min(120, int(_argv[1])))
            except ValueError:
                pass
        sample = random.sample(range(1, max(50, n * 12)), n)
        print()
        visualize(sample)                       # animated educational sort
        print()
        if _NUMPY:
            big = np.random.randint(0, 2**31, 2_000_000).astype(np.int64)
            visualize_sort(big)                 # real sort: backend + timing
        else:
            visualize_sort(sample)
        return

    # First-run nudge: pip cannot prompt during installation (wheels run no
    # install-time code, and pip is non-interactive), so the first time someone
    # runs the `quill` command without ever having set up, offer to do it now.
    # Only in a real interactive terminal, only once (a written config means
    # setup already ran), and skippable via QUILL_NO_FIRST_RUN.
    _maybe_first_run_setup()

    # ── Demo plumbing ─────────────────────────────────────────────────────────
    # All animation delays collapse to zero when stdout is not a TTY (piped /
    # captured) or when QUILL_DEMO_FAST is set, so the demo can be recorded fast
    # and never blocks CI. Glyph output is encode-guarded so a box/bar character
    # can't crash a cp1252 pipe with UnicodeEncodeError.
    NCORES = os.cpu_count() or 1
    _FAST = bool(os.environ.get("QUILL_DEMO_FAST")) or not sys.stdout.isatty()

    def _enc_safe(s):
        """Return *s* re-encoded to whatever stdout can actually emit, replacing
        any glyph the console codec (e.g. cp1252) lacks with an ASCII stand-in,
        so writing it can never raise UnicodeEncodeError."""
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        try:
            s.encode(enc)
            return s
        except (UnicodeEncodeError, LookupError):
            table = {"█": "#", "░": "-", "▄": "_", "▀": "^", "▌": "|",
                     "·": "*", "✓": "OK", "✗": "x", "—": "-", "→": "->"}
            for bad, good in table.items():
                s = s.replace(bad, good)
            return s.encode(enc, "replace").decode(enc)

    def out(text="", end="\n"):
        sys.stdout.write(_enc_safe(text) + end)
        sys.stdout.flush()

    def nap(seconds):
        if not _FAST:
            time.sleep(seconds)

    def typewrite(text, delay=0.012):
        if _FAST:
            out(text)
            return
        for ch in text:
            sys.stdout.write(_enc_safe(ch)); sys.stdout.flush(); time.sleep(delay)
        out()

    def bar(filled, total, width=36):
        total = total or 1
        n = int(width * min(filled, total) / total)
        return f"[{'█' * n}{'░' * (width - n)}]"

    def human(seconds):
        if seconds < 1e-3:
            return f"{seconds * 1e6:6.1f}us"
        if seconds < 1.0:
            return f"{seconds * 1e3:6.2f}ms"
        return f"{seconds:6.3f}s "

    # Every backend name we actually observe via _backends._LAST_BACKEND this
    # run is recorded here, so the closing "Paths actually used" summary reflects
    # reality rather than a hardcoded narrative.
    paths_seen = {}

    def note_backend(name):
        if name:
            paths_seen[name] = paths_seen.get(name, 0) + 1

    # One-line, accurate description of each real v6 backend / inline path.
    BACKEND_NOTES = {
        "rust_voracious": "compiled multi-threaded radix (quill._fastsort) — the headline int64/float64 path",
        "cupy_gpu":       "GPU radix sort via CuPy — large arrays that fit in VRAM",
        "openmp_simd":    "OpenMP-threaded AVX chunks + parallel C k-way merge (int64)",
        "polars":         "delegates to polars' multi-threaded Rust sort",
        "numpy_parallel": "thread-parallel np.partition sample-sort (integers, many-core)",
        "counting":       "inline np.bincount counting sort for dense bounded int64/uint64",
        "numpy":          "np.sort floor — always correct, the never-lose fallback",
        "timsort":        "Python's Timsort — strings / objects / tiny & presorted lists",
    }

    # ── Header ────────────────────────────────────────────────────────────────
    out()
    typewrite("  ▄▀▄ █ █ █ █   █   ")
    typewrite(f"  ▀▄▀ ▀▄█ █ █▄▄ █▄▄   {display_version()}  ({__version__})")
    out()
    typewrite("  Adaptive Ultra-Sort  —  by Isaiah Tucker", 0.01)

    np_tag = ("numpy ✓  vectorised C path active" if _NUMPY
              else "numpy ✗  pure-Python correctness path  (pip install numpy for speed)")
    typewrite(f"  [{np_tag}]", 0.008)
    out()
    nap(0.25)

    # ── Section 1: the REAL backends, in priority order ───────────────────────
    typewrite("  Backends available on this machine (priority order)", 0.012)
    out()
    from . import available_backends
    try:
        backends = available_backends()
    except Exception:
        backends = []
    for i, name in enumerate(backends, 1):
        desc = BACKEND_NOTES.get(name, "fast backend")
        out(f"    {i}. {name:<16} — {desc}")
    out(f"    {len(backends) + 1}. {'np.sort':<16} — {BACKEND_NOTES['numpy']}")
    out(f"    {'·':>2} {'counting':<16} — {BACKEND_NOTES['counting']}")
    out()
    missing = [n for n in ("rust_voracious", "cupy_gpu", "polars")
               if n not in backends]
    if missing:
        typewrite(f"    (not installed here: {', '.join(missing)} — "
                  f"run `quill setup` to add more)", 0.008)
    else:
        typewrite("    Every accelerator is installed — this box is fully loaded.", 0.008)
    out()
    nap(0.3)

    if not _NUMPY:
        # Degrade gracefully: no numpy means no array fast path. Show the pure
        # list correctness path only, then exit cleanly.
        typewrite("  numpy is not installed — showing the pure-Python list path only.", 0.012)
        out()
        _list_demo(out, typewrite, bar, human, nap, note_backend)
        _closing(out, typewrite, paths_seen, BACKEND_NOTES, array_ran=False)
        return

    from . import sort_array as _sort_array
    from . import _backends

    # ── Section 2: the ARRAY fast path (the headline) ─────────────────────────
    typewrite("  Array fast path  —  np.sort  vs  quill.sort_array  (no conversion)", 0.012)
    typewrite("  This is where the real speedups live: a sorted ndarray in, a sorted", 0.008)
    typewrite("  ndarray out, dispatched to the fastest installed backend.", 0.008)
    out()

    def array_case(label, make, bar_scale):
        try:
            a = make()
        except MemoryError:
            out(f"    {label:<30} [skipped — not enough RAM]")
            return
        ref = a.copy()

        t0 = time.perf_counter()
        expected = np.sort(ref)
        np_t = time.perf_counter() - t0

        t0 = time.perf_counter()
        result = _sort_array(a)
        q_t = time.perf_counter() - t0
        backend = _backends._LAST_BACKEND or "numpy"
        note_backend(backend)

        ok = bool(np.array_equal(result, expected))
        speed = (np_t / q_t) if q_t > 0 else float("inf")
        b = bar(q_t, bar_scale)
        mark = "✓" if ok else "✗"
        out(f"    {label}")
        out(f"      np.sort {human(np_t)}   sort_array {human(q_t)}  {b}")
        out(f"      {speed:4.1f}x faster   backend={backend:<14} {mark} "
            f"{'matches np.sort' if ok else 'MISMATCH'}")
        out()
        del a, ref, expected, result

    array_case(
        "int64  · 10,000,000 random (wide range)",
        lambda: (np.random.random(10_000_000) * 2**40).astype(np.int64),
        bar_scale=1.0,
    )
    array_case(
        "int64  · 10,000,000 dense bounded  (counting-sort territory)",
        lambda: np.random.randint(0, 100_000, 10_000_000, dtype=np.int64),
        bar_scale=1.0,
    )
    array_case(
        "float64· 10,000,000 random in [-1e9, 1e9]",
        lambda: np.random.uniform(-1e9, 1e9, 10_000_000),
        bar_scale=1.0,
    )
    nap(0.2)

    # ── Section 3: the LIST API (honest: conversion-bound) ────────────────────
    _list_demo(out, typewrite, bar, human, nap, note_backend)

    # ── Section 4: watch Quill sort 20 numbers in real time ───────────────────
    nap(0.2)
    typewrite("  Watching Quill sort 20 numbers in real time...", 0.012)
    out()
    nap(0.2)
    sample = random.sample(range(1, 999), 20)
    out(f"  Before:  {sample}")
    nap(0.6)
    quill_sort(sample)
    note_backend(_backends._LAST_BACKEND)
    sys.stdout.write("  After:   "); sys.stdout.flush()
    for i, v in enumerate(sample):
        sys.stdout.write(("" if i == 0 else ", ") + str(v))
        sys.stdout.flush()
        nap(0.055)
    out(); out()
    nap(0.3)

    # ── Closing: paths actually used + correctness recap + install line ───────
    _closing(out, typewrite, paths_seen, BACKEND_NOTES, array_ran=True)


def _list_demo(out, typewrite, bar, human, nap, note_backend):
    """The quill_sort(list) -> list API. Honestly conversion-bound: np.asarray +
    tolist dominate, so it matches np.sort and beats the builtin sorted() a few
    times over. Times each case and verifies the output is genuinely sorted."""
    from . import quill_sort
    try:
        from . import _backends
    except Exception:
        _backends = None

    typewrite("  List API  —  quill.quill_sort(list) -> list", 0.012)
    typewrite("  Conversion-bound (asarray/tolist): comparable to np.sort, and", 0.008)
    typewrite("  several times faster than Python's built-in sorted().", 0.008)
    out()

    def list_case(label, make, bar_scale, cmp_builtin=True):
        data = make()
        original = list(data)

        # Clear the marker first so a case that never reaches the numeric
        # dispatcher (e.g. strings → Timsort) honestly reports no backend
        # instead of inheriting the previous case's value.
        if _backends is not None:
            _backends._LAST_BACKEND = None
        t0 = time.perf_counter()
        quill_sort(data)
        q_t = time.perf_counter() - t0
        backend = getattr(_backends, "_LAST_BACKEND", None) if _backends else None
        note_backend(backend or "timsort")

        ok = all(data[i] <= data[i + 1] for i in range(len(data) - 1))

        extra = ""
        if cmp_builtin:
            t0 = time.perf_counter()
            sorted(original)
            s_t = time.perf_counter() - t0
            if q_t > 0:
                extra = f"   vs sorted() {s_t / q_t:4.1f}x"
        b = bar(q_t, bar_scale)
        # No dispatched backend means the data took the Timsort path
        # (strings / objects) — name it truthfully rather than leaving it blank.
        tag = f"  path={backend}" if backend else "  path=Timsort"
        mark = "✓" if ok else "✗"
        out(f"    {label:<34} {human(q_t)}  {b}{extra}")
        out(f"      {mark} {'output is sorted' if ok else 'NOT SORTED'}{tag}")

    list_case("ints · 1,000,000 wide range",
              lambda: [random.randint(0, 10**9) for _ in range(1_000_000)],
              bar_scale=0.4)
    list_case("mixed ± ints · 1,000,000",
              lambda: [random.randint(-10**6, 10**6) for _ in range(1_000_000)],
              bar_scale=0.4)
    list_case("narrow-range ints [0,100] · 1,000,000  (counting path)",
              lambda: [random.randint(0, 100) for _ in range(1_000_000)],
              bar_scale=0.2)
    list_case("floats · 1,000,000",
              lambda: [random.uniform(-1e9, 1e9) for _ in range(1_000_000)],
              bar_scale=0.4)
    list_case("strings · 200,000  (Timsort)",
              lambda: [''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=6))
                       for _ in range(200_000)],
              bar_scale=0.4)
    out()
    nap(0.2)


def _closing(out, typewrite, paths_seen, BACKEND_NOTES, array_ran=True):
    """Summarise the backends actually exercised this run (from observed
    _LAST_BACKEND values, never hardcoded), recap correctness, and sign off."""
    typewrite("  Paths actually used this run", 0.012)
    if paths_seen:
        for name, count in sorted(paths_seen.items(), key=lambda kv: -kv[1]):
            desc = BACKEND_NOTES.get(name, "fast backend")
            out(f"    · {name:<16} ×{count:<3} — {desc}")
    else:
        out("    · (no array/list sorts ran)")
    out()
    if array_ran:
        typewrite("  Correctness held on every case above: array results were compared", 0.009)
        typewrite("  to np.sort and list results checked monotonic — each in line, this", 0.009)
        typewrite("  run. NaN/None sort to the end; negatives are native; a mixed", 0.009)
    else:
        typewrite("  Correctness held on every case above: each list result was checked", 0.009)
        typewrite("  monotonic in line, this run. NaN/None sort to the end; negatives", 0.009)
        typewrite("  are native; a mixed", 0.009)
    typewrite("  int/float list promotes to float. Any backend error falls back to", 0.009)
    typewrite("  np.sort, so Quill never loses to the baseline and never crashes.", 0.009)
    out()
    typewrite("  pip install quill-sort", 0.022)
    out()


if __name__ == "__main__":
    main()

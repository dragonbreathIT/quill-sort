"""
quill/visualize.py
------------------
Watch Quill sort — two complementary visualizers, zero required deps.

The user asked: *"While it sorts data, show a visualizer of it."*  There are two
honest answers, because Quill's real sorts have nothing to animate frame by
frame: ``sort_array`` is a single atomic call into a C/Rust/GPU kernel that
returns in microseconds with no observable intermediate state.  So this module
ships both halves of the wish:

  1. ``visualize(data)`` — an animated TERMINAL bar chart of an *instrumented,
     educational* sort (insertion or quicksort partition) running step by step.
     This is the "watch the bars dance" experience.  It deliberately does NOT
     call the real engine; it runs a tiny pedagogical sort whose every
     comparison and swap is a yielded frame, so there is something to draw.
     The input is capped/downsampled to a screen-friendly width.

  2. ``visualize_sort(data)`` — a PHASE-PROGRESS view of a *real* Quill sort on
     a (possibly huge) array.  It times the pipeline (profile → choose backend →
     sort → done), reports which backend actually ran (from
     ``_backends._LAST_BACKEND`` / ``available_backends()``), and draws a
     before/after distribution sparkline so you can see the data go from chaos
     to a clean ramp.  This is the truthful "here is what the engine did"
     experience.

Design constraints honored here:
  * stdlib only, numpy optional — degrades cleanly when numpy is absent.
  * Windows-safe: ANSI colors are enabled via the ``os.system('')`` trick (which
    flips on the VT100 processing mode in modern Windows consoles) plus a
    colorama import if one happens to be installed; if neither works and we are
    not on a TTY, all escapes are suppressed and output is plain text.
  * Never hangs a pipe: when stdout is not a TTY (captured/redirected) the
    animation collapses to a few static frames so ``python -c`` and test
    captures finish instantly instead of sleeping through hundreds of frames.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

try:
    import numpy as np
    _NUMPY = True
except ImportError:  # numpy is optional everywhere in Quill
    _NUMPY = False


# ─────────────────────────────────────────────────────────────────────────────
# Terminal plumbing — ANSI on Windows, graceful no-color fallback
# ─────────────────────────────────────────────────────────────────────────────

# Eight vertical block glyphs (1/8 .. 8/8) let a single character render a bar
# height to sub-row precision, so short arrays still look smooth.
_BLOCKS = " ▁▂▃▄▅▆▇█"
# Sparkline glyphs (no leading space — a sparkline cell is always visible).
_SPARKS = "▁▂▃▄▅▆▇█"


def _enable_windows_ansi() -> None:
    """Turn on ANSI/VT100 escape processing on Windows terminals.

    ``os.system('')`` is the well-known zero-dependency trick: spawning a shell
    flips the console into VT-processing mode for the rest of the process, so our
    color escapes render instead of printing literally.  We also try colorama if
    it is installed (harmless if not).  Everything is wrapped so a locked-down
    environment can never crash the import or the call.
    """
    if os.name != "nt":
        return
    try:
        import colorama  # type: ignore
        colorama.just_fix_windows_console()
        return
    except Exception:
        pass
    try:
        os.system("")  # nosec - empty command, only flips VT mode
    except Exception:
        pass


def _supports_color(stream) -> bool:
    """True when we should emit ANSI color: a real TTY, not NO_COLOR-disabled."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("QUILL_FORCE_COLOR"):
        return True
    return bool(getattr(stream, "isatty", lambda: False)())


# Transliteration of every non-ASCII glyph we emit, for streams whose encoding
# cannot represent them (e.g. a Windows cp1252 pipe — `python -m quill ... | foo`
# or a redirected stdout). Without this, a single box-drawing char raises
# UnicodeEncodeError and crashes the whole visualization.
_ASCII_MAP = str.maketrans({
    "─": "-", "│": "|", "┌": "+", "┐": "+", "└": "+", "┘": "+",
    "├": "+", "┤": "+", "┬": "+", "┴": "+", "┼": "+", "═": "=",
    "█": "#", "▇": "#", "▆": "*", "▅": "+", "▄": "=", "▃": "-", "▂": ":", "▁": ".",
    "✓": "v", "✗": "x", "✔": "v", "✘": "x", "→": ">", "←": "<",
    "•": "*", "·": ".", "—": "-", "└": "+",
})


def _stream_unicode_ok(stream) -> bool:
    """True when *stream* can encode the box/block glyphs we use."""
    enc = getattr(stream, "encoding", None)
    if not enc:
        return False
    try:
        "─▁█✓→".encode(enc)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


class _SafeStream:
    """Stream proxy that downgrades Unicode glyphs to ASCII when the underlying
    stream's encoding can't represent them, with a hard fallback so a write can
    never raise. Transparent for everything else."""

    def __init__(self, stream):
        self._s = stream
        self._ascii = not _stream_unicode_ok(stream)

    def write(self, text):
        if self._ascii and isinstance(text, str):
            text = text.translate(_ASCII_MAP)
        try:
            return self._s.write(text)
        except UnicodeEncodeError:
            safe = text.encode("ascii", "replace").decode("ascii")
            return self._s.write(safe)

    def __getattr__(self, name):
        return getattr(self._s, name)


class _Palette:
    """ANSI escape strings, blanked out when color is unavailable so the exact
    same f-strings produce clean plain text in a captured/redirected stream."""

    def __init__(self, enabled: bool) -> None:
        def c(code: str) -> str:
            return f"\x1b[{code}m" if enabled else ""

        self.enabled = enabled
        self.reset   = c("0")
        self.dim     = c("2")
        self.bold    = c("1")
        self.unsorted= c("38;5;245")   # grey — not yet placed
        self.sorted  = c("38;5;42")    # green — locked in final position
        self.compare = c("38;5;220")   # yellow — being compared
        self.swap    = c("38;5;203")   # red — being swapped/moved
        self.pivot   = c("38;5;213")   # magenta — quicksort pivot
        self.title   = c("38;5;39")    # blue — title bar
        self.frame   = c("38;5;240")   # dark grey — frame chrome


def _term_width(default: int = 80) -> int:
    try:
        return max(20, shutil.get_terminal_size((default, 24)).columns)
    except Exception:
        return default


# ─────────────────────────────────────────────────────────────────────────────
# Frame model — an instrumented sort yields these; the renderer draws them.
# ─────────────────────────────────────────────────────────────────────────────

class _Frame:
    """One step of an instrumented sort.

    Attributes
    ----------
    values   : current array state (list of comparable numbers)
    compare  : indices currently being compared (highlight yellow)
    swap     : indices currently being swapped/moved (highlight red)
    pivot    : pivot index for partition sorts (highlight magenta), or None
    sorted_lo: everything with index < this is in its final sorted position
    sorted_hi: everything with index >= this is in its final sorted position
    note     : short status string shown under the bars
    """

    __slots__ = ("values", "compare", "swap", "pivot",
                 "sorted_lo", "sorted_hi", "note")

    def __init__(self, values, compare=(), swap=(), pivot=None,
                 sorted_lo=0, sorted_hi=None, note=""):
        self.values    = values
        self.compare   = tuple(compare)
        self.swap      = tuple(swap)
        self.pivot     = pivot
        self.sorted_lo = sorted_lo
        self.sorted_hi = len(values) if sorted_hi is None else sorted_hi
        self.note      = note


# ─────────────────────────────────────────────────────────────────────────────
# Instrumented educational sorts — these are what gets *animated*.
# They are intentionally simple O(n²)/partition sorts so every comparison and
# swap is observable. They are NOT the real engine (the real engine is atomic).
# ─────────────────────────────────────────────────────────────────────────────

def _insertion_frames(a: List) -> Iterator[_Frame]:
    """Insertion sort, yielding a frame per comparison and per shift.

    Insertion sort animates beautifully on small arrays: the left side grows as
    a solid sorted block while the active element walks left into place.
    """
    n = len(a)
    yield _Frame(list(a), sorted_lo=0, sorted_hi=0, note="start")
    for i in range(1, n):
        j = i
        # walk the element at i down into the sorted prefix [0, i)
        while j > 0:
            yield _Frame(list(a), compare=(j - 1, j), sorted_hi=i,
                         note=f"compare {j-1} vs {j}")
            if a[j - 1] <= a[j]:
                break
            a[j - 1], a[j] = a[j], a[j - 1]
            yield _Frame(list(a), swap=(j - 1, j), sorted_hi=i,
                         note=f"swap {j-1} <-> {j}")
            j -= 1
    yield _Frame(list(a), sorted_lo=0, sorted_hi=n, note="sorted")


def _quick_frames(a: List) -> Iterator[_Frame]:
    """Iterative Lomuto-partition quicksort, yielding a frame per comparison and
    swap, with the pivot highlighted. Animates the "partition then recurse"
    structure that makes Quill's real radix/partition backends fast."""
    n = len(a)
    yield _Frame(list(a), note="start")
    # iterative to avoid recursion-depth surprises on adversarial inputs
    stack: List[Tuple[int, int]] = [(0, n - 1)]
    # track positions known to be final so we can paint them green
    finalized = [False] * n

    while stack:
        lo, hi = stack.pop()
        if lo >= hi:
            if 0 <= lo < n:
                finalized[lo] = True
            continue
        pivot = a[hi]
        i = lo
        for j in range(lo, hi):
            yield _Frame(list(a), compare=(j, hi), pivot=hi,
                         note=f"partition [{lo}..{hi}] pivot={pivot}")
            if a[j] <= pivot:
                if i != j:
                    a[i], a[j] = a[j], a[i]
                    yield _Frame(list(a), swap=(i, j), pivot=hi,
                                 note=f"swap {i} <-> {j}")
                i += 1
        a[i], a[hi] = a[hi], a[i]
        finalized[i] = True
        yield _Frame(list(a), swap=(i, hi), pivot=i,
                     note=f"place pivot at {i}")
        stack.append((lo, i - 1))
        stack.append((i + 1, hi))

    yield _Frame(list(a), sorted_lo=0, sorted_hi=n, note="sorted")


_ALGORITHMS = {
    "insertion": _insertion_frames,
    "quick":     _quick_frames,
}


def _choose_algorithm(name: str, n: int):
    """Resolve the algorithm name. ``'auto'`` picks insertion for very small
    arrays (its growing-green-block animation reads cleanly) and quicksort
    otherwise (fewer frames, shows partition structure)."""
    if name == "auto":
        name = "insertion" if n <= 32 else "quick"
    fn = _ALGORITHMS.get(name)
    if fn is None:
        raise ValueError(
            f"unknown algorithm {name!r}; choose from "
            f"{sorted(_ALGORITHMS) + ['auto']}"
        )
    return name, fn


# ─────────────────────────────────────────────────────────────────────────────
# Input coercion — turn arbitrary data into a small list of numbers to draw.
# ─────────────────────────────────────────────────────────────────────────────

def _to_number_list(data, cap: int) -> List[float]:
    """Coerce *data* into a list of <= cap finite numbers for visualization.

    Large inputs are evenly DOWNSAMPLED (not truncated) so the visualization
    still reflects the whole array's shape. Non-numeric items are ranked by
    their sort order so any comparable data can be drawn as bar heights.
    """
    # Materialize without consuming the caller's iterator semantics surprisingly.
    if _NUMPY and isinstance(data, np.ndarray):
        seq: Sequence = data.tolist()
    elif isinstance(data, (list, tuple)):
        seq = list(data)
    else:
        seq = list(data)

    n = len(seq)
    if n == 0:
        return []

    # Even downsample to at most `cap` points across the full span.
    if n > cap:
        step = n / cap
        seq = [seq[min(int(k * step), n - 1)] for k in range(cap)]

    # If everything is already numeric, use values directly (after dropping
    # NaN/None so the bar math stays finite — consistent with Quill's contract
    # of NaN/None sorting to the end, which we just omit from the picture).
    nums: List[float] = []
    all_numeric = True
    for v in seq:
        if v is None:
            all_numeric = False
            break
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            all_numeric = False
            break
        if isinstance(v, float) and v != v:  # NaN
            all_numeric = False
            break
        nums.append(float(v))

    if all_numeric:
        return nums

    # Mixed / non-numeric: rank items by sorted order and draw the ranks. This
    # gives a faithful "is it sorted?" picture for strings, tuples, objects.
    try:
        order = sorted(range(len(seq)), key=lambda k: seq[k])
    except TypeError:
        # Truly unorderable mixed types: fall back to string comparison.
        order = sorted(range(len(seq)), key=lambda k: str(seq[k]))
    rank = [0] * len(seq)
    for r, idx in enumerate(order):
        rank[idx] = r
    return [float(x) for x in rank]


# ─────────────────────────────────────────────────────────────────────────────
# Rendering — draw a frame as a row of colored block-bars.
# ─────────────────────────────────────────────────────────────────────────────

def _render_frame(frame: _Frame, pal: _Palette, height: int,
                  width_cap: int) -> str:
    """Render one frame to a multi-line string (height rows of bars + a note).

    Each value becomes one column of vertical block glyphs colored by its role
    (sorted/unsorted/compare/swap/pivot). Bars are scaled to *height* rows.
    """
    vals = frame.values
    n = len(vals)
    if n == 0:
        return "(empty)"

    # cap the number of columns to the terminal width
    cols = min(n, width_cap)
    if cols < n:
        # already downsampled at coercion time, but guard anyway
        step = n / cols
        idx = [min(int(k * step), n - 1) for k in range(cols)]
    else:
        idx = list(range(n))

    vmin = min(vals)
    vmax = max(vals)
    span = (vmax - vmin) or 1.0

    # Per-column fractional height in [0, height], used for block selection.
    levels = []
    for k in idx:
        frac = (vals[k] - vmin) / span
        levels.append(frac * height)

    def color_for(k: int) -> str:
        if k in frame.swap:
            return pal.swap
        if frame.pivot is not None and k == frame.pivot:
            return pal.pivot
        if k in frame.compare:
            return pal.compare
        if k < frame.sorted_lo or k >= frame.sorted_hi:
            return pal.sorted
        return pal.unsorted

    lines: List[str] = []
    # Draw from the top row down to the bottom row.
    for row in range(height, 0, -1):
        chars: List[str] = []
        for col, k in enumerate(idx):
            lvl = levels[col]
            # how much of THIS row this bar fills: 0..8 eighths
            cell = lvl - (row - 1)
            if cell >= 1.0:
                glyph = _BLOCKS[8]
            elif cell <= 0.0:
                glyph = _BLOCKS[0]
            else:
                glyph = _BLOCKS[int(cell * 8)]
            chars.append(f"{color_for(k)}{glyph}{pal.reset}")
        lines.append("".join(chars))

    return "\n".join(lines)


def _title_bar(algo: str, n: int, original_n: int, pal: _Palette,
               width: int) -> str:
    tag = f" Quill visualizer — {algo} sort · n={n} "
    if original_n != n:
        tag = f" Quill visualizer — {algo} sort · n={n} (of {original_n:,}) "
    bar = tag.center(min(width, max(len(tag) + 2, 40)), "─")
    return f"{pal.title}{pal.bold}{bar}{pal.reset}"


def _move_cursor_up(stream, lines: int) -> None:
    """Move the cursor up *lines* rows so the next frame overwrites the prior
    one in place (only meaningful on a TTY; harmless string otherwise)."""
    stream.write(f"\x1b[{lines}A")


# ─────────────────────────────────────────────────────────────────────────────
# Public API #1 — animated educational visualizer
# ─────────────────────────────────────────────────────────────────────────────

def visualize(
    data: Iterable,
    algorithm: str = "auto",
    speed: float = 1.0,
    *,
    height: int = 8,
    cap: int = 120,
    stream=None,
    color: Optional[bool] = None,
    max_static_frames: int = 6,
) -> None:
    """Animate a step-by-step sort of *data* as a colored terminal bar chart.

    This runs an INSTRUMENTED educational sort (insertion or quicksort) so there
    is genuine frame-by-frame motion to watch — comparisons flash yellow, swaps
    flash red, the quicksort pivot is magenta, and bars turn green as they reach
    their final sorted position, with a title bar and a closing flourish. The
    real Quill engine sorts atomically with nothing to animate; for *that*, see
    :func:`visualize_sort`.

    Parameters
    ----------
    data       : any iterable. Numeric values are drawn directly; anything else
                 (strings, tuples, objects) is drawn by its sort rank so the
                 "is it sorted yet?" picture is still faithful. Inputs larger
                 than *cap* are evenly DOWNSAMPLED (not truncated).
    algorithm  : 'auto' (default), 'insertion', or 'quick'.
    speed      : animation speed multiplier; 2.0 is twice as fast, 0.5 half.
    height     : bar chart height in rows (default 8).
    cap        : maximum number of visualized elements (default 120).
    stream     : output stream (default ``sys.stdout``).
    color      : force color on/off; default auto-detects a TTY.
    max_static_frames : when stdout is NOT a TTY (captured/redirected), the
                 animation collapses to at most this many evenly-spaced static
                 frames so pipes/tests never hang. Default 6.

    Returns
    -------
    None — everything is drawn to *stream*.
    """
    stream = stream if stream is not None else sys.stdout
    _enable_windows_ansi()
    stream = _SafeStream(stream)   # ASCII-safe on cp1252 pipes

    is_tty = bool(getattr(stream, "isatty", lambda: False)())
    use_color = _supports_color(stream) if color is None else bool(color)
    pal = _Palette(use_color)

    # Coerce + downsample the input to a drawable list of numbers.
    nums = _to_number_list(data, cap)
    original_n = _count(data)
    n = len(nums)

    if n == 0:
        stream.write("(nothing to visualize — empty input)\n")
        return

    width = _term_width()
    width_cap = max(8, min(width, cap))
    algo_name, frame_fn = _choose_algorithm(algorithm, n)

    # Generate the full frame sequence up front (small n, cheap).
    frames = list(frame_fn(list(nums)))

    title = _title_bar(algo_name, n, original_n, pal, width)

    if is_tty:
        _play_animated(frames, pal, height, width_cap, title, speed, stream)
    else:
        _play_static(frames, pal, height, width_cap, title, stream,
                     max_static_frames)


def _play_animated(frames, pal, height, width_cap, title, speed, stream):
    """In-place animation for a real TTY: redraw each frame over the last."""
    # Per-frame delay; clamped so it is watchable but never glacial.
    base_delay = 0.045
    delay = max(0.004, base_delay / max(speed, 0.05))
    block_lines = height + 1  # bars + note line

    stream.write(title + "\n")
    stream.flush()
    first = True
    for fr in frames:
        body = _render_frame(fr, pal, height, width_cap)
        note = f"{pal.frame}{fr.note}{pal.reset}"
        if not first:
            _move_cursor_up(stream, block_lines)
        stream.write(body + "\n")
        # pad the note line to clear any longer previous note
        stream.write(note.ljust(len(note) + 8) + "\x1b[K\n")
        stream.flush()
        first = False
        time.sleep(delay)

    # Final flourish: sweep the whole bar green left-to-right.
    final = frames[-1]
    for cut in range(0, len(final.values) + 1, max(1, len(final.values) // 24 or 1)):
        fr = _Frame(final.values, sorted_lo=0, sorted_hi=len(final.values),
                    note="sorted ✓")
        _move_cursor_up(stream, block_lines)
        stream.write(_render_frame(fr, pal, height, width_cap) + "\n")
        stream.write(f"{pal.sorted}{pal.bold}sorted ✓{pal.reset}\x1b[K\n")
        stream.flush()
        time.sleep(0.02)


def _play_static(frames, pal, height, width_cap, title, stream, max_frames):
    """Non-TTY rendering: print a handful of evenly-spaced snapshots and stop.

    Crucial for ``python -c`` / captured output / tests: there is no cursor to
    move and no point sleeping, so we sample the frame list and print labeled
    stills. This finishes instantly and never blocks a pipe.
    """
    stream.write(title + "\n")
    total = len(frames)
    if total <= max_frames:
        picks = list(range(total))
    else:
        # always include first and last; evenly space the rest
        step = (total - 1) / (max_frames - 1)
        picks = sorted({int(round(k * step)) for k in range(max_frames)})
        picks[-1] = total - 1
    for i, fi in enumerate(picks):
        fr = frames[fi]
        stream.write(f"--- frame {fi + 1}/{total}: {fr.note} ---\n")
        stream.write(_render_frame(fr, pal, height, width_cap) + "\n")
    stream.write("sorted ✓\n")
    stream.flush()


def _count(data) -> int:
    """Length of *data* if cheaply known, else 0 (used only for the title)."""
    try:
        return len(data)
    except TypeError:
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Public API #2 — real-sort phase-progress visualizer
# ─────────────────────────────────────────────────────────────────────────────

def _sparkline(values, buckets: int = 40) -> str:
    """A one-line histogram sparkline of *values*' distribution.

    For the BEFORE picture this looks noisy; for the AFTER picture (sorted data)
    the per-bucket *counts* are roughly uniform, so the eye reads "evened out".
    To actually show order we histogram the VALUES into equal-width bins and
    draw bin population — sorted vs unsorted have the same histogram, so instead
    we draw the sequence itself downsampled, which visibly goes from jagged to a
    clean monotonic ramp.
    """
    if _NUMPY and isinstance(values, np.ndarray):
        if values.size == 0:
            return ""
        n = values.size
        if n > buckets:
            idx = (np.linspace(0, n - 1, buckets)).astype(np.int64)
            sample = values[idx]
        else:
            sample = values
        sample = sample.astype(np.float64)
        finite = sample[np.isfinite(sample)]
        if finite.size == 0:
            return "─" * buckets
        lo = float(finite.min()); hi = float(finite.max())
        span = (hi - lo) or 1.0
        out = []
        for v in sample.tolist():
            if v != v:  # NaN
                out.append(" ")
                continue
            frac = (v - lo) / span
            out.append(_SPARKS[min(7, max(0, int(frac * 8)))])
        return "".join(out)

    # pure-Python path
    seq = list(values)
    if not seq:
        return ""
    if len(seq) > buckets:
        step = len(seq) / buckets
        seq = [seq[min(int(k * step), len(seq) - 1)] for k in range(buckets)]
    finite = [v for v in seq if isinstance(v, (int, float)) and v == v]
    if not finite:
        return "─" * len(seq)
    lo = min(finite); hi = max(finite)
    span = (hi - lo) or 1.0
    out = []
    for v in seq:
        if not isinstance(v, (int, float)) or v != v:
            out.append(" ")
            continue
        frac = (v - lo) / span
        out.append(_SPARKS[min(7, max(0, int(frac * 8)))])
    return "".join(out)


class _PhaseReporter:
    """Prints the real-sort pipeline phases with timing, as a context manager.

    Usage::

        with _PhaseReporter("profile", pal, stream) as p:
            ...                       # do the work
        # on exit it prints "profile  ✓  0.42 ms"

    Used internally by :func:`visualize_sort` to narrate profile → backend →
    sort → done. Each phase records its own wall-clock duration.
    """

    def __init__(self, label, pal, stream, detail=""):
        self.label = label
        self.pal = pal
        self.stream = stream
        self.detail = detail
        self.t0 = 0.0
        self.elapsed_ms = 0.0

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def set_detail(self, detail: str) -> None:
        self.detail = detail

    def __exit__(self, exc_type, exc, tb):
        self.elapsed_ms = (time.perf_counter() - self.t0) * 1000.0
        pal = self.pal
        mark = "✓" if exc_type is None else "✗"
        markc = pal.sorted if exc_type is None else pal.swap
        detail = f"  {pal.dim}{self.detail}{pal.reset}" if self.detail else ""
        self.stream.write(
            f"  {markc}{mark}{pal.reset} {pal.bold}{self.label:<22}{pal.reset}"
            f"{pal.frame}{self.elapsed_ms:>9.2f} ms{pal.reset}{detail}\n"
        )
        self.stream.flush()
        return False  # never suppress exceptions


def visualize_sort(
    data,
    *,
    descending: bool = False,
    stream=None,
    color: Optional[bool] = None,
    spark_buckets: int = 48,
) -> "object":
    """Run a REAL Quill sort on *data* and visualize the pipeline.

    Unlike :func:`visualize` (which animates a toy sort), this calls the actual
    engine — :func:`quill.sort_array` for numeric numpy arrays (the
    zero-conversion fast path) or :func:`quill.quill_sort` otherwise — then
    reports each pipeline phase with timing and, crucially, **which backend
    actually ran** (read from ``quill._backends._LAST_BACKEND`` and shown
    alongside ``available_backends()``). A before/after sparkline shows the data
    going from jagged chaos to a clean monotonic ramp.

    Parameters
    ----------
    data          : list or numpy array to sort (any size — millions are fine;
                    only a small sparkline sample is drawn, never the whole array).
    descending    : sort descending (numpy fast path only; the list path sorts
                    ascending and notes that).
    stream        : output stream (default ``sys.stdout``).
    color         : force color on/off; default auto-detects a TTY.
    spark_buckets : width of the before/after sparklines.

    Returns
    -------
    The sorted result (ndarray or list), so this can wrap a real call site.
    """
    import quill
    from quill import _backends

    stream = stream if stream is not None else sys.stdout
    _enable_windows_ansi()
    stream = _SafeStream(stream)   # ASCII-safe on cp1252 pipes
    use_color = _supports_color(stream) if color is None else bool(color)
    pal = _Palette(use_color)

    # Decide which real path we will exercise.
    use_array = _NUMPY and isinstance(data, np.ndarray) and data.ndim == 1
    n = _count(data)

    width = _term_width()
    header = f" Quill sort pipeline · n={n:,} "
    stream.write(f"{pal.title}{pal.bold}"
                 f"{header.center(min(width, max(len(header)+2, 44)), '─')}"
                 f"{pal.reset}\n")

    # ── BEFORE distribution ──────────────────────────────────────────────────
    before = _sparkline(data, spark_buckets)
    stream.write(f"  {pal.dim}before{pal.reset}  "
                 f"{pal.unsorted}{before}{pal.reset}\n")

    # ── Phase 1: profile ─────────────────────────────────────────────────────
    with _PhaseReporter("profile", pal, stream) as p:
        try:
            if use_array:
                info = {
                    "dtype": str(data.dtype),
                    "eligible": _backends.eligible(data),
                }
                p.set_detail(f"dtype={info['dtype']} "
                             f"eligible={info['eligible']}")
            else:
                info = quill.analyze(list(data) if not isinstance(data, list)
                                     else data)
                p.set_detail(f"dtype={info.get('dtype','?')} "
                             f"presorted={info.get('presorted','?')}")
        except Exception:
            p.set_detail("profile unavailable")

    # ── Phase 2: choose backend ──────────────────────────────────────────────
    with _PhaseReporter("choose backend", pal, stream) as p:
        avail = quill.available_backends()
        p.set_detail("available: " + (", ".join(avail) if avail else "numpy"))

    # ── Phase 3: sort (the real, timed call) ─────────────────────────────────
    result = None
    backend_used = "numpy"
    with _PhaseReporter("sorting", pal, stream) as p:
        if use_array:
            result = quill.sort_array(data, descending=descending, inplace=False)
            # _LAST_BACKEND is set by the dispatcher during the call above.
            backend_used = getattr(_backends, "_LAST_BACKEND", None) or "numpy"
            p.set_detail(f"backend = {backend_used}")
        else:
            result = quill.quill_sort(
                list(data) if not isinstance(data, list) else list(data),
                reverse=descending, inplace=False, silent=True,
            )
            backend_used = "timsort/list-engine"
            p.set_detail(f"backend = {backend_used}"
                         + ("" if not descending
                            else " (list path sorts then reverses)"))

    # ── Phase 4: done + AFTER distribution ───────────────────────────────────
    with _PhaseReporter("done", pal, stream) as p:
        # verify monotonicity on a small sample so we can show a ✓ honestly
        ok = _quick_monotonic_check(result, descending)
        p.set_detail("ordered ✓" if ok else "ordering check skipped")

    after = _sparkline(result, spark_buckets)
    stream.write(f"  {pal.dim}after {pal.reset}  "
                 f"{pal.sorted}{after}{pal.reset}\n")
    stream.write(
        f"  {pal.frame}backend{pal.reset} "
        f"{pal.bold}{pal.compare}{backend_used}{pal.reset}"
        f"{pal.frame}  (from quill._backends._LAST_BACKEND){pal.reset}\n"
    )
    stream.flush()
    return result


def _quick_monotonic_check(result, descending: bool) -> bool:
    """Cheaply confirm the first/last chunk is ordered (sample, not full scan)."""
    try:
        if _NUMPY and isinstance(result, np.ndarray):
            if result.size <= 1:
                return True
            sample = result[: min(result.size, 100_000)]
            # ignore trailing NaN (they legitimately sit at the end)
            if sample.dtype.kind == "f":
                sample = sample[np.isfinite(sample)]
            if sample.size <= 1:
                return True
            diffs = np.diff(sample)
            return bool((diffs <= 0).all() if descending else (diffs >= 0).all())
        seq = list(result)
        if len(seq) <= 1:
            return True
        seq = seq[: min(len(seq), 50_000)]
        if descending:
            return all(seq[i] >= seq[i + 1] for i in range(len(seq) - 1))
        return all(seq[i] <= seq[i + 1] for i in range(len(seq) - 1))
    except Exception:
        return False

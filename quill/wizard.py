"""
quill/wizard.py
---------------
Interactive setup wizard — ``quill setup`` (or ``python -m quill setup``).

Why this exists
===============
Quill's whole performance story is *per-machine*. The same call (`sort_array`)
dispatches to a Rust radix, a GPU, polars, or a numpy parallel-partition
depending on what is installed and how big the data is — and the crossover *n*
where each of those starts to beat plain ``np.sort`` is a property of THIS box's
CPU, memory bandwidth, core count and PCIe link, not something we can hard-code.

The conservative built-in defaults in ``_config.py`` guarantee "never lose" on
an unknown machine by keeping parallelism off until well past the point it
reliably wins. That safety costs speed on a capable box. The wizard closes the
gap: it *measures* this machine and writes a calibrated ``config.json`` so the
adaptive paths engage exactly where they pay off here.

Design constraints honoured here
================================
  * **Never hangs.** Every benchmark size is capped, the whole calibration is
    time-bounded, and ``noninteractive=True`` skips the single confirmation
    prompt. There is no unbounded loop and no network access.
  * **Zero required deps.** Runs with the standard library alone. numpy, psutil,
    polars, cupy and rich are all *optional* — each is probed and the wizard
    degrades to a clear "not installed, here's how" message. ``rich`` upgrades
    the look (spinner + pip progress bars) when present; without it we use
    ``tqdm`` if available, then plain print as the bottom of the ladder.
  * **Cross-platform pretty UI.** Box-drawing headers, check/cross marks and
    ANSI colour, with automatic plain-ASCII + no-colour fallback when the
    terminal can't handle them (redirected output, Windows legacy console,
    ``NO_COLOR`` set).
  * **Safe.** Hardware detection is best-effort and never raises — a probe that
    fails just reports "unknown". Calibration only allocates small, bounded
    arrays and never touches the user's real data.
"""

from __future__ import annotations

import os
import platform
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import __version__, available_backends, display_version
from ._config import DEFAULTS, config_path, load_config, save_config


# ─────────────────────────────────────────────────────────────────────────────
# Terminal styling — color + box drawing with a graceful plain-text fallback.
# ─────────────────────────────────────────────────────────────────────────────

class _Style:
    """Resolve once whether this terminal can do ANSI colour / Unicode glyphs,
    then expose helpers that no-op cleanly when it can't. Doing the capability
    check up front keeps every print site simple and means a dumb pipe or a
    legacy console just gets readable ASCII instead of escape soup."""

    # ANSI SGR codes; emptied out when colour is disabled.
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"

    def __init__(self, stream=None) -> None:
        self.stream = stream or sys.stdout
        self.color = self._supports_color()
        self.unicode = self._supports_unicode()
        if not self.color:
            for attr in ("RESET", "BOLD", "DIM", "RED", "GREEN",
                         "YELLOW", "BLUE", "MAGENTA", "CYAN"):
                setattr(self, attr, "")
        if self.unicode:
            self.CHECK, self.CROSS, self.DOT, self.ARROW = "✓", "✗", "•", "→"
            self.TL, self.TR, self.BL, self.BR = "╭", "╮", "╰", "╯"
            self.H, self.V = "─", "│"
        else:
            self.CHECK, self.CROSS, self.DOT, self.ARROW = "OK", "X", "-", "->"
            self.TL, self.TR, self.BL, self.BR = "+", "+", "+", "+"
            self.H, self.V = "-", "|"

    def _supports_color(self) -> bool:
        if os.environ.get("NO_COLOR"):
            return False
        if os.environ.get("FORCE_COLOR"):
            return True
        if not hasattr(self.stream, "isatty") or not self.stream.isatty():
            return False
        if os.name == "nt":
            if os.environ.get("WT_SESSION") or os.environ.get("TERM"):
                return True
            return self._enable_windows_vt()
        return True

    @staticmethod
    def _enable_windows_vt() -> bool:
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)
            mode = ctypes.c_uint32()
            if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                return False
            ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
            new_mode = mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING
            return bool(kernel32.SetConsoleMode(handle, new_mode))
        except Exception:
            return False

    def _supports_unicode(self) -> bool:
        enc = (getattr(self.stream, "encoding", None) or "").lower()
        if os.name == "nt":
            if not (os.environ.get("WT_SESSION") or os.environ.get("TERM")):
                return False
        if "utf" in enc:
            return True
        try:
            "╭✓".encode(enc or "ascii")
            return True
        except (UnicodeError, LookupError, TypeError):
            return False

    # convenience colourisers ------------------------------------------------
    def ok(self, text: str) -> str:    return f"{self.GREEN}{text}{self.RESET}"
    def bad(self, text: str) -> str:   return f"{self.RED}{text}{self.RESET}"
    def warn(self, text: str) -> str:  return f"{self.YELLOW}{text}{self.RESET}"
    def info(self, text: str) -> str:  return f"{self.CYAN}{text}{self.RESET}"
    def hot(self, text: str) -> str:   return f"{self.MAGENTA}{text}{self.RESET}"
    def dim(self, text: str) -> str:   return f"{self.DIM}{text}{self.RESET}"
    def bold(self, text: str) -> str:  return f"{self.BOLD}{text}{self.RESET}"


def _visible_len(text: str) -> int:
    out, i = 0, 0
    while i < len(text):
        if text[i] == "\033":
            j = text.find("m", i)
            i = len(text) if j == -1 else j + 1
        else:
            out += 1
            i += 1
    return out


def _truncate(text: str, limit: int) -> str:
    if _visible_len(text) <= limit:
        return text
    keep = max(0, limit - 1)
    out, vis, i, had_code = [], 0, 0, False
    while i < len(text) and vis < keep:
        if text[i] == "\033":
            j = text.find("m", i)
            j = len(text) - 1 if j == -1 else j
            out.append(text[i:j + 1])
            had_code = True
            i = j + 1
        else:
            out.append(text[i])
            vis += 1
            i += 1
    out.append(".")
    if had_code:
        out.append("\033[0m")
    return "".join(out)


def _box(st: _Style, title: str, lines: List[str], width: int = 64) -> str:
    inner = width - 2
    top_label = f" {title} "
    fill = st.H * max(0, inner - _visible_len(top_label))
    out = [f"{st.TL}{top_label}{fill}{st.TR}"]
    for line in lines:
        line = _truncate(line, inner - 1)
        pad = " " * max(0, inner - _visible_len(line) - 1)
        out.append(f"{st.V} {line}{pad}{st.V}")
    out.append(f"{st.BL}{st.H * inner}{st.BR}")
    return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# UI ladder: rich (spinners, pip progress bars) → tqdm → plain print.
# ─────────────────────────────────────────────────────────────────────────────

def _have_rich() -> bool:
    """Whether the optional ``rich`` library is importable. Cached implicitly
    by the import system."""
    try:
        import rich  # noqa: F401
        return True
    except ImportError:
        return False


def _have_tqdm() -> bool:
    try:
        import tqdm  # noqa: F401
        return True
    except ImportError:
        return False


def _step(title: str, fn: Callable[[], Any],
          out: Callable[[str], None]) -> Any:
    """Run ``fn()`` while showing a spinner (rich) or animated dots (fallback)
    next to ``title``. Returns whatever ``fn`` returned.

    Never raises if ``fn`` is well-behaved — the spinner just stops and the
    line is closed with a check mark either way."""
    if _have_rich() and sys.stdout.isatty():
        try:
            from rich.console import Console
            from rich.live import Live
            from rich.spinner import Spinner

            console = Console()
            with Live(Spinner("dots", text=title),
                      console=console, refresh_per_second=12,
                      transient=True):
                result = fn()
            console.print(f"  [green]✓[/green] {title}")
            return result
        except Exception:
            pass  # fall through to the plain path below
    # Plain ladder: just print "... title" then "OK" when done. We deliberately
    # don't animate here — a single steady line is friendlier in CI/pipes than
    # a flickering spinner emulated by hand.
    out(f"  ... {title}")
    result = fn()
    # Move up one line and rewrite the marker only when we're on a TTY; on a
    # pipe a second line is fine and avoids escape soup in logs.
    return result


def _progress_pip(package: str, out: Callable[[str], None]) -> bool:
    """``pip install --upgrade --progress-bar on <package>`` with pip's own
    progress bar streamed to the terminal. Returns True on success.

    We let pip's output through (no capture) precisely so the user sees the
    progress bar live — that's the whole point. On non-TTY runs pip drops the
    bar automatically and just prints status, which is also fine."""
    import subprocess

    cmd = [sys.executable, "-m", "pip", "install", "--upgrade",
           "--progress-bar", "on", package]
    out(f"    $ {' '.join(cmd)}")
    try:
        return subprocess.run(cmd, check=False).returncode == 0
    except Exception as exc:                                # pragma: no cover
        out(f"    pip failed: {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Hardware detection — every probe wrapped so a missing dep / unsupported OS
# returns "unknown" rather than crashing the wizard.
# ─────────────────────────────────────────────────────────────────────────────

def _safe(fn: Callable[[], Any], default: Any = None) -> Any:
    """Run a detection probe; swallow *anything* it raises and return the
    default. Hardware probing is full of OS-specific edge cases (no CPUID on
    ARM, missing /proc on Windows, locked-down sandboxes) and not one of them
    should turn into a wizard crash."""
    try:
        return fn()
    except Exception:
        return default


def _detect_cpu() -> Dict[str, Any]:
    """CPU brand string, logical & physical core count, and a coarse ISA
    feature set (avx512 / avx2 / neon). Each probe is independent so a partial
    failure still yields a partial answer."""
    info: Dict[str, Any] = {
        "brand": None,
        "logical": os.cpu_count() or 1,
        "physical": None,
        "arch": platform.machine(),
        "features": [],
    }

    # Brand string. cpuinfo gives a clean answer on every OS; platform.processor
    # is empty on Linux and wordy on Windows but better than nothing as fallback.
    def _brand_via_cpuinfo() -> Optional[str]:
        import cpuinfo  # type: ignore
        d = cpuinfo.get_cpu_info()
        return d.get("brand_raw") or d.get("brand")
    info["brand"] = (_safe(_brand_via_cpuinfo)
                     or _safe(platform.processor) or None)

    # Physical core count — psutil knows; otherwise leave as None.
    def _physical() -> Optional[int]:
        import psutil  # type: ignore
        return psutil.cpu_count(logical=False)
    info["physical"] = _safe(_physical)

    # ISA features. py-cpuinfo gives a flags list; on ARM/Apple silicon we
    # synthesise NEON because every ARMv8 chip has it.
    def _flags_via_cpuinfo() -> List[str]:
        import cpuinfo  # type: ignore
        flags = cpuinfo.get_cpu_info().get("flags", []) or []
        # Normalise to the few we actually advertise.
        out = []
        for tag in ("avx512f", "avx2", "sse4_2", "neon", "asimd"):
            if tag in flags:
                out.append(tag)
        return out

    def _flags_via_numpy() -> List[str]:
        # Fallback when py-cpuinfo isn't installed (it's an OPTIONAL dep, absent
        # on a plain ``pip install quill-sort``). numpy — the base fast-path dep —
        # already detects and exposes the CPU SIMD features it can use, so this
        # keeps the ISA line honest without another dependency. Without it the
        # panel used to report "no SIMD features detected" on an obviously
        # AVX2/AVX-512 box while the x86_simd_sort backend (which keys off the
        # SAME numpy detection) happily engaged — a self-contradiction.
        try:
            from numpy._core._multiarray_umath import __cpu_features__ as feats
        except Exception:
            from numpy.core._multiarray_umath import __cpu_features__ as feats  # numpy<2
        out = []
        if feats.get("AVX512F"):              out.append("avx512f")
        if feats.get("AVX2"):                 out.append("avx2")
        if feats.get("SSE42") or feats.get("SSE4_2"): out.append("sse4_2")
        if feats.get("ASIMD") or feats.get("NEON"):   out.append("neon")
        return out

    # py-cpuinfo first (richest), then numpy's detection, then the ARMv8 floor
    # (every ARMv8 chip has NEON/ASIMD even if a probe missed it).
    flags = _safe(_flags_via_cpuinfo, []) or []
    if not flags:
        flags = _safe(_flags_via_numpy, []) or []
    if not flags and info["arch"].lower() in ("arm64", "aarch64"):
        flags = ["neon"]
    info["features"] = flags
    info["avx512"] = any(f.startswith("avx512") for f in flags)
    info["avx2"] = "avx2" in flags
    info["neon"] = ("neon" in flags) or ("asimd" in flags)
    return info


def _detect_memory() -> Dict[str, Any]:
    """Total + currently available RAM in bytes via psutil. Returns ``{}`` if
    psutil isn't available."""
    def _via_psutil() -> Dict[str, int]:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        return {"total": int(vm.total), "available": int(vm.available)}
    return _safe(_via_psutil, {}) or {}


def _detect_gpus() -> List[Dict[str, str]]:
    """Probe NVIDIA (cupy/CUDA), AMD (ROCm via cupy-rocm or rocm-smi) and
    Apple Metal (mlx / arch). Quiet about missing back-ends."""
    gpus: List[Dict[str, str]] = []

    # NVIDIA + CUDA via cupy. This also confirms a working CUDA toolkit.
    def _nvidia() -> Optional[Dict[str, str]]:
        import cupy as cp  # type: ignore
        props = cp.cuda.runtime.getDeviceProperties(0)
        raw = props["name"]
        name = raw.decode() if isinstance(raw, bytes) else str(raw)
        rt = cp.cuda.runtime.runtimeGetVersion()
        major, minor = rt // 1000, (rt % 1000) // 10
        return {"vendor": "NVIDIA", "name": name,
                "toolkit": f"CUDA {major}.{minor}"}
    g = _safe(_nvidia)
    if g:
        gpus.append(g)

    # AMD ROCm — best-effort: try rocm-smi on path, else look for /opt/rocm.
    def _rocm() -> Optional[Dict[str, str]]:
        import shutil
        if shutil.which("rocm-smi") or os.path.isdir("/opt/rocm"):
            return {"vendor": "AMD", "name": "ROCm device", "toolkit": "ROCm"}
        return None
    g = _safe(_rocm)
    if g:
        gpus.append(g)

    # Apple Metal — present on every Apple Silicon Mac.
    def _metal() -> Optional[Dict[str, str]]:
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            return {"vendor": "Apple", "name": "Apple Silicon GPU",
                    "toolkit": "Metal"}
        return None
    g = _safe(_metal)
    if g:
        gpus.append(g)

    return gpus


def _gpu_present_ignoring_config(hw: Dict[str, Any]) -> bool:
    """Is a usable NVIDIA GPU present — independent of the current config?

    Used when deciding whether to enable ``use_gpu``. It must NOT consult
    ``CuPyBackend`` (whose probe is gated on the very ``use_gpu`` flag we're about
    to set, so a stale ``use_gpu=False`` would make it self-perpetuating). We
    trust the hardware detection first, then a direct CUDA device-count probe."""
    if any(g.get("vendor") == "NVIDIA" for g in hw.get("gpus", []) or []):
        return True
    def _cuda_devices() -> bool:
        import cupy as cp  # type: ignore
        return cp.cuda.runtime.getDeviceCount() >= 1
    return bool(_safe(_cuda_devices, False))


def _detect_numa() -> Dict[str, Any]:
    """How many NUMA nodes (sockets) this machine has. We only really care
    whether it's >1; multi-socket boxes get the NUMA backend recommendation."""
    info: Dict[str, Any] = {"nodes": 1, "source": "assumed"}

    # Linux: /sys/devices/system/node/node*/ count.
    def _linux() -> Optional[int]:
        nodes = [n for n in os.listdir("/sys/devices/system/node")
                 if n.startswith("node") and n[4:].isdigit()]
        return len(nodes) or None
    n = _safe(_linux)
    if n:
        info["nodes"], info["source"] = n, "/sys"
        return info

    # Windows: GetLogicalProcessorInformationEx — expensive, skip if psutil
    # gives us a simpler hint via cpu count vs physical cores.
    # Heuristic fallback: physical cores doesn't reveal sockets directly, so
    # we just stay at 1 unless something positively tells us otherwise.
    return info


def _detect_disks() -> List[Dict[str, Any]]:
    """List mounted disks with type (NVMe / SATA / unknown), filesystem and
    free-space. Best-effort via psutil; empty list if psutil isn't there."""
    def _via_psutil() -> List[Dict[str, Any]]:
        import psutil  # type: ignore
        disks: List[Dict[str, Any]] = []
        for part in psutil.disk_partitions(all=False):
            try:
                usage = psutil.disk_usage(part.mountpoint)
            except (PermissionError, OSError):
                continue
            kind = "unknown"
            dev = (part.device or "").lower()
            if "nvme" in dev:
                kind = "NVMe"
            elif any(x in dev for x in ("sd", "ata")):
                kind = "SATA"
            disks.append({
                "mount": part.mountpoint,
                "fstype": part.fstype,
                "kind": kind,
                "free": int(usage.free),
                "total": int(usage.total),
            })
        return disks
    return _safe(_via_psutil, []) or []


def _detect_toolchain() -> Dict[str, bool]:
    """Whether a C++ compiler we recognise is on PATH. Drives whether we offer
    the IPS4O / SIMD packages that need a local build."""
    import shutil

    return {
        "msvc": bool(_safe(lambda: shutil.which("cl"))),
        "clang": bool(_safe(lambda: shutil.which("clang") or shutil.which("clang++"))),
        "gcc": bool(_safe(lambda: shutil.which("gcc") or shutil.which("g++"))),
    }


def _detect_module(name: str) -> Tuple[bool, Optional[str]]:
    """Import *name* without side effects; return (present, version-or-None)."""
    try:
        mod = __import__(name)
    except Exception:
        return False, None
    return True, getattr(mod, "__version__", None)


def _detect_rust_backend() -> bool:
    """Is the compiled Rust radix backend importable here?"""
    try:
        from . import _fastsort  # noqa: F401
        return True
    except Exception:
        pass
    try:
        import quill_fastsort  # noqa: F401
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Recommendation engine — turn detected hardware into a package wishlist.
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: (label, pip target, why). The compiled CPU backends now ship inside
# quill-sort's wheels (quill._native), so the wizard no longer installs the old
# quill-fastsort* companions — it installs the OPTIONAL dependencies that unlock
# and complement the bundled fast path.

def _recommendations(hw: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """Build the install wishlist for this machine — the optional accelerators
    that are MISSING and would help. Returns ``[(label, pip-target, why), ...]``."""
    recs: List[Tuple[str, str, str]] = []

    if not _detect_module("numpy")[0]:
        recs.append(("numpy", "numpy",
                     "the array fast path — required for the compiled backends"))
    if not _detect_module("polars")[0]:
        recs.append(("polars", "polars",
                     "extra no-compile multi-threaded parallel sort (great fallback)"))
    if not _detect_module("psutil")[0]:
        recs.append(("psutil", "psutil",
                     "accurate RAM sensing for large-data routing"))

    # If the bundled native backend didn't load (e.g. a pure-Python wheel on an
    # exotic platform), point at polars as the best no-compile fast path.
    try:
        import quill._native  # noqa: F401
    except Exception:
        if not any(t == "polars" for _, t, _ in recs) and _detect_module("polars")[0] is False:
            pass  # polars already recommended above

    for gpu in hw.get("gpus", []) or []:
        if gpu.get("vendor") == "NVIDIA" and not _detect_module("cupy")[0]:
            recs.append((
                "cupy-cuda12x", "cupy-cuda12x",
                f"GPU radix on {gpu.get('name', 'CUDA device')} ({gpu.get('toolkit', 'CUDA')})",
            ))

    return recs


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-printers for the new sections.
# ─────────────────────────────────────────────────────────────────────────────

_BANNER_LINES = [
    r"   ___        _ _ _ ____             _   ",
    r"  / _ \ _   _(_) | / ___|  ___  _ __| |_ ",
    r" | | | | | | | | | \___ \ / _ \| '__| __|",
    r" | |_| | |_| | | | |___) | (_) | |  | |_ ",
    r"  \__\_\\__,_|_|_|_|____/ \___/|_|   \__|",
]


def _print_banner(st: _Style, out: Callable[[str], None]) -> None:
    out("")
    for line in _BANNER_LINES:
        out("  " + st.hot(line))
    out("")
    out("  " + st.bold(f"QuillSort {display_version()}")
        + "  " + st.dim(f"v{__version__}"))
    out("  " + st.dim("Setup wizard - tuning the sort for your machine."))
    out("")


def _human_bytes(n: Optional[int]) -> str:
    if not n or n < 0:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if isinstance(n, float) else f"{n} {unit}"
        n = n / 1024 if unit != "B" else n // 1024
    return f"{n:.1f} EB"


def _print_hardware(st: _Style, hw: Dict[str, Any],
                    out: Callable[[str], None]) -> None:
    cpu = hw.get("cpu", {})
    mem = hw.get("memory", {})
    gpus = hw.get("gpus", []) or []
    numa = hw.get("numa", {})
    disks = hw.get("disks", []) or []

    lines: List[str] = []

    brand = cpu.get("brand") or "unknown CPU"
    cores = cpu.get("logical", "?")
    phys = cpu.get("physical")
    core_str = f"{cores} logical" + (f" / {phys} physical" if phys else "")
    feats = []
    if cpu.get("avx512"): feats.append("AVX-512")
    if cpu.get("avx2"):   feats.append("AVX2")
    if cpu.get("neon"):   feats.append("NEON")
    feat_str = ", ".join(feats) if feats else st.dim("no SIMD features detected")
    lines.append(f"{st.info('CPU    ')} {st.bold(str(brand))}")
    lines.append(f"        {core_str}  {st.dim('(' + cpu.get('arch', '?') + ')')}")
    lines.append(f"        {st.dim('ISA:')} {feat_str}")

    if mem:
        lines.append(f"{st.info('RAM    ')} {_human_bytes(mem.get('total'))} total, "
                     f"{_human_bytes(mem.get('available'))} available")
    else:
        lines.append(f"{st.info('RAM    ')} {st.dim('unknown (install psutil for accurate sensing)')}")

    # Reconcile with what the DISPATCHER actually sees. The hardware probe
    # (_detect_gpus, via cupy's runtime) and the cupy_gpu backend's own probe
    # are separate code paths that can disagree — e.g. cupy importable but a
    # deep property query hiccups, or use_gpu turned off in config. That mismatch
    # is what produced the "GPU none detected" panel line sitting above a
    # dispatch ladder that scheduled cupy_gpu. Cross-check so the panel never
    # contradicts the ladder.
    gpu_backend = False
    try:
        gpu_backend = "cupy_gpu" in available_backends()
    except Exception:
        gpu_backend = False
    if gpus:
        for gpu in gpus:
            lines.append(f"{st.info('GPU    ')} {gpu.get('vendor', '?')} - "
                         f"{gpu.get('name', '?')}  "
                         f"{st.dim('(' + gpu.get('toolkit', '?') + ')')}")
            if not gpu_backend:
                # Own line so it can't be truncated off the end of the GPU row.
                lines.append(f"        {st.dim('cupy GPU backend not active — sorting on CPU')}")
    elif gpu_backend:
        # Backend probe found a usable CUDA sort even though the descriptive
        # probe came up empty — report the truth the dispatcher will act on.
        lines.append(f"{st.info('GPU    ')} {st.dim('CUDA device present (cupy GPU backend active)')}")
    else:
        lines.append(f"{st.info('GPU    ')} {st.dim('none detected (CPU-only sort)')}")

    nodes = numa.get("nodes", 1)
    if nodes > 1:
        lines.append(f"{st.info('NUMA   ')} {nodes} sockets "
                     f"{st.dim('(' + numa.get('source', '?') + ')')}")
    else:
        lines.append(f"{st.info('NUMA   ')} single-socket")

    if disks:
        for d in disks[:4]:                  # don't flood the box on a fileserver
            lines.append(f"{st.info('Disk   ')} {d['mount']}  "
                         f"{d['kind']}  {st.dim(d.get('fstype', '') or '')}  "
                         f"{_human_bytes(d['free'])} free")
        if len(disks) > 4:
            lines.append(st.dim(f"        (+{len(disks)-4} more disks)"))
    else:
        lines.append(f"{st.info('Disk   ')} {st.dim('no disks reported')}")

    out(_box(st, "Hardware", lines, width=78))
    out("")


def _print_recommendations(st: _Style, recs: List[Tuple[str, str, str]],
                           installed: List[str],
                           out: Callable[[str], None]) -> None:
    lines: List[str] = []
    if not recs:
        lines.append(st.dim("Nothing to install — every recommended backend is here."))
    else:
        lines.append(st.dim("Based on your hardware, Quill recommends:"))
        for label, target, why in recs:
            mark = st.ok(st.CHECK) if target in installed else st.dim(st.DOT)
            lines.append(f"  {mark} {st.bold(label)}  {st.dim(st.ARROW + ' ' + why)}")
    out(_box(st, "Recommendations", lines, width=82))
    out("")


def _dispatch_ladder_rows() -> Optional[List[Tuple[str, str, str]]]:
    """Compute the REAL per-(size, dtype) dispatch on this machine by asking the
    dispatcher itself which backend it uses for representative arrays — not a
    hardcoded template.

    Earlier versions printed a fixed preference list (``cupy_gpu`` for >1M, etc.)
    regardless of what was installed or measured, so the ladder could name a
    backend the machine can't run (a GPU path on a CPU-only box) or disagree with
    the hardware panel. Driving it from ``dispatch_sort`` + ``_LAST_BACKEND``
    makes every row ground truth: it can only ever show a backend that actually
    ran, it reflects the counting-sort fast path and the self-tuning winner, and
    it honours ``sort_array``'s small-array floor (which routes < _SMALL_ARRAY_N
    straight to np.sort). Returns None when numpy isn't importable.
    """
    try:
        import numpy as np
    except ImportError:
        return None
    from . import _backends
    try:
        from . import _SMALL_ARRAY_N
    except Exception:
        _SMALL_ARRAY_N = 200_000

    def probe(arr) -> str:
        # sort_array sends anything below the small-array floor straight to
        # np.sort, so report that honestly without invoking the backend chain.
        if arr.size < _SMALL_ARRAY_N or not _backends.eligible(arr):
            return "numpy"
        last = "numpy"
        # Warm briefly so the self-tuning dispatcher reports its CONVERGED pick
        # for this (dtype, size) bucket rather than a mid-exploration probe.
        for _ in range(4):
            _backends.dispatch_sort(arr.copy())
            last = _backends._LAST_BACKEND or "numpy"
        return last

    rng = np.random.default_rng(0)

    def gen(dt, n, bounded=False):
        dt = np.dtype(dt)
        if dt.kind in "iu":
            info = np.iinfo(dt)
            hi = 100 if bounded else min(info.max, 2 ** 40)
            lo = 0 if bounded else max(info.min, -(2 ** 40))
            return rng.integers(lo, hi, n, dtype=dt)
        return (rng.random(n) * (100 if bounded else 2 ** 20)).astype(dt)

    return [
        ("n < 200K (small)",      "int / float",   probe(gen(np.int64, 100_000))),
        ("200K - 2M (medium)",    "int64",         probe(gen(np.int64, 1_000_000))),
        ("200K - 2M (medium)",    "float64",       probe(gen(np.float64, 1_000_000))),
        ("> 2M (large)",          "int64",         probe(gen(np.int64, 3_000_000))),
        ("> 2M (large)",          "float64",       probe(gen(np.float64, 3_000_000))),
        ("> 2M dense / bounded",  "int (bounded)", probe(gen(np.int64, 3_000_000, bounded=True))),
    ]


def _print_dispatch_ladder(st: _Style, backends: List[str],
                           out: Callable[[str], None]) -> None:
    """Final 'what will Quill actually pick' table, computed live from the
    dispatcher on representative arrays (see :func:`_dispatch_ladder_rows`) so it
    reflects THIS machine — never a generic template naming absent backends."""
    lines: List[str] = []
    lines.append(st.dim("Quill's per-size dispatch on this machine (measured):"))
    lines.append("")
    header = f"  {'size class':<22}  {'dtype':<14}  backend"
    lines.append(st.dim(header))

    rows = _safe(_dispatch_ladder_rows, None)
    if not rows:
        lines.append("")
        lines.append(st.dim("  numpy not installed — correctness via the "
                            "standard-library Timsort."))
        out(_box(st, "Dispatch ladder", lines, width=82))
        out("")
        return
    for size, dt, b in rows:
        lines.append(f"  {size:<22}  {dt:<14}  {st.ok(b)}")
    out(_box(st, "Dispatch ladder", lines, width=82))
    out("")


# ─────────────────────────────────────────────────────────────────────────────
# Calibration — small int64 sweep, writes per-backend crossover thresholds.
# ─────────────────────────────────────────────────────────────────────────────

_CALIB_SIZES: Tuple[int, ...] = (10_000, 100_000, 1_000_000)
_TIME_BUDGET_S = 8.0
_REPEATS = 2


def _time_sort(fn: Callable[[], object]) -> float:
    best = float("inf")
    for _ in range(_REPEATS):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _calibrate(st: _Style, hw: Dict[str, Any],
               out: Callable[[str], None]) -> Optional[Dict[str, Any]]:
    """Tiny benchmark sweep at n=10K, 100K, 1M of int64 timing each available
    backend; pick the smallest n where each one starts to win, write those
    per-backend crossover thresholds into the config."""
    try:
        import numpy as np
    except ImportError:
        out(st.warn(f"  {st.CROSS} numpy not installed - skipping calibration."))
        out("")
        return None

    from . import _backends

    out(st.dim(f"  Sweep: {', '.join(f'{n:,}' for n in _CALIB_SIZES)} int64  "
               f"(best of {_REPEATS}, <={int(_TIME_BUDGET_S)}s budget)"))
    out("")
    out(st.dim(f"  {'n':>10}  {'np.sort':>9}  {'quill':>9}  "
               f"{'speedup':>8}  backend"))

    deadline = time.perf_counter() + _TIME_BUDGET_S
    rng = np.random.default_rng(0xC0FFEE)
    backend_first_win: Dict[str, int] = {}
    crossover_n: Optional[int] = None
    best_speedup = 1.0

    for n in _CALIB_SIZES:
        if time.perf_counter() > deadline:
            out(st.dim("  (time budget reached - stopping sweep)"))
            break
        base = rng.integers(-(2**40), 2**40, size=n, dtype=np.int64)

        t_np = _time_sort(lambda: np.sort(base.copy()))
        t_q = _time_sort(lambda: _backends.dispatch_sort(base.copy()))
        used = _backends._LAST_BACKEND or "numpy"
        speedup = (t_np / t_q) if t_q > 0 else 1.0
        flag = st.ok(st.CHECK) if speedup >= 1.10 else st.dim(st.DOT)
        out(f"  {n:>10,}  {t_np*1e3:>8.2f}m  {t_q*1e3:>8.2f}m  "
            f"{speedup:>7.2f}x  {used} {flag}")
        if speedup >= 1.10 and used not in ("numpy", "counting"):
            backend_first_win.setdefault(used, n)
            if crossover_n is None:
                crossover_n = n
        best_speedup = max(best_speedup, speedup)
    out("")

    ncores = int(hw.get("cpu", {}).get("logical") or 1)
    parallel_pays = best_speedup >= 1.10 and ncores >= 4
    delta: Dict[str, Any] = {
        "auto_parallel": bool(parallel_pays),
        "parallel_min_n": int(crossover_n) if crossover_n
                          else int(DEFAULTS["parallel_min_n"]),
        "parallel_min_cores": 4 if parallel_pays else int(
            DEFAULTS["parallel_min_cores"]),
        "calibrated": True,
        # Per-backend first-win thresholds for power users to inspect.
        "backend_thresholds": backend_first_win,
    }
    # GPU: only ever POSITIVELY enable — never auto-write use_gpu=False. A flaky
    # descriptive probe (cupy importable but a property query momentarily fails)
    # previously wrote use_gpu=False, and CuPyBackend then refused to engage a
    # perfectly good card — a silent, self-perpetuating brick (the next probe,
    # gated on the now-false config, also fails). The backend's own probe runs a
    # real sort kernel and disables itself safely on CPU-only boxes, so leaving
    # use_gpu at its default (True) is correct there. Set True only when a GPU is
    # actually present (checked independently of the possibly-stale config).
    if _gpu_present_ignoring_config(hw):
        delta["use_gpu"] = True
    return delta


# ─────────────────────────────────────────────────────────────────────────────
# Install phase — uses rich progress wrapper when present, plain pip otherwise.
# ─────────────────────────────────────────────────────────────────────────────

def _do_installs(st: _Style, recs: List[Tuple[str, str, str]],
                 already: List[str],
                 out: Callable[[str], None], *,
                 interactive: bool, assume_yes: bool) -> List[str]:
    """Walk the recommendations and pip-install the ones the user wants.

    Returns the list of *pip targets* that ended up installed (so the caller
    can re-probe). When ``interactive`` is False and ``assume_yes`` is False,
    we don't install anything silently — just print the pip command per missing
    package."""
    missing = [r for r in recs if r[1] not in already]
    if not missing:
        out(_box(st, "Install", [
            st.ok(f"  {st.CHECK} Every recommended package is already installed."),
        ], width=70))
        out("")
        return []

    installed: List[str] = []
    out(st.bold("  Installing recommended accelerators"))
    out("")
    for label, target, why in missing:
        out(f"  {st.ARROW} {st.bold(label)}  {st.dim(why)}")
        if assume_yes:
            do_it = True
        elif interactive:
            try:
                ans = input(f"    install {target}? [Y/n] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                ans = "n"
                out("")
            do_it = ans not in ("n", "no")
        else:
            out(f"    {st.warn('pip install ' + target)}   {st.dim('(run this to enable)')}")
            continue

        if not do_it:
            out(f"    {st.dim('skipped')}")
            continue

        ok = _progress_pip(target, out)
        if ok:
            out(f"    {st.ok(st.CHECK + ' installed ' + label)}")
            installed.append(target)
        else:
            out(f"    {st.bad(st.CROSS + ' install failed (continuing)')}")
    out("")
    return installed


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point — interface preserved for the CLI.
# ─────────────────────────────────────────────────────────────────────────────

def run_wizard(noninteractive: bool = False, stream=None,
               assume_yes: bool = False, install: bool = True) -> Dict[str, object]:
    """Run the Quill setup wizard end-to-end and return the saved config dict.

    Steps, in order:
      1. ASCII banner + version.
      2. Hardware detection (CPU/ISA, RAM, GPU, NUMA, disks) - each probe
         runs behind a spinner via ``_step``.
      3. Recommendation table for the companion packages this hardware can use.
      4. Optional install pass (skipped with ``install=False``); uses pip's
         own progress bar so the user sees download / install progress live.
      5. Calibration sweep at n=10K, 100K, 1M of int64; per-backend crossover
         thresholds written into the config via ``save_config``.
      6. Dispatch-ladder display showing what Quill will pick per (size, dtype).
      7. Success summary with the path to the written config.

    Parameters
    ----------
    noninteractive : bool
        When True, never block on input(); install offers fall back to "show
        the pip command" unless ``assume_yes`` is also set. The wizard never
        hangs regardless.
    stream :
        Output stream (defaults to ``sys.stdout``); injectable for testing.
    assume_yes : bool
        Answer "yes" to every install offer and the config write (``--yes``).
    install : bool
        When False, skip the install-offer step entirely (``--no-install``).

    Returns
    -------
    dict
        The config that was written (or the existing/default config if there
        was nothing to calibrate or the user declined).
    """
    st = _Style(stream)
    s = stream or sys.stdout

    def out(text: str = "") -> None:
        try:
            print(text, file=s)
        except UnicodeEncodeError:
            enc = getattr(s, "encoding", None) or "ascii"
            print(text.encode(enc, "replace").decode(enc), file=s)

    # 1. Welcome banner.
    _print_banner(st, out)

    # 2. Hardware detection (each step quietly probes; never raises).
    hw: Dict[str, Any] = {}
    out(st.bold("  Detecting hardware"))
    hw["cpu"] = _step("CPU model, cores, ISA features",
                      _detect_cpu, out)
    hw["memory"] = _step("Total and available RAM",
                         _detect_memory, out)
    hw["gpus"] = _step("GPU presence (CUDA / ROCm / Metal)",
                       _detect_gpus, out)
    hw["numa"] = _step("NUMA topology",
                       _detect_numa, out)
    hw["disks"] = _step("Disks and free space",
                        _detect_disks, out)
    hw["toolchain"] = _step("Local C++ toolchain",
                            _detect_toolchain, out)
    out("")
    _print_hardware(st, hw, out)

    # 3. Recommendation list — the OPTIONAL accelerators missing on this box.
    #    (The compiled CPU backends already ship inside quill-sort's wheel.)
    recs = _recommendations(hw)
    _print_recommendations(st, recs, [], out)

    # 4. Install phase — one confirm, then a genuine PARALLEL download with live
    #    real-bytes progress bars (quill._installer). Non-interactive without
    #    --yes just prints the pip command.
    if install and recs:
        interactive = (not noninteractive) and bool(
            getattr(sys.stdin, "isatty", lambda: False)())
        if interactive or assume_yes:
            try:
                from ._installer import install_packages
                results = install_packages(
                    [(target, label) for (label, target, _why) in recs],
                    out=out, assume_yes=assume_yes)
                if any(results.values()):
                    from . import _backends
                    _backends.reset_availability()
            except Exception:
                if os.environ.get("QUILL_BACKEND_DEBUG"):
                    raise
                out("  installer unavailable — run: pip install "
                    + " ".join(t for _, t, _ in recs))
        else:
            out("  To install these accelerators, run:")
            out("    pip install " + " ".join(t for _, t, _ in recs))
            out("")

    # 5. Calibration sweep.
    out(st.bold("  Calibration"))
    before = dict(load_config(force=True))
    delta = _calibrate(st, hw, out)
    if delta is None:
        out(st.dim("  No measurements — keeping existing config."))
        out("")
        _print_dispatch_ladder(st, available_backends(), out)
        out(st.ok(f"  {st.CHECK} Setup complete. quill is ready."))
        out(f"  {st.dim('config path:')} {st.bold(config_path())}")
        out("")
        return before

    # Show what we'd save, then ask once (unless we're noninteractive or -y).
    after_preview = dict(before)
    after_preview.update(delta)
    if not noninteractive and not assume_yes:
        try:
            ans = input("  Write this calibrated config? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = "n"
            out("")
        if ans in ("n", "no"):
            out(st.warn(f"  {st.CROSS} Aborted - no changes written."))
            out("")
            return before

    path = save_config(delta)
    out("")
    out(st.ok(f"  {st.CHECK} Calibrated config written to:"))
    out(f"    {st.bold(path)}")
    out("")

    # 6. Dispatch ladder display.
    _print_dispatch_ladder(st, available_backends(), out)

    # 7. Success summary.
    out(st.ok(f"  {st.CHECK} Setup complete! quill is ready."))
    out(f"  {st.dim('config:')}  {path}")
    out(f"  {st.dim('re-run')}  {st.bold('quill setup')}  "
        f"{st.dim('any time your hardware changes.')}")
    out("")
    return dict(load_config(force=True))


if __name__ == "__main__":  # pragma: no cover - convenience for direct runs
    _yes = "--yes" in sys.argv or "-y" in sys.argv
    run_wizard(noninteractive=_yes or "--noninteractive" in sys.argv,
               assume_yes=_yes, install="--no-install" not in sys.argv)

"""
quill/_installer.py — honest, parallel wheel downloader with live progress.

Used by ``quill setup`` to fetch optional accelerators (numpy / polars / psutil /
cupy, and the native backend on the rare platform without a prebuilt wheel).

Truth-in-progress
=================
The ``#####`` bars show REAL bytes off the wire, not a fake animation. For each
package we resolve the exact wheel URL and its size from PyPI's JSON API, then
stream-download it ourselves, updating the byte counter as data arrives:

    numpy            [###########                 ]  6.2mb/16.4mb
    polars           [######################      ] 24.9mb/31.0mb

If a server sends no length (rare), we show the bytes downloaded with a moving
marker instead of inventing a percentage — no lying to the user. Downloads run
in parallel; once fetched, ``pip install --find-links`` installs them offline.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Tuple

_BAR_WIDTH = 30
_PYPI = "https://pypi.org/pypi/{}/json"


# ─────────────────────────────────────────────────────────────────────────────
# Wheel resolution — find the exact wheel this interpreter/platform needs.
# ─────────────────────────────────────────────────────────────────────────────

def _compatible_tags() -> Optional[set]:
    try:
        from packaging.tags import sys_tags
        return {str(t) for t in sys_tags()}
    except Exception:
        return None


def _wheel_matches(filename: str, tagset: Optional[set]) -> bool:
    if not filename.endswith(".whl"):
        return False
    stem = filename[:-4]
    parts = stem.split("-")
    if len(parts) < 3:
        return False
    py, abi, plat = parts[-3], parts[-2], parts[-1]
    if tagset is None:
        # No packaging.tags available: accept pure-python wheels, else a coarse
        # interpreter/platform match. Conservative — pip is the real installer.
        if plat == "any":
            return True
        return sys.platform[:3] in plat or "manylinux" in plat
    for p in py.split("."):
        for a in abi.split("."):
            for pl in plat.split("."):
                if f"{p}-{a}-{pl}" in tagset:
                    return True
    return False


def resolve_wheel(pkg: str, timeout: float = 20.0
                  ) -> Optional[Tuple[str, str, Optional[int]]]:
    """Return (url, filename, size_bytes) for the best wheel of *pkg* on this
    platform, or None if it can't be resolved (caller falls back to plain pip)."""
    try:
        with urllib.request.urlopen(_PYPI.format(pkg), timeout=timeout) as r:
            data = json.load(r)
    except Exception:
        return None
    tagset = _compatible_tags()
    # The latest release's files are under info.version -> releases[version].
    version = data.get("info", {}).get("version")
    files = data.get("releases", {}).get(version, []) or data.get("urls", [])
    best = None
    for f in files:
        if f.get("packagetype") != "bdist_wheel":
            continue
        if f.get("yanked"):
            continue
        if _wheel_matches(f.get("filename", ""), tagset):
            # Prefer a platform wheel over a pure-python one if both match.
            plat_specific = not f["filename"].endswith("-any.whl")
            if best is None or (plat_specific and best[3] is False):
                best = (f["url"], f["filename"], f.get("size"), plat_specific)
    if best is None:
        return None
    return best[0], best[1], best[2]


# ─────────────────────────────────────────────────────────────────────────────
# Live multi-bar rendering (real bytes)
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_mb(n: Optional[int]) -> str:
    if n is None:
        return "  ? mb"
    return f"{n / 1_048_576:.1f}mb"


def _bar(label: str, done: int, total: Optional[int], status: str, tick: int) -> str:
    if total and total > 0:
        frac = min(1.0, done / total)
        filled = int(round(frac * _BAR_WIDTH))
        bar = "#" * filled + " " * (_BAR_WIDTH - filled)
        right = f"{_fmt_mb(done)}/{_fmt_mb(total)}"
    elif status == "done":
        bar = "#" * _BAR_WIDTH
        right = f"{_fmt_mb(done)}/{_fmt_mb(done)}"
    else:
        # Unknown size: honest moving marker, no invented percentage.
        pos = tick % _BAR_WIDTH
        bar = " " * pos + "#" + " " * (_BAR_WIDTH - pos - 1)
        right = f"{_fmt_mb(done)} so far"
    tag = {"error": " FAILED", "done": " done"}.get(status, "")
    return f"  {label:<16}[{bar}] {right}{tag}"


class _Progress:
    def __init__(self, labels: List[str]):
        self.state: Dict[str, dict] = {
            lbl: {"done": 0, "total": None, "status": "waiting"} for lbl in labels
        }
        self._labels = labels
        self._lock = threading.Lock()
        self._tty = sys.stderr.isatty()
        self._printed = False
        self._tick = 0

    def update(self, label: str, done: int, total: Optional[int], status: str):
        with self._lock:
            s = self.state[label]
            s["done"], s["status"] = done, status
            if total is not None:
                s["total"] = total

    def render(self):
        with self._lock:
            lines = [_bar(l, self.state[l]["done"], self.state[l]["total"],
                          self.state[l]["status"], self._tick) for l in self._labels]
            self._tick += 1
        block = "\n".join(lines)
        if self._tty:
            if self._printed:
                sys.stderr.write(f"\033[{len(lines)}A")  # cursor up N lines
            for ln in lines:
                sys.stderr.write("\033[K" + ln + "\n")   # clear line + write
            sys.stderr.flush()
            self._printed = True
        # non-TTY: caller prints snapshots explicitly (avoid log spam)
        return block


# ─────────────────────────────────────────────────────────────────────────────
# Download + install
# ─────────────────────────────────────────────────────────────────────────────

def _download(label: str, url: str, dest: str, prog: _Progress,
              declared_size: Optional[int]) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "quill-setup"})
        with urllib.request.urlopen(req, timeout=60) as r:
            total = declared_size
            cl = r.headers.get("Content-Length")
            if cl and cl.isdigit():
                total = int(cl)
            prog.update(label, 0, total, "downloading")
            done = 0
            with open(dest, "wb") as fh:
                while True:
                    chunk = r.read(65536)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    prog.update(label, done, total, "downloading")
        prog.update(label, done, total or done, "done")
        return True
    except Exception:
        prog.update(label, prog.state[label]["done"],
                    prog.state[label]["total"], "error")
        return False


def download_parallel(targets: List[Tuple[str, str, str, Optional[int]]],
                      out: Callable[[str], None] = print) -> Dict[str, bool]:
    """targets: [(label, url, dest_path, declared_size)]. Streams all in parallel
    with a live real-bytes progress block. Returns {label: ok}."""
    labels = [t[0] for t in targets]
    prog = _Progress(labels)
    # Seed the totals we already learned from PyPI so the bar shows the real
    # size from the very first frame (honest: we know it before we connect).
    for (lbl, _url, _dest, size) in targets:
        if size:
            prog.state[lbl]["total"] = size
    results: Dict[str, bool] = {}
    stop = threading.Event()

    def renderer():
        while not stop.is_set():
            prog.render()
            time.sleep(1 / 15)
        prog.render()  # final frame

    rt = threading.Thread(target=renderer, daemon=True)
    if prog._tty:
        # reserve the block
        for _ in labels:
            sys.stderr.write("\n")
        sys.stderr.write(f"\033[{len(labels)}A")
        rt.start()

    with ThreadPoolExecutor(max_workers=min(8, len(targets))) as pool:
        futs = {lbl: pool.submit(_download, lbl, url, dest, prog, size)
                for (lbl, url, dest, size) in targets}
        if not prog._tty:
            # non-TTY: emit periodic honest snapshots instead of animating
            while any(not f.done() for f in futs.values()):
                out(prog.render())
                out("")
                time.sleep(1.0)
        for lbl, f in futs.items():
            results[lbl] = f.result()

    stop.set()
    if prog._tty:
        rt.join(timeout=1.0)
        sys.stderr.write("\n")
    else:
        out(prog.render())
    return results


def install_packages(packages: List[Tuple[str, str]],
                     out: Callable[[str], None] = print,
                     assume_yes: bool = False) -> Dict[str, bool]:
    """packages: [(pip_name, friendly_label)]. Resolves wheels, shows what will
    be installed, confirms (unless assume_yes), downloads with genuine progress,
    then installs offline. Returns {friendly_label: ok}."""
    resolved: List[Tuple[str, str, str, Optional[int]]] = []
    unresolved: List[Tuple[str, str]] = []
    tmp = tempfile.mkdtemp(prefix="quill-setup-")
    for pip_name, label in packages:
        info = resolve_wheel(pip_name)
        if info is None:
            unresolved.append((pip_name, label))
            continue
        url, filename, size = info
        resolved.append((label, url, os.path.join(tmp, filename), size))

    total_mb = sum((t[3] or 0) for t in resolved) / 1_048_576
    out("")
    out("  Quill will download and install:")
    for (label, _url, dest, size) in resolved:
        out(f"    - {label:<16} {_fmt_mb(size)}")
    for (pip_name, label) in unresolved:
        out(f"    - {label:<16} (via pip)")
    out(f"    total ~{total_mb:.1f} mb")
    out("")

    if not assume_yes:
        try:
            ans = input("  Proceed? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = "n"
        if ans not in ("y", "yes"):
            out("  cancelled.")
            shutil.rmtree(tmp, ignore_errors=True)
            return {}

    results: Dict[str, bool] = {}
    if resolved:
        dl = download_parallel(resolved, out=out)
        results.update(dl)
        # Install everything we successfully downloaded, offline (prefer local
        # wheels; pip pulls any missing pure-python deps).
        ok_files = [t[2] for t in resolved if dl.get(t[0])]
        if ok_files:
            rc = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--find-links", tmp,
                 "--quiet"] + ok_files,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            ).returncode
            for t in resolved:
                results[t[0]] = results.get(t[0], False) and rc == 0

    # Fall back to plain pip for anything we couldn't resolve a wheel for.
    for (pip_name, label) in unresolved:
        out(f"  installing {label} via pip ...")
        rc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", pip_name],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode
        results[label] = rc == 0

    shutil.rmtree(tmp, ignore_errors=True)
    out("")
    for label, ok in results.items():
        out(f"  {'✓' if ok else '✗'} {label}")
    return results


if __name__ == "__main__":
    # Demo: download (don't install) whatever packages are named on argv, showing
    # the genuine progress bars. Defaults to a couple of small ones.
    pkgs = sys.argv[1:] or ["psutil", "polars"]
    tmp = tempfile.mkdtemp(prefix="quill-demo-")
    targets = []
    for p in pkgs:
        info = resolve_wheel(p)
        if info:
            url, fn, size = info
            targets.append((p, url, os.path.join(tmp, fn), size))
        else:
            print(f"(could not resolve a wheel for {p})")
    if targets:
        res = download_parallel(targets)
        print("results:", res)
    shutil.rmtree(tmp, ignore_errors=True)

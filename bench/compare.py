#!/usr/bin/env python3
"""
bench/compare.py HEAD.json BASELINE.json — decide whether the post-merge build
regressed vs the previous stable release (or fell badly behind numpy).

Exit 0 = OK, 1 = regression (CI then pings Claude to investigate/fix).

Why the gate is deliberately hard to trip
------------------------------------------
GitHub runners are noisy and Quill's self-tuning dispatcher can pick a different
backend from run to run, so a single case drifting 15-20% is almost always noise,
not a real regression. A gate that fires on that is worse than no gate — it burns
a Claude run chasing a phantom. So we require a *convincing* signal to fire:

  * >= 2 cases each slower than the previous stable by more than --tol (25%), OR
  * any single case slower than the previous stable by more than --severe (50%)
    (a doubling-class regression is unambiguous), OR
  * any case where HEAD's quill is more than --numpy-tol (1.30x) slower than
    HEAD's own numpy — i.e. we stopped being competitive with the baseline numpy.

A single mild drift is reported but does NOT fail the build. Every threshold is
CLI-tunable. Output is ASCII-only so it renders on any console and in a GH issue.
"""
from __future__ import annotations

import argparse
import json
import sys


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("head")
    ap.add_argument("baseline")
    ap.add_argument("--tol", type=float, default=0.25,
                    help="per-case slowdown vs previous stable that counts as a regression")
    ap.add_argument("--severe", type=float, default=0.50,
                    help="a single case this much slower than previous stable fails on its own")
    ap.add_argument("--numpy-tol", type=float, default=1.30,
                    help="max quill/numpy ratio before flagging loss of numpy-competitiveness")
    ap.add_argument("--report", default=None, help="write a markdown report here")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    head = load(args.head)
    base = load(args.baseline)
    hc, bc = head["cases"], base["cases"]

    rows = []
    mild = []       # regressed > tol vs baseline
    severe = []     # regressed > severe vs baseline
    numpy_bad = []  # lost to numpy beyond numpy_tol
    for cid in sorted(hc):
        h = hc[cid]
        b = bc.get(cid)
        hq, hn = h["quill_ms"], h["numpy_ms"]
        bq = b["quill_ms"] if b else None
        notes = []
        delta_str = "-"
        if bq:
            delta = hq / bq - 1.0
            delta_str = f"{delta:+.0%}"
            if hq > bq * (1.0 + args.severe):
                severe.append(cid); notes.append(f"SEVERE vs {base['quill_version']} ({delta:+.0%})")
            elif hq > bq * (1.0 + args.tol):
                mild.append(cid); notes.append(f"slower vs {base['quill_version']} ({delta:+.0%})")
        ratio = hq / hn if hn else float("inf")
        if ratio > args.numpy_tol:
            numpy_bad.append(cid); notes.append(f"SLOW vs numpy ({ratio:.2f}x)")
        rows.append((cid, hq, bq, delta_str, hn, ratio, "; ".join(notes) if notes else "ok"))

    # Fire only on a convincing signal (see module docstring).
    fail = (len(mild) + len(severe) >= 2) or bool(severe) or bool(numpy_bad)

    lines = [
        "# Post-merge benchmark",
        "",
        f"- HEAD quill **{head['quill_version']}**, numpy {head['numpy_version']}",
        f"- baseline quill **{base['quill_version']}** (previous stable), numpy {base['numpy_version']}",
        f"- thresholds: regression>{args.tol:.0%}, severe>{args.severe:.0%}, numpy-floor {args.numpy_tol:.2f}x",
        "",
        "| case | HEAD quill | baseline quill | delta | HEAD numpy | quill/numpy | verdict |",
        "|---|--:|--:|--:|--:|--:|:--|",
    ]
    for cid, hq, bq, delta_str, hn, ratio, verdict in rows:
        bq_str = f"{bq:.3f}" if bq else "-"
        lines.append(f"| {cid} | {hq:.3f} | {bq_str} | {delta_str} | {hn:.3f} | {ratio:.2f}x | {verdict} |")
    lines.append("")
    if fail:
        lines.append(f"## FAIL: regression detected")
        if severe:    lines.append(f"- severe (>{args.severe:.0%}) vs previous stable: {', '.join(severe)}")
        if mild:      lines.append(f"- slower (>{args.tol:.0%}) vs previous stable: {', '.join(mild)}")
        if numpy_bad: lines.append(f"- lost to numpy (>{args.numpy_tol:.2f}x): {', '.join(numpy_bad)}")
    elif mild:
        lines.append(f"## OK (with a caveat): one mild drift ({', '.join(mild)}) - within noise, not failing.")
    else:
        lines.append("## OK: no regressions - all cases within tolerance.")

    report = "\n".join(lines)
    print(report)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(report)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())

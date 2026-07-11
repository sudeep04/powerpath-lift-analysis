"""Gate report: compare hand-labeled bar positions (tools/label.py) against
the pipeline's overlay output and PASS/FAIL the M1 accuracy gates.

Usage (engine venv)::

    python tools/gate_report.py VIDEO.labels.json overlay.json --scale-mm-per-px F

Inputs:

- labels JSON: ``{"video": ..., "clicks": [{"name", "t", "frame_index",
  "x", "y"}]}``, optionally carrying ``"rep_count"``.
- overlay JSON (pipeline): ``{"frames": [{"t": <pts seconds>,
  "bar": [x, y] | null, "skeleton": {...}}], "reps": [...]}``.

Matching: the bar position at a label's t is taken from the overlay frame
with the nearest t among frames that HAVE a bar (null-bar frames are
tolerated and skipped), and only within a 50 ms tolerance -- a keyframe
with no bar-bearing overlay frame within 50 ms is reported UNMATCHED and
fails the gate.

Gates (M1): PASS requires every labeled keyframe |bar - label| <= 1.0 cm
(``cm = px * scale_mm_per_px / 10``) AND matching rep counts when BOTH
files carry one (labels ``rep_count`` vs ``len(overlay["reps"])``); the
rep gate is skipped if either side lacks a count. Prints a per-keyframe
table (name, t, dx px, dy px, dist px, dist cm, PASS/FAIL) and exits 0 on
PASS, 1 on FAIL.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

MATCH_TOLERANCE_S = 0.050
GATE_THRESHOLD_CM = 1.0


@dataclass
class KeyframeRow:
    """One labeled keyframe's comparison. The px/cm fields are None when the
    keyframe is UNMATCHED (no bar-bearing overlay frame within tolerance)."""

    name: str
    t: float
    matched: bool
    dx_px: float | None
    dy_px: float | None
    dist_px: float | None
    dist_cm: float | None
    passed: bool


@dataclass
class GateResult:
    rows: list[KeyframeRow]
    label_rep_count: int | None
    overlay_rep_count: int | None
    rep_count_checked: bool
    rep_count_ok: bool  # True when the rep gate is skipped
    passed: bool


def px_to_cm(px: float, scale_mm_per_px: float) -> float:
    return px * scale_mm_per_px / 10.0


def nearest_bar_frame(
    overlay_frames: list[dict], t: float, tolerance_s: float = MATCH_TOLERANCE_S
) -> dict | None:
    """The overlay frame with a non-null bar whose t is nearest to `t`,
    or None if the nearest such frame is more than `tolerance_s` away."""
    best: dict | None = None
    best_dt = math.inf
    for entry in overlay_frames:
        if entry.get("bar") is None:
            continue
        dt = abs(entry["t"] - t)
        if dt < best_dt:
            best, best_dt = entry, dt
    if best is None or best_dt > tolerance_s:
        return None
    return best


def evaluate(
    labels: dict,
    overlay: dict,
    scale_mm_per_px: float,
    tolerance_s: float = MATCH_TOLERANCE_S,
    threshold_cm: float = GATE_THRESHOLD_CM,
) -> GateResult:
    """Apply the M1 gates to hand-built labels/overlay dicts (see module
    docstring for the exact rules). An empty click list fails the gate --
    a report proving nothing must not pass."""
    rows: list[KeyframeRow] = []
    overlay_frames = overlay.get("frames", [])
    for click in labels.get("clicks", []):
        match = nearest_bar_frame(overlay_frames, click["t"], tolerance_s)
        if match is None:
            rows.append(
                KeyframeRow(
                    name=click["name"],
                    t=click["t"],
                    matched=False,
                    dx_px=None,
                    dy_px=None,
                    dist_px=None,
                    dist_cm=None,
                    passed=False,
                )
            )
            continue
        dx_px = float(match["bar"][0]) - float(click["x"])
        dy_px = float(match["bar"][1]) - float(click["y"])
        dist_px = math.hypot(dx_px, dy_px)
        dist_cm = px_to_cm(dist_px, scale_mm_per_px)
        rows.append(
            KeyframeRow(
                name=click["name"],
                t=click["t"],
                matched=True,
                dx_px=dx_px,
                dy_px=dy_px,
                dist_px=dist_px,
                dist_cm=dist_cm,
                passed=dist_cm <= threshold_cm,
            )
        )

    label_rep_count = labels.get("rep_count")
    reps = overlay.get("reps")
    overlay_rep_count = len(reps) if isinstance(reps, list) else None
    rep_count_checked = label_rep_count is not None and overlay_rep_count is not None
    rep_count_ok = (not rep_count_checked) or label_rep_count == overlay_rep_count

    passed = bool(rows) and all(row.passed for row in rows) and rep_count_ok
    return GateResult(
        rows=rows,
        label_rep_count=label_rep_count,
        overlay_rep_count=overlay_rep_count,
        rep_count_checked=rep_count_checked,
        rep_count_ok=rep_count_ok,
        passed=passed,
    )


def format_report(result: GateResult) -> str:
    header = (
        f"{'keyframe':<12} {'t (s)':>8} {'dx px':>8} {'dy px':>8}"
        f" {'dist px':>8} {'dist cm':>8}  status"
    )
    lines = [header, "-" * len(header)]
    for row in result.rows:
        if not row.matched:
            lines.append(
                f"{row.name:<12} {row.t:>8.3f} {'-':>8} {'-':>8} {'-':>8} {'-':>8}  UNMATCHED"
            )
        else:
            status = "PASS" if row.passed else "FAIL"
            lines.append(
                f"{row.name:<12} {row.t:>8.3f} {row.dx_px:>8.1f} {row.dy_px:>8.1f}"
                f" {row.dist_px:>8.1f} {row.dist_cm:>8.2f}  {status}"
            )
    if not result.rows:
        lines.append("(no labeled keyframes)")
    if result.rep_count_checked:
        rep_status = "PASS" if result.rep_count_ok else "FAIL"
        lines.append(
            f"reps: labels={result.label_rep_count}"
            f" overlay={result.overlay_rep_count}  {rep_status}"
        )
    else:
        lines.append("reps: not compared (count missing from labels and/or overlay)")
    lines.append(f"gate: {'PASS' if result.passed else 'FAIL'}")
    return "\n".join(lines)


def _positive_float(raw: str) -> float:
    value = float(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {raw}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="PASS/FAIL the M1 accuracy gates: labeled keyframes vs overlay bar path."
    )
    parser.add_argument("labels_json", type=Path, help="VIDEO.labels.json from tools/label.py")
    parser.add_argument("overlay_json", type=Path, help="overlay.json from the pipeline")
    parser.add_argument(
        "--scale-mm-per-px",
        type=_positive_float,
        required=True,
        help="bar-plane scale in mm per pixel (from calibration)",
    )
    args = parser.parse_args(argv)

    labels = json.loads(args.labels_json.read_text())
    overlay = json.loads(args.overlay_json.read_text())
    result = evaluate(labels, overlay, args.scale_mm_per_px)
    print(format_report(result))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

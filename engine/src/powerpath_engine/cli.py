"""The ``powerpath`` command-line interface (argparse; no extra deps).

Three subcommands:

* ``powerpath analyze VIDEO --movement KEY --load KG [--height CM]
  [--pose fake|rtmlib|mediapipe] [--out DIR]`` -- run the full pipeline and
  write ``metrics.json``, ``overlay.json`` and ``annotated.mp4`` into the
  output directory (default: next to the video), then print a rep table
  (rep, made, score, top fault). Progress goes to stderr so stdout stays
  the result.
* ``powerpath extract-fixtures VIDEO --movement KEY --out DIR [--pose ...]``
  -- freeze the extracted bar + landmark series as JSON golden fixtures
  (full-rate pose, stride 1): the manually-run fixture-extraction step the
  global constraints call for, so future regression tests replay real-model
  output without re-running inference.
* ``powerpath movements`` -- list the registry's movement keys.

Exit codes: 0 on success, 2 on any expected failure (bad movement key,
undecodable video, failed calibration, missing pose extra), with the reason
on stderr.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from powerpath_engine import decode, overlay, registry
from powerpath_engine.calibration import CalibrationError
from powerpath_engine.decode import DecodeError
from powerpath_engine.pipeline import AnalysisResult, analyze
from powerpath_engine.pose import PoseUnavailableError, StridedPose, make_pose_backend

_POSE_CHOICES = ("fake", "rtmlib", "mediapipe")
_DEFAULT_HEIGHT_CM = 175.0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (
        registry.UnknownMovementError,
        DecodeError,
        CalibrationError,
        PoseUnavailableError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="powerpath", description="PowerPath lift analysis")
    sub = parser.add_subparsers(required=True)

    p_analyze = sub.add_parser("analyze", help="analyze a lift video")
    p_analyze.add_argument("video", help="path to the video file")
    p_analyze.add_argument("--movement", required=True, help="movement key (see `movements`)")
    p_analyze.add_argument("--load", required=True, type=float, help="bar load in kg")
    p_analyze.add_argument("--height", type=float, default=_DEFAULT_HEIGHT_CM, help="athlete cm")
    p_analyze.add_argument("--pose", choices=_POSE_CHOICES, default="rtmlib")
    p_analyze.add_argument("--out", default=None, help="output dir (default: next to VIDEO)")
    p_analyze.set_defaults(func=_cmd_analyze)

    p_fixtures = sub.add_parser("extract-fixtures", help="freeze bar+landmark series JSON")
    p_fixtures.add_argument("video", help="path to the video file")
    p_fixtures.add_argument("--movement", required=True, help="movement key (recorded only)")
    p_fixtures.add_argument("--out", required=True, help="output dir for the fixture JSONs")
    p_fixtures.add_argument("--pose", choices=_POSE_CHOICES, default="rtmlib")
    p_fixtures.set_defaults(func=_cmd_extract_fixtures)

    p_movements = sub.add_parser("movements", help="list movement registry keys")
    p_movements.set_defaults(func=_cmd_movements)

    return parser


# --- analyze ----------------------------------------------------------------


def _cmd_analyze(args: argparse.Namespace) -> int:
    video_path = Path(args.video)
    out_dir = Path(args.out) if args.out is not None else video_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    backend = make_pose_backend(args.pose)
    last_stage: str | None = None

    def progress(stage: str, pct: int) -> None:
        nonlocal last_stage
        if stage != last_stage:
            print(f"  {stage}...", file=sys.stderr)
            last_stage = stage

    result = analyze(
        str(video_path),
        args.movement,
        args.load,
        args.height,
        backend,
        progress_cb=progress,
    )

    metrics_path = out_dir / "metrics.json"
    overlay_path = out_dir / "overlay.json"
    annotated_path = out_dir / "annotated.mp4"
    overlay.write_metrics_json(result, metrics_path)
    overlay.write_overlay_json(result, result.bar_px, result.landmarks_px, overlay_path)
    overlay.write_annotated_mp4(video_path, json.loads(overlay_path.read_text()), annotated_path)

    _print_summary(result, metrics_path, overlay_path, annotated_path)
    return 0


def _print_summary(
    result: AnalysisResult, metrics_path: Path, overlay_path: Path, annotated_path: Path
) -> None:
    calibration = result.calibration
    print(f"{result.movement} @ {result.load_kg:g}kg -- {len(result.reps)} rep(s)")
    print(f"calibration: {calibration.source} ({calibration.bar_scale.cm_per_px * 10.0:.2f} mm/px)")
    if calibration.warning:
        print(f"warning: {calibration.warning}")
    print()
    print(f"{'rep':>3}  {'made':<4}  {'score':>5}  top fault")
    for rep in result.reps:
        made = "yes" if rep.made else "no"
        score = "-" if rep.score is None else f"{int(round(rep.score))}"
        if rep.unanalyzed_reason is not None:
            fault = "unanalyzed"
        elif rep.faults:
            fault = rep.faults[0].code
        else:
            fault = "-"
        print(f"{rep.window.rep_index + 1:>3}  {made:<4}  {score:>5}  {fault}")
    print()
    print(f"wrote {metrics_path}")
    print(f"wrote {overlay_path}")
    print(f"wrote {annotated_path}")


# --- extract-fixtures -------------------------------------------------------


def _cmd_extract_fixtures(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    backend = make_pose_backend(args.pose)

    from powerpath_engine.bar import MarkerTracker  # local: reuses pipeline's per-frame consumers

    tracker = MarkerTracker()
    pose = StridedPose(backend, stride=1)  # fixtures are full-rate
    for frame in decode.frames(args.video):
        tracker.feed(frame.t, frame.image)
        pose.feed(frame.t, frame.image, frame.index)

    bar_payload = {
        "video": str(args.video),
        "movement": args.movement,
        "samples": [
            {"t": d.sample.t, "x": d.sample.x, "y": d.sample.y, "visibility": d.sample.visibility}
            for d in tracker.detections
        ],
    }
    landmark_payload = {
        "video": str(args.video),
        "movement": args.movement,
        "frames": [
            {
                "t": frame.t,
                "points": {
                    name: {"x": s.x, "y": s.y, "visibility": s.visibility}
                    for name, s in frame.points.items()
                },
            }
            for frame in pose.series().frames
        ],
    }
    bar_path = out_dir / "bar_series.json"
    landmark_path = out_dir / "landmark_series.json"
    bar_path.write_text(json.dumps(bar_payload))
    landmark_path.write_text(json.dumps(landmark_payload))
    print(f"wrote {bar_path} ({len(bar_payload['samples'])} samples)")
    print(f"wrote {landmark_path} ({len(landmark_payload['frames'])} frames)")
    return 0


# --- movements ---------------------------------------------------------------


def _cmd_movements(args: argparse.Namespace) -> int:
    for config in registry.all_configs():
        print(f"{config.key}\t{config.display_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

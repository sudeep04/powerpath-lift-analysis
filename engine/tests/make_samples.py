"""Generate playable sample videos + their full analysis, for manual testing.

Renders synthetic-but-real .mp4 clips (magenta bar marker + 450mm plate disc) and
runs the REAL pipeline (decode -> track -> calibrate -> segment -> phases -> metrics
-> faults -> score) with pose scripted from the synthetic body, then writes the
annotated video + metrics.json + overlay.json. Not a test; run it directly:

    cd engine && uv run python tests/make_samples.py [OUT_DIR]

Default OUT_DIR: ../samples
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from render_utils import render_lift_video, scripted_pose_backend  # noqa: E402
from synthetic import back_squat, clean  # noqa: E402

from powerpath_engine import overlay, pipeline  # noqa: E402
from powerpath_engine.pipeline import analyze  # noqa: E402

FPS = 60
SAMPLES = [
    ("power_clean", "power_clean", lambda: clean(5, fps=FPS), 60.0),
    ("back_squat", "back_squat", lambda: back_squat(5, fps=FPS), 100.0),
]


def main(out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    for name, movement, make_lift, load_kg in SAMPLES:
        d = out_root / name
        d.mkdir(parents=True, exist_ok=True)
        lift = make_lift()
        video_path = d / f"{name}.mp4"
        spec = render_lift_video(video_path, lift, FPS)
        backend = scripted_pose_backend(lift, spec, stride=pipeline.POSE_STRIDE)
        result = analyze(str(video_path), movement, load_kg, 175.0, backend)

        overlay.write_metrics_json(result, d / "metrics.json")
        overlay.write_overlay_json(
            result, result.bar_px, result.landmarks_px, d / "overlay.json"
        )
        overlay_data = json.loads((d / "overlay.json").read_text())
        overlay.write_annotated_mp4(video_path, overlay_data, d / "annotated.mp4")

        made = sum(1 for r in result.reps if r.made)
        scores = [int(round(r.score)) for r in result.reps if r.score is not None]
        print(f"\n{name}  ({movement}, {load_kg}kg)")
        print(f"  video:     {video_path}")
        print(f"  annotated: {d / 'annotated.mp4'}")
        print(f"  metrics:   {d / 'metrics.json'}")
        print(f"  reps: {len(result.reps)} ({made} made)  scores: {scores}")
        print(f"  calibration: {result.calibration.source} "
              f"{result.calibration.bar_scale.cm_per_px:.4f} cm/px")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parents[2] / "samples"
    main(out)

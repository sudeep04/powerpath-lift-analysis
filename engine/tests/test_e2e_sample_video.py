"""The sample-video end-to-end test: the whole engine on a real video file.

A synthetic 5-rep clean is rendered to an actual h264/mp4 (magenta bar
marker + 450mm plate disc, via render_utils), pose is scripted from the
matching synthetic landmark series (FakePoseBackend -- the global
no-model-inference constraint), and ``pipeline.analyze`` runs the REAL
decode -> marker-tracking -> calibration -> segmentation -> phases ->
metrics -> faults -> scoring path. The outputs are then written through
overlay.py and validated against the FROZEN JSON contracts plus the
annotated-video render.

Asserted here (the Task 8 headline deliverable):
* 5 reps detected, all made, all scored;
* metrics.json + overlay.json parse and match the frozen contract
  (contract_utils: exact key sets, strictly increasing frames[].t,
  missed-rep score rules, ...);
* annotated.mp4 exists with >= 0.9x the input frame count;
* progress_cb fired all five stages in order;
* peak-RSS growth across analyze stays far below the ~500MB streaming
  budget (resource.getrusage; ru_maxrss is bytes on macOS, KB on Linux).
"""

from __future__ import annotations

import json
import resource
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from contract_utils import assert_metrics_contract, assert_overlay_contract
from render_utils import render_lift_video, scripted_pose_backend
from synthetic import SyntheticLift, clean

from powerpath_engine import decode, overlay, pipeline
from powerpath_engine.pipeline import AnalysisResult, analyze

FPS = 60
N_REPS = 5
LOAD_KG = 80.0
HEIGHT_CM = 157.0  # the synthetic body's nose-to-ankle span (see test_pipeline)

# ru_maxrss units differ by platform: bytes on macOS, kilobytes on Linux.
_RSS_TO_BYTES = 1 if sys.platform == "darwin" else 1024
_RSS_BUDGET_BYTES = 500 * 1024 * 1024


def _peak_rss_bytes() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _RSS_TO_BYTES


@dataclass(frozen=True)
class E2ERun:
    lift: SyntheticLift
    video_path: Path
    result: AnalysisResult
    metrics: dict
    overlay_data: dict
    annotated_path: Path
    input_frame_count: int
    annotated_frame_count: int
    progress_calls: list[tuple[str, int]]
    rss_growth_bytes: int


@pytest.fixture(scope="module")
def run(tmp_path_factory) -> E2ERun:
    """Render, analyze and serialize ONCE; every test asserts on the result."""
    out_dir = tmp_path_factory.mktemp("e2e")
    lift = clean(N_REPS, fps=FPS)
    video_path = out_dir / "clean5.mp4"
    spec = render_lift_video(video_path, lift, FPS)
    backend = scripted_pose_backend(lift, spec, stride=pipeline.POSE_STRIDE)

    progress_calls: list[tuple[str, int]] = []
    rss_before = _peak_rss_bytes()
    result = analyze(
        str(video_path),
        "power_clean",
        LOAD_KG,
        HEIGHT_CM,
        backend,
        progress_cb=lambda stage, pct: progress_calls.append((stage, pct)),
    )
    rss_growth = _peak_rss_bytes() - rss_before

    metrics_path = out_dir / "metrics.json"
    overlay_path = out_dir / "overlay.json"
    annotated_path = out_dir / "annotated.mp4"
    overlay.write_metrics_json(result, metrics_path)
    overlay.write_overlay_json(result, result.bar_px, result.landmarks_px, overlay_path)
    overlay_data = json.loads(overlay_path.read_text())
    overlay.write_annotated_mp4(video_path, overlay_data, annotated_path)

    return E2ERun(
        lift=lift,
        video_path=video_path,
        result=result,
        metrics=json.loads(metrics_path.read_text()),
        overlay_data=overlay_data,
        annotated_path=annotated_path,
        input_frame_count=sum(1 for _ in decode.frames(video_path)),
        annotated_frame_count=sum(1 for _ in decode.frames(annotated_path)),
        progress_calls=progress_calls,
        rss_growth_bytes=rss_growth,
    )


def test_five_reps_detected_all_made_and_scored(run: E2ERun) -> None:
    assert len(run.result.reps) == N_REPS
    for rep in run.result.reps:
        assert rep.made is True
        assert rep.score is not None and 0.0 < rep.score <= 100.0
        assert rep.unanalyzed_reason is None
        assert rep.excluded_from_templates is False


def test_rep_windows_cover_each_synthetic_rep_in_order(run: E2ERun) -> None:
    """Each detected window brackets its rep's ground-truth catch instant."""
    for rep, catch_t in zip(run.result.reps, run.lift.truth["catch"], strict=True):
        assert rep.window.t_start < catch_t < rep.window.t_end
        assert rep.phases["catch"] == pytest.approx(catch_t, abs=0.1)


def test_metrics_json_validates_against_frozen_contract(run: E2ERun) -> None:
    assert_metrics_contract(run.metrics)
    assert run.metrics["movement"] == "power_clean"
    assert run.metrics["load_kg"] == LOAD_KG
    assert run.metrics["extraction_version"] == 1
    assert run.metrics["rules_version"] == 1
    assert run.metrics["calibration"]["source"] == "plate"
    assert len(run.metrics["reps"]) == N_REPS
    for rep in run.metrics["reps"]:
        assert rep["made"] is True
        assert isinstance(rep["score"], int) and rep["score"] > 0


def test_overlay_json_validates_against_frozen_contract(run: E2ERun) -> None:
    assert_overlay_contract(run.overlay_data)  # includes strictly-increasing t
    assert len(run.overlay_data["reps"]) == N_REPS
    for rep in run.overlay_data["reps"]:
        assert isinstance(rep["score"], int) and rep["score"] > 0
        assert len(rep["bar_path"]) > 0
        assert rep["phases"]  # at least one detected phase to annotate
    # The player needs dense per-frame data: the bar channel covers nearly
    # every decoded frame (marker tracking + gap interpolation).
    frames = run.overlay_data["frames"]
    assert len(frames) >= 0.9 * run.input_frame_count
    assert sum(1 for f in frames if f["bar"] is not None) >= 0.9 * run.input_frame_count
    assert any(f["skeleton"] for f in frames)


def test_annotated_video_written_near_frame_for_frame(run: E2ERun) -> None:
    assert run.annotated_path.exists() and run.annotated_path.stat().st_size > 0
    assert run.annotated_frame_count >= 0.9 * run.input_frame_count


def test_progress_reported_all_five_stages_in_order(run: E2ERun) -> None:
    first_seen: list[str] = []
    for stage, pct in run.progress_calls:
        assert isinstance(pct, int) and 0 <= pct <= 100
        if stage not in first_seen:
            first_seen.append(stage)
    assert first_seen == ["decode", "pose", "bar", "segment", "metrics"]


def test_peak_rss_growth_stays_within_streaming_budget(run: E2ERun) -> None:
    assert run.rss_growth_bytes < _RSS_BUDGET_BYTES, (
        f"analyze grew peak RSS by {run.rss_growth_bytes / 1e6:.0f}MB "
        f"(budget {_RSS_BUDGET_BYTES / 1e6:.0f}MB) -- is something buffering frames?"
    )

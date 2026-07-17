"""Tests for overlay.py: the frozen-contract JSON writers + annotated mp4.

The JSON writers are exercised on hand-constructed AnalysisResults (every
field controlled, no video needed) and validated with the shared contract
assertions in contract_utils.py. The annotated-video writer runs a real
decode->draw->encode round trip over a tiny synthetic clip.
"""

from __future__ import annotations

import json

import pytest
from contract_utils import assert_metrics_contract, assert_overlay_contract
from video_utils import moving_square_frames, write_test_video

from powerpath_engine import decode, overlay
from powerpath_engine.calibration import CalibrationResult
from powerpath_engine.decode import VideoMeta
from powerpath_engine.faults import FaultFinding
from powerpath_engine.geometry import PlaneScale
from powerpath_engine.metrics import RepMetrics
from powerpath_engine.pipeline import AnalysisResult, RepResult
from powerpath_engine.segmentation import RepWindow
from powerpath_engine.series import LandmarkFrame, LandmarkSeries, Sample, TimeSeries

FPS = 30.0
DT = 1.0 / FPS


def _bar_px(n: int = 30) -> TimeSeries:
    return TimeSeries(
        [Sample(t=i * DT, x=100.0 + i, y=200.0 - 3.0 * i, visibility=1.0) for i in range(n)]
    )


def _landmarks_px(n: int = 30, stride: int = 2) -> LandmarkSeries:
    frames = []
    for i in range(0, n, stride):
        t = i * DT
        frames.append(
            LandmarkFrame(
                t=t,
                points={
                    "left_hip": Sample(t=t, x=90.0, y=150.0),
                    "right_hip": Sample(t=t, x=110.0, y=150.0),
                    "left_knee": Sample(t=t, x=88.0, y=190.0),
                    "right_knee": Sample(t=t, x=112.0, y=190.0),
                },
            )
        )
    return LandmarkSeries(frames)


def _fault() -> FaultFinding:
    return FaultFinding(
        code="bar_drift",
        message="Bar drifted 7.2cm from vertical (envelope 6cm).",
        phase=None,
        value=7.2,
        threshold=6.0,
    )


def _rep(
    index: int = 0,
    *,
    made: bool = True,
    score: float | None = 84.38,
    metrics: RepMetrics | None = None,
    faults: list[FaultFinding] | None = None,
    unanalyzed_reason: str | None = None,
) -> RepResult:
    t0 = index * 0.5
    return RepResult(
        window=RepWindow(t_start=t0, t_end=t0 + 0.4, rep_index=index),
        made=made,
        score=score,
        excluded_from_templates=not made,
        metrics=metrics if metrics is not None else RepMetrics(rep_index=index),
        faults=faults if faults is not None else [],
        phases={"knee_pass": t0 + 0.1, "catch": t0 + 0.3, "second_pull": None},
        unanalyzed_reason=unanalyzed_reason,
    )


def _result(reps: list[RepResult]) -> AnalysisResult:
    return AnalysisResult(
        video=VideoMeta(width=320, height=240, rotation_deg=0, fps_avg=FPS, duration_s=1.0),
        movement="power_clean",
        load_kg=60.0,
        calibration=CalibrationResult(
            bar_scale=PlaneScale(cm_per_px=0.2), source="plate", warning=None
        ),
        reps=reps,
        extraction_version=1,
        rules_version=1,
        bar_px=_bar_px(),
        landmarks_px=_landmarks_px(),
    )


# --- metrics.json -----------------------------------------------------------


def test_metrics_json_matches_frozen_contract(tmp_path) -> None:
    result = _result([_rep(0, faults=[_fault()]), _rep(1, made=False, score=None)])
    path = tmp_path / "metrics.json"
    overlay.write_metrics_json(result, path)
    data = json.loads(path.read_text())

    assert_metrics_contract(data)
    assert data["movement"] == "power_clean"
    assert data["load_kg"] == 60.0
    assert data["calibration"] == {
        "source": "plate",
        "bar_scale_cm_per_px": 0.2,
        "warning": None,
    }
    made, missed = data["reps"]
    assert made["score"] == 84  # rounded to int
    assert made["faults"][0]["code"] == "bar_drift"
    assert missed["made"] is False and missed["score"] is None
    assert missed["excluded_from_templates"] is True


def test_metrics_json_smoothness_is_the_scorer_fraction(tmp_path) -> None:
    """The contract's `smoothness` is 0-1 (1.0 = perfectly smooth), not the
    raw normalized jerk."""
    smooth = _rep(0, metrics=RepMetrics(smoothness_normalized_jerk=0.0))
    jerky = _rep(1, metrics=RepMetrics(smoothness_normalized_jerk=1000.0))
    path = tmp_path / "metrics.json"
    overlay.write_metrics_json(_result([smooth, jerky]), path)
    reps = json.loads(path.read_text())["reps"]
    assert reps[0]["metrics"]["smoothness"] == 1.0
    assert reps[1]["metrics"]["smoothness"] == pytest.approx(0.5)  # NJ_HALF_CREDIT


def test_metrics_json_phases_drop_undetected(tmp_path) -> None:
    path = tmp_path / "metrics.json"
    overlay.write_metrics_json(_result([_rep(0)]), path)
    phases = json.loads(path.read_text())["reps"][0]["phases"]
    assert "second_pull" not in phases  # was None
    assert set(phases) == {"knee_pass", "catch"}


def test_json_never_emits_nan_or_infinity(tmp_path) -> None:
    bad = RepMetrics(
        bar_drift_cm=float("nan"),
        peak_concentric_velocity_ms=float("inf"),
        path_length_ratio=float("-inf"),
        hip_angle_at_phase={"catch": float("nan")},
    )
    path = tmp_path / "metrics.json"
    overlay.write_metrics_json(_result([_rep(0, metrics=bad)]), path)
    text = path.read_text()
    assert "NaN" not in text and "Infinity" not in text
    metrics = json.loads(text)["reps"][0]["metrics"]
    assert metrics["bar_drift_cm"] is None
    assert metrics["peak_concentric_velocity_ms"] is None
    assert metrics["path_length_ratio"] is None
    assert metrics["hip_angle_at_phase"]["catch"] is None


# --- overlay.json -----------------------------------------------------------


def test_overlay_json_matches_frozen_contract(tmp_path) -> None:
    result = _result([_rep(0, faults=[_fault()]), _rep(1, made=False, score=None)])
    path = tmp_path / "overlay.json"
    overlay.write_overlay_json(result, result.bar_px, result.landmarks_px, path)
    data = json.loads(path.read_text())

    assert_overlay_contract(data)
    assert data["movement"] == "power_clean"
    assert data["reps"][0]["score"] == 84
    assert data["reps"][1]["score"] is None
    assert data["reps"][0]["unanalyzed_reason"] is None


def test_overlay_frames_merge_bar_and_skeleton_channels(tmp_path) -> None:
    """Bar samples exist every frame, skeletons every 2nd (strided pose):
    shared instants merge into ONE frame; pose-less frames carry the bar
    with an empty skeleton."""
    result = _result([_rep(0)])
    path = tmp_path / "overlay.json"
    overlay.write_overlay_json(result, result.bar_px, result.landmarks_px, path)
    frames = json.loads(path.read_text())["frames"]

    assert len(frames) == 30  # union == bar cadence; no duplicated instants
    with_skeleton = [f for f in frames if f["skeleton"]]
    without_skeleton = [f for f in frames if not f["skeleton"]]
    assert len(with_skeleton) == 15 and len(without_skeleton) == 15
    assert all(f["bar"] is not None for f in frames)
    assert set(with_skeleton[0]["skeleton"]) == {
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
    }


def test_overlay_bar_path_is_the_reps_window_slice(tmp_path) -> None:
    result = _result([_rep(0)])  # window [0.0, 0.4] of a 1s bar series
    path = tmp_path / "overlay.json"
    overlay.write_overlay_json(result, result.bar_px, result.landmarks_px, path)
    rep = json.loads(path.read_text())["reps"][0]
    # Samples at t = 0, 1/30, ..., 12/30 fall inside [0, 0.4].
    assert len(rep["bar_path"]) == 13
    assert rep["bar_path"][0] == [100.0, 200.0]


def test_overlay_unanalyzed_reason_serialized(tmp_path) -> None:
    result = _result([_rep(0, unanalyzed_reason="bar marker lost for more than 5 frames")])
    path = tmp_path / "overlay.json"
    overlay.write_overlay_json(result, result.bar_px, result.landmarks_px, path)
    rep = json.loads(path.read_text())["reps"][0]
    assert "marker lost" in rep["unanalyzed_reason"]


# --- annotated mp4 ----------------------------------------------------------


def test_annotated_mp4_written_frame_for_frame(tmp_path) -> None:
    video_path = tmp_path / "input.mp4"
    write_test_video(video_path, moving_square_frames(30, width=320, height=240), fps=30)

    result = _result([_rep(0, faults=[_fault()])])
    overlay_path = tmp_path / "overlay.json"
    overlay.write_overlay_json(result, result.bar_px, result.landmarks_px, overlay_path)
    overlay_data = json.loads(overlay_path.read_text())

    out_path = tmp_path / "annotated.mp4"
    overlay.write_annotated_mp4(video_path, overlay_data, out_path)

    assert out_path.exists() and out_path.stat().st_size > 0
    out_frames = sum(1 for _ in decode.frames(out_path))
    assert out_frames >= 0.9 * 30
    meta = decode.probe(out_path)
    assert (meta.width, meta.height) == (320, 240)

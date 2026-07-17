"""Tests for pipeline.analyze: the decode->track->pose->segment->phases->
metrics->faults->score orchestrator (Task 8).

These run the REAL streaming pipeline over small rendered videos
(render_utils draws the magenta marker + calibration plate from a synthetic
lift) with a scripted FakePoseBackend -- no model inference (global
constraint). The full 5-rep sample-video E2E with its RSS/overlay/annotated
assertions lives in test_e2e_sample_video.py; this file covers the
pipeline's per-feature behavior on a single-rep clip plus the pure helpers.
"""

from __future__ import annotations

import pytest
from render_utils import render_lift_video, scripted_pose_backend
from synthetic import single_rep

from powerpath_engine import pipeline, registry
from powerpath_engine.geometry import PlaneScale
from powerpath_engine.pipeline import _body_plane_scale, _unanalyzed_reason, analyze
from powerpath_engine.pose import make_pose_backend
from powerpath_engine.segmentation import RepWindow
from powerpath_engine.series import Gap, LandmarkFrame, LandmarkSeries, Sample

FPS = 60
LOAD_KG = 60.0
# The synthetic body's nose-to-ankle span (165cm - 8cm): what the pipeline's
# apparent-height estimator measures, so passing it as the athlete height
# makes the body-plane scale match the render scale.
HEIGHT_CM = 157.0


@pytest.fixture(scope="module")
def rep_video(tmp_path_factory):
    """One rendered single-rep clean, shared by the read-only tests."""
    lift = single_rep(fps=FPS)
    path = tmp_path_factory.mktemp("pipeline") / "single_rep.mp4"
    spec = render_lift_video(path, lift, FPS)
    return lift, str(path), spec


def _analyze(lift, path, spec, **kwargs):
    backend = scripted_pose_backend(lift, spec, stride=pipeline.POSE_STRIDE)
    return analyze(path, "power_clean", LOAD_KG, HEIGHT_CM, backend, **kwargs)


# --- happy path -------------------------------------------------------------


def test_analyze_single_rep_made_and_scored(rep_video) -> None:
    lift, path, spec = rep_video
    result = _analyze(lift, path, spec)

    assert len(result.reps) == 1
    rep = result.reps[0]
    assert rep.made is True
    assert rep.score is not None and 0.0 < rep.score <= 100.0
    assert rep.excluded_from_templates is False
    assert rep.unanalyzed_reason is None
    assert result.movement == "power_clean"
    assert result.load_kg == LOAD_KG
    assert result.extraction_version == 1
    assert result.rules_version == 1


def test_analyze_calibrates_from_the_drawn_plate(rep_video) -> None:
    lift, path, spec = rep_video
    result = _analyze(lift, path, spec)
    assert result.calibration.source == "plate"
    assert result.calibration.warning is None
    # The recovered bar scale must match the render transform within a few
    # percent (Hough radius quantization) -- this is what makes every
    # downstream cm metric trustworthy.
    assert result.calibration.bar_scale.cm_per_px == pytest.approx(spec.cm_per_px, rel=0.05)


def test_analyze_detects_phases_near_truth(rep_video) -> None:
    """Key phase events land within 0.1s of the generator's ground truth
    even after the round trip through video encode + marker tracking +
    calibration (the detectors' own +/-2-frame accuracy is covered in
    test_phases.py on pristine series)."""
    lift, path, spec = rep_video
    rep = _analyze(lift, path, spec).reps[0]
    assert rep.phases["knee_pass"] == pytest.approx(lift.truth["knee_pass"][0], abs=0.1)
    assert rep.phases["catch"] == pytest.approx(lift.truth["catch"][0], abs=0.1)


def test_analyze_image_space_series_kept_for_overlay(rep_video) -> None:
    """bar_px / landmarks_px stay in image px (y-down) for overlay drawing."""
    lift, path, spec = rep_video
    result = _analyze(lift, path, spec)
    assert len(result.bar_px.samples) > 0
    # Image space: the bar's highest point has a SMALL y (y grows down) and
    # everything sits inside the frame.
    ys = [s.y for s in result.bar_px.samples]
    assert 0.0 <= min(ys) <= result.video.height
    assert len(result.landmarks_px.frames) > 0
    wrist_ys = [
        f.points["left_wrist"].y for f in result.landmarks_px.frames if "left_wrist" in f.points
    ]
    assert min(wrist_ys) < spec.y_px(100.0)  # wrists ride the bar above 100cm


def test_progress_cb_reports_all_stages_in_order(rep_video) -> None:
    lift, path, spec = rep_video
    calls: list[tuple[str, int]] = []
    _analyze(lift, path, spec, progress_cb=lambda stage, pct: calls.append((stage, pct)))

    first_seen: list[str] = []
    for stage, pct in calls:
        assert isinstance(pct, int) and 0 <= pct <= 100
        if stage not in first_seen:
            first_seen.append(stage)
    assert first_seen == list(pipeline.STAGES)
    pcts = [pct for _stage, pct in calls]
    assert pcts == sorted(pcts)
    assert pcts[-1] == 100


def test_velocity_history_consulted_with_load_and_tolerance(rep_video) -> None:
    lift, path, spec = rep_video

    class RecordingHistory:
        def __init__(self) -> None:
            self.calls: list[tuple[float, float]] = []

        def peak_velocities_near_load(self, load_kg: float, tolerance_frac: float) -> list[float]:
            self.calls.append((load_kg, tolerance_frac))
            return [1.0] * 5

        # 5 history reps -> the velocity component is live (not redistributed).

    history = RecordingHistory()
    result = _analyze(lift, path, spec, velocity_history=history)
    assert history.calls == [(LOAD_KG, 0.10)]
    assert result.reps[0].score is not None


# --- degraded inputs --------------------------------------------------------


def test_unknown_movement_raises_before_decoding(tmp_path) -> None:
    backend = make_pose_backend("fake")
    with pytest.raises(registry.UnknownMovementError):
        analyze(str(tmp_path / "missing.mp4"), "bench_press", LOAD_KG, HEIGHT_CM, backend)


def test_markerless_video_yields_zero_reps(tmp_path) -> None:
    """No magenta marker anywhere: calibration still works off the plate,
    but there is no bar trajectory, hence no reps -- not an error."""
    lift = single_rep(fps=30)
    path = tmp_path / "no_marker.mp4"
    spec = render_lift_video(path, lift, 30, draw_marker=False)
    backend = scripted_pose_backend(lift, spec, stride=pipeline.POSE_STRIDE)
    result = analyze(str(path), "power_clean", LOAD_KG, HEIGHT_CM, backend)
    assert result.reps == []
    assert result.bar_px.samples == []
    assert result.calibration.source == "plate"


def test_long_marker_gap_marks_rep_unanalyzed(tmp_path) -> None:
    """A >MAX_GAP_FRAMES marker dropout inside the rep window is not
    interpolated; the rep is flagged with a reason and excluded from
    templates (the global gap rule)."""
    lift = single_rep(fps=FPS)
    path = tmp_path / "gap.mp4"
    # 9 missing frames right after the catch (t ~ 1.17-1.30s), mid-window.
    spec = render_lift_video(path, lift, FPS, skip_marker_frames=set(range(70, 79)))
    backend = scripted_pose_backend(lift, spec, stride=pipeline.POSE_STRIDE)
    result = analyze(str(path), "power_clean", LOAD_KG, HEIGHT_CM, backend)

    assert len(result.reps) == 1
    rep = result.reps[0]
    assert rep.unanalyzed_reason is not None
    assert "marker lost" in rep.unanalyzed_reason
    assert rep.excluded_from_templates is True


def test_noop_pose_backend_still_produces_bar_only_reps(rep_video) -> None:
    """make_pose_backend('fake') never detects landmarks; the clean's made
    criteria (catch) is bar-only, so the rep still counts -- and the body
    scale falls back to the bar scale without erroring."""
    _lift, path, _spec = rep_video
    result = analyze(path, "power_clean", LOAD_KG, HEIGHT_CM, make_pose_backend("fake"))
    assert len(result.reps) == 1
    assert result.reps[0].made is True
    assert result.landmarks_px.frames == []


# --- pure helpers -----------------------------------------------------------


def test_unanalyzed_reason_only_for_overlapping_gaps() -> None:
    window = RepWindow(t_start=1.0, t_end=2.0, rep_index=0)
    before = Gap(t_start=0.1, t_end=0.9, filled=False)
    inside = Gap(t_start=1.4, t_end=1.6, filled=False)
    after = Gap(t_start=2.1, t_end=3.0, filled=False)
    assert _unanalyzed_reason(window, [before, after]) is None
    reason = _unanalyzed_reason(window, [before, inside, after])
    assert reason is not None and "1.40-1.60" in reason


def _frame(t: float, points: dict[str, tuple[float, float]]) -> LandmarkFrame:
    return LandmarkFrame(t=t, points={n: Sample(t=t, x=x, y=y) for n, (x, y) in points.items()})


def test_body_plane_scale_from_nose_to_ankle_extent() -> None:
    # Nose 100px above the ankles in image space -> 180cm athlete = 1.8 cm/px.
    frames = [
        _frame(0.0, {"nose": (50.0, 100.0), "left_ankle": (50.0, 200.0)}),
        _frame(1.0, {"nose": (50.0, 120.0), "left_ankle": (50.0, 200.0)}),  # crouched
    ]
    scale = _body_plane_scale(LandmarkSeries(frames), 180.0, fallback=PlaneScale(cm_per_px=0.5))
    # 95th percentile of extents [100, 80] ~= the standing 100px.
    assert scale.cm_per_px == pytest.approx(180.0 / 99.0, rel=0.02)


def test_body_plane_scale_falls_back_without_landmarks() -> None:
    fallback = PlaneScale(cm_per_px=0.25)
    assert _body_plane_scale(LandmarkSeries([]), 180.0, fallback=fallback) is fallback
    no_ankles = LandmarkSeries([_frame(0.0, {"nose": (50.0, 100.0)})])
    assert _body_plane_scale(no_ankles, 180.0, fallback=fallback) is fallback
    assert _body_plane_scale(no_ankles, 0.0, fallback=fallback) is fallback

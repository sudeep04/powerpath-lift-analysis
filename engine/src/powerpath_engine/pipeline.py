"""The analysis pipeline: orchestrate decode -> track -> pose -> segment ->
phases -> metrics -> faults -> score into one :class:`AnalysisResult`.

This module owns *iteration and wiring only* -- every algorithm lives in the
module that owns it (bar.py, pose.py, calibration.py, segmentation.py,
phases.py, metrics.py, faults.py, scoring.py) and geometry.py remains the
sole owner of coordinate conversions. The pipeline's job is to feed them in
the right order under the global constraints:

* **Streaming single pass**: ``decode.frames`` yields one frame at a time;
  ``MarkerTracker.feed`` and ``StridedPose.feed`` consume it and keep only
  extracted time series. The sole frame buffering is the first
  ``CALIBRATION_FRAMES`` (~30) images kept for plate calibration and
  released immediately after ``calibrate`` runs. Peak RSS stays far under
  the ~500MB budget on a 30s 1080p video.
* **PTS timebase**: every series sample and rep boundary is keyed by the
  decoded frame's PTS seconds, never a frame index.
* **Two-plane calibration**: the bar plane comes from the 450mm plate (with
  the date-fallback/manual ladder); the body plane from athlete height,
  estimated against the athlete's apparent nose-to-ankle pixel extent. When
  no usable landmark extent exists (e.g. a bar-only run with the no-op
  pose backend) the bar scale is reused for the body plane -- angles are
  scale-invariant so joint math is unaffected; only cross-checks that
  compare bar cm to landmark cm degrade, and that approximation is
  confined to this documented fallback.

Simplification noted per the task brief: rep windows are analyzed from the
STRIDED pose series (pose runs every ``pose_stride`` frames during the single
streaming pass); the optional full-rate second decode pass over rep windows
(``StridedPose.rerun_full_rate``) is not wired up in v1. Per-landmark gap
interpolation runs at the strided cadence instead, which is sufficient for
the M1 phase/angle reads.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from powerpath_engine import decode, registry
from powerpath_engine.bar import MarkerTracker, estimate_marker_diameter_px
from powerpath_engine.calibration import CalibrationResult, calibrate
from powerpath_engine.faults import FaultFinding, evaluate_faults
from powerpath_engine.geometry import PlaneScale, body_plane_scale_from_height, to_y_up
from powerpath_engine.metrics import RepMetrics, compute_rep_metrics
from powerpath_engine.phases import detect_phases
from powerpath_engine.pose import PoseBackend, StridedPose
from powerpath_engine.scoring import VelocityHistory, evaluate_made, score_rep
from powerpath_engine.segmentation import RepWindow, segment
from powerpath_engine.series import (
    Gap,
    LandmarkFrame,
    LandmarkSeries,
    Sample,
    TimeSeries,
)
from powerpath_engine.versions import EXTRACTION_VERSION, RULES_VERSION

# How many leading frames are buffered for plate calibration (and released
# right after `calibrate` runs) -- the only frame buffering in the pipeline.
CALIBRATION_FRAMES = 30

# Bar-marker gap rule (global constraint): holes of <= this many missing
# frames are linearly interpolated; longer holes leave the series unfilled
# and mark any rep whose window overlaps them as unanalyzed.
MAX_GAP_FRAMES = 5

# Pose runs every Nth frame during the streaming pass (see pose.StridedPose).
POSE_STRIDE = 2

# The progress stage vocabulary, in reporting order (kept in sync with
# api.jobs.STAGES and the UI's JobStage type).
STAGES = ("decode", "pose", "bar", "segment", "metrics")

# Fallback frame period when the container reports no average rate.
_FALLBACK_DT = 1.0 / 30.0

ProgressCallback = Callable[[str, int], None]


class _EmptyVelocityHistory:
    """The no-history default: velocity credit is always redistributed."""

    def peak_velocities_near_load(self, load_kg: float, tolerance_frac: float) -> list[float]:
        return []


@dataclass(frozen=True)
class RepResult:
    """One rep's full judgement: window, made/score, metrics, faults, phases.

    ``phases`` maps phase name -> event PTS seconds (None = not detected).
    ``unanalyzed_reason`` is a human-readable string when the bar marker was
    lost for more than :data:`MAX_GAP_FRAMES` frames somewhere inside the
    rep's window (the rep is also excluded from templates then), else None.
    """

    window: RepWindow
    made: bool
    score: float | None
    excluded_from_templates: bool
    metrics: RepMetrics
    faults: list[FaultFinding]
    phases: dict[str, float | None]
    unanalyzed_reason: str | None = None


@dataclass(frozen=True)
class AnalysisResult:
    """Everything ``analyze`` produced for one video.

    ``bar_px`` (gap-filled bar marker samples) and ``landmarks_px`` (the
    strided pose detections) are in UPRIGHT IMAGE pixel space (y-down) --
    they exist for overlay.py, which draws directly in image coordinates.
    All rep-level analysis inside ``reps`` was computed on the calibrated
    centimeter, y-up versions of the same series.
    """

    video: decode.VideoMeta
    movement: str
    load_kg: float
    calibration: CalibrationResult
    reps: list[RepResult]
    extraction_version: int
    rules_version: int
    bar_px: TimeSeries
    landmarks_px: LandmarkSeries


def analyze(
    path: str,
    movement_key: str,
    load_kg: float,
    athlete_height_cm: float,
    pose_backend: PoseBackend,
    *,
    date_fallback_scale: PlaneScale | None = None,
    manual_scale: PlaneScale | None = None,
    velocity_history: VelocityHistory | None = None,
    progress_cb: ProgressCallback | None = None,
    pose_stride: int = POSE_STRIDE,
) -> AnalysisResult:
    """Run the full analysis pipeline over the video at ``path``.

    Stages (progress_cb fires each stage name from :data:`STAGES` in order,
    with a monotonically nondecreasing overall percentage):

    1. ``decode``: the single streaming pass -- every decoded frame is fed
       to the marker tracker and the strided pose scheduler; the first
       ~:data:`CALIBRATION_FRAMES` images are kept for calibration.
    2. ``pose``: the pass is over, so pose extraction is complete.
    3. ``bar``: calibration + bar gap interpolation + px->cm conversion.
    4. ``segment``: rep windows from the calibrated bar trajectory.
    5. ``metrics``: per-rep phases, metrics, faults, made/missed, score.

    Raises ``registry.UnknownMovementError`` for a bad ``movement_key``,
    ``decode.DecodeError`` for an unreadable video, and
    ``calibration.CalibrationError`` when no trustworthy bar scale exists.
    A video where the marker is never detected yields an empty ``reps`` list
    rather than an error (there is simply nothing to segment).
    """
    config = registry.get(movement_key)
    report = progress_cb if progress_cb is not None else (lambda stage, pct: None)

    meta = decode.probe(path)
    expected_dt = 1.0 / meta.fps_avg if meta.fps_avg > 0.0 else _FALLBACK_DT
    total_frames_est = meta.duration_s / expected_dt if meta.duration_s > 0.0 else 0.0

    # --- stage 1: the single streaming pass --------------------------------
    report("decode", 0)
    tracker = MarkerTracker()
    strided = StridedPose(pose_backend, stride=pose_stride)
    calibration_frames: list[np.ndarray] = []

    for frame in decode.frames(path):
        if frame.index < CALIBRATION_FRAMES:
            calibration_frames.append(frame.image)
        tracker.feed(frame.t, frame.image)
        strided.feed(frame.t, frame.image, frame.index)
        if total_frames_est > 0.0 and frame.index % 60 == 0:
            report("decode", min(39, int(40.0 * frame.index / total_frames_est)))

    report("pose", 50)

    # --- stage 3: calibration + bar series in cm, y-up ----------------------
    marker_diameter_px = (
        estimate_marker_diameter_px(tracker.detections) if tracker.detections else None
    )
    calibration = calibrate(
        calibration_frames,
        date_fallback=date_fallback_scale,
        manual=manual_scale,
        marker_diameter_px=marker_diameter_px,
    )
    del calibration_frames  # release the only buffered images
    bar_scale = calibration.bar_scale

    bar_px_raw = TimeSeries([d.sample for d in tracker.detections])
    bar_px, bar_gaps = bar_px_raw.interpolate_gaps(MAX_GAP_FRAMES, expected_dt)
    unfilled_gaps = [gap for gap in bar_gaps if not gap.filled]
    bar_cm = _bar_to_cm(bar_px, bar_scale, meta.height)

    landmarks_px = strided.series()
    landmarks_filled, _ = landmarks_px.interpolate_gaps(MAX_GAP_FRAMES, expected_dt * pose_stride)
    body_scale = _body_plane_scale(landmarks_filled, athlete_height_cm, fallback=bar_scale)
    landmarks_cm = _landmarks_to_cm(landmarks_filled, body_scale, meta.height)
    report("bar", 65)

    # --- stage 4: segmentation ----------------------------------------------
    windows = segment(bar_cm, config, expected_dt)
    report("segment", 75)

    # --- stage 5: per-rep phases, metrics, faults, made, score --------------
    history = velocity_history if velocity_history is not None else _EmptyVelocityHistory()
    reps: list[RepResult] = []
    for window in windows:
        phases = detect_phases(window, bar_cm, landmarks_cm, config)
        metrics = compute_rep_metrics(
            window, bar_cm, landmarks_cm, phases, config, PlaneScale(cm_per_px=1.0)
        )
        faults = evaluate_faults(metrics, config)
        made = evaluate_made(window, bar_cm, phases, metrics, config)
        rep_score = score_rep(metrics, faults, made, history, load_kg)
        reason = _unanalyzed_reason(window, unfilled_gaps)
        reps.append(
            RepResult(
                window=window,
                made=rep_score.made,
                score=rep_score.score,
                excluded_from_templates=rep_score.excluded_from_templates or reason is not None,
                metrics=metrics,
                faults=faults,
                phases=phases,
                unanalyzed_reason=reason,
            )
        )
    report("metrics", 100)

    return AnalysisResult(
        video=meta,
        movement=movement_key,
        load_kg=load_kg,
        calibration=calibration,
        reps=reps,
        extraction_version=EXTRACTION_VERSION,
        rules_version=RULES_VERSION,
        bar_px=bar_px,
        landmarks_px=landmarks_px,
    )


def _bar_to_cm(bar_px: TimeSeries, scale: PlaneScale, frame_height: int) -> TimeSeries:
    """Bar samples from image px (y-down) to bar-plane cm, y-up."""
    return TimeSeries(
        [
            Sample(
                t=s.t,
                x=scale.px_to_cm(s.x),
                y=scale.px_to_cm(to_y_up(s.y, frame_height)),
                visibility=s.visibility,
            )
            for s in bar_px.samples
        ]
    )


def _landmarks_to_cm(
    landmarks: LandmarkSeries, scale: PlaneScale, frame_height: int
) -> LandmarkSeries:
    """Landmark samples from image px (y-down) to body-plane cm, y-up."""
    return LandmarkSeries(
        [
            LandmarkFrame(
                t=frame.t,
                points={
                    name: Sample(
                        t=s.t,
                        x=scale.px_to_cm(s.x),
                        y=scale.px_to_cm(to_y_up(s.y, frame_height)),
                        visibility=s.visibility,
                    )
                    for name, s in frame.points.items()
                },
            )
            for frame in landmarks.frames
        ]
    )


def _body_plane_scale(
    landmarks: LandmarkSeries, athlete_height_cm: float, fallback: PlaneScale
) -> PlaneScale:
    """Body-plane scale from athlete height vs apparent pixel height.

    The apparent height is the 95th percentile over pose frames of the
    nose-to-ankle vertical extent in px (near the athlete's tallest,
    standing moments; the percentile rejects the crouched frames and
    outliers). ``athlete_height_cm`` should be the athlete's nose-to-ankle
    span rather than their full height for exactness, but the difference is
    a few percent and only affects cross-plane comparisons, not angles.
    Falls back to ``fallback`` (the bar scale -- see the module docstring)
    when no frame carries both a nose and an ankle, or the height input is
    unusable.
    """
    if not (athlete_height_cm > 0.0):
        return fallback
    extents: list[float] = []
    for frame in landmarks.frames:
        points = frame.points
        if "nose" not in points:
            continue
        ankle_ys = [points[n].y for n in ("left_ankle", "right_ankle") if n in points]
        if not ankle_ys:
            continue
        extent = float(np.mean(ankle_ys)) - points["nose"].y  # y-down: ankle below nose
        if extent > 0.0:
            extents.append(extent)
    if not extents:
        return fallback
    height_px = float(np.percentile(extents, 95))
    return body_plane_scale_from_height(athlete_height_cm, height_px)


def _unanalyzed_reason(window: RepWindow, unfilled_gaps: list[Gap]) -> str | None:
    """The reason string when an uninterpolated marker gap overlaps the rep."""
    for gap in unfilled_gaps:
        if gap.t_start < window.t_end and gap.t_end > window.t_start:
            return (
                f"bar marker lost for more than {MAX_GAP_FRAMES} frames during the rep "
                f"(t={gap.t_start:.2f}-{gap.t_end:.2f}s)"
            )
    return None

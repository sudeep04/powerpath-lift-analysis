"""Serialize an AnalysisResult to the FROZEN JSON contracts + annotated mp4.

The two JSON writers here produce EXACTLY the shapes frozen in
``docs/contracts/overlay-metrics-contract.md`` (the interface Task 10
serves verbatim from disk and Task 12's player renders):

* ``metrics.json`` -- the full analysis record: video meta, movement,
  load, versions, calibration provenance, and one entry per rep carrying
  made/score/excluded_from_templates, the scalar + per-phase-angle
  metrics, fault findings, and detected phase times. The ``smoothness``
  key is the scorer's smoothness fraction (1.0 = perfectly smooth, 0.5 at
  scoring.NJ_HALF_CREDIT normalized jerk) -- the 0-1 number the contract
  shows, not the raw normalized jerk (which lives on RepMetrics).
* ``overlay.json`` -- what the player canvas binary-searches while the
  video plays: ``frames[]`` keyed by strictly-increasing PTS ``t`` with the
  bar ``[x, y]`` and the ``skeleton`` landmark map in UPRIGHT IMAGE pixels
  (y-down, drawable directly), plus ``reps[]`` with the per-rep
  ``bar_path`` polyline, phases, faults, score/made and
  ``unanalyzed_reason``. Overlay faults carry ``severity``
  ("fault" | "informational" -- the UI mutes informational findings);
  metrics faults stay the frozen 5-key shape without it.

Serialization rules (contract): snake_case keys; never NaN/Infinity --
every value is sanitized to null (``json.dump(allow_nan=False)`` enforces
it); missed reps stay present with ``made: false, score: null``; phase maps
carry only detected phases; scores are rounded to integers.

``write_annotated_mp4`` is a SECOND streaming decode pass (the single-pass
rule constrains the analysis pass, not rendering; frames are still
processed one at a time): each decoded frame gets the bar trail so far in
the current rep, the nearest skeleton, and the rep/score/fault banner, then
goes straight to an OpenCV VideoWriter.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from powerpath_engine import decode
from powerpath_engine.faults import FaultFinding
from powerpath_engine.pipeline import AnalysisResult, RepResult
from powerpath_engine.scoring import smoothness_fraction
from powerpath_engine.series import LandmarkSeries, TimeSeries

# Two frame timestamps within this are "the same instant" when merging the
# bar and skeleton channels into overlay frames[] (matches series.py's
# per-landmark merge tolerance; far below any real frame spacing).
_MERGE_TOLERANCE_S = 1e-6

# --- annotated-video styling ------------------------------------------------
_TRAIL_COLOR = (0, 255, 255)  # yellow bar trail
_BAR_COLOR = (255, 0, 255)  # magenta current bar position
_SKELETON_COLOR = (0, 220, 0)  # green joints/bones
_TEXT_COLOR = (255, 255, 255)
_FAULT_COLOR = (0, 0, 255)
_BONES = (
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
)


def write_metrics_json(result: AnalysisResult, path: str | Path) -> None:
    """Write the frozen ``metrics.json`` contract for ``result`` to ``path``."""
    payload = {
        "video": _video_dict(result),
        "movement": result.movement,
        "load_kg": result.load_kg,
        "extraction_version": result.extraction_version,
        "rules_version": result.rules_version,
        "calibration": {
            "source": result.calibration.source,
            "bar_scale_cm_per_px": result.calibration.bar_scale.cm_per_px,
            "warning": result.calibration.warning,
        },
        "reps": [_metrics_rep(rep) for rep in result.reps],
    }
    _dump(payload, path)


def write_overlay_json(
    result: AnalysisResult,
    bar_series: TimeSeries,
    landmark_series: LandmarkSeries,
    path: str | Path,
) -> None:
    """Write the frozen ``overlay.json`` contract to ``path``.

    ``bar_series`` and ``landmark_series`` are the image-pixel-space series
    the pipeline kept for exactly this purpose (``result.bar_px`` /
    ``result.landmarks_px``). ``frames[]`` is their union keyed by PTS
    ``t`` (strictly increasing): a frame missing one channel carries
    ``bar: null`` / an empty ``skeleton``.
    """
    payload = {
        "video": _video_dict(result),
        "movement": result.movement,
        "frames": _merged_frames(bar_series, landmark_series),
        "reps": [_overlay_rep(rep, bar_series) for rep in result.reps],
    }
    _dump(payload, path)


def write_annotated_mp4(
    video_path: str | Path, overlay_data: dict[str, Any], out_path: str | Path
) -> None:
    """Render ``video_path`` with ``overlay_data`` drawn on, to ``out_path``.

    ``overlay_data`` is the (already-sanitized) dict ``write_overlay_json``
    writes -- pass it straight or ``json.load`` it back. A second streaming
    decode pass: one frame decoded, annotated (bar trail within the current
    rep up to now, nearest skeleton, rep/score banner + first fault line),
    and written at a time via OpenCV's VideoWriter; nothing is buffered.
    """
    video = overlay_data.get("video", {})
    fps = float(video.get("fps_avg") or 0.0) or 30.0
    frames = overlay_data.get("frames", [])
    reps = overlay_data.get("reps", [])
    bar_points = [(f["t"], f["bar"][0], f["bar"][1]) for f in frames if f.get("bar")]
    bar_ts = [p[0] for p in bar_points]
    skeleton_frames = [(f["t"], f["skeleton"]) for f in frames if f.get("skeleton")]
    skeleton_ts = [t for t, _ in skeleton_frames]

    writer: cv2.VideoWriter | None = None
    try:
        for frame in decode.frames(video_path):
            if writer is None:
                height, width = frame.image.shape[:2]
                writer = cv2.VideoWriter(
                    str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                )
                if not writer.isOpened():
                    raise RuntimeError(f"could not open VideoWriter for {out_path}")
            image = frame.image.copy()
            rep = _active_rep(reps, frame.t)
            _draw_trail(image, bar_points, bar_ts, rep, frame.t)
            _draw_skeleton(image, skeleton_frames, skeleton_ts, frame.t)
            _draw_banner(image, rep, len(reps))
            writer.write(image)
    finally:
        if writer is not None:
            writer.release()


# --- JSON assembly ----------------------------------------------------------


def _video_dict(result: AnalysisResult) -> dict[str, Any]:
    return {
        "width": result.video.width,
        "height": result.video.height,
        "fps_avg": result.video.fps_avg,
        "duration_s": result.video.duration_s,
    }


def _score_int(rep: RepResult) -> int | None:
    return int(round(rep.score)) if rep.score is not None else None


def _metrics_fault_dict(fault: FaultFinding) -> dict[str, Any]:
    """The frozen 5-key metrics.json fault shape (no severity)."""
    return {
        "code": fault.code,
        "message": fault.message,
        "phase": fault.phase,
        "value": fault.value,
        "threshold": fault.threshold,
    }


def _overlay_fault_dict(fault: FaultFinding) -> dict[str, Any]:
    """The 6-key overlay.json fault shape: the metrics keys plus ``severity``
    ("fault" | "informational") so the UI can mute informational findings."""
    return {**_metrics_fault_dict(fault), "severity": fault.severity}


def _detected_phases(rep: RepResult) -> dict[str, float]:
    return {name: t for name, t in rep.phases.items() if t is not None}


def _metrics_rep(rep: RepResult) -> dict[str, Any]:
    m = rep.metrics
    return {
        "rep_index": rep.window.rep_index,
        "made": rep.made,
        "score": _score_int(rep),
        "excluded_from_templates": rep.excluded_from_templates,
        "metrics": {
            "bar_drift_cm": m.bar_drift_cm,
            "peak_concentric_velocity_ms": m.peak_concentric_velocity_ms,
            "path_length_ratio": m.path_length_ratio,
            "smoothness": smoothness_fraction(m.smoothness_normalized_jerk),
            "hip_angle_at_phase": dict(m.hip_angle_at_phase),
            "knee_angle_at_phase": dict(m.knee_angle_at_phase),
            "elbow_angle_at_phase": dict(m.elbow_angle_at_phase),
        },
        "faults": [_metrics_fault_dict(f) for f in rep.faults],
        "phases": _detected_phases(rep),
    }


def _overlay_rep(rep: RepResult, bar_series: TimeSeries) -> dict[str, Any]:
    window = rep.window
    bar_path = [
        [round(s.x, 2), round(s.y, 2)]
        for s in bar_series.samples
        if window.t_start <= s.t <= window.t_end
    ]
    return {
        "rep_index": window.rep_index,
        "t_start": window.t_start,
        "t_end": window.t_end,
        "made": rep.made,
        "score": _score_int(rep),
        "bar_path": bar_path,
        "phases": _detected_phases(rep),
        "faults": [_overlay_fault_dict(f) for f in rep.faults],
        "unanalyzed_reason": rep.unanalyzed_reason,
    }


def _merged_frames(bar_series: TimeSeries, landmark_series: LandmarkSeries) -> list[dict[str, Any]]:
    """Union of the bar and skeleton channels as frames[], strictly
    increasing in ``t`` (channels within _MERGE_TOLERANCE_S share a frame)."""
    events: list[tuple[float, str, Any]] = [
        (s.t, "bar", [round(s.x, 2), round(s.y, 2)]) for s in bar_series.samples
    ]
    events.extend(
        (
            frame.t,
            "skeleton",
            {name: [round(p.x, 2), round(p.y, 2)] for name, p in frame.points.items()},
        )
        for frame in landmark_series.frames
    )
    events.sort(key=lambda event: event[0])

    frames: list[dict[str, Any]] = []
    for t, channel, payload in events:
        if not frames or t - frames[-1]["t"] > _MERGE_TOLERANCE_S:
            frames.append({"t": t, "bar": None, "skeleton": {}})
        frames[-1][channel] = payload
    return frames


def _dump(payload: dict[str, Any], path: str | Path) -> None:
    """Sanitize (never NaN/Infinity -- the contract) and write JSON."""
    with open(path, "w") as fh:
        json.dump(_sanitize(payload), fh, allow_nan=False)


def _sanitize(value: Any) -> Any:
    """Recursively convert non-finite floats to None (JSON null)."""
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _sanitize(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_sanitize(inner) for inner in value]
    return value


# --- annotated-video drawing -------------------------------------------------


def _active_rep(reps: list[dict[str, Any]], t: float) -> dict[str, Any] | None:
    for rep in reps:
        if rep["t_start"] <= t <= rep["t_end"]:
            return rep
    return None


def _draw_trail(
    image: np.ndarray,
    bar_points: list[tuple[float, float, float]],
    bar_ts: list[float],
    rep: dict[str, Any] | None,
    t: float,
) -> None:
    """The bar path from the current rep's start up to ``t`` + the bar dot."""
    if not bar_points:
        return
    end = bisect_right(bar_ts, t)
    start = bisect_right(bar_ts, rep["t_start"]) if rep is not None else max(0, end - 1)
    trail = bar_points[start:end]
    if len(trail) >= 2:
        points = np.array([[int(round(x)), int(round(y))] for _t, x, y in trail], dtype=np.int32)
        cv2.polylines(image, [points], isClosed=False, color=_TRAIL_COLOR, thickness=2)
    if end > 0:
        _t, x, y = bar_points[end - 1]
        cv2.circle(image, (int(round(x)), int(round(y))), 6, _BAR_COLOR, -1)


def _draw_skeleton(
    image: np.ndarray,
    skeleton_frames: list[tuple[float, dict[str, Any]]],
    skeleton_ts: list[float],
    t: float,
) -> None:
    """The most recent skeleton at/before ``t`` (pose may be strided)."""
    index = bisect_right(skeleton_ts, t) - 1
    if index < 0:
        return
    skeleton = skeleton_frames[index][1]
    points = {name: (int(round(xy[0])), int(round(xy[1]))) for name, xy in skeleton.items() if xy}
    for a, b in _BONES:
        if a in points and b in points:
            cv2.line(image, points[a], points[b], _SKELETON_COLOR, 2)
    for xy in points.values():
        cv2.circle(image, xy, 3, _SKELETON_COLOR, -1)


def _draw_banner(image: np.ndarray, rep: dict[str, Any] | None, total_reps: int) -> None:
    if rep is None:
        return
    if rep.get("unanalyzed_reason"):
        status = "UNANALYZED"
    elif rep.get("made"):
        status = f"score {rep['score']}" if rep.get("score") is not None else "made"
    else:
        status = "MISSED"
    banner = f"rep {rep['rep_index'] + 1}/{total_reps}  {status}"
    cv2.putText(image, banner, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, _TEXT_COLOR, 2)
    faults = rep.get("faults") or []
    if faults:
        cv2.putText(
            image, faults[0]["code"], (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.6, _FAULT_COLOR, 2
        )

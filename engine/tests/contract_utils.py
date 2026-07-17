"""Assertions for the FROZEN overlay.json / metrics.json contracts.

One place encodes `.superpowers/sdd/overlay-metrics-contract.md` as
executable checks so every producer test (overlay writer unit tests, the
sample-video E2E, the API process-pool integration test and the fake-engine
runner) validates against the identical shape. Key sets are compared with
EQUALITY -- the contract is frozen, so a missing key and an extra key are
both violations.
"""

from __future__ import annotations

from typing import Any

VIDEO_KEYS = {"width", "height", "fps_avg", "duration_s"}
CALIBRATION_KEYS = {"source", "bar_scale_cm_per_px", "warning"}
# metrics.json faults are the frozen 5-key shape; overlay.json faults add
# `severity` ("fault" | "informational") so the UI can mute informational
# findings.
METRICS_FAULT_KEYS = {"code", "message", "phase", "value", "threshold"}
OVERLAY_FAULT_KEYS = METRICS_FAULT_KEYS | {"severity"}

METRICS_TOP_KEYS = {
    "video",
    "movement",
    "load_kg",
    "extraction_version",
    "rules_version",
    "calibration",
    "reps",
}
METRICS_REP_KEYS = {
    "rep_index",
    "made",
    "score",
    "excluded_from_templates",
    "metrics",
    "faults",
    "phases",
}
METRICS_METRICS_KEYS = {
    "bar_drift_cm",
    "peak_concentric_velocity_ms",
    "path_length_ratio",
    "smoothness",
    "hip_angle_at_phase",
    "knee_angle_at_phase",
    "elbow_angle_at_phase",
}

OVERLAY_TOP_KEYS = {"video", "movement", "frames", "reps"}
OVERLAY_FRAME_KEYS = {"t", "bar", "skeleton"}
OVERLAY_REP_KEYS = {
    "rep_index",
    "t_start",
    "t_end",
    "made",
    "score",
    "bar_path",
    "phases",
    "faults",
    "unanalyzed_reason",
}


def _assert_point(value: Any, label: str) -> None:
    assert (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(v, int | float) for v in value)
    ), f"{label} must be [x, y], got {value!r}"


def _assert_video(video: Any) -> None:
    assert set(video) == VIDEO_KEYS
    for key in VIDEO_KEYS:
        assert isinstance(video[key], int | float), f"video.{key} must be numeric"


def _assert_faults(faults: Any, keys: set[str]) -> None:
    assert isinstance(faults, list)
    for fault in faults:
        assert set(fault) == keys
        assert isinstance(fault["code"], str)
        assert isinstance(fault["message"], str)
        if "severity" in keys:
            assert fault["severity"] in ("fault", "informational")


def _assert_phases(phases: Any) -> None:
    """Phase maps carry ONLY detected phases: name -> numeric t, never null."""
    assert isinstance(phases, dict)
    for name, t in phases.items():
        assert isinstance(name, str)
        assert isinstance(t, int | float), f"phase {name} must have a numeric t, got {t!r}"


def assert_metrics_contract(data: dict[str, Any]) -> None:
    """Assert ``data`` (a parsed metrics.json) matches the frozen contract."""
    assert set(data) == METRICS_TOP_KEYS
    _assert_video(data["video"])
    assert isinstance(data["movement"], str)
    assert isinstance(data["load_kg"], int | float)
    assert isinstance(data["extraction_version"], int)
    assert isinstance(data["rules_version"], int)
    calibration = data["calibration"]
    assert set(calibration) == CALIBRATION_KEYS
    assert calibration["source"] in ("plate", "date_fallback", "manual")
    assert isinstance(calibration["bar_scale_cm_per_px"], int | float)

    assert isinstance(data["reps"], list)
    for rep in data["reps"]:
        assert set(rep) == METRICS_REP_KEYS
        assert isinstance(rep["rep_index"], int)
        assert isinstance(rep["made"], bool)
        assert rep["score"] is None or isinstance(rep["score"], int | float)
        if not rep["made"]:
            assert rep["score"] is None, "missed reps carry score: null"
        assert isinstance(rep["excluded_from_templates"], bool)
        assert set(rep["metrics"]) == METRICS_METRICS_KEYS
        for angle_key in ("hip_angle_at_phase", "knee_angle_at_phase", "elbow_angle_at_phase"):
            assert isinstance(rep["metrics"][angle_key], dict)
        _assert_faults(rep["faults"], METRICS_FAULT_KEYS)
        _assert_phases(rep["phases"])


def assert_overlay_contract(data: dict[str, Any]) -> None:
    """Assert ``data`` (a parsed overlay.json) matches the frozen contract."""
    assert set(data) == OVERLAY_TOP_KEYS
    _assert_video(data["video"])
    assert isinstance(data["movement"], str)

    frames = data["frames"]
    assert isinstance(frames, list)
    previous_t: float | None = None
    for frame in frames:
        assert set(frame) == OVERLAY_FRAME_KEYS
        t = frame["t"]
        assert isinstance(t, int | float)
        if previous_t is not None:
            assert t > previous_t, (
                f"frames[].t must be strictly increasing ({t} after {previous_t})"
            )
        previous_t = t
        if frame["bar"] is not None:
            _assert_point(frame["bar"], "frames[].bar")
        assert isinstance(frame["skeleton"], dict)
        for name, point in frame["skeleton"].items():
            assert isinstance(name, str)
            if point is not None:
                _assert_point(point, f"skeleton[{name}]")

    assert isinstance(data["reps"], list)
    for rep in data["reps"]:
        assert set(rep) == OVERLAY_REP_KEYS
        assert isinstance(rep["rep_index"], int)
        assert isinstance(rep["t_start"], int | float)
        assert isinstance(rep["t_end"], int | float)
        assert rep["t_start"] <= rep["t_end"]
        assert isinstance(rep["made"], bool)
        assert rep["score"] is None or isinstance(rep["score"], int | float)
        if not rep["made"]:
            assert rep["score"] is None, "missed reps carry score: null"
        assert isinstance(rep["bar_path"], list)
        for point in rep["bar_path"]:
            _assert_point(point, "reps[].bar_path[]")
        _assert_phases(rep["phases"])
        _assert_faults(rep["faults"], OVERLAY_FAULT_KEYS)
        assert rep["unanalyzed_reason"] is None or isinstance(rep["unanalyzed_reason"], str)

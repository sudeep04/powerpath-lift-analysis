"""Tests for powerpath_engine.metrics.

The bar-trajectory metrics (drift, peak velocity, path ratio, smoothness) are
well-defined on realistic bar motion, so they are exercised both on the
synthetic generators (via the real segment -> detect_phases pipeline) and on
hand-built bar series where the quantity has a known closed form. The joint
angles are degenerate on the synthetic body (see the metrics module docstring)
so they are only checked for shape here and driven with controlled values in
the fault/score tests.
"""

from __future__ import annotations

import numpy as np
from synthetic import back_squat, clean, deadlift

from powerpath_engine import registry
from powerpath_engine.geometry import PlaneScale
from powerpath_engine.metrics import RepMetrics, compute_rep_metrics
from powerpath_engine.phases import detect_phases
from powerpath_engine.segmentation import RepWindow, segment
from powerpath_engine.series import Sample, TimeSeries

EXPECTED_DT = 1.0 / 60.0
IDENTITY = PlaneScale(cm_per_px=1.0)


def _linear_bar(
    y0: float, y1: float, *, x0: float = 0.0, x1: float = 0.0, n: int = 60
) -> TimeSeries:
    """A straight-line bar move from (x0, y0) to (x1, y1) over 1 second."""
    ts = np.linspace(0.0, 1.0, n)
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    return TimeSeries(
        [Sample(t=float(t), x=float(x), y=float(y)) for t, x, y in zip(ts, xs, ys, strict=True)]
    )


def _window(bar: TimeSeries) -> RepWindow:
    return RepWindow(t_start=bar.samples[0].t, t_end=bar.samples[-1].t + 1e-6, rep_index=0)


# --- hand-built bar series: closed-form metrics ----------------------------


def test_straight_vertical_bar_has_unit_path_ratio_and_no_drift() -> None:
    bar = _linear_bar(0.0, 100.0)
    config = registry.get("deadlift")
    m = compute_rep_metrics(_window(bar), bar, _empty_landmarks(bar), {}, config, IDENTITY)
    assert m.bar_drift_cm < 1e-6
    assert abs(m.path_length_ratio - 1.0) < 1e-6


def test_bar_drift_is_max_horizontal_deviation_cm() -> None:
    bar = _linear_bar(0.0, 100.0, x0=0.0, x1=10.0)
    config = registry.get("deadlift")
    m = compute_rep_metrics(_window(bar), bar, _empty_landmarks(bar), {}, config, IDENTITY)
    assert abs(m.bar_drift_cm - 10.0) < 0.2
    # Diagonal path is longer than the pure vertical travel.
    assert m.path_length_ratio > 1.0


def test_scale_converts_drift_pixels_to_cm() -> None:
    bar = _linear_bar(0.0, 100.0, x0=0.0, x1=5.0)
    config = registry.get("deadlift")
    identity = compute_rep_metrics(_window(bar), bar, _empty_landmarks(bar), {}, config, IDENTITY)
    doubled = compute_rep_metrics(
        _window(bar), bar, _empty_landmarks(bar), {}, config, PlaneScale(cm_per_px=2.0)
    )
    assert abs(doubled.bar_drift_cm - 2.0 * identity.bar_drift_cm) < 1e-6


def test_peak_concentric_velocity_from_known_rise_rate() -> None:
    # 100 cm over 1 s = 100 cm/s = 1.0 m/s upward.
    bar = _linear_bar(0.0, 100.0)
    config = registry.get("deadlift")
    m = compute_rep_metrics(_window(bar), bar, _empty_landmarks(bar), {}, config, IDENTITY)
    assert abs(m.peak_concentric_velocity_ms - 1.0) < 0.05


def test_descending_only_bar_has_zero_concentric_velocity() -> None:
    bar = _linear_bar(100.0, 0.0)
    config = registry.get("deadlift")
    m = compute_rep_metrics(_window(bar), bar, _empty_landmarks(bar), {}, config, IDENTITY)
    assert m.peak_concentric_velocity_ms == 0.0


def test_jittery_bar_is_less_smooth_than_clean_bar() -> None:
    ts = np.linspace(0.0, 1.0, 61)
    ys = 100.0 * np.sin(np.pi * ts)  # smooth up-and-over bump
    smooth = TimeSeries(
        [Sample(t=float(t), x=0.0, y=float(y)) for t, y in zip(ts, ys, strict=True)]
    )
    rng = np.random.default_rng(0)
    jit = ys + rng.normal(0.0, 3.0, size=ys.shape)
    jittery = TimeSeries(
        [Sample(t=float(t), x=0.0, y=float(y)) for t, y in zip(ts, jit, strict=True)]
    )
    config = registry.get("deadlift")
    m_smooth = compute_rep_metrics(
        _window(smooth), smooth, _empty_landmarks(smooth), {}, config, IDENTITY
    )
    m_jit = compute_rep_metrics(
        _window(jittery), jittery, _empty_landmarks(jittery), {}, config, IDENTITY
    )
    assert m_jit.smoothness_normalized_jerk > m_smooth.smoothness_normalized_jerk


# --- synthetic pipeline: shape + plausibility ------------------------------


def _empty_landmarks(bar: TimeSeries):
    from powerpath_engine.series import LandmarkSeries

    return LandmarkSeries([])


def _first_rep_metrics(lift, config_key):
    config = registry.get(config_key)
    windows = segment(lift.bar, config, EXPECTED_DT)
    assert windows
    phases = detect_phases(windows[0], lift.bar, lift.landmarks, config)
    return compute_rep_metrics(windows[0], lift.bar, lift.landmarks, phases, config, IDENTITY)


def test_clean_metrics_are_plausible_and_have_phase_angles() -> None:
    lift = clean(3, seed=1)
    m = _first_rep_metrics(lift, "power_clean")
    # A clean pull rises fast: peak concentric velocity is a few m/s.
    assert 1.0 < m.peak_concentric_velocity_ms < 10.0
    # No lateral noise in the fixture -> near-vertical path, tiny drift.
    assert m.bar_drift_cm < 1.0
    assert m.path_length_ratio < 1.05
    assert m.smoothness_normalized_jerk >= 0.0
    # Angles are recorded for each detected phase (values may be near 180).
    assert set(m.elbow_angle_at_phase) == {"first_pull", "knee_pass", "second_pull", "catch"}
    for a in m.hip_angle_at_phase.values():
        assert a is None or 0.0 <= a <= 180.0
    # A clean has a front-rack catch, so the catch-height ratio is defined.
    assert m.catch_height_ratio is not None and m.catch_height_ratio > 0.0


def test_squat_metrics_capture_bottom_hip_and_knee_height() -> None:
    lift = back_squat(2, seed=1)
    m = _first_rep_metrics(lift, "back_squat")
    assert m.bottom_hip_y_cm is not None
    assert m.bottom_knee_y_cm is not None
    # No catch phase in a squat.
    assert m.catch_height_ratio is None


def test_deadlift_has_no_catch_ratio() -> None:
    lift = deadlift(2, seed=1)
    m = _first_rep_metrics(lift, "deadlift")
    assert m.catch_height_ratio is None


def test_metrics_are_deterministic() -> None:
    lift = clean(2, seed=3)
    a = _first_rep_metrics(lift, "power_clean")
    b = _first_rep_metrics(lift, "power_clean")
    assert a == b


def test_repmetrics_is_directly_constructible_with_defaults() -> None:
    # The fault/score tests rely on this: build with only the fields they need.
    m = RepMetrics(elbow_angle_at_phase={"second_pull": 160.0})
    assert m.bar_drift_cm == 0.0
    assert m.path_length_ratio == 1.0
    assert m.elbow_angle_at_phase["second_pull"] == 160.0

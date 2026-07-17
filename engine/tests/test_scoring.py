"""Tests for powerpath_engine.scoring.

Made/missed is exercised both as the pure :func:`is_made` core (with the two
trajectory booleans supplied directly, so the angle gates -- degenerate on the
synthetic body -- are driven with controlled RepMetrics) and end-to-end via
:func:`evaluate_made` on the clean/dumped-clean fixtures. Scoring is driven with
constructed RepMetrics + fake velocity histories so every component and the
redistribution branch have exact expected values.
"""

from __future__ import annotations

import synthetic

from powerpath_engine import registry
from powerpath_engine.geometry import PlaneScale
from powerpath_engine.metrics import RepMetrics, compute_rep_metrics
from powerpath_engine.phases import detect_phases
from powerpath_engine.scoring import (
    MIN_HISTORY_REPS,
    VELOCITY_LOAD_TOLERANCE,
    RepScore,
    evaluate_made,
    is_made,
    score_rep,
)
from powerpath_engine.segmentation import segment

EXPECTED_DT = 1.0 / 60.0
IDENTITY = PlaneScale(cm_per_px=1.0)

CLEAN = registry.get("power_clean")
SQUAT = registry.get("back_squat")
PRESS = registry.get("push_press")
SNATCH = registry.get("power_snatch")
DEADLIFT = registry.get("deadlift")


class FakeHistory:
    """A VelocityHistory that returns a fixed list and records the call args."""

    def __init__(self, velocities: list[float]) -> None:
        self.velocities = velocities
        self.calls: list[tuple[float, float]] = []

    def peak_velocities_near_load(self, load_kg: float, tolerance_frac: float) -> list[float]:
        self.calls.append((load_kg, tolerance_frac))
        return list(self.velocities)


def _faults(n: int) -> list:
    return [f"fault{i}" for i in range(n)]


# --- is_made: pure decision core -------------------------------------------


def test_is_made_true_when_required_phase_present_and_controlled() -> None:
    assert is_made(
        {"catch": 1.1}, RepMetrics(), CLEAN, returned_to_start=True, terminal_freefall=False
    )


def test_is_made_false_on_terminal_freefall() -> None:
    assert not is_made(
        {"catch": 1.1}, RepMetrics(), CLEAN, returned_to_start=True, terminal_freefall=True
    )


def test_is_made_false_when_required_phase_missing() -> None:
    assert not is_made(
        {"catch": None}, RepMetrics(), CLEAN, returned_to_start=True, terminal_freefall=False
    )


def test_squat_eventless_standing_needs_return_to_start() -> None:
    # back_squat requires ("bottom", "standing"); "standing" is eventless so it
    # is satisfied only by returning to the start band. Knee gate <= 90 met.
    m = RepMetrics(knee_angle_at_phase={"bottom": 85.0})
    phases = {"bottom": 0.7}
    assert is_made(phases, m, SQUAT, returned_to_start=True, terminal_freefall=False)
    assert not is_made(phases, m, SQUAT, returned_to_start=False, terminal_freefall=False)


def test_squat_depth_gate_blocks_made() -> None:
    phases = {"bottom": 0.7}
    shallow = RepMetrics(knee_angle_at_phase={"bottom": 100.0})  # > 90 -> too shallow
    assert not is_made(phases, shallow, SQUAT, returned_to_start=True, terminal_freefall=False)
    missing = RepMetrics()
    assert not is_made(phases, missing, SQUAT, returned_to_start=True, terminal_freefall=False)


def test_press_lockout_angle_gate() -> None:
    phases = {"lockout": 0.55}
    locked = RepMetrics(
        hip_angle_at_phase={"lockout": 178.0},
        knee_angle_at_phase={"lockout": 178.0},
        elbow_angle_at_phase={"lockout": 170.0},
    )
    assert is_made(phases, locked, PRESS, returned_to_start=True, terminal_freefall=False)
    soft = RepMetrics(
        hip_angle_at_phase={"lockout": 178.0},
        knee_angle_at_phase={"lockout": 178.0},
        elbow_angle_at_phase={"lockout": 150.0},  # < 160
    )
    assert not is_made(phases, soft, PRESS, returned_to_start=True, terminal_freefall=False)


def test_snatch_lockout_uses_receive_elbow() -> None:
    phases = {"receive": 1.15}
    m = RepMetrics(elbow_angle_at_phase={"receive": 168.0})
    assert is_made(phases, m, SNATCH, returned_to_start=True, terminal_freefall=False)
    m2 = RepMetrics(elbow_angle_at_phase={"receive": 150.0})
    assert not is_made(phases, m2, SNATCH, returned_to_start=True, terminal_freefall=False)


def test_deadlift_lockout_uses_standing_hip_knee() -> None:
    phases = {"lockout": 0.85}
    m = RepMetrics(hip_angle_at_phase={"lockout": 170.0}, knee_angle_at_phase={"lockout": 172.0})
    assert is_made(phases, m, DEADLIFT, returned_to_start=True, terminal_freefall=False)
    m2 = RepMetrics(hip_angle_at_phase={"lockout": 160.0}, knee_angle_at_phase={"lockout": 172.0})
    assert not is_made(phases, m2, DEADLIFT, returned_to_start=True, terminal_freefall=False)


# --- evaluate_made: end-to-end on fixtures ---------------------------------


def _pipeline_made(lift, config_key):
    config = registry.get(config_key)
    windows = segment(lift.bar, config, EXPECTED_DT)
    results = []
    for w in windows:
        phases = detect_phases(w, lift.bar, lift.landmarks, config)
        metrics = compute_rep_metrics(w, lift.bar, lift.landmarks, phases, config, IDENTITY)
        results.append(evaluate_made(w, lift.bar, phases, metrics, config))
    return results


def test_good_clean_is_made() -> None:
    assert _pipeline_made(synthetic.clean(1, seed=1), "power_clean") == [True]


def test_dumped_clean_is_missed() -> None:
    # The one good pull reaches the rack but the bar is then dumped into
    # free-fall -> terminal free-fall marks the rep missed.
    assert _pipeline_made(synthetic.dumped_clean(), "power_clean") == [False]


# --- score_rep -------------------------------------------------------------


def test_missed_rep_is_unscored_and_excluded() -> None:
    s = score_rep(RepMetrics(), _faults(0), made=False, history=FakeHistory([]), load_kg=100.0)
    assert s == RepScore(
        made=False,
        score=None,
        excluded_from_templates=True,
        smoothness_component=None,
        path_component=None,
        velocity_component=None,
        fault_component=None,
        velocity_redistributed=False,
    )


def test_perfect_rep_with_full_history_scores_100() -> None:
    m = RepMetrics(
        smoothness_normalized_jerk=0.0, path_length_ratio=1.0, peak_concentric_velocity_ms=2.0
    )
    history = FakeHistory([2.0] * MIN_HISTORY_REPS)
    s = score_rep(m, _faults(0), made=True, history=history, load_kg=100.0)
    assert s.score == 100.0
    assert not s.velocity_redistributed
    assert s.velocity_component == 20.0
    assert s.smoothness_component == 30.0 and s.path_component == 30.0
    # velocity was queried at +/-10% load.
    assert history.calls == [(100.0, VELOCITY_LOAD_TOLERANCE)]


def test_velocity_redistribution_below_five_history_reps() -> None:
    m = RepMetrics(
        smoothness_normalized_jerk=0.0, path_length_ratio=1.0, peak_concentric_velocity_ms=2.0
    )
    s = score_rep(m, _faults(0), made=True, history=FakeHistory([2.0] * 4), load_kg=80.0)
    assert s.velocity_redistributed
    assert s.velocity_component is None
    assert s.smoothness_component == 40.0 and s.path_component == 40.0
    assert s.score == 100.0


def test_history_boundary_at_five_reps_uses_velocity() -> None:
    m = RepMetrics(smoothness_normalized_jerk=0.0, path_length_ratio=1.0)
    below = score_rep(m, _faults(0), made=True, history=FakeHistory([1.0] * 4), load_kg=1.0)
    at = score_rep(m, _faults(0), made=True, history=FakeHistory([1.0] * 5), load_kg=1.0)
    assert below.velocity_redistributed and not at.velocity_redistributed


def test_slow_rep_earns_partial_velocity() -> None:
    m = RepMetrics(
        smoothness_normalized_jerk=0.0, path_length_ratio=1.0, peak_concentric_velocity_ms=1.0
    )
    # median history 2.0 m/s, rep at 1.0 -> half the velocity credit (10 of 20).
    s = score_rep(m, _faults(0), made=True, history=FakeHistory([2.0] * 5), load_kg=100.0)
    assert s.velocity_component == 10.0
    assert s.score == 90.0


def test_each_fault_costs_seven_with_a_floor_of_zero() -> None:
    m = RepMetrics(smoothness_normalized_jerk=0.0, path_length_ratio=1.0)
    two = score_rep(m, _faults(2), made=True, history=FakeHistory([]), load_kg=50.0)
    assert two.fault_component == 6.0  # 20 - 2*7
    many = score_rep(m, _faults(4), made=True, history=FakeHistory([]), load_kg=50.0)
    assert many.fault_component == 0.0  # floored, not negative


def test_terrible_rep_scores_near_zero_and_is_clamped() -> None:
    m = RepMetrics(smoothness_normalized_jerk=1e9, path_length_ratio=3.0)
    s = score_rep(m, _faults(5), made=True, history=FakeHistory([]), load_kg=50.0)
    assert s.score is not None
    assert 0.0 <= s.score <= 100.0
    assert s.score < 1.0


def test_score_is_deterministic() -> None:
    m = RepMetrics(
        smoothness_normalized_jerk=500.0, path_length_ratio=1.2, peak_concentric_velocity_ms=1.5
    )
    a = score_rep(m, _faults(1), made=True, history=FakeHistory([1.6] * 6), load_kg=100.0)
    b = score_rep(m, _faults(1), made=True, history=FakeHistory([1.6] * 6), load_kg=100.0)
    assert a == b
    assert a.score is not None and 0.0 <= a.score <= 100.0

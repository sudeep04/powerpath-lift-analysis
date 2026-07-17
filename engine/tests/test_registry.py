"""Tests for powerpath_engine.registry.

The registry is pure declarative data (Task 6a). These tests pin the config
data model, the six M1 movement configs, the lookup API, and the invariant
that every phase detector is drawn from the fixed Task-6b vocabulary. No
segmentation or detector *behaviour* is exercised here -- that is Task 6b.
"""

from __future__ import annotations

import pytest

from powerpath_engine.registry import (
    UnknownMovementError,
    all_configs,
    all_keys,
    get,
)
from powerpath_engine.registry.base import (
    KNOWN_DETECTORS,
    MadeCriteria,
    MovementConfig,
    PhaseDef,
)
from powerpath_engine.series import LANDMARK_NAMES

# The full set of M1 movement keys the registry must expose.
EXPECTED_KEYS = {
    "power_clean",
    "power_snatch",
    "back_squat",
    "push_press",
    "deadlift",
    "hang_power_clean",
}

# fault_rules per movement: only rules implemented in faults._FAULT_RULES and
# detectable from the single-side-view camera premise (dropped names are noted
# as v2 in each config module).
EXPECTED_FAULT_RULES = {
    "power_clean": ("early_arm_bend", "bar_drift", "catch_above_parallel"),
    "hang_power_clean": ("early_arm_bend", "bar_drift", "catch_above_parallel"),
    "power_snatch": ("early_arm_bend", "bar_drift", "no_lockout"),
    "back_squat": ("squat_depth", "bar_drift"),
    "push_press": ("early_press_out", "bar_drift", "no_lockout"),
    "deadlift": ("bar_drift", "no_lockout"),
}

# min_disp_cm per family: 20 for squat/hinge, 40 for the explosive family.
EXPECTED_MIN_DISP = {
    "power_clean": 40.0,
    "power_snatch": 40.0,
    "push_press": 40.0,
    "back_squat": 20.0,
    "deadlift": 20.0,
    "hang_power_clean": 40.0,
}


# ---------------------------------------------------------------------------
# lookup API
# ---------------------------------------------------------------------------


def test_all_keys_are_the_expected_six() -> None:
    assert set(all_keys()) == EXPECTED_KEYS


def test_all_configs_matches_all_keys() -> None:
    configs = all_configs()
    assert {c.key for c in configs} == EXPECTED_KEYS
    assert all(isinstance(c, MovementConfig) for c in configs)
    assert len(configs) == len(all_keys())


@pytest.mark.parametrize("key", sorted(EXPECTED_KEYS))
def test_get_returns_config_with_matching_key(key: str) -> None:
    config = get(key)
    assert isinstance(config, MovementConfig)
    assert config.key == key


def test_get_unknown_raises_and_names_available_keys() -> None:
    with pytest.raises(UnknownMovementError) as excinfo:
        get("clean_and_jerk")
    message = str(excinfo.value)
    # the error must help the caller by listing what *is* available
    for key in EXPECTED_KEYS:
        assert key in message


# ---------------------------------------------------------------------------
# config validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config", all_configs(), ids=lambda c: c.key)
def test_config_fields_are_valid(config: MovementConfig) -> None:
    assert config.family in {"squat", "clean", "press", "snatch", "hinge"}
    assert config.starts_from in {"floor", "hang", "rack", "shoulders"}
    assert config.bar_travel in {"up", "down_up", "up_down"}
    assert config.phases  # non-empty
    assert config.min_disp_cm > 0.0
    assert isinstance(config.made_criteria, MadeCriteria)
    # made criteria required phases must all be real phases of this movement
    phase_names = {p.name for p in config.phases}
    assert set(config.made_criteria.required_phases) <= phase_names


@pytest.mark.parametrize("config", all_configs(), ids=lambda c: c.key)
def test_comparison_landmarks_are_valid_landmark_names(config: MovementConfig) -> None:
    assert config.comparison_landmarks  # non-empty
    for name in config.comparison_landmarks:
        assert name in LANDMARK_NAMES


@pytest.mark.parametrize("config", all_configs(), ids=lambda c: c.key)
def test_min_disp_cm_matches_family_expectation(config: MovementConfig) -> None:
    assert config.min_disp_cm == EXPECTED_MIN_DISP[config.key]


@pytest.mark.parametrize("config", all_configs(), ids=lambda c: c.key)
def test_fault_rules_are_the_reconciled_side_view_vocabulary(config: MovementConfig) -> None:
    assert config.fault_rules == EXPECTED_FAULT_RULES[config.key]


def test_movement_config_rejects_bad_family() -> None:
    with pytest.raises(ValueError, match="family"):
        MovementConfig(
            key="bogus",
            display_name="Bogus",
            family="jerk",  # type: ignore[arg-type]
            starts_from="floor",
            bar_travel="up",
            phases=(PhaseDef(name="setup", detector=""),),
            fault_rules=(),
            made_criteria=MadeCriteria(required_phases=()),
            comparison_landmarks=("left_hip",),
            min_disp_cm=40.0,
        )


def test_movement_config_rejects_unknown_comparison_landmark() -> None:
    with pytest.raises(ValueError, match="comparison_landmarks"):
        MovementConfig(
            key="bogus",
            display_name="Bogus",
            family="clean",
            starts_from="floor",
            bar_travel="up",
            phases=(PhaseDef(name="setup", detector=""),),
            fault_rules=(),
            made_criteria=MadeCriteria(required_phases=()),
            comparison_landmarks=("left_elbow_tip",),
            min_disp_cm=40.0,
        )


def test_movement_config_rejects_empty_phases() -> None:
    with pytest.raises(ValueError, match="phases"):
        MovementConfig(
            key="bogus",
            display_name="Bogus",
            family="clean",
            starts_from="floor",
            bar_travel="up",
            phases=(),
            fault_rules=(),
            made_criteria=MadeCriteria(required_phases=()),
            comparison_landmarks=("left_hip",),
            min_disp_cm=40.0,
        )


# ---------------------------------------------------------------------------
# detector vocabulary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config", all_configs(), ids=lambda c: c.key)
def test_every_non_empty_detector_is_in_the_known_vocabulary(config: MovementConfig) -> None:
    detectors = {p.detector for p in config.phases if p.detector}
    assert detectors <= KNOWN_DETECTORS


def test_known_detectors_is_the_fixed_task6b_vocabulary() -> None:
    assert KNOWN_DETECTORS == {
        "bar_leaves_floor",
        "knee_pass",
        "hip_contact",
        "peak_hip_extension_velocity",
        "catch_rack",
        "receive_overhead",
        "bottom",
        "lockout_top",
        "dip_turnaround",
    }


def test_power_clean_pulls_from_the_floor_with_bar_leaves_floor() -> None:
    detectors = [p.detector for p in get("power_clean").phases]
    assert "bar_leaves_floor" in detectors


def test_hang_power_clean_has_no_bar_leaves_floor_phase() -> None:
    """The hang variant starts above the knee, so it must not carry a
    first-pull / bar_leaves_floor phase -- this is the extensibility proof."""
    hang = get("hang_power_clean")
    detectors = [p.detector for p in hang.phases]
    assert "bar_leaves_floor" not in detectors
    # but it does share the rest of the clean's keyframes
    assert "knee_pass" in detectors
    assert "peak_hip_extension_velocity" in detectors
    assert "catch_rack" in detectors


def test_power_snatch_receives_overhead_not_racked() -> None:
    detectors = [p.detector for p in get("power_snatch").phases]
    assert "receive_overhead" in detectors
    assert "catch_rack" not in detectors


def test_deadlift_has_no_catch_phase() -> None:
    deadlift = get("deadlift")
    phase_names = {p.name for p in deadlift.phases}
    assert "catch" not in phase_names
    detectors = [p.detector for p in deadlift.phases]
    assert "catch_rack" not in detectors
    assert "lockout_top" in detectors

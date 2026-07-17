"""Tests for powerpath_engine.faults.

Every rule is a pure ``(RepMetrics, MovementConfig) -> FaultFinding | None``
function, so each is exercised directly with a :class:`RepMetrics` built to trip
it and one built not to (the synthetic body's angles are too degenerate to drive
the angle rules through the generators -- see the metrics module docstring).
:func:`evaluate_faults` dispatch is tested against both real registry configs
and hand-built configs.
"""

from __future__ import annotations

from powerpath_engine import registry
from powerpath_engine.faults import (
    BAR_DRIFT_ENVELOPE_CM,
    RULES_VERSION,
    FaultFinding,
    bar_drift,
    catch_above_parallel,
    early_arm_bend,
    early_press_out,
    evaluate_faults,
    no_lockout,
    squat_depth,
)
from powerpath_engine.metrics import RepMetrics
from powerpath_engine.registry import MadeCriteria, MovementConfig, PhaseDef

CLEAN = registry.get("power_clean")
SNATCH = registry.get("power_snatch")
SQUAT = registry.get("back_squat")
PRESS = registry.get("push_press")
DEADLIFT = registry.get("deadlift")


def _config(family: str, fault_rules: tuple[str, ...]) -> MovementConfig:
    """A minimal valid config of ``family`` naming ``fault_rules``."""
    return MovementConfig(
        key="test",
        display_name="Test",
        family=family,  # type: ignore[arg-type]
        starts_from="floor",
        bar_travel="up",
        phases=(PhaseDef(name="setup", detector=""),),
        fault_rules=fault_rules,
        made_criteria=MadeCriteria(required_phases=()),
        comparison_landmarks=("left_hip",),
        min_disp_cm=20.0,
    )


# --- early_arm_bend --------------------------------------------------------


def test_early_arm_bend_fires_when_elbow_bent_in_pull() -> None:
    m = RepMetrics(elbow_angle_at_phase={"second_pull": 160.0})
    f = early_arm_bend(m, CLEAN)
    assert f is not None
    assert f.code == "early_arm_bend"
    assert f.value == 160.0
    assert f.threshold == 170.0


def test_early_arm_bend_silent_when_arms_straight() -> None:
    m = RepMetrics(elbow_angle_at_phase={"first_pull": 178.0, "second_pull": 175.0})
    assert early_arm_bend(m, CLEAN) is None


def test_early_arm_bend_not_applicable_to_squat() -> None:
    m = RepMetrics(elbow_angle_at_phase={"second_pull": 100.0})
    assert early_arm_bend(m, SQUAT) is None


def test_early_arm_bend_none_without_pull_angles() -> None:
    assert early_arm_bend(RepMetrics(), CLEAN) is None


# --- bar_drift -------------------------------------------------------------


def test_bar_drift_fires_beyond_family_envelope() -> None:
    f = bar_drift(RepMetrics(bar_drift_cm=7.0), CLEAN)  # clean envelope 6
    assert f is not None and f.threshold == 6.0 and f.value == 7.0


def test_bar_drift_silent_within_envelope() -> None:
    assert bar_drift(RepMetrics(bar_drift_cm=5.0), CLEAN) is None


def test_bar_drift_envelope_is_family_specific() -> None:
    assert BAR_DRIFT_ENVELOPE_CM == {
        "clean": 6.0,
        "snatch": 6.0,
        "squat": 4.0,
        "hinge": 5.0,
        "press": 4.0,
    }
    # press envelope 4: 4.5 trips, squat envelope 4: 3.0 does not.
    assert bar_drift(RepMetrics(bar_drift_cm=4.5), PRESS) is not None
    assert bar_drift(RepMetrics(bar_drift_cm=3.0), SQUAT) is None
    # deadlift is the hinge family, envelope 5.
    assert bar_drift(RepMetrics(bar_drift_cm=5.5), DEADLIFT) is not None


# --- squat_depth -----------------------------------------------------------


def test_squat_depth_fires_when_hip_above_knee() -> None:
    f = squat_depth(RepMetrics(bottom_hip_y_cm=100.0, bottom_knee_y_cm=90.0), SQUAT)
    assert f is not None and f.phase == "bottom" and f.value == 10.0


def test_squat_depth_silent_when_hip_below_knee() -> None:
    assert squat_depth(RepMetrics(bottom_hip_y_cm=80.0, bottom_knee_y_cm=90.0), SQUAT) is None


def test_squat_depth_not_applicable_to_clean() -> None:
    assert squat_depth(RepMetrics(bottom_hip_y_cm=100.0, bottom_knee_y_cm=90.0), CLEAN) is None


def test_squat_depth_none_without_bottom_landmarks() -> None:
    assert squat_depth(RepMetrics(), SQUAT) is None


# --- early_press_out -------------------------------------------------------


def test_early_press_out_fires_when_elbow_leads_drive() -> None:
    f = early_press_out(RepMetrics(press_elbow_extends_before_drive=True), PRESS)
    assert f is not None and f.code == "early_press_out"


def test_early_press_out_silent_when_flag_false_or_none() -> None:
    assert early_press_out(RepMetrics(press_elbow_extends_before_drive=False), PRESS) is None
    assert early_press_out(RepMetrics(press_elbow_extends_before_drive=None), PRESS) is None


def test_early_press_out_not_applicable_to_clean() -> None:
    assert early_press_out(RepMetrics(press_elbow_extends_before_drive=True), CLEAN) is None


# --- catch_above_parallel --------------------------------------------------


def test_catch_above_parallel_fires_on_deep_hip_at_catch() -> None:
    f = catch_above_parallel(RepMetrics(hip_angle_at_phase={"catch": 80.0}), CLEAN)
    assert f is not None and f.value == 80.0 and f.threshold == 90.0


def test_catch_above_parallel_uses_receive_phase_for_snatch() -> None:
    deep = RepMetrics(hip_angle_at_phase={"receive": 85.0})
    assert catch_above_parallel(deep, SNATCH) is not None
    # A snatch reads "receive", not "catch".
    assert catch_above_parallel(RepMetrics(hip_angle_at_phase={"catch": 85.0}), SNATCH) is None


def test_catch_above_parallel_silent_on_upright_catch() -> None:
    assert catch_above_parallel(RepMetrics(hip_angle_at_phase={"catch": 120.0}), CLEAN) is None


# --- severity ----------------------------------------------------------------


def test_catch_above_parallel_finding_is_informational() -> None:
    # The brief calls this informational: the rep became a squat rep, which is
    # worth surfacing but is not a form error to penalize.
    f = catch_above_parallel(RepMetrics(hip_angle_at_phase={"catch": 80.0}), CLEAN)
    assert f is not None and f.severity == "informational"


def test_other_findings_default_to_fault_severity() -> None:
    drift = bar_drift(RepMetrics(bar_drift_cm=9.0), CLEAN)
    assert drift is not None and drift.severity == "fault"
    bend = early_arm_bend(RepMetrics(elbow_angle_at_phase={"second_pull": 150.0}), CLEAN)
    assert bend is not None and bend.severity == "fault"


def test_fault_finding_severity_defaults_to_fault() -> None:
    f = FaultFinding(code="x", message="m", phase=None, value=None, threshold=None)
    assert f.severity == "fault"


# --- no_lockout ------------------------------------------------------------


def test_no_lockout_fires_when_press_elbow_short() -> None:
    m = RepMetrics(
        hip_angle_at_phase={"lockout": 178.0},
        knee_angle_at_phase={"lockout": 178.0},
        elbow_angle_at_phase={"lockout": 150.0},
    )
    f = no_lockout(m, PRESS)
    assert f is not None and f.value == 150.0


def test_no_lockout_silent_when_all_press_joints_locked() -> None:
    m = RepMetrics(
        hip_angle_at_phase={"lockout": 178.0},
        knee_angle_at_phase={"lockout": 178.0},
        elbow_angle_at_phase={"lockout": 175.0},
    )
    assert no_lockout(m, PRESS) is None


def test_no_lockout_snatch_checks_only_overhead_elbow() -> None:
    # A power snatch legitimately receives in a partial squat: bent hips/knees
    # must NOT trip no_lockout as long as the elbows are locked overhead.
    m = RepMetrics(
        hip_angle_at_phase={"receive": 80.0},
        knee_angle_at_phase={"receive": 80.0},
        elbow_angle_at_phase={"receive": 176.0},
    )
    assert no_lockout(m, SNATCH) is None
    bent = RepMetrics(elbow_angle_at_phase={"receive": 150.0})
    assert no_lockout(bent, SNATCH) is not None


def test_no_lockout_deadlift_checks_standing_hip_knee() -> None:
    assert (
        no_lockout(
            RepMetrics(
                hip_angle_at_phase={"lockout": 160.0}, knee_angle_at_phase={"lockout": 178.0}
            ),
            DEADLIFT,
        )
        is not None
    )
    assert (
        no_lockout(
            RepMetrics(
                hip_angle_at_phase={"lockout": 178.0}, knee_angle_at_phase={"lockout": 178.0}
            ),
            DEADLIFT,
        )
        is None
    )


def test_no_lockout_not_applicable_to_squat() -> None:
    assert no_lockout(RepMetrics(hip_angle_at_phase={"lockout": 100.0}), SQUAT) is None


# --- evaluate_faults dispatch ----------------------------------------------


def test_evaluate_faults_runs_only_the_rules_the_rep_trips() -> None:
    # Real power_clean names three implemented rules; a rep tripping only the
    # arm bend yields exactly that finding.
    m = RepMetrics(elbow_angle_at_phase={"second_pull": 160.0}, bar_drift_cm=0.0)
    findings = evaluate_faults(m, CLEAN)
    assert [f.code for f in findings] == ["early_arm_bend"]


def test_back_squat_config_dispatches_squat_depth() -> None:
    # The reconciled back_squat config names squat_depth, so a hip-above-knee
    # bottom now produces a finding through the real registry config.
    m = RepMetrics(bottom_hip_y_cm=100.0, bottom_knee_y_cm=90.0)
    assert [f.code for f in evaluate_faults(m, SQUAT)] == ["squat_depth"]


def test_evaluate_faults_dispatches_all_named_rules_in_order() -> None:
    config = _config("clean", ("bar_drift", "early_arm_bend", "catch_above_parallel", "nope"))
    m = RepMetrics(
        bar_drift_cm=9.0,
        elbow_angle_at_phase={"second_pull": 150.0},
        hip_angle_at_phase={"catch": 70.0},
    )
    findings = evaluate_faults(m, config)
    assert [f.code for f in findings] == ["bar_drift", "early_arm_bend", "catch_above_parallel"]


def test_evaluate_faults_is_deterministic() -> None:
    config = _config("clean", ("bar_drift", "early_arm_bend"))
    m = RepMetrics(bar_drift_cm=9.0, elbow_angle_at_phase={"second_pull": 150.0})
    assert evaluate_faults(m, config) == evaluate_faults(m, config)


def test_rules_version_is_an_int() -> None:
    assert isinstance(RULES_VERSION, int) and RULES_VERSION >= 1

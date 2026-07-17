"""Fault rules: judge one rep's :class:`~powerpath_engine.metrics.RepMetrics`.

Each rule is a pure function ``(RepMetrics, MovementConfig) -> FaultFinding |
None``: it reads a few fields off the already-computed metrics, decides whether
this movement is even subject to the rule (via ``config.family``), and returns
a :class:`FaultFinding` when the rep trips it or ``None`` otherwise. Because the
rules never touch the raw series, they are exhaustively testable by handing them
a :class:`RepMetrics` built with controlled field values -- which is how the
angle-based rules are exercised, since the synthetic body's joint angles barely
move (see the metrics module docstring).

:func:`evaluate_faults` is the dispatcher: it walks ``config.fault_rules`` and
runs each named rule through an explicit name->function table (never ``eval``).

Registry note: every M1 movement config names only rules implemented in the
``_FAULT_RULES`` table below -- ``early_arm_bend``, ``bar_drift``,
``squat_depth``, ``early_press_out``, ``catch_above_parallel``, ``no_lockout``.
Faults that cannot be detected from the single-side-view camera premise (or
that need signals not extracted yet -- ``knees_cave``, ``rounded_back``, ...)
were dropped from the configs and are noted there as v2. Any config fault-rule
name without an implementation here is skipped by :func:`evaluate_faults`. Bump
:data:`RULES_VERSION` when a rule's threshold or logic changes so stored
analyses stay comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from powerpath_engine.metrics import PULL_PHASES, RepMetrics
from powerpath_engine.registry import MovementConfig

# Bump when any rule's threshold or decision logic changes (see module
# docstring); recorded alongside analyses so old results remain interpretable.
RULES_VERSION = 1

# An arm counts as bent (early pull) below this interior elbow angle.
EARLY_ARM_BEND_MIN_DEG = 170.0
# Max allowed horizontal bar drift, in cm, per movement family (deadlift is the
# "hinge" family). From the brief: clean/snatch 6, squat 4, deadlift 5, press 4.
BAR_DRIFT_ENVELOPE_CM: dict[str, float] = {
    "clean": 6.0,
    "snatch": 6.0,
    "squat": 4.0,
    "hinge": 5.0,
    "press": 4.0,
}
# A catch received with the hip angle below this is a full squat, not a power
# rep (informational -- the rep became a different movement).
CATCH_ABOVE_PARALLEL_HIP_DEG = 90.0
# A lockout joint below this interior angle is not locked out.
NO_LOCKOUT_MIN_DEG = 170.0
# The joints that constitute "the lockout" for each family. The brief lists
# hip/knee/elbow generically; we check each family's actual lockout joints so a
# legitimate partial-squat overhead receive (bent hips/knees) is not flagged:
# a snatch locks the elbows overhead, a deadlift the standing hips/knees, and a
# press both.
_LOCKOUT_JOINTS: dict[str, tuple[str, ...]] = {
    "press": ("hip", "knee", "elbow"),
    "snatch": ("elbow",),
    "hinge": ("hip", "knee"),
}


@dataclass(frozen=True)
class FaultFinding:
    """One detected fault on a rep.

    Attributes:
        code: Stable machine name of the rule that fired (e.g. ``"bar_drift"``).
        message: Human-facing description of the fault.
        phase: The phase the fault is anchored to, or ``None`` if rep-wide.
        value: The measured value that tripped the rule, or ``None`` if the
            rule is not a simple numeric comparison.
        threshold: The threshold ``value`` was compared against, or ``None``.
        severity: ``"fault"`` for a form error that costs score points;
            ``"informational"`` for a finding that is surfaced but never
            penalized (e.g. ``catch_above_parallel`` -- the rep became a
            different movement, not a form error).
    """

    code: str
    message: str
    phase: str | None
    value: float | None
    threshold: float | None
    severity: Literal["fault", "informational"] = "fault"


def early_arm_bend(metrics: RepMetrics, config: MovementConfig) -> FaultFinding | None:
    """Clean/snatch: the elbows bent during the pull (before triple extension)."""
    if config.family not in ("clean", "snatch"):
        return None
    present = [metrics.elbow_angle_at_phase.get(p) for p in PULL_PHASES]
    angles = [a for a in present if a is not None]
    if not angles:
        return None
    worst = min(angles)
    if worst < EARLY_ARM_BEND_MIN_DEG:
        return FaultFinding(
            code="early_arm_bend",
            message=(
                f"Elbows bent to {worst:.0f}deg during the pull "
                f"(should stay above {EARLY_ARM_BEND_MIN_DEG:.0f}deg until triple extension)."
            ),
            phase="second_pull",
            value=worst,
            threshold=EARLY_ARM_BEND_MIN_DEG,
        )
    return None


def bar_drift(metrics: RepMetrics, config: MovementConfig) -> FaultFinding | None:
    """The bar drifted horizontally past the movement's envelope."""
    envelope = BAR_DRIFT_ENVELOPE_CM.get(config.family)
    if envelope is None:
        return None
    if metrics.bar_drift_cm > envelope:
        return FaultFinding(
            code="bar_drift",
            message=(
                f"Bar drifted {metrics.bar_drift_cm:.1f}cm from vertical "
                f"(envelope {envelope:.0f}cm)."
            ),
            phase=None,
            value=metrics.bar_drift_cm,
            threshold=envelope,
        )
    return None


def squat_depth(metrics: RepMetrics, config: MovementConfig) -> FaultFinding | None:
    """Squat: the hip stayed above the knee at the bottom (above parallel)."""
    if config.family != "squat":
        return None
    hip_y = metrics.bottom_hip_y_cm
    knee_y = metrics.bottom_knee_y_cm
    if hip_y is None or knee_y is None:
        return None
    if hip_y > knee_y:
        return FaultFinding(
            code="squat_depth",
            message=(
                f"Hip crease {hip_y - knee_y:.1f}cm above the knee at the bottom "
                "(did not reach parallel)."
            ),
            phase="bottom",
            value=hip_y - knee_y,
            threshold=0.0,
        )
    return None


def early_press_out(metrics: RepMetrics, config: MovementConfig) -> FaultFinding | None:
    """Press: the elbows began extending before the leg drive finished."""
    if config.family != "press":
        return None
    if metrics.press_elbow_extends_before_drive is True:
        return FaultFinding(
            code="early_press_out",
            message="Elbows began pressing before the leg drive was complete.",
            phase="drive",
            value=None,
            threshold=None,
        )
    return None


def catch_above_parallel(metrics: RepMetrics, config: MovementConfig) -> FaultFinding | None:
    """Clean/snatch (informational): received in a full squat, not a power rep."""
    if config.family not in ("clean", "snatch"):
        return None
    phase = "catch" if config.family == "clean" else "receive"
    hip = metrics.hip_angle_at_phase.get(phase)
    if hip is None:
        return None
    if hip < CATCH_ABOVE_PARALLEL_HIP_DEG:
        return FaultFinding(
            code="catch_above_parallel",
            message=(
                f"Received with the hip at {hip:.0f}deg "
                f"(below {CATCH_ABOVE_PARALLEL_HIP_DEG:.0f}deg -- became a squat rep)."
            ),
            phase=phase,
            value=hip,
            threshold=CATCH_ABOVE_PARALLEL_HIP_DEG,
            severity="informational",
        )
    return None


def no_lockout(metrics: RepMetrics, config: MovementConfig) -> FaultFinding | None:
    """Press/snatch/deadlift: a lockout joint did not reach full extension."""
    joints = _LOCKOUT_JOINTS.get(config.family)
    if joints is None:
        return None
    phase = "receive" if config.family == "snatch" else "lockout"
    by_joint = {
        "hip": metrics.hip_angle_at_phase,
        "knee": metrics.knee_angle_at_phase,
        "elbow": metrics.elbow_angle_at_phase,
    }
    angles = [by_joint[j].get(phase) for j in joints]
    present = [a for a in angles if a is not None]
    if not present:
        return None
    worst = min(present)
    if worst < NO_LOCKOUT_MIN_DEG:
        return FaultFinding(
            code="no_lockout",
            message=(
                f"Lockout joint only reached {worst:.0f}deg at {phase} "
                f"(needs {NO_LOCKOUT_MIN_DEG:.0f}deg)."
            ),
            phase=phase,
            value=worst,
            threshold=NO_LOCKOUT_MIN_DEG,
        )
    return None


# Explicit rule-name -> function dispatch (no eval). Keys are the canonical
# Task 7 rule vocabulary (see the module docstring).
_FAULT_RULES = {
    "early_arm_bend": early_arm_bend,
    "bar_drift": bar_drift,
    "squat_depth": squat_depth,
    "early_press_out": early_press_out,
    "catch_above_parallel": catch_above_parallel,
    "no_lockout": no_lockout,
}


def evaluate_faults(metrics: RepMetrics, config: MovementConfig) -> list[FaultFinding]:
    """Run ``config.fault_rules`` against ``metrics`` and collect the findings.

    Each name in ``config.fault_rules`` is resolved through the explicit
    :data:`_FAULT_RULES` table; names without an implementation in Task 7 are
    skipped (see the module docstring). Findings are returned in the order the
    rules are listed on the config.
    """
    findings: list[FaultFinding] = []
    for name in config.fault_rules:
        rule = _FAULT_RULES.get(name)
        if rule is None:
            continue
        finding = rule(metrics, config)
        if finding is not None:
            findings.append(finding)
    return findings

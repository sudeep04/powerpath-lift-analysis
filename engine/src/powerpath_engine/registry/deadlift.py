"""Deadlift: pull the bar from the floor to a standing lockout, then lower.

Phases: setup / knee_pass / lockout (hips and knees fully extended at the
top) / descent back to the floor. Bar travel is ``up_down``; there is NO
catch phase (the bar is never received -- it is held and lowered).
"""

from __future__ import annotations

from .base import MadeCriteria, MovementConfig, PhaseDef

CONFIG = MovementConfig(
    key="deadlift",
    display_name="Deadlift",
    family="hinge",
    starts_from="floor",
    bar_travel="up_down",
    phases=(
        PhaseDef(name="setup", detector=""),
        PhaseDef(name="knee_pass", detector="knee_pass"),
        PhaseDef(name="lockout", detector="lockout_top"),
        PhaseDef(name="descent", detector=""),
    ),
    fault_rules=("rounded_back", "hips_shoot_up", "incomplete_lockout"),
    made_criteria=MadeCriteria(required_phases=("lockout",), min_lockout_angle_deg=165.0),
    comparison_landmarks=(
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_shoulder",
        "right_shoulder",
    ),
    min_disp_cm=20.0,
)

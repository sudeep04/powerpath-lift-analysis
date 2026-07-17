"""Power snatch: clean-family pull, but the bar is received OVERHEAD.

Same first_pull / knee_pass / second_pull as the clean, then the bar
continues overhead and is received with locked elbows above the head
(``receive_overhead``) rather than racked on the shoulders.
"""

from __future__ import annotations

from .base import MadeCriteria, MovementConfig, PhaseDef

CONFIG = MovementConfig(
    key="power_snatch",
    display_name="Power Snatch",
    family="snatch",
    starts_from="floor",
    bar_travel="up",
    phases=(
        PhaseDef(name="setup", detector=""),
        PhaseDef(name="first_pull", detector="bar_leaves_floor"),
        PhaseDef(name="knee_pass", detector="knee_pass"),
        PhaseDef(name="second_pull", detector="peak_hip_extension_velocity"),
        PhaseDef(name="receive", detector="receive_overhead"),
        PhaseDef(name="recovery", detector=""),
    ),
    # dropped: no_triple_extension (v2 / not single-side-view detectable)
    fault_rules=("early_arm_bend", "bar_drift", "no_lockout"),
    made_criteria=MadeCriteria(required_phases=("receive",), min_lockout_angle_deg=160.0),
    comparison_landmarks=(
        "nose",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
    ),
    min_disp_cm=40.0,
)

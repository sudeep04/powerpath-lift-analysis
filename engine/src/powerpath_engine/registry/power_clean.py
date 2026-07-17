"""Power clean: pull from the floor, receive in a partial-squat front rack.

Phases: setup / first_pull (bar leaves floor) / knee_pass / second_pull
(peak hip extension velocity) / catch (received in front rack) / recovery.
"""

from __future__ import annotations

from .base import MadeCriteria, MovementConfig, PhaseDef

CONFIG = MovementConfig(
    key="power_clean",
    display_name="Power Clean",
    family="clean",
    starts_from="floor",
    bar_travel="up",
    phases=(
        PhaseDef(name="setup", detector=""),
        PhaseDef(name="first_pull", detector="bar_leaves_floor"),
        PhaseDef(name="knee_pass", detector="knee_pass"),
        PhaseDef(name="second_pull", detector="peak_hip_extension_velocity"),
        PhaseDef(name="catch", detector="catch_rack"),
        PhaseDef(name="recovery", detector=""),
    ),
    # dropped: no_triple_extension (v2 / not single-side-view detectable)
    fault_rules=("early_arm_bend", "bar_drift", "catch_above_parallel"),
    made_criteria=MadeCriteria(required_phases=("catch",)),
    comparison_landmarks=(
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
    ),
    min_disp_cm=40.0,
)

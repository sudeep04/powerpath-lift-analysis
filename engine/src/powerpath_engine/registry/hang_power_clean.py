"""Hang power clean: a power clean started from the hang (above the knee).

Minimal config that proves the registry's extensibility -- it is the power
clean minus its first pull. Because the bar starts above the knee, the phase
list has NO ``first_pull`` / ``bar_leaves_floor`` phase: setup / knee_pass /
second_pull / catch / recovery.
"""

from __future__ import annotations

from .base import MadeCriteria, MovementConfig, PhaseDef

CONFIG = MovementConfig(
    key="hang_power_clean",
    display_name="Hang Power Clean",
    family="clean",
    starts_from="hang",
    bar_travel="up",
    phases=(
        PhaseDef(name="setup", detector=""),
        PhaseDef(name="knee_pass", detector="knee_pass"),
        PhaseDef(name="second_pull", detector="peak_hip_extension_velocity"),
        PhaseDef(name="catch", detector="catch_rack"),
        PhaseDef(name="recovery", detector=""),
    ),
    fault_rules=("early_arm_bend", "bar_loops_away", "no_triple_extension"),
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

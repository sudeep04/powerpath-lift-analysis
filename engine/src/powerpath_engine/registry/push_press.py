"""Push press: dip, drive, and press the bar to an overhead lockout.

Phases: dip (dip turnaround, the bottom of the leg dip) / drive / press_out /
lockout (bar locked overhead) / standing. Bar travel is ``up`` from the
shoulders to overhead.
"""

from __future__ import annotations

from .base import MadeCriteria, MovementConfig, PhaseDef

CONFIG = MovementConfig(
    key="push_press",
    display_name="Push Press",
    family="press",
    starts_from="shoulders",
    bar_travel="up",
    phases=(
        PhaseDef(name="dip", detector="dip_turnaround"),
        PhaseDef(name="drive", detector=""),
        PhaseDef(name="press_out", detector=""),
        PhaseDef(name="lockout", detector="lockout_top"),
        PhaseDef(name="standing", detector=""),
    ),
    fault_rules=("shallow_dip", "press_forward", "incomplete_lockout"),
    made_criteria=MadeCriteria(required_phases=("lockout",), min_lockout_angle_deg=160.0),
    comparison_landmarks=(
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
    ),
    min_disp_cm=40.0,
)

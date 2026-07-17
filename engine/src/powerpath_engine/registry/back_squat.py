"""Back squat: bar racked on the back, descend to depth, drive back up.

Phases: descent / bottom (deepest point) / ascent / standing. Bar travel is
``down_up`` -- the only keyframe detector is the ``bottom`` turnaround.
"""

from __future__ import annotations

from .base import MadeCriteria, MovementConfig, PhaseDef

CONFIG = MovementConfig(
    key="back_squat",
    display_name="Back Squat",
    family="squat",
    starts_from="rack",
    bar_travel="down_up",
    phases=(
        PhaseDef(name="descent", detector=""),
        PhaseDef(name="bottom", detector="bottom"),
        PhaseDef(name="ascent", detector=""),
        PhaseDef(name="standing", detector=""),
    ),
    # dropped: knees_cave, heels_rise (frontal/foot plane, not single-side-view
    # detectable); excessive_forward_lean (v2)
    fault_rules=("squat_depth", "bar_drift"),
    made_criteria=MadeCriteria(
        required_phases=("bottom", "standing"),
        max_bottom_knee_angle_deg=90.0,
    ),
    comparison_landmarks=(
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ),
    min_disp_cm=20.0,
)

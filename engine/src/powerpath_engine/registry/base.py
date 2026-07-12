"""Data model for the movement registry.

A :class:`MovementConfig` is a *declarative* description of one barbell
movement: the ordered phases the segmenter walks through, the named
keyframe detectors that mark phase boundaries, the fault rules that apply,
and the criteria that make a rep count as "made". It is pure data -- no
behaviour lives here. The algorithms that consume it are split across later
tasks:

* the ``detector`` string on each :class:`PhaseDef` names a keyframe-finding
  strategy that ``phases.py`` (Task 6b) resolves and runs; this module never
  imports or implements those strategies. The known detector vocabulary is
  documented on :data:`KNOWN_DETECTORS` for reference only.
* :class:`MadeCriteria` is read by the made-rep evaluator (Task 7).
* ``fault_rules`` name fault-checking strategies resolved by the fault engine
  (a later task), exactly like detectors -- plain strings here.
* ``min_disp_cm`` is read by the segmenter (Task 6b) as the minimum bar
  vertical displacement that distinguishes a real rep from walkout jitter.

A phase whose boundary is simply the end of the previous phase (``setup``,
``recovery``, ``ascent`` ...) carries an empty ``detector`` string: it has no
distinct keyframe event of its own, its start is the previous phase's end and
its end is the next phase's detector. Only non-empty detectors name a
strategy, and only those are required to be in :data:`KNOWN_DETECTORS`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from powerpath_engine.series import LANDMARK_NAMES

# The fixed vocabulary of keyframe detector strategies. These are implemented
# in phases.py (Task 6b); this module only records the names so configs and
# tests share a single spelling. A phase with no distinct keyframe event uses
# the empty string instead of one of these.
KNOWN_DETECTORS: frozenset[str] = frozenset(
    {
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
)

# Allowed values for the small closed-vocabulary MovementConfig fields.
FAMILIES: frozenset[str] = frozenset({"squat", "clean", "press", "snatch", "hinge"})
STARTS_FROM: frozenset[str] = frozenset({"floor", "hang", "rack", "shoulders"})
BAR_TRAVELS: frozenset[str] = frozenset({"up", "down_up", "up_down"})


@dataclass(frozen=True)
class PhaseDef:
    """One phase of a movement.

    Attributes:
        name: Human/stable identifier for the phase (e.g. ``"second_pull"``).
        detector: Name of the keyframe strategy that marks this phase's
            boundary, resolved later in ``phases.py``. Must be a member of
            :data:`KNOWN_DETECTORS` when non-empty. The empty string means the
            phase has no keyframe event of its own -- its start is the previous
            phase's end (used for ``setup``, ``recovery``, ``ascent``, ...).
        params: Free-form tuning parameters passed to the detector strategy
            (thresholds, windows). Opaque to this module.
    """

    name: str
    detector: str
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MadeCriteria:
    """Declarative criteria for whether a rep counts as "made".

    Pure data -- the evaluator that reads it is Task 7. Optional angle gates
    default to ``None`` meaning "not gated for this movement".

    Attributes:
        required_phases: Names of phases that must all be detected in a rep for
            it to be a candidate made rep (e.g. a snatch must reach
            ``"receive"``). Ordering is not implied; presence is.
        min_lockout_angle_deg: If set, the finishing lockout joint (elbow for
            overhead receives/presses, hip/knee for a standing finish) must
            reach at least this interior angle in degrees. ``None`` = no
            lockout gate.
        max_bottom_knee_angle_deg: If set, the knee interior angle at the
            bottom of the movement must be at most this many degrees (a smaller
            angle is a deeper position) -- a depth gate for squats. ``None`` =
            no depth gate.
    """

    required_phases: tuple[str, ...]
    min_lockout_angle_deg: float | None = None
    max_bottom_knee_angle_deg: float | None = None


@dataclass(frozen=True)
class MovementConfig:
    """The full declarative configuration for one barbell movement.

    Attributes:
        key: Stable registry key (e.g. ``"power_clean"``).
        display_name: Human-facing name.
        family: Movement family, one of :data:`FAMILIES`.
        starts_from: Where the bar starts, one of :data:`STARTS_FROM`.
        bar_travel: Net bar travel signature, one of :data:`BAR_TRAVELS`
            (``"up"`` = floor/shoulders to a higher finish, ``"down_up"`` =
            down then back up as in a squat, ``"up_down"`` = up then returned
            as in a deadlift).
        phases: Ordered phases the segmenter walks through.
        fault_rules: Names of fault-checking strategies (resolved later).
        made_criteria: Criteria for a made rep.
        comparison_landmarks: Landmark names (subset of
            :data:`~powerpath_engine.series.LANDMARK_NAMES`) that rep-to-rep
            comparison and analysis focus on for this movement.
        min_disp_cm: Minimum bar vertical displacement (cm) the segmenter uses
            to tell a real rep from walkout jitter. 20 for squat/hinge, 40 for
            the explosive clean/snatch/press family.
    """

    key: str
    display_name: str
    family: Literal["squat", "clean", "press", "snatch", "hinge"]
    starts_from: Literal["floor", "hang", "rack", "shoulders"]
    bar_travel: Literal["up", "down_up", "up_down"]
    phases: tuple[PhaseDef, ...]
    fault_rules: tuple[str, ...]
    made_criteria: MadeCriteria
    comparison_landmarks: tuple[str, ...]
    min_disp_cm: float

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"{self.key!r}: family {self.family!r} not in {sorted(FAMILIES)}")
        if self.starts_from not in STARTS_FROM:
            raise ValueError(
                f"{self.key!r}: starts_from {self.starts_from!r} not in {sorted(STARTS_FROM)}"
            )
        if self.bar_travel not in BAR_TRAVELS:
            raise ValueError(
                f"{self.key!r}: bar_travel {self.bar_travel!r} not in {sorted(BAR_TRAVELS)}"
            )
        if not self.phases:
            raise ValueError(f"{self.key!r}: phases must be non-empty")
        unknown = [name for name in self.comparison_landmarks if name not in LANDMARK_NAMES]
        if unknown:
            raise ValueError(
                f"{self.key!r}: comparison_landmarks {unknown} not in series.LANDMARK_NAMES"
            )
        if not (self.min_disp_cm > 0.0):
            raise ValueError(
                f"{self.key!r}: min_disp_cm must be positive, got {self.min_disp_cm!r}"
            )

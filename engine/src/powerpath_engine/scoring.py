"""Made/missed judgement and the 0-100 rep quality score.

Two steps sit on top of :mod:`~powerpath_engine.metrics` and
:mod:`~powerpath_engine.faults`:

1. **Made vs missed** (:func:`evaluate_made`): a rep counts as *made* when it
   satisfies its movement's :class:`~powerpath_engine.registry.MadeCriteria`
   -- every required phase is present (a required phase that is *eventless*,
   e.g. a squat's ``standing``, is satisfied by the bar returning to its start
   band), the finishing lockout angle clears ``min_lockout_angle_deg``, and the
   squat bottom clears ``max_bottom_knee_angle_deg`` -- and the rep did not end
   in a terminal free-fall (a dumped bar). The pure decision core
   :func:`is_made` takes the two trajectory-derived booleans (returned-to-start,
   terminal-free-fall) as arguments so it is exhaustively testable without a bar
   series; :func:`evaluate_made` derives those from the bar and calls it.

2. **Quality score** (:func:`score_rep`), for made reps only. 100 points:

       smoothness 30 + path efficiency 30 + velocity 20 + faults 20

   * ``smoothness`` scales with :func:`_smoothness_fraction` of the normalized
     jerk (1.0 at zero jerk, 0.5 at :data:`NJ_HALF_CREDIT`).
   * ``path`` scales with :func:`_path_fraction` of the path-length ratio (1.0
     at a dead-vertical ratio of 1.0, 0.0 at :data:`PATH_RATIO_ZERO`).
   * ``velocity`` compares this rep's peak concentric velocity to the athlete's
     own history at +/-10% load (via the :class:`VelocityHistory` protocol);
     full credit once the rep matches or beats the median of that history.
   * ``faults`` starts at 20 and loses :data:`FAULT_PENALTY` (7) per finding,
     floored at 0.

   **Velocity redistribution:** with fewer than :data:`MIN_HISTORY_REPS` (5)
   history reps at +/-10% load there is no trustworthy velocity baseline, so the
   velocity 20 is split evenly onto smoothness and path -- each worth 40 instead
   of 30 -- and ``velocity_component`` is reported as ``None``. Concretely:

       history >= 5 reps:  score = 30*sm + 30*path + 20*vel + faults
       history <  5 reps:  score = 40*sm + 40*path +   0    + faults

   Both paths total 100 at full marks; the final score is clamped to [0, 100].
   Missed reps get ``score=None`` and ``excluded_from_templates=True``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from powerpath_engine.metrics import RepMetrics
from powerpath_engine.registry import MovementConfig
from powerpath_engine.segmentation import V_FREEFALL_CMS, RepWindow
from powerpath_engine.series import TimeSeries

# --- score weights ---------------------------------------------------------
SMOOTHNESS_MAX = 30.0
PATH_MAX = 30.0
VELOCITY_MAX = 20.0
FAULTS_MAX = 20.0
FAULT_PENALTY = 7.0
# The velocity weight split onto smoothness and path when history is too thin.
VELOCITY_REDISTRIBUTION_HALF = VELOCITY_MAX / 2.0

# --- component tuning ------------------------------------------------------
# Normalized jerk at which smoothness earns half credit (see module docstring).
# A tunable placeholder pending calibration against labeled lifts.
NJ_HALF_CREDIT = 1000.0
# Path-length ratio at (and above) which path efficiency earns zero credit.
PATH_RATIO_ZERO = 1.5
# Minimum same-load history reps before the velocity component is trusted.
MIN_HISTORY_REPS = 5
# The +/- load window (fraction) that counts as "same load" for velocity.
VELOCITY_LOAD_TOLERANCE = 0.10

# --- made-rep trajectory checks -------------------------------------------
# The bar must finish within this fraction of the movement's expected
# displacement of its start height to count as "returned to the start band".
RETURN_BAND_FRACTION = 0.5
# Window (s) after the rep in which a sustained bar free-fall means a dumped rep.
FREEFALL_LOOKAHEAD_S = 0.4
# Consecutive free-fall samples needed to trust a terminal drop.
FREEFALL_MIN_SAMPLES = 3


class VelocityHistory(Protocol):
    """The athlete's peak-velocity history, supplied by the API layer.

    Tests provide a fake. The scorer asks only for the peak concentric
    velocities (m/s) of prior reps of the same movement within ``tolerance_frac``
    of the current rep's load.
    """

    def peak_velocities_near_load(self, load_kg: float, tolerance_frac: float) -> list[float]:
        """Peak concentric velocities (m/s) at loads within ``+/-tolerance_frac``."""
        ...


@dataclass(frozen=True)
class RepScore:
    """The quality score (and its breakdown) for one rep.

    Missed reps carry ``score=None``, ``excluded_from_templates=True`` and
    ``None`` components. For made reps ``velocity_component`` is ``None`` exactly
    when the velocity weight was redistributed (``velocity_redistributed``).
    """

    made: bool
    score: float | None
    excluded_from_templates: bool
    smoothness_component: float | None
    path_component: float | None
    velocity_component: float | None
    fault_component: float | None
    velocity_redistributed: bool


# --- made / missed ---------------------------------------------------------


def _lockout_angle(metrics: RepMetrics, config: MovementConfig) -> float | None:
    """The finishing lockout angle to gate on, by family (min of its joints)."""
    if config.family == "snatch":
        return metrics.elbow_angle_at_phase.get("receive")
    if config.family == "press":
        candidates = [
            metrics.hip_angle_at_phase.get("lockout"),
            metrics.knee_angle_at_phase.get("lockout"),
            metrics.elbow_angle_at_phase.get("lockout"),
        ]
    elif config.family == "hinge":
        candidates = [
            metrics.hip_angle_at_phase.get("lockout"),
            metrics.knee_angle_at_phase.get("lockout"),
        ]
    else:
        candidates = [metrics.elbow_angle_at_phase.get("lockout")]
    present = [c for c in candidates if c is not None]
    return min(present) if present else None


def is_made(
    phases: dict[str, float | None],
    metrics: RepMetrics,
    config: MovementConfig,
    *,
    returned_to_start: bool,
    terminal_freefall: bool,
) -> bool:
    """Pure made-rep decision from phases, metrics and two trajectory booleans.

    See the module docstring for the criteria. ``returned_to_start`` and
    ``terminal_freefall`` are derived from the bar series by
    :func:`evaluate_made`; kept as explicit arguments here so the decision is
    testable without a trajectory.
    """
    if terminal_freefall:
        return False

    crit = config.made_criteria
    for name in crit.required_phases:
        if name in phases:
            if phases[name] is None:
                return False
        # An eventless required phase (no keyframe of its own, e.g. a squat's
        # ``standing``) is satisfied by the bar returning to the start band.
        elif not returned_to_start:
            return False

    if crit.min_lockout_angle_deg is not None:
        angle = _lockout_angle(metrics, config)
        if angle is None or angle < crit.min_lockout_angle_deg:
            return False

    if crit.max_bottom_knee_angle_deg is not None:
        knee = metrics.knee_angle_at_phase.get("bottom")
        if knee is None or knee > crit.max_bottom_knee_angle_deg:
            return False

    return True


def _returned_to_start(bar: TimeSeries, rep: RepWindow, config: MovementConfig) -> bool:
    """Whether the bar ends the rep back within a band of its start height."""
    t = bar.ts()
    y = bar.ys()
    if len(t) == 0:
        return False
    y_start = float(np.interp(rep.t_start, t, y))
    y_end = float(np.interp(rep.t_end, t, y))
    band = RETURN_BAND_FRACTION * config.min_disp_cm
    return abs(y_end - y_start) <= band


def _terminal_freefall(bar: TimeSeries, rep: RepWindow) -> bool:
    """Whether the bar enters a sustained free-fall just after the rep window."""
    t = bar.ts()
    y = bar.ys()
    if len(t) < 3:
        return False
    v = np.gradient(y, t)
    mask = (t >= rep.t_end) & (t <= rep.t_end + FREEFALL_LOOKAHEAD_S)
    return int(np.sum(v[mask] < V_FREEFALL_CMS)) >= FREEFALL_MIN_SAMPLES


def evaluate_made(
    rep: RepWindow,
    bar: TimeSeries,
    phases: dict[str, float | None],
    metrics: RepMetrics,
    config: MovementConfig,
) -> bool:
    """Made-rep judgement: derive the trajectory booleans and call :func:`is_made`."""
    return is_made(
        phases,
        metrics,
        config,
        returned_to_start=_returned_to_start(bar, rep, config),
        terminal_freefall=_terminal_freefall(bar, rep),
    )


# --- quality score ---------------------------------------------------------


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _smoothness_fraction(normalized_jerk: float) -> float:
    """1.0 at zero jerk, 0.5 at ``NJ_HALF_CREDIT``, decaying toward 0."""
    if normalized_jerk <= 0.0:
        return 1.0
    return 1.0 / (1.0 + normalized_jerk / NJ_HALF_CREDIT)


def _path_fraction(path_length_ratio: float) -> float:
    """1.0 at a dead-vertical ratio of 1.0, 0.0 at/after ``PATH_RATIO_ZERO``."""
    return _clamp((PATH_RATIO_ZERO - path_length_ratio) / (PATH_RATIO_ZERO - 1.0), 0.0, 1.0)


def _velocity_fraction(peak_velocity_ms: float, history: list[float]) -> float:
    """Peak velocity over the median of the same-load history, clamped to [0, 1]."""
    reference = float(np.median(history))
    if reference <= 0.0:
        return 1.0
    return _clamp(peak_velocity_ms / reference, 0.0, 1.0)


def _missed_score() -> RepScore:
    return RepScore(
        made=False,
        score=None,
        excluded_from_templates=True,
        smoothness_component=None,
        path_component=None,
        velocity_component=None,
        fault_component=None,
        velocity_redistributed=False,
    )


def score_rep(
    metrics: RepMetrics,
    faults: list,
    made: bool,
    history: VelocityHistory,
    load_kg: float,
) -> RepScore:
    """Score one rep 0-100 (made reps only); see the module docstring for the formula.

    ``faults`` is the finding list from
    :func:`~powerpath_engine.faults.evaluate_faults`; only its length matters.
    A missed rep (``made`` is False) is returned unscored and excluded from
    templates.
    """
    if not made:
        return _missed_score()

    sm_frac = _smoothness_fraction(metrics.smoothness_normalized_jerk)
    path_frac = _path_fraction(metrics.path_length_ratio)
    fault_component = max(0.0, FAULTS_MAX - FAULT_PENALTY * len(faults))

    velocities = history.peak_velocities_near_load(load_kg, VELOCITY_LOAD_TOLERANCE)
    if len(velocities) >= MIN_HISTORY_REPS:
        smoothness_component = SMOOTHNESS_MAX * sm_frac
        path_component = PATH_MAX * path_frac
        velocity_component: float | None = VELOCITY_MAX * _velocity_fraction(
            metrics.peak_concentric_velocity_ms, velocities
        )
        redistributed = False
    else:
        smoothness_component = (SMOOTHNESS_MAX + VELOCITY_REDISTRIBUTION_HALF) * sm_frac
        path_component = (PATH_MAX + VELOCITY_REDISTRIBUTION_HALF) * path_frac
        velocity_component = None
        redistributed = True

    total = smoothness_component + path_component + (velocity_component or 0.0) + fault_component
    return RepScore(
        made=True,
        score=_clamp(total, 0.0, 100.0),
        excluded_from_templates=False,
        smoothness_component=smoothness_component,
        path_component=path_component,
        velocity_component=velocity_component,
        fault_component=fault_component,
        velocity_redistributed=redistributed,
    )

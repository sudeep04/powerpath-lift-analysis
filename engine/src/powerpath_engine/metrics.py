"""Per-rep measurements: turn one rep's windows + phases into numbers.

Segmentation carves the bar trajectory into one
:class:`~powerpath_engine.segmentation.RepWindow` per rep and ``phases.py``
locates each rep's keyframe events. This module reduces one rep to a
:class:`RepMetrics` bundle: a handful of scalar bar-trajectory measurements
(drift, peak velocity, path efficiency, smoothness) plus the hip/knee/elbow
joint angles at every detected phase landmark. Those numbers are the sole
input to the fault rules (``faults.py``) and the quality score
(``scoring.py``), which is why every measurement lives on one frozen,
directly-constructible dataclass -- tests build a :class:`RepMetrics` with
controlled field values and feed it straight to a rule or the scorer.

Coordinate convention (matching ``phases.py`` and ``tests/synthetic.py``):
the bar ``TimeSeries`` and the ``LandmarkSeries`` handed to
:func:`compute_rep_metrics` are both **y-up** (larger ``y`` is physically
higher) and share a common time grid. The bar series is already calibrated to
centimeters; the ``scale`` argument is the bar-plane :class:`PlaneScale` and is
applied only through :meth:`geometry.PlaneScale.px_to_cm` and
:func:`geometry.horizontal_deviation_cm` so that this module never touches a
raw scale factor directly (the geometry-owns-conversions constraint). When the
bar series is already in cm -- as it is throughout the
calibrated pipeline and in the fixtures -- the caller passes an identity
``PlaneScale(cm_per_px=1.0)``; a px-valued caller would pass the real bar-plane
scale instead. Angles are read via :func:`geometry.joint_angle`.

A note on synthetic fixtures: the generated body translates vertically by a
crouch factor rather than hinging, so its interior joint angles barely change
across a rep (see ``phases.py``). The *bar-trajectory* metrics (drift, peak
velocity, path ratio, smoothness) are well-defined on the fixtures because the
bar positions/velocities are realistic; the *angle* fields are not
meaningfully exercised by the generators and are instead validated by
constructing :class:`RepMetrics` directly in the fault/score tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from powerpath_engine.geometry import PlaneScale, horizontal_deviation_cm, joint_angle
from powerpath_engine.registry import MovementConfig
from powerpath_engine.segmentation import RepWindow
from powerpath_engine.series import LandmarkSeries, TimeSeries

# Pull-phase keyframes over which an early arm bend is judged (clean/snatch):
# the elbow should stay straight from the floor through the second pull.
PULL_PHASES = ("first_pull", "knee_pass", "second_pull")

# The catch/receive keyframe name differs by family (front-rack catch vs
# overhead receive); catch_height_ratio and the catch angle read the first of
# these that a rep actually detected.
CATCH_PHASES = ("catch", "receive")

# press-family early-press-out heuristic tuning. The elbow has "begun
# extending" once it opens this many degrees past its angle at the dip; the leg
# drive is "complete" once the knee reaches within this margin of its window
# maximum. Both are degenerate on the near-collinear synthetic body (see the
# module docstring); the rule that consumes the resulting flag is tested with a
# constructed RepMetrics.
_ELBOW_ONSET_MARGIN_DEG = 5.0
_DRIVE_COMPLETE_TOL_DEG = 3.0

# Below this many samples a rep slice is too short for a stable third
# derivative, so normalized jerk is reported as 0.0 (treated as smooth).
_MIN_JERK_SAMPLES = 5


@dataclass(frozen=True)
class RepMetrics:
    """Per-rep measurements consumed by the fault rules and the scorer.

    Every field has a default so tests can construct a metrics object with
    only the values a particular rule or score component reads. Angle values
    are interior joint angles in degrees (``None`` where the landmarks needed
    were absent at that instant); ``*_at_phase`` maps are keyed by phase name.

    Attributes:
        rep_index: The rep's ordinal (from its :class:`RepWindow`).
        bar_drift_cm: Maximum absolute horizontal deviation of the bar from
            its setup vertical over the rep, in centimeters.
        peak_concentric_velocity_ms: Peak upward (concentric) bar velocity in
            meters per second (0.0 if the bar never moves up).
        path_length_ratio: Total bar path length divided by its net vertical
            excursion; 1.0 is a perfectly vertical path, larger is wanderier.
        smoothness_normalized_jerk: Dimensionless normalized jerk of the bar
            trajectory; smaller is smoother.
        catch_height_ratio: Bar height at the catch/receive divided by the
            hip-to-ankle height proxy, or ``None`` if there is no catch.
        hip_angle_at_phase: shoulder-hip-knee angle at each detected phase.
        knee_angle_at_phase: hip-knee-ankle angle at each detected phase.
        elbow_angle_at_phase: shoulder-elbow-wrist angle at each detected phase.
        bottom_hip_y_cm: Hip landmark height (y-up) at the squat bottom.
        bottom_knee_y_cm: Knee landmark height (y-up) at the squat bottom.
        press_elbow_extends_before_drive: For the press family, whether the
            elbows began extending before the leg drive completed (``None``
            when not a press or the dip/lockout were not both found).
    """

    rep_index: int = 0
    bar_drift_cm: float = 0.0
    peak_concentric_velocity_ms: float = 0.0
    path_length_ratio: float = 1.0
    smoothness_normalized_jerk: float = 0.0
    catch_height_ratio: float | None = None
    hip_angle_at_phase: dict[str, float | None] = field(default_factory=dict)
    knee_angle_at_phase: dict[str, float | None] = field(default_factory=dict)
    elbow_angle_at_phase: dict[str, float | None] = field(default_factory=dict)
    bottom_hip_y_cm: float | None = None
    bottom_knee_y_cm: float | None = None
    press_elbow_extends_before_drive: bool | None = None


# --- landmark helpers ------------------------------------------------------


def _named_arrays(
    landmarks: LandmarkSeries, name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(t, x, y)`` arrays for a single full landmark name (frames present)."""
    ts: list[float] = []
    xs: list[float] = []
    ys: list[float] = []
    for frame in landmarks.frames:
        if name in frame.points:
            s = frame.points[name]
            ts.append(frame.t)
            xs.append(s.x)
            ys.append(s.y)
    return np.array(ts), np.array(xs), np.array(ys)


def _interp_point(landmarks: LandmarkSeries, name: str, t: float) -> tuple[float, float] | None:
    """Interpolated ``(x, y)`` of a landmark at time ``t``, or ``None``."""
    ts, xs, ys = _named_arrays(landmarks, name)
    if len(ts) == 0:
        return None
    return float(np.interp(t, ts, xs)), float(np.interp(t, ts, ys))


def _interp_y(landmarks: LandmarkSeries, joint: str, t: float) -> float | None:
    """Interpolated left/right-averaged ``y`` of a sided joint at ``t``."""
    left = _interp_point(landmarks, f"left_{joint}", t)
    right = _interp_point(landmarks, f"right_{joint}", t)
    ys = [p[1] for p in (left, right) if p is not None]
    return float(np.mean(ys)) if ys else None


def _angle_at(landmarks: LandmarkSeries, a: str, b: str, c: str, t: float) -> float | None:
    """Left/right-averaged interior angle at vertex ``b`` at time ``t``."""
    vals: list[float] = []
    for side in ("left", "right"):
        pa = _interp_point(landmarks, f"{side}_{a}", t)
        pb = _interp_point(landmarks, f"{side}_{b}", t)
        pc = _interp_point(landmarks, f"{side}_{c}", t)
        if pa is None or pb is None or pc is None:
            continue
        if pa == pb or pc == pb:
            continue
        vals.append(joint_angle(pa, pb, pc))
    return float(np.mean(vals)) if vals else None


def _angle_series(
    landmarks: LandmarkSeries, a: str, b: str, c: str
) -> tuple[np.ndarray, np.ndarray]:
    """``(t, angle)`` of the left/right-averaged interior angle at ``b``."""
    ts: list[float] = []
    angs: list[float] = []
    for frame in landmarks.frames:
        p = frame.points
        vals: list[float] = []
        for side in ("left", "right"):
            an, bn, cn = f"{side}_{a}", f"{side}_{b}", f"{side}_{c}"
            if an in p and bn in p and cn in p:
                pa, pb, pc = (p[an].x, p[an].y), (p[bn].x, p[bn].y), (p[cn].x, p[cn].y)
                if pa != pb and pc != pb:
                    vals.append(joint_angle(pa, pb, pc))
        if vals:
            ts.append(frame.t)
            angs.append(float(np.mean(vals)))
    return np.array(ts), np.array(angs)


# --- bar-trajectory metrics ------------------------------------------------


def _bar_drift_cm(x_smoothed: np.ndarray, scale: PlaneScale) -> float:
    """Max absolute horizontal deviation from the setup vertical, in cm."""
    deviations = horizontal_deviation_cm(list(x_smoothed), scale)
    return max((abs(d) for d in deviations), default=0.0)


def _peak_concentric_velocity_ms(t: np.ndarray, y_cm: np.ndarray) -> float:
    """Peak upward bar velocity over the rep, in meters per second."""
    if len(t) < 2:
        return 0.0
    vy = np.gradient(y_cm, t)  # cm/s, y-up so upward is positive
    return max(0.0, float(np.max(vy))) / 100.0


def _path_length_ratio(x_cm: np.ndarray, y_cm: np.ndarray) -> float:
    """Bar path length over its net vertical travel (1.0 = dead vertical).

    "Net vertical" is the total vertical distance travelled (``sum |dy|``), not
    the top-to-bottom excursion, so the ratio is 1.0 for any perfectly vertical
    path -- whether it only rises, or rises and returns -- and climbs above 1.0
    exactly to the extent the bar also wanders horizontally.
    """
    if len(x_cm) < 2:
        return 1.0
    dx = np.diff(x_cm)
    dy = np.diff(y_cm)
    path = float(np.sum(np.hypot(dx, dy)))
    vertical = float(np.sum(np.abs(dy)))
    if vertical <= 1e-9:
        return 1.0
    return path / vertical


def _smoothness_normalized_jerk(t: np.ndarray, x_cm: np.ndarray, y_cm: np.ndarray) -> float:
    """Dimensionless normalized jerk of the 2D bar trajectory (smaller = smoother).

    ``NJ = sqrt( (duration^5 / (2 * L^2)) * integral(||d3r/dt3||^2 dt) )`` where
    ``L`` is the total path length and ``r(t)`` the bar position; the factors of
    duration and length make it scale- and duration-invariant.
    """
    n = len(t)
    if n < _MIN_JERK_SAMPLES:
        return 0.0
    duration = float(t[-1] - t[0])
    steps = np.hypot(np.diff(x_cm), np.diff(y_cm))
    length = float(np.sum(steps))
    if duration <= 0.0 or length <= 1e-9:
        return 0.0
    jx = np.gradient(np.gradient(np.gradient(x_cm, t), t), t)
    jy = np.gradient(np.gradient(np.gradient(y_cm, t), t), t)
    jerk_sq = jx**2 + jy**2
    integral = float(np.trapezoid(jerk_sq, t))
    return float(np.sqrt(0.5 * integral * duration**5 / length**2))


def _catch_height_ratio(
    landmarks: LandmarkSeries, t: np.ndarray, y_cm: np.ndarray, catch_t: float | None
) -> float | None:
    """Bar height at the catch over the hip-to-ankle height proxy."""
    if catch_t is None or len(t) == 0:
        return None
    bar_y = float(np.interp(catch_t, t, y_cm))
    hip_y = _interp_y(landmarks, "hip", catch_t)
    ankle_y = _interp_y(landmarks, "ankle", catch_t)
    if hip_y is None or ankle_y is None:
        return None
    proxy = hip_y - ankle_y
    if proxy <= 1e-9:
        return None
    return bar_y / proxy


def _first_catch_t(phases: dict[str, float | None]) -> float | None:
    """First detected catch/receive time across the family's naming."""
    for name in CATCH_PHASES:
        if phases.get(name) is not None:
            return phases[name]
    return None


def _press_elbow_extends_before_drive(
    landmarks: LandmarkSeries, phases: dict[str, float | None], config: MovementConfig
) -> bool | None:
    """Whether the elbows began extending before the leg drive completed.

    Best-effort press-family heuristic; ``None`` for non-press movements or
    when the dip and lockout were not both detected. Degenerate on the
    synthetic body (the elbow angle barely moves), so the consuming rule is
    tested with a constructed :class:`RepMetrics`.
    """
    if config.family != "press":
        return None
    dip_t = phases.get("dip")
    lock_t = phases.get("lockout")
    if dip_t is None or lock_t is None or not (dip_t < lock_t):
        return None
    et, ea = _angle_series(landmarks, "shoulder", "elbow", "wrist")
    kt, ka = _angle_series(landmarks, "hip", "knee", "ankle")
    if len(et) == 0 or len(kt) == 0:
        return None

    elbow_at_dip = float(np.interp(dip_t, et, ea))
    window = (kt >= dip_t) & (kt <= lock_t)
    if not np.any(window):
        return None
    knee_max = float(np.max(ka[window]))

    elbow_onset: float | None = None
    for tt, aa in zip(et, ea, strict=True):
        if dip_t < tt <= lock_t and aa > elbow_at_dip + _ELBOW_ONSET_MARGIN_DEG:
            elbow_onset = float(tt)
            break
    if elbow_onset is None:
        return False

    drive_complete: float | None = None
    for tt, aa in zip(kt, ka, strict=True):
        if dip_t < tt <= lock_t and aa >= knee_max - _DRIVE_COMPLETE_TOL_DEG:
            drive_complete = float(tt)
            break
    return drive_complete is None or elbow_onset < drive_complete


# --- top-level assembly ----------------------------------------------------


def _slice_landmarks(landmarks: LandmarkSeries, t0: float, t1: float) -> LandmarkSeries:
    """Frames with ``t0 <= t < t1`` (mirrors ``TimeSeries.slice_time``)."""
    return LandmarkSeries([f for f in landmarks.frames if t0 <= f.t < t1])


def compute_rep_metrics(
    rep: RepWindow,
    bar: TimeSeries,
    landmarks: LandmarkSeries,
    phases: dict[str, float | None],
    config: MovementConfig,
    scale: PlaneScale,
) -> RepMetrics:
    """Reduce one rep to a :class:`RepMetrics` bundle.

    ``bar`` and ``landmarks`` are y-up on a shared grid (see the module
    docstring); ``phases`` is the ``phases.detect_phases`` result for this rep
    (phase name -> event PTS or ``None``); ``scale`` is the bar-plane
    :class:`PlaneScale` (pass ``PlaneScale(1.0)`` when the bar is already in
    cm). Bar metrics are computed on the rep's smoothed slice; joint angles are
    interpolated to each detected phase instant.
    """
    bar_slice = bar.slice_time(rep.t_start, rep.t_end)
    lm_slice = _slice_landmarks(landmarks, rep.t_start, rep.t_end)

    smoothed = bar_slice.smooth()
    t = smoothed.ts()
    x = smoothed.xs()
    y = smoothed.ys()
    x_cm = scale.px_to_cm(x)
    y_cm = scale.px_to_cm(y)

    bar_drift = _bar_drift_cm(x, scale)
    peak_v = _peak_concentric_velocity_ms(t, y_cm)
    path_ratio = _path_length_ratio(x_cm, y_cm)
    smoothness = _smoothness_normalized_jerk(t, x_cm, y_cm)

    catch_t = _first_catch_t(phases)
    catch_ratio = _catch_height_ratio(lm_slice, t, y_cm, catch_t)

    hip_angles: dict[str, float | None] = {}
    knee_angles: dict[str, float | None] = {}
    elbow_angles: dict[str, float | None] = {}
    for name, event_t in phases.items():
        if event_t is None:
            continue
        hip_angles[name] = _angle_at(lm_slice, "shoulder", "hip", "knee", event_t)
        knee_angles[name] = _angle_at(lm_slice, "hip", "knee", "ankle", event_t)
        elbow_angles[name] = _angle_at(lm_slice, "shoulder", "elbow", "wrist", event_t)

    bottom_t = phases.get("bottom")
    bottom_hip_y = _interp_y(lm_slice, "hip", bottom_t) if bottom_t is not None else None
    bottom_knee_y = _interp_y(lm_slice, "knee", bottom_t) if bottom_t is not None else None

    return RepMetrics(
        rep_index=rep.rep_index,
        bar_drift_cm=bar_drift,
        peak_concentric_velocity_ms=peak_v,
        path_length_ratio=path_ratio,
        smoothness_normalized_jerk=smoothness,
        catch_height_ratio=catch_ratio,
        hip_angle_at_phase=hip_angles,
        knee_angle_at_phase=knee_angles,
        elbow_angle_at_phase=elbow_angles,
        bottom_hip_y_cm=bottom_hip_y,
        bottom_knee_y_cm=bottom_knee_y,
        press_elbow_extends_before_drive=_press_elbow_extends_before_drive(
            lm_slice, phases, config
        ),
    )

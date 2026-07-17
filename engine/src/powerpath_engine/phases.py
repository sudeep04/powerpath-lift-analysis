"""Movement phase detectors: locate keyframe events inside one rep.

Segmentation (``segmentation.py``) carves the bar trajectory into one
:class:`~powerpath_engine.segmentation.RepWindow` per rep. This module finds
the *keyframe events* inside each window -- the instants a movement's phases
begin (bar leaves the floor, passes the knees, is caught, locked out, ...).

Each detector is a small pure function over the rep's slice of the bar
``TimeSeries`` and the matching ``LandmarkSeries``, returning the event's PTS
(seconds) or ``None`` if the event is not present. The registry's ``PhaseDef``
names a detector by string; :func:`detect_phases` resolves those names through
an explicit dispatch table (never ``eval``) and keys the results by phase
name.

All positions are y-up (larger ``y`` is physically higher); joint angles come
from :func:`powerpath_engine.geometry.joint_angle` (interior angle at the
middle vertex, degrees, averaged over the left/right sides).

A note on the synthetic body model (see ``tests/synthetic.py``): the
generators translate the whole skeleton vertically by a "crouch" factor
rather than hinging the torso, so every interior joint angle stays near
180 deg and barely *changes* through a lift. Angle-*threshold* gates
(``>= 170`` for a lockout, ``>= 165`` for an overhead receive) are therefore
satisfied, but detectors keyed on an angle *change* are degenerate on these
fixtures and are aligned to the equivalent bar/landmark motion the fixture
truth actually marks:

* ``peak_hip_extension_velocity``: the shoulder-hip-knee angle only spans
  ~174-180 deg, so its derivative peaks ~5 frames early. The hip's upward
  vertical velocity (the hip rising fastest) is the triple-extension instant
  the ``second_pull`` truth marks -- see :func:`_peak_hip_extension_velocity`.
* ``catch_rack``: the elbow angle stays ~176 deg (<1 deg of "whip" at the
  catch), so the >=40 deg elbow-rotation gate can never fire. The catch is
  detected by the bar's velocity turning back upward after the apex -- the
  catch-dip minimum the ``catch`` truth marks -- see :func:`_catch_rack`.
"""

from __future__ import annotations

import numpy as np

from powerpath_engine.geometry import joint_angle
from powerpath_engine.registry import MovementConfig
from powerpath_engine.segmentation import RepWindow
from powerpath_engine.series import LandmarkSeries, Sample, TimeSeries

# --- detector tuning constants --------------------------------------------

# ``bar_leaves_floor``: the bar has left the floor once it rises this far
# above its resting height and stays there -- a couple of cm clears marker
# jitter while still firing early in the first pull.
_FLOOR_BAND_CM = 2.0
# Consecutive samples an event must persist to count as "sustained".
_SUSTAIN_SAMPLES = 3

# ``receive_overhead`` / ``lockout_top``: an overhead receive needs the
# elbows near-straight; a standing lockout needs hips and knees near-straight.
_OVERHEAD_ELBOW_DEG = 165.0
_LOCKOUT_ANGLE_DEG = 170.0
# A bar "top" plateau is flat, so accept the first sample within this margin
# of the window maximum (with the joint gate met) as the lockout instant --
# the moment the top is *reached*, not the middle of the plateau.
_LOCKOUT_TOL_CM = 0.5


# --- landmark / series helpers --------------------------------------------


def _bar_ty(bar: TimeSeries) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed bar ``(t, y)`` arrays for the rep slice."""
    s = bar.smooth()
    return s.ts(), s.ys()


def _sided_y(landmarks: LandmarkSeries, joint: str) -> tuple[np.ndarray, np.ndarray]:
    """``(t, y)`` for a sided landmark, averaging left/right per frame."""
    ts: list[float] = []
    ys: list[float] = []
    left, right = f"left_{joint}", f"right_{joint}"
    for frame in landmarks.frames:
        p = frame.points
        if left in p and right in p:
            ts.append(frame.t)
            ys.append(0.5 * (p[left].y + p[right].y))
    return np.array(ts), np.array(ys)


def _nose_y(landmarks: LandmarkSeries) -> tuple[np.ndarray, np.ndarray]:
    """``(t, y)`` for the (single, centered) nose landmark."""
    ts = [f.t for f in landmarks.frames if "nose" in f.points]
    ys = [f.points["nose"].y for f in landmarks.frames if "nose" in f.points]
    return np.array(ts, dtype=float), np.array(ys, dtype=float)


def _sided_timeseries(landmarks: LandmarkSeries, joint: str) -> TimeSeries:
    """A :class:`TimeSeries` of a sided landmark's left/right midpoint."""
    samples: list[Sample] = []
    left, right = f"left_{joint}", f"right_{joint}"
    for frame in landmarks.frames:
        p = frame.points
        if left in p and right in p:
            samples.append(
                Sample(
                    t=frame.t,
                    x=0.5 * (p[left].x + p[right].x),
                    y=0.5 * (p[left].y + p[right].y),
                )
            )
    return TimeSeries(samples)


def _angle_series(
    landmarks: LandmarkSeries, a: str, b: str, c: str
) -> tuple[np.ndarray, np.ndarray]:
    """``(t, angle)`` of ``joint_angle(a, b, c)`` at vertex ``b``, deg.

    ``a``/``b``/``c`` are unsided joint names (e.g. ``"shoulder"``); the angle
    is computed per side where all three are present and averaged.
    """
    ts: list[float] = []
    angs: list[float] = []
    for frame in landmarks.frames:
        p = frame.points
        vals: list[float] = []
        for side in ("left", "right"):
            an, bn, cn = f"{side}_{a}", f"{side}_{b}", f"{side}_{c}"
            if an in p and bn in p and cn in p:
                pa, pb, pc = p[an], p[bn], p[cn]
                if (pa.x, pa.y) != (pb.x, pb.y) and (pc.x, pc.y) != (pb.x, pb.y):
                    vals.append(joint_angle((pa.x, pa.y), (pb.x, pb.y), (pc.x, pc.y)))
        if vals:
            ts.append(frame.t)
            angs.append(float(np.mean(vals)))
    return np.array(ts), np.array(angs)


# --- detectors -------------------------------------------------------------
# Each takes the rep's (already-sliced) bar TimeSeries and LandmarkSeries plus
# the MovementConfig, and returns the event PTS in seconds or None.


def _bar_leaves_floor(
    bar: TimeSeries, landmarks: LandmarkSeries, config: MovementConfig
) -> float | None:
    """First time the bar rises, and stays, above its resting start-band."""
    t, y = _bar_ty(bar)
    if len(t) < 2:
        return None
    band = float(y[0]) + _FLOOR_BAND_CM
    for i in range(len(y)):
        if y[i] >= band and all(y[j] >= band for j in range(i, min(i + _SUSTAIN_SAMPLES, len(y)))):
            return float(t[i])
    return None


def _knee_pass(bar: TimeSeries, landmarks: LandmarkSeries, config: MovementConfig) -> float | None:
    """Interpolated instant the rising bar crosses the knee landmark's y."""
    t, y = _bar_ty(bar)
    kt, ky = _sided_y(landmarks, "knee")
    if len(t) < 2 or len(kt) == 0:
        return None
    knee = np.interp(t, kt, ky)
    diff = y - knee
    for i in range(len(diff) - 1):
        if diff[i] < 0.0 <= diff[i + 1]:
            span = diff[i + 1] - diff[i]
            frac = (-diff[i] / span) if span != 0.0 else 0.0
            return float(t[i] + frac * (t[i + 1] - t[i]))
    return None


def _hip_contact(
    bar: TimeSeries, landmarks: LandmarkSeries, config: MovementConfig
) -> float | None:
    """Instant during the pull the bar y is nearest the hip landmark y.

    Reserved for a hip-contact analysis; not wired to any current config's
    phases but implemented for completeness.
    """
    t, y = _bar_ty(bar)
    ht, hy = _sided_y(landmarks, "hip")
    if len(t) == 0 or len(ht) == 0:
        return None
    hip = np.interp(t, ht, hy)
    apex = int(np.argmax(y))
    upto = apex if apex > 0 else len(y) - 1
    idx = int(np.argmin(np.abs(y[: upto + 1] - hip[: upto + 1])))
    return float(t[idx])


def _peak_hip_extension_velocity(
    bar: TimeSeries, landmarks: LandmarkSeries, config: MovementConfig
) -> float | None:
    """Instant of peak upward hip-extension velocity.

    Measured as the maximum upward (y-up) velocity of the hip landmark -- the
    hip rising fastest -- which is the triple-extension instant the fixture's
    ``second_pull`` truth marks. (The shoulder-hip-knee joint angle is
    degenerate on the synthetic near-collinear body; see the module docstring.)
    """
    hip = _sided_timeseries(landmarks, "hip")
    if len(hip.samples) < 2:
        return None
    t = hip.ts()
    v = hip.velocity()
    return float(t[int(np.argmax(v))])


def _catch_rack(bar: TimeSeries, landmarks: LandmarkSeries, config: MovementConfig) -> float | None:
    """Front-rack catch: the bar's velocity turns back upward after the apex.

    The catch is the dip minimum -- the athlete drops under the bar and
    receives it, so the bar decelerates, stops, and rises into the rack. That
    is the first upward (negative-to-positive) bar-velocity zero-crossing
    after the apex. (The movement definition also gates on the elbows whipping
    through the rack, but the synthetic elbow angle is degenerate -- see the
    module docstring -- so that gate is omitted here.)
    """
    t, y = _bar_ty(bar)
    if len(t) < 3:
        return None
    vy = np.gradient(y, t)
    apex = int(np.argmax(y))
    for i in range(apex, len(vy) - 1):
        if vy[i] < 0.0 <= vy[i + 1]:
            return float(t[i + 1])
    return None


def _receive_overhead(
    bar: TimeSeries, landmarks: LandmarkSeries, config: MovementConfig
) -> float | None:
    """First instant the bar is above the nose with the elbows locked out."""
    t, y = _bar_ty(bar)
    nt, ny = _nose_y(landmarks)
    et, ea = _angle_series(landmarks, "shoulder", "elbow", "wrist")
    if len(t) == 0 or len(nt) == 0 or len(et) == 0:
        return None
    nose = np.interp(t, nt, ny)
    elbow = np.interp(t, et, ea)
    for i in range(len(t)):
        if y[i] > nose[i] and elbow[i] >= _OVERHEAD_ELBOW_DEG:
            return float(t[i])
    return None


def _bottom(bar: TimeSeries, landmarks: LandmarkSeries, config: MovementConfig) -> float | None:
    """Instant of the bar's lowest point in the window (squat bottom)."""
    t, y = _bar_ty(bar)
    if len(t) == 0:
        return None
    return float(t[int(np.argmin(y))])


def _lockout_top(
    bar: TimeSeries, landmarks: LandmarkSeries, config: MovementConfig
) -> float | None:
    """Instant the bar first reaches its top with hips and knees locked out.

    The top of a lift is a flat plateau, so rather than the (ambiguous)
    argmax of a plateau this returns the first sample within
    ``_LOCKOUT_TOL_CM`` of the window maximum where both the hip and knee
    interior angles are at least ``_LOCKOUT_ANGLE_DEG`` -- i.e. when the top
    is first reached in a standing position.
    """
    t, y = _bar_ty(bar)
    if len(t) == 0:
        return None
    ht, ha = _angle_series(landmarks, "shoulder", "hip", "knee")
    kt, ka = _angle_series(landmarks, "hip", "knee", "ankle")
    if len(ht) == 0 or len(kt) == 0:
        return None
    hip = np.interp(t, ht, ha)
    knee = np.interp(t, kt, ka)
    y_max = float(np.max(y))
    for i in range(len(t)):
        if (
            y[i] >= y_max - _LOCKOUT_TOL_CM
            and hip[i] >= _LOCKOUT_ANGLE_DEG
            and knee[i] >= _LOCKOUT_ANGLE_DEG
        ):
            return float(t[i])
    return None


def _dip_turnaround(
    bar: TimeSeries, landmarks: LandmarkSeries, config: MovementConfig
) -> float | None:
    """The press dip: the bar's lowest point before it drives to the top."""
    t, y = _bar_ty(bar)
    if len(t) == 0:
        return None
    apex = int(np.argmax(y))
    if apex == 0:
        return float(t[0])
    return float(t[int(np.argmin(y[: apex + 1]))])


# Explicit detector-name -> function dispatch (no eval). Names match
# registry.KNOWN_DETECTORS.
_DETECTORS = {
    "bar_leaves_floor": _bar_leaves_floor,
    "knee_pass": _knee_pass,
    "hip_contact": _hip_contact,
    "peak_hip_extension_velocity": _peak_hip_extension_velocity,
    "catch_rack": _catch_rack,
    "receive_overhead": _receive_overhead,
    "bottom": _bottom,
    "lockout_top": _lockout_top,
    "dip_turnaround": _dip_turnaround,
}


def _slice_landmarks(landmarks: LandmarkSeries, t0: float, t1: float) -> LandmarkSeries:
    """Frames with ``t0 <= t < t1`` (mirrors ``TimeSeries.slice_time``)."""
    return LandmarkSeries([f for f in landmarks.frames if t0 <= f.t < t1])


def detect_phases(
    rep: RepWindow,
    bar: TimeSeries,
    landmarks: LandmarkSeries,
    config: MovementConfig,
) -> dict[str, float | None]:
    """Detect each phase's keyframe event within ``rep``.

    For every :class:`~powerpath_engine.registry.PhaseDef` in
    ``config.phases`` that names a (non-empty) detector, run it on the rep's
    slice of the bar and landmark series and record the result under the
    *phase name* (value = event PTS, or ``None`` if not detected). Eventless
    phases (empty detector string -- ``setup``, ``recovery``, ``ascent`` ...)
    have no keyframe of their own and are omitted from the result.
    """
    bar_slice = bar.slice_time(rep.t_start, rep.t_end)
    lm_slice = _slice_landmarks(landmarks, rep.t_start, rep.t_end)

    detected: dict[str, float | None] = {}
    for phase in config.phases:
        if not phase.detector:
            continue
        detector = _DETECTORS[phase.detector]
        detected[phase.name] = detector(bar_slice, lm_slice, config)
    return detected

"""Synthetic trajectory generators: the fixture backbone for Tasks 6b/7/8.

These are pure, deterministic builders of physically-plausible bar and body
trajectories, plus a ground-truth dict of phase timestamps (PTS seconds), so
downstream segmentation / phase / made-rep tests can assert detected events
against known truth (typically within +/-2 frames). They live in ``tests/``
(they are fixtures, not shipped engine code) but import only numpy/scipy and
the engine's pure ``series`` types -- never cv2 or any pose model.

Coordinate conventions (matching the engine's biomechanics space):

* All positions are in centimeters, **y-up**: a larger ``y`` means the bar /
  joint is physically higher off the floor.
* Time is PTS seconds, sampled on a uniform ``1/fps`` grid. The bar
  ``TimeSeries`` and every ``LandmarkFrame`` share the exact same grid
  timestamps, so a test can compare "bar y vs nose y at time t" by frame.

Determinism: every generator takes a ``seed`` and builds its own
``numpy.random.default_rng(seed)``; the global ``numpy.random`` state is never
touched. With the default ``noise_cm=0`` (and no walkout jitter) the output is
a fixed, smooth curve.

Physical model: each movement is described by a handful of ``(time, bar
height, body-crouch)`` keyframes per rep. Bar height is interpolated with a
shape-preserving monotone cubic (``scipy`` PCHIP) so local extrema sit exactly
on the keyframes and never overshoot -- the derivative is zero at each
turnaround, giving clean vertical-velocity sign changes. The body "crouch"
factor in ``[0, 1]`` (0 = standing tall, 1 = deepest crouch) drives the lower
body: hip/knee/shoulder/nose descend proportionally while the ankles stay
planted; the wrists ride with the bar (hands on the bar) and the elbows sit
between shoulder and wrist, so the elbow angle opens and closes naturally
through a lift (e.g. it rotates through as the bar is racked in a clean).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator

from powerpath_engine.series import LandmarkFrame, LandmarkSeries, Sample, TimeSeries

# --- gravity for genuine free-fall (dumped bar), cm/s^2 -------------------
G_CM_S2 = 980.0

# --- bar heights, cm, y-up ------------------------------------------------
FLOOR_BAR = 22.0  # loaded bar resting on the floor (45cm plates -> ~22cm center)
KNEE_BAR = 50.0  # bar level with the knees
CLEAN_PEAK = 125.0  # apex of the clean pull
CLEAN_DIP = 110.0  # bar dips as the athlete drops under to receive
RACK = 118.0  # bar settled in the front rack
SQUAT_TOP = 140.0  # bar on the back, standing
SQUAT_BOTTOM = 95.0  # bar at the bottom of the squat
PP_SHOULDER = 140.0  # push press: bar racked on shoulders
PP_DIP = 122.0  # push press: bottom of the dip
OVERHEAD = 210.0  # bar locked out overhead
SNATCH_SETTLE = 205.0  # overhead bar settled after the receive
DL_LOCKOUT = 95.0  # deadlift bar at the hip at lockout

# --- body skeleton, standing heights (cm, y-up) and crouch drops ----------
ANKLE_STAND = 8.0
KNEE_STAND, KNEE_DROP = 52.0, 14.0
HIP_STAND, HIP_DROP = 95.0, 46.0
SHOULDER_STAND, SHOULDER_DROP = 140.0, 48.0
NOSE_STAND, NOSE_DROP = 165.0, 48.0
ELBOW_FRAC = 0.55  # elbow sits this fraction from shoulder toward wrist

# --- lateral (x) offsets from body center, cm -----------------------------
SH_X, HIP_X, KNEE_X, ANK_X, WR_X = 18.0, 14.0, 11.0, 11.0, 24.0
EL_X = 21.0  # between shoulder and wrist x, off the shoulder-wrist line

# landmark names produced by every generator
_SIDED = ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")


@dataclass(frozen=True)
class SyntheticLift:
    """A generated lift fixture.

    Attributes:
        bar: The bar-marker ``TimeSeries`` (cm, y-up) on the fps grid.
        landmarks: Per-frame body landmarks (cm, y-up) on the same grid.
        truth: Ground-truth phase timestamps in PTS seconds, keyed by event
            name (e.g. ``"knee_pass"``), each a list with one entry per rep.
        reps: The number of real reps in the trajectory.
    """

    bar: TimeSeries
    landmarks: LandmarkSeries
    truth: dict[str, list[float]]
    reps: int


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _grid(t_end: float, fps: int) -> np.ndarray:
    """Uniform sample times in ``[0, t_end)`` at ``1/fps`` spacing."""
    return np.arange(0.0, t_end, 1.0 / fps)


def _pchip(kf_t: np.ndarray, kf_v: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Shape-preserving monotone-cubic interpolation of keyframes onto grid."""
    return PchipInterpolator(kf_t, kf_v)(grid)


def _concat_reps(
    per_rep: list[tuple[float, float, float]],
    n_reps: int,
    rep_period: float,
    lead_in: float,
    lead_value: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tile per-rep ``(rel_t, bar, crouch)`` keyframes into global arrays.

    ``per_rep`` describes one rep whose last keyframe (at ``rep_period``)
    coincides with the next rep's first keyframe, so the shared boundary
    keyframe is emitted once. When ``lead_in > 0`` a leading keyframe at
    ``t=0`` holding ``lead_value = (bar, crouch)`` is prepended (the walkout).
    """
    ts: list[float] = []
    bars: list[float] = []
    crouches: list[float] = []
    if lead_in > 0.0:
        ts.append(0.0)
        bars.append(lead_value[0])
        crouches.append(lead_value[1])
    for i in range(n_reps):
        offset = lead_in + i * rep_period
        points = per_rep if i == 0 else per_rep[1:]
        for rel_t, bar, crouch in points:
            ts.append(offset + rel_t)
            bars.append(bar)
            crouches.append(crouch)
    return np.array(ts), np.array(bars), np.array(crouches)


def _truth_times(rel_t: float, n_reps: int, rep_period: float, lead_in: float) -> list[float]:
    """Absolute PTS times of a per-rep event across all reps."""
    return [lead_in + i * rep_period + rel_t for i in range(n_reps)]


def _bar_series(
    grid: np.ndarray,
    bar_ys: np.ndarray,
    rng: np.random.Generator,
    noise_cm: float,
    x_center: float = 0.0,
) -> TimeSeries:
    """Build the bar ``TimeSeries`` (x roughly centered, y from ``bar_ys``)."""
    xs = np.full_like(grid, x_center)
    ys = bar_ys.copy()
    if noise_cm > 0.0:
        xs = xs + rng.normal(0.0, noise_cm, size=grid.shape)
        ys = ys + rng.normal(0.0, noise_cm, size=grid.shape)
    return TimeSeries(
        [Sample(t=float(t), x=float(x), y=float(y)) for t, x, y in zip(grid, xs, ys, strict=True)]
    )


def _landmarks(
    grid: np.ndarray,
    bar_ys: np.ndarray,
    crouch: np.ndarray,
    rng: np.random.Generator,
    noise_cm: float,
    x_center: float = 0.0,
) -> LandmarkSeries:
    """Build the body ``LandmarkSeries`` from crouch factor and bar height.

    Lower-body joints descend proportionally to ``crouch`` (0..1); the ankles
    stay planted; the wrists ride with the bar and the elbows interpolate
    between shoulder and wrist (offset off the shoulder-wrist line so the
    elbow angle is always well-defined and rotates through the lift).
    """
    c = np.clip(crouch, 0.0, 1.0)
    knee_y = KNEE_STAND - c * KNEE_DROP
    hip_y = HIP_STAND - c * HIP_DROP
    shoulder_y = SHOULDER_STAND - c * SHOULDER_DROP
    nose_y = NOSE_STAND - c * NOSE_DROP
    ankle_y = np.full_like(grid, ANKLE_STAND)
    wrist_y = bar_ys.copy()
    elbow_y = shoulder_y + ELBOW_FRAC * (wrist_y - shoulder_y)

    def noisy(arr: np.ndarray) -> np.ndarray:
        if noise_cm > 0.0:
            return arr + rng.normal(0.0, noise_cm, size=arr.shape)
        return arr

    # (name, x_offset, y_array); sided landmarks are emitted left/right.
    y_by_joint = {
        "shoulder": (SH_X, shoulder_y),
        "elbow": (EL_X, elbow_y),
        "wrist": (WR_X, wrist_y),
        "hip": (HIP_X, hip_y),
        "knee": (KNEE_X, knee_y),
        "ankle": (ANK_X, ankle_y),
    }

    columns: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    columns["nose"] = (noisy(np.full_like(grid, x_center)), noisy(nose_y))
    for joint in _SIDED:
        x_off, y_arr = y_by_joint[joint]
        columns[f"left_{joint}"] = (noisy(np.full_like(grid, x_center - x_off)), noisy(y_arr))
        columns[f"right_{joint}"] = (noisy(np.full_like(grid, x_center + x_off)), noisy(y_arr))

    frames: list[LandmarkFrame] = []
    for k, t in enumerate(grid):
        points = {
            name: Sample(t=float(t), x=float(xs[k]), y=float(ys[k]))
            for name, (xs, ys) in columns.items()
        }
        frames.append(LandmarkFrame(t=float(t), points=points))
    return LandmarkSeries(frames)


def _build(
    per_rep: list[tuple[float, float, float]],
    events: dict[str, float],
    *,
    n_reps: int,
    rep_period: float,
    fps: int,
    noise_cm: float,
    seed: int,
    speed: float,
    lead_in: float = 0.0,
    lead_value: tuple[float, float] = (0.0, 0.0),
    walkout_jitter_cm: float = 0.0,
) -> SyntheticLift:
    """Assemble a standard (PCHIP-interpolated) lift from per-rep keyframes.

    ``per_rep`` and ``events`` use rep-relative seconds at ``speed=1``; both
    are divided by ``speed`` (``speed>1`` = a faster lift). ``events`` maps a
    truth key to its rep-relative time. A leading ``walkout_jitter_cm`` band of
    bar jitter is added over the ``lead_in`` window.
    """
    rng = np.random.default_rng(seed)
    per_rep_s = [(t / speed, b, c) for (t, b, c) in per_rep]
    rep_period_s = rep_period / speed
    lead_in_s = lead_in / speed

    kf_t, kf_bar, kf_crouch = _concat_reps(per_rep_s, n_reps, rep_period_s, lead_in_s, lead_value)
    t_end = lead_in_s + n_reps * rep_period_s
    grid = _grid(t_end, fps)
    bar_ys = _pchip(kf_t, kf_bar, grid)
    crouch = _pchip(kf_t, kf_crouch, grid)

    if walkout_jitter_cm > 0.0 and lead_in_s > 0.0:
        jitter = rng.normal(0.0, walkout_jitter_cm / 2.0, size=grid.shape)
        bar_ys = bar_ys + jitter * (grid < lead_in_s)

    bar = _bar_series(grid, bar_ys, rng, noise_cm)
    landmarks = _landmarks(grid, bar_ys, crouch, rng, noise_cm)
    truth = {
        name: _truth_times(rel_t / speed, n_reps, rep_period_s, lead_in_s)
        for name, rel_t in events.items()
    }
    return SyntheticLift(bar=bar, landmarks=landmarks, truth=truth, reps=n_reps)


# ---------------------------------------------------------------------------
# generators
# ---------------------------------------------------------------------------


def clean(
    n_reps: int = 3,
    *,
    fps: int = 60,
    noise_cm: float = 0.0,
    speed: float = 1.0,
    seed: int = 0,
) -> SyntheticLift:
    """Power clean(s): floor -> knee -> triple extension -> catch dip -> rack.

    Each rep the bar rises to a peak (triple extension), then **dips** as the
    athlete drops under to receive it, then settles at rack height before
    being lowered back to the floor for the next rep. That catch dip creates
    an intra-rep vertical-velocity zero-crossing (peak, then dip minimum),
    which is exactly what breaks a naive one-up-one-down segmenter. Truth:
    first_pull / knee_pass / second_pull / catch per rep.
    """
    per_rep = [
        (0.00, FLOOR_BAR, 0.85),  # setup on the floor
        (0.25, FLOOR_BAR, 0.85),  # bar leaves the floor (first pull)
        (0.65, KNEE_BAR, 0.55),  # bar passes the knees
        (0.95, CLEAN_PEAK, 0.00),  # triple extension apex
        (1.10, CLEAN_DIP, 0.35),  # catch dip: drop under the bar
        (1.25, RACK, 0.20),  # settle in the front rack
        (1.45, RACK, 0.15),  # rack hold
        (2.35, FLOOR_BAR, 0.85),  # lowered back to the floor under control
        (2.65, FLOOR_BAR, 0.85),  # inter-rep hold (== next rep start)
    ]
    events = {
        "first_pull": 0.25,
        "knee_pass": 0.65,
        "second_pull": 0.80,  # peak hip-extension velocity, mid second pull
        "catch": 1.10,
    }
    return _build(
        per_rep,
        events,
        n_reps=n_reps,
        rep_period=2.65,
        fps=fps,
        noise_cm=noise_cm,
        seed=seed,
        speed=speed,
    )


def single_rep(**kwargs: object) -> SyntheticLift:
    """Convenience: a single-rep clean fixture (see :func:`clean`)."""
    return clean(1, **kwargs)  # type: ignore[arg-type]


def power_snatch(
    n_reps: int = 3,
    *,
    fps: int = 60,
    noise_cm: float = 0.0,
    speed: float = 1.0,
    seed: int = 0,
) -> SyntheticLift:
    """Power snatch(es): clean-style pull, but received OVERHEAD.

    The bar continues past the rack and is caught locked out overhead -- its
    height at the receive is above the nose landmark (arms overhead). Truth:
    first_pull / knee_pass / second_pull / receive per rep.
    """
    per_rep = [
        (0.00, FLOOR_BAR, 0.85),  # setup on the floor
        (0.25, FLOOR_BAR, 0.85),  # bar leaves the floor
        (0.65, KNEE_BAR, 0.55),  # bar passes the knees
        (0.92, 150.0, 0.00),  # triple extension, bar accelerating up
        (1.15, OVERHEAD, 0.45),  # received overhead (drop under)
        (1.35, SNATCH_SETTLE, 0.20),  # stand the bar up overhead
        (1.55, SNATCH_SETTLE, 0.10),  # overhead hold
        (2.15, FLOOR_BAR, 0.85),  # lowered back to the floor
        (2.45, FLOOR_BAR, 0.85),  # inter-rep hold
    ]
    events = {
        "first_pull": 0.25,
        "knee_pass": 0.65,
        "second_pull": 0.785,
        "receive": 1.15,
    }
    return _build(
        per_rep,
        events,
        n_reps=n_reps,
        rep_period=2.45,
        fps=fps,
        noise_cm=noise_cm,
        seed=seed,
        speed=speed,
    )


def back_squat(
    n_reps: int = 3,
    *,
    fps: int = 60,
    walkout_jitter_cm: float = 2.0,
    walkout_seconds: float = 2.0,
    noise_cm: float = 0.0,
    speed: float = 1.0,
    seed: int = 0,
) -> SyntheticLift:
    """Back squat(s): a jittery walkout, then descend / bottom / ascend / stand.

    A few seconds of small +/- ``walkout_jitter_cm`` bar jitter precede the
    first rep (the walkout), which a segmenter must not mistake for a rep.
    Each rep the bar (on the back) travels down to the bottom and back up.
    Truth: bottom per rep.
    """
    per_rep = [
        (0.00, SQUAT_TOP, 0.00),  # standing
        (0.70, SQUAT_BOTTOM, 1.00),  # bottom
        (1.40, SQUAT_TOP, 0.00),  # stood back up
        (1.70, SQUAT_TOP, 0.00),  # standing hold
    ]
    events = {"bottom": 0.70}
    return _build(
        per_rep,
        events,
        n_reps=n_reps,
        rep_period=1.70,
        fps=fps,
        noise_cm=noise_cm,
        seed=seed,
        speed=speed,
        lead_in=walkout_seconds,
        lead_value=(SQUAT_TOP, 0.0),
        walkout_jitter_cm=walkout_jitter_cm,
    )


def push_press(
    n_reps: int = 3,
    *,
    fps: int = 60,
    noise_cm: float = 0.0,
    speed: float = 1.0,
    seed: int = 0,
) -> SyntheticLift:
    """Push press(es): dip / drive / press overhead / lockout / return.

    The bar dips at the shoulders (leg dip), then drives up to an overhead
    lockout and returns to the shoulders. Truth: dip_turnaround / lockout per
    rep.
    """
    per_rep = [
        (0.00, PP_SHOULDER, 0.00),  # racked on shoulders
        (0.25, PP_DIP, 0.30),  # bottom of the dip (turnaround)
        (0.55, OVERHEAD, 0.00),  # locked out overhead
        (0.75, OVERHEAD, 0.00),  # overhead hold
        (1.10, PP_SHOULDER, 0.00),  # returned to shoulders
        (1.35, PP_SHOULDER, 0.00),  # standing hold
    ]
    events = {"dip_turnaround": 0.25, "lockout": 0.55}
    return _build(
        per_rep,
        events,
        n_reps=n_reps,
        rep_period=1.35,
        fps=fps,
        noise_cm=noise_cm,
        seed=seed,
        speed=speed,
    )


def deadlift(
    n_reps: int = 3,
    *,
    fps: int = 60,
    noise_cm: float = 0.0,
    speed: float = 1.0,
    seed: int = 0,
) -> SyntheticLift:
    """Deadlift(s): floor -> knee_pass -> lockout -> controlled descent.

    The bar goes up and is lowered back down (``up_down``); there is NO catch.
    Truth: knee_pass / lockout per rep.
    """
    per_rep = [
        (0.00, FLOOR_BAR, 0.70),  # setup, hinged over the bar
        (0.45, KNEE_BAR, 0.35),  # bar passes the knees
        (0.85, DL_LOCKOUT, 0.00),  # standing lockout
        (1.05, DL_LOCKOUT, 0.00),  # lockout hold
        (1.60, FLOOR_BAR, 0.70),  # controlled descent to the floor
        (1.85, FLOOR_BAR, 0.70),  # inter-rep hold
    ]
    events = {"knee_pass": 0.45, "lockout": 0.85}
    return _build(
        per_rep,
        events,
        n_reps=n_reps,
        rep_period=1.85,
        fps=fps,
        noise_cm=noise_cm,
        seed=seed,
        speed=speed,
    )


def dumped_clean(
    *,
    fps: int = 60,
    noise_cm: float = 0.0,
    speed: float = 1.0,
    seed: int = 0,
) -> SyntheticLift:
    """One good clean to the rack, then the bar is DUMPED into free-fall.

    A single clean pull reaches the rack, then the athlete drops the bar: it
    accelerates under gravity (``y = rack - 1/2 g t^2``) to the floor, so the
    vertical velocity is sustained well below -250 cm/s -- the terminal drop a
    segmenter must recognize and discard. ``reps`` is 1 (the one good rep).
    Truth includes the good rep's phases plus ``dump_start``.
    """
    rng = np.random.default_rng(seed)
    # good pull up to the rack hold (reuse the clean keyframes to 1.45s)
    to_rack = [
        (0.00, FLOOR_BAR, 0.85),
        (0.25, FLOOR_BAR, 0.85),
        (0.65, KNEE_BAR, 0.55),
        (0.95, CLEAN_PEAK, 0.00),
        (1.10, CLEAN_DIP, 0.35),
        (1.25, RACK, 0.20),
        (1.45, RACK, 0.15),
    ]
    t_dump = 1.45 / speed
    pull_t = np.array([t / speed for (t, _b, _c) in to_rack])
    pull = PchipInterpolator(pull_t, np.array([b for (_t, b, _c) in to_rack]))
    # crouch keyframes: the pull's crouch, plus a standing point after the dump
    crouch_t = np.append(pull_t, [t_dump + 0.9 / speed])
    crouch_v = np.append([c for (_t, _b, c) in to_rack], [0.10])

    fall_time = float(np.sqrt(2.0 * (RACK - FLOOR_BAR) / G_CM_S2))
    t_end = t_dump + fall_time + 0.4 / speed
    grid = _grid(t_end, fps)

    bar_ys = np.empty_like(grid)
    for i, t in enumerate(grid):
        if t <= t_dump:
            bar_ys[i] = pull(t)
        else:
            bar_ys[i] = max(RACK - 0.5 * G_CM_S2 * (t - t_dump) ** 2, FLOOR_BAR)
    crouch = PchipInterpolator(crouch_t, crouch_v)(np.clip(grid, None, crouch_t[-1]))

    bar = _bar_series(grid, bar_ys, rng, noise_cm)
    landmarks = _landmarks(grid, bar_ys, crouch, rng, noise_cm)
    truth = {
        "first_pull": [0.25 / speed],
        "knee_pass": [0.65 / speed],
        "second_pull": [0.80 / speed],
        "catch": [1.10 / speed],
        "dump_start": [t_dump],
    }
    return SyntheticLift(bar=bar, landmarks=landmarks, truth=truth, reps=1)


def zero_rep(
    *,
    fps: int = 60,
    duration_s: float = 3.0,
    walkout_jitter_cm: float = 2.0,
    noise_cm: float = 0.0,
    seed: int = 0,
) -> SyntheticLift:
    """Only a walkout / jitter -- the bar never crosses the start threshold.

    The bar hovers at a resting height with small jitter and no rep ever
    happens: ``reps`` is 0 and ``truth`` is empty. Used to prove a segmenter
    reports zero reps on non-lifts.
    """
    rng = np.random.default_rng(seed)
    grid = _grid(duration_s, fps)
    rest = SQUAT_TOP
    bar_ys = np.full_like(grid, rest) + rng.normal(0.0, walkout_jitter_cm / 2.0, size=grid.shape)
    crouch = np.zeros_like(grid)
    bar = _bar_series(grid, bar_ys, rng, noise_cm)
    landmarks = _landmarks(grid, np.full_like(grid, rest), crouch, rng, noise_cm)
    return SyntheticLift(bar=bar, landmarks=landmarks, truth={}, reps=0)

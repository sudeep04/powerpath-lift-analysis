"""Rep segmentation via prominence-based peak/valley detection.

The problem this module solves is *counting reps* and carving the bar
trajectory into one time window per rep, so the phase detectors in
``phases.py`` each get a clean slice containing exactly one rep's
excursion.

Why peak detection and NOT a velocity-gap / dwell state machine: the quiet
gaps *between* reps vary wildly by movement (a clean rests ~600ms on the
floor, a squat ~370ms standing, a push press ~270ms at the shoulders) and
those inter-rep gaps *collide* with quiet moments *inside* a single rep (a
deadlift holds ~230ms at the top). There is no fixed dwell threshold that
separates "between reps" from "inside a rep". A rep, however, always has
exactly one dominant bar apex -- the rack/overhead/lockout height for an
upward lift, or the depth for a squat -- and that apex stands out from its
surroundings by a *prominence* proportional to the movement's expected
displacement. Counting sufficiently-prominent apices counts reps, and
sidesteps the dwell problem entirely.

``config.bar_travel`` picks the apex orientation: ``"up"`` and ``"up_down"``
lifts (clean rack, snatch/press overhead, deadlift top) apex at a bar
*maximum*, so we detect peaks of ``+y``; a ``"down_up"`` squat apexes at the
*bottom*, so we detect peaks of ``-y`` (valleys of ``+y``). Everything after
detection is expressed against a single ``signal`` array (``+y`` or ``-y``)
whose peaks are the apices and whose intervening minima are the rep
boundaries, so the two orientations share one code path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from powerpath_engine.registry import MovementConfig
from powerpath_engine.series import TimeSeries

# A detected apex must rise this fraction of the movement's expected bar
# displacement (``config.min_disp_cm``) above its surroundings to count as a
# rep. 0.6 is comfortably above walkout / rack jitter (a few cm) yet well
# below a real rep's excursion (tens of cm), so genuine reps clear it and
# noise never does.
PROMINENCE_FRACTION = 0.6

# Two apices closer together in time than this are treated as one rep: no
# barbell lift repeats faster than ~2Hz, so this rejects a doubled detection
# on a jittery apex without ever merging two real reps.
MIN_REP_SEPARATION_S = 0.5

# Vertical bar velocity (cm/s, y-up so downward is negative) below which the
# bar is considered to be in free-fall -- a dropped/dumped bar accelerating
# under gravity, not a controlled lower. A controlled descent of even a
# tall lift stays well above this (a clean lowered from the rack is only
# ~-110 cm/s); genuine free-fall blows past it (~-430 cm/s near the floor).
V_FREEFALL_CMS = -250.0

# A free-fall must persist at least this many consecutive samples before we
# trust it as a real drop (rather than a one-sample velocity spike).
_FREEFALL_MIN_SAMPLES = 3


@dataclass(frozen=True)
class RepWindow:
    """One rep's time span, ``[t_start, t_end]`` in PTS seconds.

    ``rep_index`` is the rep's ordinal in detection (== chronological)
    order, starting at 0. The window is built to fully contain the rep's
    bar excursion -- its apex plus the run-up and run-down on either side --
    so a phase detector handed this slice sees the whole rep.
    """

    t_start: float
    t_end: float
    rep_index: int


def segment(bar: TimeSeries, config: MovementConfig, expected_dt: float) -> list[RepWindow]:
    """Split ``bar`` into one :class:`RepWindow` per detected rep.

    Prominence-based apex detection (see the module docstring): the smoothed
    bar height is oriented by ``config.bar_travel`` into a ``signal`` whose
    peaks are rep apices, gated by a prominence of
    ``PROMINENCE_FRACTION * config.min_disp_cm`` and a minimum separation of
    ``MIN_REP_SEPARATION_S``. Each surviving apex is exactly one rep. Window
    boundaries are the minima of ``signal`` between consecutive apices (the
    quietest point between reps); the first window opens at the minimum
    before the first apex and the last closes at the minimum after the last
    apex. Finally, any window whose tail runs into a sustained bar free-fall
    (a dumped bar) has its ``t_end`` clamped back to the onset of the drop so
    the window never extends into the crash -- this only trims a window end,
    never changes the rep count.

    ``expected_dt`` is the nominal frame period; the actual sample spacing is
    preferred (median of the series timestamps) and ``expected_dt`` is only
    used as a fallback when the series has fewer than two samples.
    """
    smoothed = bar.smooth()
    t = smoothed.ts()
    y = smoothed.ys()

    if len(t) >= 2:
        dt = float(np.median(np.diff(t)))
        if not (dt > 0.0):
            dt = expected_dt
    else:
        dt = expected_dt

    if len(t) < 2 or not (dt > 0.0):
        return []

    # Orient so apices are always *peaks* of ``signal``.
    signal = y if config.bar_travel != "down_up" else -y

    prominence = PROMINENCE_FRACTION * config.min_disp_cm
    distance = max(1, round(MIN_REP_SEPARATION_S / dt))
    peaks, _props = find_peaks(signal, prominence=prominence, distance=distance)

    if len(peaks) == 0:
        return []

    velocity = np.gradient(y, t)

    windows: list[RepWindow] = []
    n = len(signal)
    for i, apex in enumerate(peaks):
        # Left boundary: the minimum of ``signal`` from the previous apex
        # (or the series start) up to and including this apex.
        left_from = 0 if i == 0 else int(peaks[i - 1])
        left_idx = left_from + int(np.argmin(signal[left_from : apex + 1]))
        # Right boundary: the minimum of ``signal`` from this apex to the
        # next apex (or the series end).
        right_to = n - 1 if i == len(peaks) - 1 else int(peaks[i + 1])
        right_idx = apex + int(np.argmin(signal[apex : right_to + 1]))

        t_start = float(t[left_idx])
        t_end = float(t[right_idx])
        t_end = _trim_freefall(t, velocity, apex, right_idx, t_end, dt)

        windows.append(RepWindow(t_start=t_start, t_end=t_end, rep_index=i))

    return windows


def _trim_freefall(
    t: np.ndarray,
    velocity: np.ndarray,
    apex_idx: int,
    end_idx: int,
    t_end: float,
    dt: float,
) -> float:
    """Clamp ``t_end`` back to the onset of a *terminal* free-fall.

    A dumped bar is distinguished from a merely fast (but controlled)
    descent by *where* the plunge lands: a dump is still in free-fall as it
    crashes into the window end, whereas a controlled lower -- even the
    quick drop from an overhead lockout -- decelerates into a turnaround
    well before the boundary. So a trim applies only when the bar is still
    moving below :data:`V_FREEFALL_CMS` within the last ~0.1s of samples
    before ``end_idx`` (and that plunge lasts at least
    ``_FREEFALL_MIN_SAMPLES``). When it does, ``t_end`` is pulled back to the
    top of that terminal descent -- the last sample before the bar began
    dropping -- so the window stops before the crash. Returns ``t_end``
    unchanged otherwise, and never extends the window.
    """
    tail = max(1, round(0.1 / dt))
    tail_lo = max(apex_idx + 1, end_idx - tail)
    if not any(velocity[k] < V_FREEFALL_CMS for k in range(tail_lo, end_idx + 1)):
        return t_end

    falling = sum(1 for k in range(apex_idx + 1, end_idx + 1) if velocity[k] < V_FREEFALL_CMS)
    if falling < _FREEFALL_MIN_SAMPLES:
        return t_end

    # Seek the last fast-falling sample at/before the end (skipping any near-
    # stationary samples once the bar has already hit the floor), then walk
    # back over the whole descent to its top (last non-descending sample).
    k = end_idx
    while k > apex_idx and velocity[k] >= V_FREEFALL_CMS:
        k -= 1
    onset = k
    while onset > apex_idx and velocity[onset] < 0.0:
        onset -= 1
    return min(t_end, float(t[onset]))

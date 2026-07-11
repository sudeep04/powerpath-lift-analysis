"""Time series core: the data spine of the engine.

`TimeSeries` is a PTS-keyed (see the global PTS-timebase constraint: `t` is
always seconds, from PyAV, never a bare frame index -- iPhone video is
variable frame rate) sequence of `Sample`s for a single scalar-ish 2D
trajectory (bar marker position, one pose landmark, ...). `LandmarkSeries`
is the per-frame, multi-landmark counterpart produced by pose estimation:
one `LandmarkFrame` per decoded/pose-processed frame, each carrying a
`{landmark_name: Sample}` dict (a landmark absent from a frame simply has
no key -- it was not detected there).

Every operation here is pure: methods return new `TimeSeries` /
`LandmarkSeries` objects and never mutate `self.samples` / `self.frames`.
Bar tracking (marker gaps), pose (missed-landmark gaps), and segmentation
(smoothed velocity for state-machine thresholds) all build on this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from scipy.signal import savgol_filter

# Savitzky-Golay smoothing is fit with a quadratic local polynomial (captures
# the curvature of a bar-path turnaround without over-fitting to noise); the
# window is always forced odd (scipy requirement) and at least this many
# samples so the fit is over-determined (polyorder 2 needs >= 3, 5 leaves
# margin against noise amplification at the small end).
SG_POLYORDER = 2
SG_MIN_WINDOW = 5

# Landmark names used downstream (Task 5 pose backends map their native
# keypoint sets onto this vocabulary). Documentation only -- LandmarkFrame.
# points is a plain dict[str, Sample], nothing here enforces membership.
LANDMARK_NAMES = (
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)


@dataclass
class Sample:
    """A single 2D observation at time `t`.

    `t` is PTS seconds (float). `x`/`y` are in px, image space (y grows
    down), at whatever layer produced this sample -- geometry.py is the
    sole place that converts these to cm or flips to y-up.
    """

    t: float
    x: float
    y: float
    visibility: float = 1.0


@dataclass
class Gap:
    """A detected hole in a TimeSeries between two consecutive samples.

    `t_start`/`t_end` are the timestamps of the samples bracketing the
    hole (not the missing samples themselves). `filled` is True if the
    hole was small enough (<= max_gap_frames missing samples) to be
    linearly interpolated, False if it was left unfilled and is only
    being reported.
    """

    t_start: float
    t_end: float
    filled: bool


@dataclass
class TimeSeries:
    """A time-ordered sequence of `Sample`s for one trajectory."""

    samples: list[Sample]

    def ts(self) -> np.ndarray:
        return np.array([s.t for s in self.samples], dtype=float)

    def xs(self) -> np.ndarray:
        return np.array([s.x for s in self.samples], dtype=float)

    def ys(self) -> np.ndarray:
        return np.array([s.y for s in self.samples], dtype=float)

    def interpolate_gaps(
        self, max_gap_frames: int, expected_dt: float
    ) -> tuple[TimeSeries, list[Gap]]:
        """Fill short holes linearly, report longer ones unfilled.

        A "hole" is a run of `n` missing samples between two consecutive
        existing samples, inferred from how many `expected_dt`-sized steps
        fit in the actual time delta between them (`n = round(dt /
        expected_dt) - 1`). Holes with `n <= max_gap_frames` are filled
        with samples linearly interpolated (t, x, y, visibility) at the
        expected spacing; longer holes are left as-is in the output and
        reported as an unfilled `Gap`. Consecutive samples with no
        detected hole are passed through untouched.
        """
        if len(self.samples) < 2:
            return TimeSeries(list(self.samples)), []

        filled_samples: list[Sample] = [self.samples[0]]
        gaps: list[Gap] = []
        for prev, curr in pairwise(self.samples):
            dt = curr.t - prev.t
            missing = round(dt / expected_dt) - 1
            if missing >= 1:
                is_filled = missing <= max_gap_frames
                gaps.append(Gap(t_start=prev.t, t_end=curr.t, filled=is_filled))
                if is_filled:
                    steps = missing + 1
                    for i in range(1, steps):
                        frac = i / steps
                        filled_samples.append(
                            Sample(
                                t=prev.t + frac * dt,
                                x=prev.x + frac * (curr.x - prev.x),
                                y=prev.y + frac * (curr.y - prev.y),
                                visibility=prev.visibility
                                + frac * (curr.visibility - prev.visibility),
                            )
                        )
            filled_samples.append(curr)

        return TimeSeries(filled_samples), gaps

    def smooth(self, window_s: float = 0.15) -> TimeSeries:
        """Savitzky-Golay smoothing (polyorder 2) of x and y.

        The window length is derived from the median sample spacing
        (`window_s / median_dt`), forced odd, and floored at
        `SG_MIN_WINDOW`. Too few samples to support even the minimum
        window (or a degenerate/zero median dt) returns a copy of the
        series unchanged rather than raising -- callers may hand this a
        short rep window.
        """
        n = len(self.samples)
        if n < SG_MIN_WINDOW:
            return TimeSeries(list(self.samples))

        ts = self.ts()
        dt_values = np.diff(ts)
        median_dt = float(np.median(dt_values))
        if not (median_dt > 0.0):
            return TimeSeries(list(self.samples))

        window_length = int(round(window_s / median_dt))
        if window_length % 2 == 0:
            window_length += 1
        window_length = max(window_length, SG_MIN_WINDOW)
        if window_length > n:
            window_length = n if n % 2 == 1 else n - 1
        if window_length <= SG_POLYORDER or window_length < SG_MIN_WINDOW:
            return TimeSeries(list(self.samples))

        xs_smooth = savgol_filter(self.xs(), window_length, SG_POLYORDER)
        ys_smooth = savgol_filter(self.ys(), window_length, SG_POLYORDER)
        visibilities = [s.visibility for s in self.samples]

        return TimeSeries(
            [
                Sample(t=float(t), x=float(x), y=float(y), visibility=v)
                for t, x, y, v in zip(ts, xs_smooth, ys_smooth, visibilities, strict=True)
            ]
        )

    def velocity(self) -> np.ndarray:
        """Central-difference velocity of y, per second, in the caller's units.

        Smooths first (via `smooth()`, default window) so differencing
        does not amplify per-sample noise, then takes `np.gradient` over
        the (possibly non-uniform, VFR) timestamps -- central differences
        in the interior, second-order-accurate one-sided differences at
        the boundaries.
        """
        smoothed = self.smooth()
        t = smoothed.ts()
        y = smoothed.ys()
        if len(t) < 2:
            return np.zeros_like(y)
        return np.gradient(y, t)

    def slice_time(self, t0: float, t1: float) -> TimeSeries:
        """Samples with `t0 <= t < t1` (inclusive start, exclusive end)."""
        return TimeSeries([s for s in self.samples if t0 <= s.t < t1])


@dataclass
class LandmarkFrame:
    """All landmarks detected for a single pose-processed frame at time `t`."""

    t: float
    points: dict[str, Sample]


@dataclass
class LandmarkSeries:
    """Per-frame, multi-landmark trajectories, produced by pose estimation."""

    frames: list[LandmarkFrame]

    def landmark_names(self) -> set[str]:
        """Union of landmark names appearing in any frame."""
        names: set[str] = set()
        for frame in self.frames:
            names.update(frame.points.keys())
        return names

    def series_for(self, name: str) -> TimeSeries:
        """The single-landmark `TimeSeries` for `name` across all frames.

        Only frames where `name` was detected contribute a sample -- this
        is the extraction step that lets `interpolate_gaps`/`smooth` reuse
        TimeSeries's implementation per landmark.
        """
        return TimeSeries([frame.points[name] for frame in self.frames if name in frame.points])

    def interpolate_gaps(
        self, max_gap_frames: int, expected_dt: float
    ) -> tuple[LandmarkSeries, dict[str, list[Gap]]]:
        """Apply `TimeSeries.interpolate_gaps` independently per landmark name.

        Each landmark's own timeline is gap-filled on its own terms, so
        interpolated samples for different landmarks are not necessarily
        time-aligned with each other unless they share neighboring
        original frame timestamps. The result's frames are the union of
        all output timestamps across landmarks; each frame carries
        whichever landmark samples exist at that instant.
        """
        gaps_by_name: dict[str, list[Gap]] = {}
        filled_by_name: dict[str, TimeSeries] = {}
        for name in self.landmark_names():
            filled_ts, gaps = self.series_for(name).interpolate_gaps(max_gap_frames, expected_dt)
            filled_by_name[name] = filled_ts
            gaps_by_name[name] = gaps
        return _merge_landmark_series(filled_by_name), gaps_by_name

    def smooth(self, window_s: float = 0.15) -> LandmarkSeries:
        """Apply `TimeSeries.smooth` independently per landmark name."""
        smoothed_by_name = {
            name: self.series_for(name).smooth(window_s) for name in self.landmark_names()
        }
        return _merge_landmark_series(smoothed_by_name)


# Timestamps within this many seconds of each other are treated as "the same
# instant" when merging per-landmark series back together (see
# _merge_landmark_series). Two landmarks anchored on the same original frame
# times can each independently compute an interpolated t via prev.t + frac *
# (curr.t - prev.t); float arithmetic doesn't guarantee those land on exactly
# the same bit pattern even when they're mathematically identical, so exact
# dict-key equality would needlessly split one instant into two frames. This
# tolerance (1us) is many orders of magnitude below any real frame spacing
# (milliseconds) and many orders above float64 rounding noise on second-scale
# timestamps, so it only ever merges values that are the same instant.
_MERGE_TOLERANCE_S = 1e-6


def _merge_landmark_series(series_by_name: dict[str, TimeSeries]) -> LandmarkSeries:
    """Recombine independent per-landmark TimeSeries into one LandmarkSeries.

    Groups samples across landmarks by timestamp (within
    `_MERGE_TOLERANCE_S`): frames that share an instant (e.g. every
    original frame, or landmarks whose gap-filled timestamps coincide
    because they share the same neighboring anchors) are merged into a
    single LandmarkFrame; otherwise each timestamp gets its own frame
    carrying only the landmarks present at that instant.
    """
    entries: list[tuple[float, str, Sample]] = []
    for name, ts_obj in series_by_name.items():
        for sample in ts_obj.samples:
            entries.append((sample.t, name, sample))
    entries.sort(key=lambda entry: entry[0])

    frames: list[LandmarkFrame] = []
    group_t: float | None = None
    group_points: dict[str, Sample] = {}
    for t, name, sample in entries:
        if group_t is None or t - group_t > _MERGE_TOLERANCE_S:
            if group_t is not None:
                frames.append(LandmarkFrame(t=group_t, points=group_points))
            group_t = t
            group_points = {}
        group_points[name] = sample
    if group_t is not None:
        frames.append(LandmarkFrame(t=group_t, points=group_points))

    return LandmarkSeries(frames)

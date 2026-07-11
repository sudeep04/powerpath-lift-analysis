"""Tests for powerpath_engine.series.

series.py is the data spine for the engine: bar tracking, pose, and
segmentation all produce/consume Sample/TimeSeries and
LandmarkFrame/LandmarkSeries. Every operation must be pure (return new
objects, never mutate the receiver).
"""

import numpy as np
import pytest

from powerpath_engine.series import (
    Gap,
    LandmarkFrame,
    LandmarkSeries,
    Sample,
    TimeSeries,
)

# ---------------------------------------------------------------------------
# Sample / TimeSeries basics
# ---------------------------------------------------------------------------


def test_sample_defaults_visibility_to_one() -> None:
    s = Sample(t=1.0, x=2.0, y=3.0)
    assert s.visibility == 1.0


def test_ts_xs_ys_return_numpy_arrays_in_order() -> None:
    samples = [Sample(t=0.0, x=1.0, y=2.0), Sample(t=1.0, x=3.0, y=4.0)]
    ts_obj = TimeSeries(samples)
    assert isinstance(ts_obj.ts(), np.ndarray)
    assert isinstance(ts_obj.xs(), np.ndarray)
    assert isinstance(ts_obj.ys(), np.ndarray)
    assert ts_obj.ts().tolist() == [0.0, 1.0]
    assert ts_obj.xs().tolist() == [1.0, 3.0]
    assert ts_obj.ys().tolist() == [2.0, 4.0]


# ---------------------------------------------------------------------------
# interpolate_gaps
# ---------------------------------------------------------------------------


def test_interpolate_gaps_fills_short_gap_and_reports_long_gap_unfilled() -> None:
    """A 3-frame hole (missing indices 3,4,5) is filled exactly linearly;
    a 6-frame hole (missing indices 11..16) is left unfilled and reported,
    with max_gap_frames=5 as the threshold between the two.
    """
    dt = 1.0 / 30.0
    present_indices = [0, 1, 2, 6, 7, 8, 9, 10, 17, 18]
    samples = [Sample(t=i * dt, x=float(i) * 2.0, y=float(i) * 3.0) for i in present_indices]
    ts_obj = TimeSeries(samples)

    filled, gaps = ts_obj.interpolate_gaps(max_gap_frames=5, expected_dt=dt)

    assert len(gaps) == 2
    small_gap, large_gap = gaps
    assert small_gap == Gap(t_start=2 * dt, t_end=6 * dt, filled=True)
    assert large_gap == Gap(t_start=10 * dt, t_end=17 * dt, filled=False)

    filled_by_index = {round(s.t / dt): s for s in filled.samples}
    for missing_i in (3, 4, 5):
        assert missing_i in filled_by_index
        s = filled_by_index[missing_i]
        assert s.x == pytest.approx(missing_i * 2.0)
        assert s.y == pytest.approx(missing_i * 3.0)

    for missing_i in range(11, 17):
        assert missing_i not in filled_by_index

    # nothing else was dropped or duplicated
    assert len(filled.samples) == len(samples) + 3


def test_interpolate_gaps_no_gap_returns_equivalent_series() -> None:
    dt = 1.0 / 60.0
    samples = [Sample(t=i * dt, x=float(i), y=float(i)) for i in range(10)]
    ts_obj = TimeSeries(samples)
    filled, gaps = ts_obj.interpolate_gaps(max_gap_frames=5, expected_dt=dt)
    assert gaps == []
    assert [s.t for s in filled.samples] == [s.t for s in samples]


def test_interpolate_gaps_does_not_mutate_original() -> None:
    dt = 1.0 / 30.0
    samples = [Sample(t=0.0, x=0.0, y=0.0), Sample(t=4 * dt, x=8.0, y=8.0)]
    ts_obj = TimeSeries(list(samples))
    ts_obj.interpolate_gaps(max_gap_frames=5, expected_dt=dt)
    assert len(ts_obj.samples) == 2


# ---------------------------------------------------------------------------
# smooth
# ---------------------------------------------------------------------------


def test_smooth_reduces_added_white_noise_variance_and_preserves_peak() -> None:
    """A single-peaked cosine (not a multi-cycle sine) so there is exactly
    one true maximum -- a periodic signal with several equal peaks would
    let added noise flip *which* peak is the global max without the
    smoothing itself failing to track any individual peak's location.
    """
    rng = np.random.default_rng(42)
    dt = 1.0 / 60.0
    n = 90
    t = np.arange(n) * dt
    freq_hz = 0.5
    t_peak = 0.75
    clean_y = np.cos(2 * np.pi * freq_hz * (t - t_peak))
    noise = rng.normal(scale=0.05, size=n)
    noisy_y = clean_y + noise

    samples = [Sample(t=float(tt), x=0.0, y=float(yy)) for tt, yy in zip(t, noisy_y, strict=True)]
    smoothed = TimeSeries(samples).smooth(window_s=0.2)
    smoothed_y = smoothed.ys()

    var_before = np.var(noisy_y - clean_y)
    var_after = np.var(smoothed_y - clean_y)
    assert var_after <= var_before / 5.0

    true_peak_idx = int(np.argmax(clean_y))
    smoothed_peak_idx = int(np.argmax(smoothed_y))
    assert abs(smoothed_peak_idx - true_peak_idx) <= 1


def test_smooth_preserves_linear_signal_exactly() -> None:
    dt = 1.0 / 60.0
    n = 60
    t = [i * dt for i in range(n)]
    slope = 12.0
    samples = [Sample(t=tt, x=0.0, y=slope * tt + 5.0) for tt in t]
    smoothed = TimeSeries(samples).smooth()
    assert smoothed.ys() == pytest.approx([slope * tt + 5.0 for tt in t], abs=1e-6)


def test_smooth_is_pure() -> None:
    dt = 1.0 / 60.0
    samples = [Sample(t=i * dt, x=float(i), y=float(i)) for i in range(20)]
    ts_obj = TimeSeries(list(samples))
    ts_obj.smooth()
    assert ts_obj.samples == samples


# ---------------------------------------------------------------------------
# velocity
# ---------------------------------------------------------------------------


def test_velocity_of_linear_ramp_is_constant() -> None:
    dt = 1.0 / 60.0
    n = 60
    slope = 25.0
    samples = [Sample(t=i * dt, x=0.0, y=slope * (i * dt) + 100.0) for i in range(n)]
    vel = TimeSeries(samples).velocity()
    assert vel == pytest.approx([slope] * n, abs=1e-6)


# ---------------------------------------------------------------------------
# slice_time
# ---------------------------------------------------------------------------


def test_slice_time_boundaries_inclusive_exclusive() -> None:
    samples = [Sample(t=float(i), x=0.0, y=0.0) for i in range(5)]  # t = 0,1,2,3,4
    sliced = TimeSeries(samples).slice_time(1.0, 3.0)
    assert [s.t for s in sliced.samples] == [1.0, 2.0]


def test_slice_time_is_pure() -> None:
    samples = [Sample(t=float(i), x=0.0, y=0.0) for i in range(5)]
    ts_obj = TimeSeries(list(samples))
    ts_obj.slice_time(1.0, 3.0)
    assert len(ts_obj.samples) == 5


# ---------------------------------------------------------------------------
# LandmarkFrame / LandmarkSeries
# ---------------------------------------------------------------------------


def test_landmark_series_interpolate_gaps_applies_per_landmark_independently() -> None:
    dt = 1.0 / 30.0
    frames = []
    for i in range(6):
        points = {"nose": Sample(t=i * dt, x=1.0, y=2.0)}
        if i != 2:  # left_wrist missing exactly one frame -> a fillable 1-frame gap
            points["left_wrist"] = Sample(t=i * dt, x=float(i), y=float(i) * 2.0)
        frames.append(LandmarkFrame(t=i * dt, points=points))
    series = LandmarkSeries(frames)

    filled, gaps_by_name = series.interpolate_gaps(max_gap_frames=2, expected_dt=dt)

    assert gaps_by_name["nose"] == []
    assert len(gaps_by_name["left_wrist"]) == 1
    assert gaps_by_name["left_wrist"][0].filled is True

    frame_at_2 = next(f for f in filled.frames if f.t == pytest.approx(2 * dt))
    assert "left_wrist" in frame_at_2.points
    assert frame_at_2.points["left_wrist"].x == pytest.approx(2.0)
    assert frame_at_2.points["left_wrist"].y == pytest.approx(4.0)


def test_landmark_series_smooth_applies_per_landmark() -> None:
    dt = 1.0 / 30.0
    n = 20
    frames = [
        LandmarkFrame(
            t=i * dt,
            points={"nose": Sample(t=i * dt, x=float(i) * 0.5, y=10.0)},
        )
        for i in range(n)
    ]
    smoothed = LandmarkSeries(frames).smooth()
    nose_ts = smoothed.series_for("nose")
    assert nose_ts.xs() == pytest.approx([i * 0.5 for i in range(n)], abs=1e-6)


def test_landmark_series_series_for_extracts_single_landmark_timeline() -> None:
    frames = [
        LandmarkFrame(t=0.0, points={"nose": Sample(t=0.0, x=1.0, y=1.0)}),
        LandmarkFrame(
            t=1.0,
            points={
                "nose": Sample(t=1.0, x=2.0, y=2.0),
                "left_knee": Sample(t=1.0, x=9.0, y=9.0),
            },
        ),
    ]
    series = LandmarkSeries(frames)
    nose_ts = series.series_for("nose")
    assert nose_ts.ts().tolist() == [0.0, 1.0]
    knee_ts = series.series_for("left_knee")
    assert knee_ts.ts().tolist() == [1.0]

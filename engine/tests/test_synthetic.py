"""Tests for the synthetic trajectory generators (tests/synthetic.py).

These pin the *physical* invariants downstream tasks (6b segmentation, 7 made
reps, 8 faults) rely on: correct rep counts, y-up orientation, the clean's
catch-dip velocity zero-crossing, the dumped bar's free-fall, the snatch's
overhead receive, and the zero-rep non-lift. The generators themselves are
fixtures; here we assert they behave.
"""

from __future__ import annotations

import numpy as np
import pytest
from synthetic import (
    SyntheticLift,
    back_squat,
    clean,
    deadlift,
    dumped_clean,
    power_snatch,
    push_press,
    single_rep,
    zero_rep,
)

from powerpath_engine.series import TimeSeries

REP_GENERATORS = (clean, power_snatch, back_squat, push_press, deadlift)


def _y_at(series: TimeSeries, t: float) -> float:
    """Bar/landmark y at the grid sample nearest ``t``."""
    ts = series.ts()
    return float(series.ys()[int(np.argmin(np.abs(ts - t)))])


def _longest_run(mask: np.ndarray) -> int:
    run = best = 0
    for flag in mask:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


# ---------------------------------------------------------------------------
# structure / rep counts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gen", REP_GENERATORS, ids=lambda g: g.__name__)
@pytest.mark.parametrize("n_reps", [1, 2, 4])
def test_generator_returns_requested_rep_count(gen, n_reps: int) -> None:
    lift = gen(n_reps)
    assert isinstance(lift, SyntheticLift)
    assert lift.reps == n_reps
    # every truth channel has exactly one timestamp per rep
    assert lift.truth  # non-empty
    for name, times in lift.truth.items():
        assert len(times) == n_reps, name
    # bar and landmarks share the same grid timestamps
    assert lift.bar.ts().tolist() == lift.landmarks.series_for("nose").ts().tolist()


def test_single_rep_is_a_one_rep_clean() -> None:
    lift = single_rep()
    assert lift.reps == 1
    assert set(lift.truth) == {"first_pull", "knee_pass", "second_pull", "catch"}
    assert all(len(v) == 1 for v in lift.truth.values())


def test_truth_times_are_strictly_increasing_within_a_rep() -> None:
    lift = clean(1)
    ordered = [lift.truth[k][0] for k in ("first_pull", "knee_pass", "second_pull", "catch")]
    assert ordered == sorted(ordered)


def test_dumped_clean_is_one_rep_with_dump_marker() -> None:
    lift = dumped_clean()
    assert lift.reps == 1
    assert "dump_start" in lift.truth
    assert lift.truth["dump_start"][0] > lift.truth["catch"][0]


def test_zero_rep_has_no_reps_and_empty_truth() -> None:
    lift = zero_rep()
    assert lift.reps == 0
    assert lift.truth == {}


# ---------------------------------------------------------------------------
# y-up orientation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gen", REP_GENERATORS, ids=lambda g: g.__name__)
def test_series_are_y_up_standing_body_stacks_upward(gen) -> None:
    lift = gen(1)
    frame = lift.landmarks.frames[0]
    p = frame.points
    # a standing skeleton: nose highest, then shoulder, hip, knee, ankle
    assert p["nose"].y > p["left_shoulder"].y > p["left_hip"].y
    assert p["left_hip"].y > p["left_knee"].y > p["left_ankle"].y


def test_clean_bar_is_y_up_peak_above_floor() -> None:
    lift = clean(2)
    ys = lift.bar.ys()
    # the pulled bar rises far above its floor rest -> larger y is higher
    assert ys.max() - ys.min() > 40.0
    floor_y = _y_at(lift.bar, lift.truth["first_pull"][0])
    peak_y = _y_at(lift.bar, lift.truth["catch"][0] - 0.15)
    assert peak_y > floor_y


# ---------------------------------------------------------------------------
# clean catch dip: intra-rep local-max-then-dip + velocity zero-crossing
# ---------------------------------------------------------------------------


def test_clean_has_local_max_then_dip_near_each_catch() -> None:
    lift = clean(3)
    t = lift.bar.ts()
    y = lift.bar.ys()
    for tc in lift.truth["catch"]:
        window = (t >= tc - 0.35) & (t <= tc + 0.25)
        yw = y[window]
        i_peak = int(np.argmax(yw))
        after = yw[i_peak:]
        i_dip = i_peak + int(np.argmin(after))
        # a genuine dip of several cm below the pull's apex
        assert yw[i_dip] < yw[i_peak] - 3.0
        # and the bar settles back up (rack) after the dip
        assert yw[-1] > yw[i_dip] + 1.0


def test_clean_catch_dip_creates_intra_rep_velocity_zero_crossing() -> None:
    lift = clean(2)
    t = lift.bar.ts()
    vy = lift.bar.velocity()
    for tc in lift.truth["catch"]:
        window = (t >= tc - 0.3) & (t <= tc + 0.2)
        signs = np.sign(vy[window])
        signs = signs[signs != 0]
        # up (peak) then down (into the dip) then up (settle) -> >=2 sign flips
        assert np.count_nonzero(np.diff(signs) != 0) >= 1


# ---------------------------------------------------------------------------
# dumped clean free-fall
# ---------------------------------------------------------------------------


def test_dumped_clean_has_sustained_free_fall_below_minus_250() -> None:
    lift = dumped_clean()
    vy = lift.bar.velocity()
    below = vy < -250.0
    assert below.any()
    assert _longest_run(below) >= 3  # a sustained drop, not a single spike
    # the free-fall is faster than anything in a normal clean lower
    assert vy.min() < -250.0


def test_normal_clean_lower_never_free_falls() -> None:
    lift = clean(2)
    vy = lift.bar.velocity()
    # a controlled lower stays well above the free-fall threshold, with no
    # sustained run below it (the discriminator 6b uses vs a dumped bar)
    assert vy.min() > -250.0
    assert _longest_run(vy < -250.0) == 0


# ---------------------------------------------------------------------------
# snatch overhead receive
# ---------------------------------------------------------------------------


def test_power_snatch_bar_ends_above_nose_at_receive() -> None:
    lift = power_snatch(2)
    for tr in lift.truth["receive"]:
        bar_y = _y_at(lift.bar, tr)
        nose_y = _y_at(lift.landmarks.series_for("nose"), tr)
        assert bar_y > nose_y


# ---------------------------------------------------------------------------
# zero rep stays inside the start band
# ---------------------------------------------------------------------------


def test_zero_rep_never_exceeds_start_band() -> None:
    lift = zero_rep(walkout_jitter_cm=2.0)
    ys = lift.bar.ys()
    # deviation stays far under the smallest rep start threshold (min_disp 20cm)
    assert float(ys.max() - ys.min()) < 15.0


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


def test_generators_are_deterministic_for_a_given_seed() -> None:
    a = clean(2, noise_cm=1.0, seed=7)
    b = clean(2, noise_cm=1.0, seed=7)
    assert a.bar.ys().tolist() == b.bar.ys().tolist()
    assert a.bar.xs().tolist() == b.bar.xs().tolist()


def test_different_seeds_change_the_noise() -> None:
    a = clean(2, noise_cm=1.0, seed=1)
    b = clean(2, noise_cm=1.0, seed=2)
    assert a.bar.ys().tolist() != b.bar.ys().tolist()


def test_zero_noise_default_is_smooth_and_repeatable() -> None:
    a = clean(1)
    b = clean(1)
    assert a.bar.ys().tolist() == b.bar.ys().tolist()


# ---------------------------------------------------------------------------
# per-movement truth channels
# ---------------------------------------------------------------------------


def test_each_movement_exposes_its_expected_truth_channels() -> None:
    assert set(clean(1).truth) == {"first_pull", "knee_pass", "second_pull", "catch"}
    assert set(power_snatch(1).truth) >= {"receive"}
    assert set(back_squat(1).truth) == {"bottom"}
    assert set(push_press(1).truth) == {"dip_turnaround", "lockout"}
    assert set(deadlift(1).truth) == {"knee_pass", "lockout"}


def test_back_squat_walkout_precedes_first_rep() -> None:
    lift = back_squat(1, walkout_seconds=2.0)
    # the first bottom truth is after the 2s walkout
    assert lift.truth["bottom"][0] > 2.0


def test_speed_compresses_timeline_and_truth_consistently() -> None:
    slow = clean(1, speed=1.0)
    fast = clean(1, speed=2.0)
    # a 2x-faster lift finishes in ~half the time, truth times scale with it
    assert fast.bar.ts()[-1] == pytest.approx(slow.bar.ts()[-1] / 2.0, rel=0.05)
    assert fast.truth["catch"][0] == pytest.approx(slow.truth["catch"][0] / 2.0, rel=0.02)
    # truth stays inside the generated series (no double-scaling drift)
    assert fast.truth["catch"][0] < fast.bar.ts()[-1]

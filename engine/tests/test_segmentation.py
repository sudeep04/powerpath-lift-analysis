"""Tests for powerpath_engine.segmentation.

Segmentation counts reps and carves the bar trajectory into one window per
rep via prominence-based peak/valley detection (see the module docstring for
why this beats a velocity-gap state machine). These tests pin the exact rep
count on every synthetic generator (the controller-validated counts), the
free-fall trim on a dumped bar, and the structural invariants the phase
detectors rely on: windows are ordered, non-overlapping, and each brackets
its own apex.
"""

from __future__ import annotations

import pytest
from synthetic import (
    back_squat,
    clean,
    deadlift,
    dumped_clean,
    power_snatch,
    push_press,
    zero_rep,
)

from powerpath_engine import registry
from powerpath_engine.segmentation import (
    MIN_REP_SEPARATION_S,
    PROMINENCE_FRACTION,
    V_FREEFALL_CMS,
    RepWindow,
    segment,
)

# Every generator runs on a 60fps grid.
EXPECTED_DT = 1.0 / 60.0

# (label, lift, movement-config key, expected rep count). Counts are the
# controller-validated exact rep counts, all with seed=1.
COUNT_CASES = [
    ("clean_5", clean(5, seed=1), "power_clean", 5),
    ("clean_1", clean(1, seed=1), "power_clean", 1),
    ("back_squat_3", back_squat(3, seed=1), "back_squat", 3),
    ("back_squat_5", back_squat(5, seed=1), "back_squat", 5),
    ("push_press_3", push_press(3, seed=1), "push_press", 3),
    ("deadlift_3", deadlift(3, seed=1), "deadlift", 3),
    ("power_snatch_2", power_snatch(2, seed=1), "power_snatch", 2),
    ("dumped_clean", dumped_clean(seed=1), "power_clean", 1),
]


def _apex_time(window: RepWindow, lift, config) -> float:
    """PTS of the bar extremum (apex) inside ``window``."""
    sl = lift.bar.slice_time(window.t_start, window.t_end)
    t = sl.ts()
    y = sl.ys()
    if config.bar_travel == "down_up":
        idx = int(y.argmin())  # squat: apex is the bottom
    else:
        idx = int(y.argmax())  # up / up_down: apex is the top
    return float(t[idx])


# ---------------------------------------------------------------------------
# module constants
# ---------------------------------------------------------------------------


def test_module_constants_have_expected_values() -> None:
    assert PROMINENCE_FRACTION == 0.6
    assert MIN_REP_SEPARATION_S == 0.5
    assert V_FREEFALL_CMS == -250.0


# ---------------------------------------------------------------------------
# exact rep counts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "lift", "config_key", "expected"),
    COUNT_CASES,
    ids=[c[0] for c in COUNT_CASES],
)
def test_segment_counts_reps_exactly(label, lift, config_key, expected) -> None:
    config = registry.get(config_key)
    windows = segment(lift.bar, config, EXPECTED_DT)
    assert len(windows) == expected


def test_zero_rep_yields_no_windows() -> None:
    # A walkout with only jitter (no rep ever crosses the prominence gate).
    lift = zero_rep(seed=1)
    windows = segment(lift.bar, registry.get("back_squat"), EXPECTED_DT)
    assert windows == []


# ---------------------------------------------------------------------------
# structural invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "lift", "config_key", "expected"),
    COUNT_CASES,
    ids=[c[0] for c in COUNT_CASES],
)
def test_windows_are_ordered_indexed_and_non_overlapping(label, lift, config_key, expected) -> None:
    config = registry.get(config_key)
    windows = segment(lift.bar, config, EXPECTED_DT)

    for i, w in enumerate(windows):
        assert w.rep_index == i
        assert w.t_start < w.t_end
    # boundaries touch at most (half-open slices), never overlap.
    for prev, curr in zip(windows, windows[1:], strict=False):
        assert prev.t_end <= curr.t_start


@pytest.mark.parametrize(
    ("label", "lift", "config_key", "expected"),
    COUNT_CASES,
    ids=[c[0] for c in COUNT_CASES],
)
def test_each_window_contains_its_apex(label, lift, config_key, expected) -> None:
    config = registry.get(config_key)
    windows = segment(lift.bar, config, EXPECTED_DT)
    for w in windows:
        apex_t = _apex_time(w, lift, config)
        assert w.t_start < apex_t < w.t_end


def test_repwindow_is_frozen() -> None:
    w = RepWindow(t_start=0.0, t_end=1.0, rep_index=0)
    with pytest.raises(AttributeError):
        w.t_start = 2.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# free-fall trim (dumped bar)
# ---------------------------------------------------------------------------


def test_dumped_clean_window_trimmed_before_freefall_crash() -> None:
    lift = dumped_clean(seed=1)
    windows = segment(lift.bar, registry.get("power_clean"), EXPECTED_DT)
    assert len(windows) == 1
    dump_start = lift.truth["dump_start"][0]
    # The single window must stop at (not deep past) the dump onset: it may
    # not run into the free-fall crash to the floor.
    assert windows[0].t_end < dump_start + 0.1


def test_controlled_fast_descent_is_not_trimmed_as_freefall() -> None:
    # The push press drops quickly from the overhead lockout (peak velocity
    # briefly steeper than V_FREEFALL) but recovers into the next dip -- it is
    # NOT a dumped bar, so windows stay contiguous (no spurious trim).
    lift = push_press(3, seed=1)
    windows = segment(lift.bar, registry.get("push_press"), EXPECTED_DT)
    assert len(windows) == 3
    for prev, curr in zip(windows, windows[1:], strict=False):
        assert prev.t_end == pytest.approx(curr.t_start)

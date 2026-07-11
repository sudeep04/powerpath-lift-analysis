"""Tests for powerpath_engine.geometry.

geometry.py is the sole owner of coordinate conversions in the codebase:
px<->cm scaling for the bar and body planes, the image-y (y-down) to
biomechanics-y (y-up) flip, and joint-angle helpers.
"""

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from powerpath_engine.geometry import (
    PlaneScale,
    bar_plane_scale_from_plate,
    body_plane_scale_from_height,
    from_y_up,
    horizontal_deviation_cm,
    joint_angle,
    to_y_up,
)

# ---------------------------------------------------------------------------
# PlaneScale
# ---------------------------------------------------------------------------


def test_plane_scale_px_to_cm() -> None:
    scale = PlaneScale(cm_per_px=0.5)
    assert scale.px_to_cm(10.0) == pytest.approx(5.0)


def test_plane_scale_cm_to_px() -> None:
    scale = PlaneScale(cm_per_px=0.5)
    assert scale.cm_to_px(5.0) == pytest.approx(10.0)


def test_plane_scale_rejects_non_positive_scale() -> None:
    with pytest.raises(ValueError):
        PlaneScale(cm_per_px=0.0)
    with pytest.raises(ValueError):
        PlaneScale(cm_per_px=-1.0)


@given(
    cm_per_px=st.floats(min_value=0.01, max_value=1.0),
    px=st.floats(min_value=0.0, max_value=1e5),
)
def test_plane_scale_px_round_trip(cm_per_px: float, px: float) -> None:
    """px -> cm -> px is identity for an arbitrary PlaneScale."""
    scale = PlaneScale(cm_per_px=cm_per_px)
    assert scale.cm_to_px(scale.px_to_cm(px)) == pytest.approx(px, rel=1e-9, abs=1e-9)


@given(plate_diameter_px=st.floats(min_value=1.0, max_value=1e4))
def test_bar_plane_round_trip(plate_diameter_px: float) -> None:
    """px -> cm -> px round trip through a bar-plane scale derived from a plate."""
    scale = bar_plane_scale_from_plate(plate_diameter_px)
    px = plate_diameter_px
    assert scale.cm_to_px(scale.px_to_cm(px)) == pytest.approx(px, rel=1e-9, abs=1e-9)


@given(
    athlete_height_cm=st.floats(min_value=100.0, max_value=220.0),
    athlete_height_px=st.floats(min_value=1.0, max_value=1e4),
)
def test_body_plane_round_trip(athlete_height_cm: float, athlete_height_px: float) -> None:
    """px -> cm -> px round trip through a body-plane scale derived from height."""
    scale = body_plane_scale_from_height(athlete_height_cm, athlete_height_px)
    px = athlete_height_px
    assert scale.cm_to_px(scale.px_to_cm(px)) == pytest.approx(px, rel=1e-9, abs=1e-9)


def test_bar_plane_scale_from_plate_sanity() -> None:
    """A 450px plate (450mm standard plate) implies 1.0 mm/px == 0.1 cm/px."""
    scale = bar_plane_scale_from_plate(450.0)
    assert scale.cm_per_px == pytest.approx(0.1)
    assert scale.px_to_cm(450.0) == pytest.approx(45.0)


def test_bar_plane_scale_from_plate_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        bar_plane_scale_from_plate(0.0)


def test_body_plane_scale_from_height_sanity() -> None:
    scale = body_plane_scale_from_height(athlete_height_cm=180.0, athlete_height_px=900.0)
    assert scale.cm_per_px == pytest.approx(0.2)
    assert scale.px_to_cm(900.0) == pytest.approx(180.0)


def test_body_plane_scale_from_height_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        body_plane_scale_from_height(athlete_height_cm=180.0, athlete_height_px=0.0)
    with pytest.raises(ValueError):
        body_plane_scale_from_height(athlete_height_cm=0.0, athlete_height_px=900.0)


# ---------------------------------------------------------------------------
# y-up / y-down conversion
# ---------------------------------------------------------------------------


def test_to_y_up_flips_around_frame_height() -> None:
    assert to_y_up(0.0, frame_height_px=1080) == pytest.approx(1080.0)
    assert to_y_up(1080.0, frame_height_px=1080) == pytest.approx(0.0)
    assert to_y_up(540.0, frame_height_px=1080) == pytest.approx(540.0)


def test_from_y_up_is_the_inverse() -> None:
    y_px = 300.0
    frame_height_px = 1080
    y_up = to_y_up(y_px, frame_height_px)
    assert from_y_up(y_up, frame_height_px) == pytest.approx(y_px)


@given(
    frame_height_px=st.integers(min_value=1, max_value=10_000),
    y_px=st.floats(min_value=-1e5, max_value=1e5, allow_nan=False, allow_infinity=False),
)
def test_to_y_up_is_an_involution(frame_height_px: int, y_px: float) -> None:
    """Applying the y-flip twice with the same frame height returns the original value."""
    once = to_y_up(y_px, frame_height_px)
    twice = to_y_up(once, frame_height_px)
    assert twice == pytest.approx(y_px, rel=1e-9, abs=1e-6)


# ---------------------------------------------------------------------------
# joint_angle
# ---------------------------------------------------------------------------


def test_joint_angle_right_angle() -> None:
    a = (0.0, 1.0)
    b = (0.0, 0.0)
    c = (1.0, 0.0)
    assert joint_angle(a, b, c) == pytest.approx(90.0)


def test_joint_angle_straight() -> None:
    a = (-1.0, 0.0)
    b = (0.0, 0.0)
    c = (1.0, 0.0)
    assert joint_angle(a, b, c) == pytest.approx(180.0)


def test_joint_angle_known_45_degrees() -> None:
    a = (1.0, 0.0)
    b = (0.0, 0.0)
    c = (1.0, 1.0)
    assert joint_angle(a, b, c) == pytest.approx(45.0)


def test_joint_angle_collinear_folded_is_stable_zero() -> None:
    """a and c on the same side of b (folded/overlapping): stable 0 degrees, no NaN."""
    a = (1.0, 0.0)
    b = (0.0, 0.0)
    c = (2.0, 0.0)
    assert joint_angle(a, b, c) == pytest.approx(0.0, abs=1e-9)


def test_joint_angle_coincident_with_vertex_raises() -> None:
    with pytest.raises(ValueError):
        joint_angle((0.0, 0.0), (0.0, 0.0), (1.0, 0.0))
    with pytest.raises(ValueError):
        joint_angle((0.0, 0.0), (1.0, 0.0), (1.0, 0.0))


def test_joint_angle_never_nan_or_infinite() -> None:
    """Sanity: no combination in these degenerate-adjacent cases produces NaN/inf."""
    for a, b, c in [
        ((0.0, 1.0), (0.0, 0.0), (1.0, 0.0)),
        ((-1.0, 0.0), (0.0, 0.0), (1.0, 0.0)),
        ((1.0, 0.0), (0.0, 0.0), (2.0, 0.0)),
    ]:
        result = joint_angle(a, b, c)
        assert math.isfinite(result)


# ---------------------------------------------------------------------------
# horizontal_deviation_cm
# ---------------------------------------------------------------------------


def test_horizontal_deviation_cm_signed_from_first_sample() -> None:
    scale = PlaneScale(cm_per_px=0.1)
    x_px_series = [100.0, 110.0, 90.0, 100.0]
    result = horizontal_deviation_cm(x_px_series, scale)
    assert result == pytest.approx([0.0, 1.0, -1.0, 0.0])


def test_horizontal_deviation_cm_empty_series() -> None:
    scale = PlaneScale(cm_per_px=0.1)
    assert horizontal_deviation_cm([], scale) == []


def test_horizontal_deviation_cm_single_sample_is_zero() -> None:
    scale = PlaneScale(cm_per_px=0.1)
    assert horizontal_deviation_cm([123.4], scale) == pytest.approx([0.0])

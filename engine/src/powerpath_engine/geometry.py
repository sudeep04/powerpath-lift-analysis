"""Coordinate conversions for PowerPath.

This module is the SOLE owner of coordinate conversions in the codebase:
pixel<->centimeter scaling for the bar plane and the body plane, the
image-space (y-down) to biomechanics-space (y-up) flip, and joint-angle
math. No other module should touch raw scale factors directly -- convert
through :class:`PlaneScale` and the helpers below.

Two-plane calibration: the bar plane is calibrated from a 450mm standard
plate (or a manual/date fallback, added in a later task); the body plane
is calibrated from athlete height. The two planes are not directly
comparable in raw cm (camera perspective differs between them) -- express
bar-vs-body relationships as angles or ratios, never raw cross-plane cm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Standard bumper/competition plate diameter, in cm. Used to derive the
# bar-plane px->cm scale from a detected plate radius/diameter in pixels.
STANDARD_PLATE_DIAMETER_CM = 45.0


@dataclass(frozen=True)
class PlaneScale:
    """A linear pixel<->centimeter scale factor for one image plane.

    A "plane" here is whatever depth the pixels were measured at (the bar
    plane or the body plane) -- the two are calibrated independently and
    are not interchangeable.
    """

    cm_per_px: float

    def __post_init__(self) -> None:
        if not (self.cm_per_px > 0.0) or not math.isfinite(self.cm_per_px):
            raise ValueError(f"cm_per_px must be a positive finite number, got {self.cm_per_px!r}")

    def px_to_cm(self, px: float) -> float:
        """Convert a pixel length/distance to centimeters."""
        return px * self.cm_per_px

    def cm_to_px(self, cm: float) -> float:
        """Convert a centimeter length/distance to pixels."""
        return cm / self.cm_per_px


def bar_plane_scale_from_plate(plate_diameter_px: float) -> PlaneScale:
    """Derive the bar-plane scale from a detected plate diameter in pixels.

    Uses the 450mm (45.0cm) standard competition plate diameter as the
    real-world reference.
    """
    if not (plate_diameter_px > 0.0) or not math.isfinite(plate_diameter_px):
        raise ValueError(
            f"plate_diameter_px must be a positive finite number, got {plate_diameter_px!r}"
        )
    return PlaneScale(cm_per_px=STANDARD_PLATE_DIAMETER_CM / plate_diameter_px)


def body_plane_scale_from_height(athlete_height_cm: float, athlete_height_px: float) -> PlaneScale:
    """Derive the body-plane scale from athlete height in cm and in pixels."""
    if not (athlete_height_cm > 0.0) or not math.isfinite(athlete_height_cm):
        raise ValueError(
            f"athlete_height_cm must be a positive finite number, got {athlete_height_cm!r}"
        )
    if not (athlete_height_px > 0.0) or not math.isfinite(athlete_height_px):
        raise ValueError(
            f"athlete_height_px must be a positive finite number, got {athlete_height_px!r}"
        )
    return PlaneScale(cm_per_px=athlete_height_cm / athlete_height_px)


def to_y_up(y_px: float, frame_height_px: int) -> float:
    """Flip an image-space (y-down) y coordinate to biomechanics-space (y-up).

    Image space has y growing downward from the top of the frame; all
    biomechanics math (bar height, joint positions, etc.) uses y growing
    upward. The flip is a reflection around the frame height, so it is
    its own inverse: see :func:`from_y_up`.
    """
    return frame_height_px - y_px


def from_y_up(y_up: float, frame_height_px: int) -> float:
    """Inverse of :func:`to_y_up`: biomechanics-space (y-up) back to image-space (y-down)."""
    return frame_height_px - y_up


def joint_angle(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> float:
    """Interior angle at vertex ``b``, in degrees, formed by rays b->a and b->c.

    Returns a value in [0, 180]. Uses ``atan2(|cross|, dot)`` rather than
    ``acos(dot / (|v1| |v2|))`` so it stays numerically stable (no domain
    errors / NaN from floating-point drift past +/-1) for collinear input:
    a straight joint (a and c on opposite sides of b) is stable at 180.0,
    and a fully folded joint (a and c on the same side of b) is stable at
    0.0. Raises ValueError only when ``b`` coincides with ``a`` or ``c``,
    since the angle is undefined for a zero-length ray.
    """
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    if v1 == (0.0, 0.0) or v2 == (0.0, 0.0):
        raise ValueError("joint_angle is undefined when b coincides with a or c")

    dot = v1[0] * v2[0] + v1[1] * v2[1]
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    return math.degrees(math.atan2(abs(cross), dot))


def horizontal_deviation_cm(x_px_series: list[float], scale: PlaneScale) -> list[float]:
    """Signed horizontal deviation, in cm, of each sample from the first sample.

    Positive values are to the right of the first sample (in pixel-x
    terms), negative to the left. Used for bar-path drift analysis.
    """
    if not x_px_series:
        return []
    x0 = x_px_series[0]
    return [scale.px_to_cm(x - x0) for x in x_px_series]

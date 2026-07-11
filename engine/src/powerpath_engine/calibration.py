"""Bar-plane scale calibration from a visible 450mm plate.

Two-plane calibration (global constraint): the BAR plane is calibrated
here, from the standard 450mm competition/bumper plate detected via Hough
circles in the first frames of the video; the body plane is calibrated
separately from athlete height. Calibration runs per video because
stations change the bar plane mid-session.

The never-silently-wrong-scale rule shapes everything in this module. A
Hough circle can easily be the WRONG circle -- a plate on a background
rack, a wall clock -- and a wrong scale poisons every downstream cm
metric while looking perfectly healthy. So a plate-derived scale is only
accepted when it (a) sits inside the plausible band PLAUSIBLE_MM_PER_PX
for a phone filming a lift from a few meters, and (b) agrees within
MAX_MARKER_DISAGREEMENT with the independent 50mm-sleeve estimate from the
tracked marker diameter, when one is available. Anything else walks the
fallback ladder plate -> date_fallback -> manual, each step carrying a
human-readable warning, and running out of fallbacks raises
CalibrationError with that same reason -- never a silent guess.

Scale math itself lives in geometry.py (the sole owner): this module only
builds scales through geometry's helpers and compares their magnitudes for
plausibility; it never converts measurements with raw factors.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Literal

import cv2
import numpy as np

from powerpath_engine.geometry import (
    PlaneScale,
    bar_plane_scale_from_plate,
    bar_plane_scale_from_sleeve,
)

# Plausible bar-plane resolution band for real footage, in mm per pixel.
# 0.5 mm/px is a 450mm plate filling ~900px (phone unusually close /
# high-res); 3.0 mm/px is the plate down at ~150px (about as far as a
# usable lift video gets). A "plate" outside this band is not the working
# bar's plate.
PLAUSIBLE_MM_PER_PX = (0.5, 3.0)

# Maximum tolerated relative disagreement between the plate scale and the
# marker-derived 50mm sleeve scale before the plate detection is distrusted.
MAX_MARKER_DISAGREEMENT = 0.20

# Hough runs on a frame downscaled to (at most) this height: plate-sized
# structure survives easily, the accumulator gets ~5x cheaper for 1080p
# input, and small high-frequency clutter drops out.
_HOUGH_TARGET_HEIGHT = 480

# The plate circle must span 15-45% of frame height (brief threshold) --
# smaller is background clutter, larger cannot be a whole in-frame plate.
_HOUGH_MIN_RADIUS_FRAC = 0.15
_HOUGH_MAX_RADIUS_FRAC = 0.45

# Classic HOUGH_GRADIENT parameters, tuned on synthetic plates at both
# extremes of the radius band: dp=1.5 (coarser accumulator gathers enough
# votes for small circles that dp=1 misses), Canny high threshold 120,
# accumulator threshold 30.
_HOUGH_DP = 1.5
_HOUGH_CANNY_HIGH = 120.0
_HOUGH_ACCUMULATOR = 30.0
_MEDIAN_BLUR_KSIZE = 5


class CalibrationError(Exception):
    """No trustworthy bar-plane scale could be produced.

    Raised only after the whole fallback ladder (plate -> date_fallback ->
    manual) is exhausted; the message preserves the human-readable reason
    the plate scale was rejected or not found.
    """


@dataclass(frozen=True)
class CalibrationResult:
    """The bar-plane scale plus where it came from.

    `warning` is None only for a clean plate calibration; every fallback
    carries the human-readable reason the plate path failed, so the UI can
    surface it (never-silently-wrong-scale rule).
    """

    bar_scale: PlaneScale
    source: Literal["plate", "date_fallback", "manual"]
    warning: str | None


def detect_plate_circle(image_bgr: np.ndarray) -> tuple[tuple[float, float], float] | None:
    """Find the plate circle in one BGR frame: ((cx, cy), radius) px, or None.

    Hough circles on a downscaled grayscale copy; among the circles whose
    radius is 15-45% of frame height (enforced via min/maxRadius -- the
    fraction is invariant under the uniform downscale), the largest is
    returned, mapped back to full-resolution pixel coordinates.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    scale = 1.0
    if gray.shape[0] > _HOUGH_TARGET_HEIGHT:
        scale = _HOUGH_TARGET_HEIGHT / gray.shape[0]
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.medianBlur(gray, _MEDIAN_BLUR_KSIZE)

    height = gray.shape[0]
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=_HOUGH_DP,
        minDist=float(height),
        param1=_HOUGH_CANNY_HIGH,
        param2=_HOUGH_ACCUMULATOR,
        minRadius=int(_HOUGH_MIN_RADIUS_FRAC * height),
        maxRadius=int(_HOUGH_MAX_RADIUS_FRAC * height),
    )
    if circles is None:
        return None

    cx, cy, radius = max(circles[0], key=lambda circle: circle[2])
    return ((float(cx) / scale, float(cy) / scale), float(radius) / scale)


def calibrate(
    first_frames: list[np.ndarray],
    date_fallback: PlaneScale | None = None,
    manual: PlaneScale | None = None,
    marker_diameter_px: float | None = None,
) -> CalibrationResult:
    """Produce the bar-plane scale for a video, or raise CalibrationError.

    `first_frames` is a small batch from the start of the video (the
    pipeline grabs ~30): plate detection runs on each and the MEDIAN radius
    of the successful detections is used, so a single spurious circle in
    one frame cannot set the scale. The plate scale must pass the
    PLAUSIBLE_MM_PER_PX band and, when `marker_diameter_px` is given, agree
    within MAX_MARKER_DISAGREEMENT with the 50mm-sleeve marker estimate.
    On rejection, falls back to `date_fallback` (a same-day accepted
    calibration), then `manual`, attaching the rejection reason as the
    warning; with no fallback available, raises CalibrationError carrying
    the same reason.
    """
    plate_scale, reason = _plate_scale(first_frames, marker_diameter_px)
    if plate_scale is not None:
        return CalibrationResult(bar_scale=plate_scale, source="plate", warning=None)
    if date_fallback is not None:
        return CalibrationResult(
            bar_scale=date_fallback,
            source="date_fallback",
            warning=f"{reason}; using same-date fallback scale",
        )
    if manual is not None:
        return CalibrationResult(
            bar_scale=manual,
            source="manual",
            warning=f"{reason}; using manual scale",
        )
    raise CalibrationError(f"{reason}; no date-fallback or manual scale available")


def _plate_scale(
    first_frames: list[np.ndarray],
    marker_diameter_px: float | None,
) -> tuple[PlaneScale | None, str | None]:
    """The validated plate scale, or (None, human-readable rejection reason)."""
    radii = []
    for frame in first_frames:
        found = detect_plate_circle(frame)
        if found is not None:
            radii.append(found[1])
    if not radii:
        return None, f"no plate circle detected in {len(first_frames)} calibration frame(s)"

    median_radius = median(radii)
    scale = bar_plane_scale_from_plate(2.0 * median_radius)
    mm_per_px = scale.cm_per_px * 10.0

    band_low, band_high = PLAUSIBLE_MM_PER_PX
    if not (band_low <= mm_per_px <= band_high):
        return None, (
            f"plate scale {mm_per_px:.2f} mm/px (median plate radius {median_radius:.0f}px) "
            f"is outside the plausible band {band_low}-{band_high} mm/px; the detected "
            "circle is probably not the working bar's plate"
        )

    if marker_diameter_px is not None:
        marker_mm_per_px = bar_plane_scale_from_sleeve(marker_diameter_px).cm_per_px * 10.0
        disagreement = abs(mm_per_px - marker_mm_per_px) / marker_mm_per_px
        if disagreement > MAX_MARKER_DISAGREEMENT:
            return None, (
                f"plate scale {mm_per_px:.2f} mm/px disagrees with the marker-derived "
                f"50mm-sleeve scale {marker_mm_per_px:.2f} mm/px by {disagreement:.0%} "
                f"(tolerance {MAX_MARKER_DISAGREEMENT:.0%})"
            )

    return scale, None

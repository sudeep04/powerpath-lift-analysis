"""Bar marker detection and per-frame tracking.

Bar tracking is marker-ONLY (global constraint): the user tapes a saturated
pink/magenta dot or ring, CENTERED, on the bar's end cap. Centered matters
-- sleeves spin freely during a lift, so off-center tape orbits the bar
axis by ~2.5cm and would corrupt the bar path; a centered marker stays on
the axis no matter how the sleeve rotates. There is deliberately no
fallback tracker: when the marker is not found, the frame is a miss, and
downstream gap rules (interpolate <=5 frames, else mark the rep
unanalyzed) decide what that means.

This module stays strictly per-frame: `detect_marker` looks at one image,
`MarkerTracker.feed(t, image)` consumes one frame at a time and keeps only
O(1)-per-frame state plus a few floats per detection. The pipeline (Task 8)
owns iteration, per the streaming single-pass rule -- there is no
`track(frames_iter)` here on purpose.

Coordinates: `Sample.x`/`Sample.y` are px in image space (y grows down), at
the bar plane. geometry.py is the sole owner of px->cm conversion and the
y-flip; nothing here touches scale factors.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median

import cv2
import numpy as np

from powerpath_engine.series import Sample

# Smallest blob (px, after morphological open) accepted as the marker; the
# brief's threshold. Below this the "detection" is indistinguishable from
# color noise.
MIN_BLOB_AREA_PX = 20

# Consecutive misses after which MarkerTracker abandons its search ROI and
# goes back to scanning the full frame. Matches the downstream gap rule:
# gaps of <=5 frames are interpolated, so a marker that reappears within 5
# frames is overwhelmingly likely to still be near the old ROI; past that,
# assume we genuinely lost it (occlusion, bar dropped out of frame) and
# search everywhere.
ROI_RESET_MISSES = 5

# The search ROI is the last blob's bounding box scaled up by this factor
# (about its center). 3x comfortably covers realistic inter-frame bar
# motion -- at 60fps even a 3m/s bar moves well under one marker-width per
# frame -- while cropping the HSV threshold + blob search to a tiny window.
ROI_EXPAND_FACTOR = 3.0

# A detection must read at least this visible before its blob area becomes
# the new `expected_area` reference. Below it (partial occlusion by the
# athlete's body, plates, chalk dust) the area is not a trustworthy "fully
# visible marker" size and would drag the visibility denominator down.
CONFIDENT_VISIBILITY = 0.8

# 3x3 ellipse for the morphological open that strips speckle noise from the
# HSV mask before blob extraction. Kept small so a legitimately tiny/far
# marker (down at MIN_BLOB_AREA_PX) survives the open.
_OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


@dataclass(frozen=True)
class Rect:
    """Axis-aligned pixel rectangle (top-left corner + size) in image space.

    Local to bar tracking on purpose: geometry.py owns scale/angle math
    only, and an ROI is neither.
    """

    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class MarkerSpec:
    """Inclusive HSV threshold range (OpenCV convention) for the marker tape.

    The default targets saturated pink/magenta gaffer tape. OpenCV hue runs
    over [0, 180): magenta sits at ~150 and deep pink at ~165, so the whole
    pink/magenta family fits in a single [140, 179] range that stops just
    BELOW the 180->0 hue wrap. That is a deliberate choice, not an
    oversight: crossing the wrap would require a two-range threshold whose
    only additional catch is saturated RED (hue ~178-182 wrapped) -- and
    red is the single most common strong color in a gym (plates, shirts,
    equipment), exactly what the marker must never be confused with. Users
    are told to use pink/magenta tape, not red, for the same reason.

    S/V floors of 100 reject washed-out pinkish walls and dim shadows while
    accepting real tape under normal gym lighting.
    """

    hsv_low: tuple[int, int, int] = (140, 100, 100)
    hsv_high: tuple[int, int, int] = (179, 255, 255)


@dataclass(frozen=True)
class MarkerDetection:
    """One accepted marker detection plus the blob geometry behind it.

    `sample` is what feeds the bar-path TimeSeries; `area_px` and `bbox`
    are kept so `estimate_marker_diameter_px` can turn a run of detections
    into an apparent marker diameter for the 50mm-sleeve calibration
    cross-check.
    """

    sample: Sample
    area_px: float
    bbox: Rect


@dataclass(frozen=True)
class _Blob:
    """Internal: largest above-threshold mask blob, in full-frame px."""

    cx: float
    cy: float
    area_px: float
    bbox: Rect


def detect_marker(
    image_bgr: np.ndarray,
    spec: MarkerSpec | None = None,
    roi: Rect | None = None,
    *,
    expected_area: float | None = None,
    t: float = 0.0,
) -> Sample | None:
    """Detect the marker in one BGR uint8 frame; None if it is not there.

    HSV threshold (`spec`) -> morphological open -> largest connected blob
    -> centroid. Returns None when no blob reaches MIN_BLOB_AREA_PX. With
    `roi` given, only that window is searched (the returned coordinates are
    always full-frame px regardless).

    `visibility` is blob_area / expected_area clamped to [0, 1]. Bootstrap:
    with no `expected_area` yet (first detection, or stateless one-shot
    use) there is no reference size, so any accepted blob reports
    visibility 1.0. `t` is stamped onto the returned Sample (PTS seconds;
    defaults to 0.0 for callers inspecting a lone image with no clock).
    """
    blob = _find_blob(image_bgr, spec if spec is not None else MarkerSpec(), roi)
    if blob is None:
        return None
    return Sample(t=t, x=blob.cx, y=blob.cy, visibility=_visibility(blob.area_px, expected_area))


class MarkerTracker:
    """Per-frame marker tracker with a search-ROI fast path.

    Feed frames one at a time (`feed(t, image)`); the pipeline owns
    iteration. After each hit the tracker narrows the next search to
    ROI_EXPAND_FACTOR x the blob's bbox (centered on it) for speed; after
    ROI_RESET_MISSES consecutive misses it resets to full-frame search so a
    marker reappearing anywhere is re-acquired.

    `expected_area` (the visibility denominator) tracks the last CONFIDENT
    detection's blob area: the first accepted detection bootstraps it at
    visibility 1.0, and only detections with visibility >=
    CONFIDENT_VISIBILITY update it afterwards, so a partially occluded
    marker never shrinks the reference size it is judged against.

    `detections` accumulates every accepted detection (a few floats per
    frame -- within the "only extracted time series survive the pass"
    budget) for `estimate_marker_diameter_px`.
    """

    def __init__(self, spec: MarkerSpec | None = None) -> None:
        self.spec = spec if spec is not None else MarkerSpec()
        self.roi: Rect | None = None
        self.expected_area: float | None = None
        self.consecutive_misses = 0
        self.detections: list[MarkerDetection] = []

    def feed(self, t: float, image: np.ndarray) -> Sample | None:
        """Process one frame at PTS `t` seconds; Sample on a hit, else None."""
        blob = _find_blob(image, self.spec, self.roi)
        if blob is None:
            self.consecutive_misses += 1
            if self.consecutive_misses >= ROI_RESET_MISSES:
                self.roi = None
            return None

        visibility = _visibility(blob.area_px, self.expected_area)
        sample = Sample(t=t, x=blob.cx, y=blob.cy, visibility=visibility)

        self.consecutive_misses = 0
        self.roi = _expanded_roi(blob.bbox, image.shape[0], image.shape[1])
        if visibility >= CONFIDENT_VISIBILITY:
            self.expected_area = blob.area_px
        self.detections.append(MarkerDetection(sample=sample, area_px=blob.area_px, bbox=blob.bbox))
        return sample


def estimate_marker_diameter_px(samples: Sequence[MarkerDetection]) -> float:
    """Apparent marker diameter in px, from a run of tracker detections.

    Each detection's blob area is converted to its equivalent-circle
    diameter (2 * sqrt(area / pi)); the median over all detections is
    returned, so up to half the frames can be partially occluded or noisy
    without moving the estimate. Used for the calibration sanity
    cross-check against the 50mm bar sleeve (see
    geometry.bar_plane_scale_from_sleeve). Raises ValueError on an empty
    sequence -- there is no meaningful diameter to report.
    """
    if not samples:
        raise ValueError("estimate_marker_diameter_px requires at least one detection")
    return float(median(2.0 * math.sqrt(d.area_px / math.pi) for d in samples))


def _visibility(area_px: float, expected_area: float | None) -> float:
    """blob_area / expected_area clamped to [0, 1]; 1.0 with no reference yet."""
    if expected_area is None or expected_area <= 0.0:
        return 1.0
    return min(1.0, area_px / expected_area)


def _find_blob(image_bgr: np.ndarray, spec: MarkerSpec, roi: Rect | None) -> _Blob | None:
    """HSV threshold -> open -> largest blob >= MIN_BLOB_AREA_PX, or None.

    All returned coordinates are full-frame px (the ROI offset is added
    back). An ROI that clamps to an empty window counts as a miss.
    """
    frame_h, frame_w = image_bgr.shape[:2]
    x0, y0 = 0, 0
    search = image_bgr
    if roi is not None:
        x0 = min(max(roi.x, 0), frame_w)
        y0 = min(max(roi.y, 0), frame_h)
        x1 = min(max(roi.x + roi.w, 0), frame_w)
        y1 = min(max(roi.y + roi.h, 0), frame_h)
        if x1 <= x0 or y1 <= y0:
            return None
        search = image_bgr[y0:y1, x0:x1]

    hsv = cv2.cvtColor(search, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, spec.hsv_low, spec.hsv_high)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _OPEN_KERNEL)

    n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)
    if n_labels <= 1:  # label 0 is background
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    area = float(stats[best, cv2.CC_STAT_AREA])
    if area < MIN_BLOB_AREA_PX:
        return None

    return _Blob(
        cx=float(centroids[best][0]) + x0,
        cy=float(centroids[best][1]) + y0,
        area_px=area,
        bbox=Rect(
            x=int(stats[best, cv2.CC_STAT_LEFT]) + x0,
            y=int(stats[best, cv2.CC_STAT_TOP]) + y0,
            w=int(stats[best, cv2.CC_STAT_WIDTH]),
            h=int(stats[best, cv2.CC_STAT_HEIGHT]),
        ),
    )


def _expanded_roi(bbox: Rect, frame_height: int, frame_width: int) -> Rect:
    """`bbox` scaled by ROI_EXPAND_FACTOR about its center, clamped to frame."""
    cx = bbox.x + bbox.w / 2.0
    cy = bbox.y + bbox.h / 2.0
    half_w = bbox.w * ROI_EXPAND_FACTOR / 2.0
    half_h = bbox.h * ROI_EXPAND_FACTOR / 2.0
    x0 = max(0, int(round(cx - half_w)))
    y0 = max(0, int(round(cy - half_h)))
    x1 = min(frame_width, int(round(cx + half_w)))
    y1 = min(frame_height, int(round(cy + half_h)))
    return Rect(x=x0, y=y0, w=x1 - x0, h=y1 - y0)

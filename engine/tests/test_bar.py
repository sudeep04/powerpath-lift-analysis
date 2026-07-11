"""Tests for powerpath_engine.bar.

Bar tracking is marker-ONLY (global constraint): a saturated pink/magenta
dot/ring taped to the center of the bar's end cap. These tests draw
synthetic markers with OpenCV -- no real footage, no model inference.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from powerpath_engine.bar import (
    MIN_BLOB_AREA_PX,
    ROI_RESET_MISSES,
    MarkerDetection,
    MarkerSpec,
    MarkerTracker,
    Rect,
    detect_marker,
    estimate_marker_diameter_px,
)
from powerpath_engine.series import Sample

# BGR for saturated magenta tape (OpenCV HSV hue 150, S=V=255).
MAGENTA_BGR = (255, 0, 255)
GRAY_LEVEL = 128


def gray_frame(width: int = 640, height: int = 480) -> np.ndarray:
    return np.full((height, width, 3), GRAY_LEVEL, dtype=np.uint8)


def marker_frame(
    center: tuple[int, int],
    radius: int = 12,
    width: int = 640,
    height: int = 480,
) -> np.ndarray:
    frame = gray_frame(width, height)
    cv2.circle(frame, center, radius, MAGENTA_BGR, -1)
    return frame


# ---------------------------------------------------------------------------
# detect_marker
# ---------------------------------------------------------------------------


def test_detect_marker_centroid_on_clean_frame() -> None:
    frame = marker_frame((150, 100))
    sample = detect_marker(frame, MarkerSpec())
    assert sample is not None
    assert sample.x == pytest.approx(150.0, abs=0.5)
    assert sample.y == pytest.approx(100.0, abs=0.5)


def test_detect_marker_bootstrap_visibility_is_one() -> None:
    """First/stateless detection: no expected_area yet, so visibility
    bootstraps to 1.0 for any blob at or above the minimum area."""
    sample = detect_marker(marker_frame((150, 100)), MarkerSpec())
    assert sample is not None
    assert sample.visibility == 1.0


def test_detect_marker_centroid_within_1px_under_noise_and_motion_blur() -> None:
    """The brief's named accuracy test: gaussian noise + 9x1 motion blur
    (cv2.blur) must not pull the detected centroid more than 1px off."""
    frame = marker_frame((320, 240)).astype(np.int16)
    rng = np.random.default_rng(42)
    frame = np.clip(frame + rng.normal(0.0, 12.0, frame.shape), 0, 255).astype(np.uint8)
    frame = cv2.blur(frame, (9, 1))

    sample = detect_marker(frame, MarkerSpec())
    assert sample is not None
    assert sample.x == pytest.approx(320.0, abs=1.0)
    assert sample.y == pytest.approx(240.0, abs=1.0)


def test_detect_marker_none_when_no_marker() -> None:
    assert detect_marker(gray_frame(), MarkerSpec()) is None


def test_detect_marker_none_when_blob_below_min_area() -> None:
    """A radius-2 dot rasterizes to ~13px, under the 20px minimum."""
    frame = marker_frame((150, 100), radius=2)
    assert detect_marker(frame, MarkerSpec()) is None


def test_detect_marker_accepts_small_blob_at_or_above_min_area() -> None:
    frame = marker_frame((150, 100), radius=4)
    sample = detect_marker(frame, MarkerSpec())
    assert sample is not None
    assert sample.x == pytest.approx(150.0, abs=1.0)


def test_detect_marker_picks_largest_blob() -> None:
    frame = marker_frame((400, 300), radius=12)
    cv2.circle(frame, (100, 100), 6, MAGENTA_BGR, -1)
    sample = detect_marker(frame, MarkerSpec())
    assert sample is not None
    assert sample.x == pytest.approx(400.0, abs=0.5)
    assert sample.y == pytest.approx(300.0, abs=0.5)


def test_detect_marker_roi_restricts_search_and_maps_back_to_full_frame() -> None:
    """With an ROI around the small dot, the bigger dot outside the ROI is
    invisible, and the returned centroid is in FULL-frame coordinates."""
    frame = marker_frame((400, 300), radius=12)
    cv2.circle(frame, (100, 100), 6, MAGENTA_BGR, -1)
    roi = Rect(x=70, y=70, w=60, h=60)
    sample = detect_marker(frame, MarkerSpec(), roi)
    assert sample is not None
    assert sample.x == pytest.approx(100.0, abs=0.5)
    assert sample.y == pytest.approx(100.0, abs=0.5)


def test_detect_marker_default_spec_rejects_red() -> None:
    """The default hue range deliberately stops below the 180/0 red wrap:
    a saturated red blob (hue ~0) must NOT match the magenta marker spec."""
    frame = gray_frame()
    cv2.circle(frame, (150, 100), 12, (0, 0, 255), -1)  # BGR red
    assert detect_marker(frame, MarkerSpec()) is None


def test_detect_marker_visibility_is_area_ratio_clamped() -> None:
    full = detect_marker(marker_frame((150, 100)), MarkerSpec())
    assert full is not None

    # Occlude the left half of the marker: visibility ~ 0.5.
    occluded = marker_frame((150, 100))
    cv2.rectangle(occluded, (150 - 14, 100 - 14), (150, 100 + 14), (GRAY_LEVEL,) * 3, -1)
    half = detect_marker(occluded, MarkerSpec(), expected_area=441.0)
    assert half is not None
    assert 0.3 <= half.visibility <= 0.65

    # A blob larger than expected clamps to 1.0, never exceeds it.
    grown = detect_marker(marker_frame((150, 100), radius=16), MarkerSpec(), expected_area=441.0)
    assert grown is not None
    assert grown.visibility == 1.0


# ---------------------------------------------------------------------------
# MarkerTracker
# ---------------------------------------------------------------------------


def test_tracker_follows_dot_moving_15px_per_frame() -> None:
    """The brief's named tracking test: a dot moving 15px/frame stays inside
    the 3x-bbox search ROI and is hit on every frame."""
    tracker = MarkerTracker()
    for i in range(20):
        x = 60 + 15 * i
        t = i / 30.0
        sample = tracker.feed(t, marker_frame((x, 240)))
        assert sample is not None, f"lost the marker at frame {i}"
        assert sample.t == t
        assert sample.x == pytest.approx(float(x), abs=1.0)
        assert sample.y == pytest.approx(240.0, abs=1.0)
        # The ROI fast path must actually engage after the first hit.
        assert tracker.roi is not None


def test_tracker_roi_is_3x_blob_bbox_centered_on_hit() -> None:
    tracker = MarkerTracker()
    sample = tracker.feed(0.0, marker_frame((320, 240), radius=12))
    assert sample is not None
    roi = tracker.roi
    assert roi is not None
    # A radius-12 rasterized dot has a 25x25 bbox -> 75x75 ROI.
    assert roi.w == pytest.approx(75, abs=2)
    assert roi.h == pytest.approx(75, abs=2)
    assert roi.x + roi.w / 2.0 == pytest.approx(320.0, abs=1.5)
    assert roi.y + roi.h / 2.0 == pytest.approx(240.0, abs=1.5)


def test_tracker_roi_restricts_search_between_resets() -> None:
    """While the ROI is live (fewer than 5 misses), a dot far outside it is
    deliberately NOT found -- that is the marker-only, ROI-restricted
    contract, not a bug."""
    tracker = MarkerTracker()
    assert tracker.feed(0.0, marker_frame((100, 240))) is not None
    assert tracker.feed(1 / 30, marker_frame((540, 400))) is None


def test_tracker_recovers_after_5_blank_frames() -> None:
    """The brief's named recovery test: after 5 consecutive misses the ROI
    resets to full frame, so a marker reappearing anywhere is re-acquired."""
    tracker = MarkerTracker()
    for i in range(3):
        assert tracker.feed(i / 30.0, marker_frame((100, 240))) is not None

    blank = gray_frame()
    for i in range(ROI_RESET_MISSES):
        assert tracker.feed((3 + i) / 30.0, blank) is None
        if i < ROI_RESET_MISSES - 1:
            assert tracker.roi is not None, "ROI must persist until the 5th miss"
    assert tracker.roi is None, "ROI must reset to full frame after 5 misses"

    reacquired = tracker.feed(9 / 30.0, marker_frame((540, 400)))
    assert reacquired is not None
    assert reacquired.x == pytest.approx(540.0, abs=1.0)
    assert reacquired.y == pytest.approx(400.0, abs=1.0)


def test_tracker_visibility_ratio_against_last_confident_area() -> None:
    tracker = MarkerTracker()
    first = tracker.feed(0.0, marker_frame((150, 100)))
    assert first is not None
    assert first.visibility == 1.0  # bootstrap: no expected_area yet

    full_area = tracker.expected_area
    assert full_area is not None

    occluded = marker_frame((150, 100))
    cv2.rectangle(occluded, (150 - 14, 100 - 14), (150, 100 + 14), (GRAY_LEVEL,) * 3, -1)
    half = tracker.feed(1 / 30, occluded)
    assert half is not None
    assert 0.3 <= half.visibility <= 0.65

    # The unconfident (occluded) detection must NOT drag expected_area down.
    # (Asserting on visibility alone could not catch that: a halved
    # denominator would clamp a full marker right back to 1.0.)
    assert tracker.expected_area == full_area

    # The next (nearly) full marker reads as fully visible again; the 3x ROI
    # from the occluded half-disc may clip a pixel column, hence >= 0.95
    # rather than == 1.0.
    full_again = tracker.feed(2 / 30, marker_frame((150, 100)))
    assert full_again is not None
    assert full_again.visibility >= 0.95


# ---------------------------------------------------------------------------
# estimate_marker_diameter_px
# ---------------------------------------------------------------------------


def _detection(area_px: float) -> MarkerDetection:
    return MarkerDetection(
        sample=Sample(t=0.0, x=0.0, y=0.0, visibility=1.0),
        area_px=area_px,
        bbox=Rect(x=0, y=0, w=1, h=1),
    )


def test_estimate_marker_diameter_px_is_median_equivalent_diameter() -> None:
    areas = [math.pi * (d / 2.0) ** 2 for d in (20.0, 22.0, 24.0)]
    detections = [_detection(a) for a in areas]
    assert estimate_marker_diameter_px(detections) == pytest.approx(22.0)


def test_estimate_marker_diameter_px_empty_raises() -> None:
    with pytest.raises(ValueError):
        estimate_marker_diameter_px([])


def test_tracker_detections_feed_diameter_estimate() -> None:
    tracker = MarkerTracker()
    for i in range(5):
        assert tracker.feed(i / 30.0, marker_frame((150 + 5 * i, 100))) is not None
    # A rasterized radius-12 dot covers ~441px -> equivalent diameter ~23.7.
    diameter = estimate_marker_diameter_px(tracker.detections)
    assert diameter == pytest.approx(2.0 * math.sqrt(441.0 / math.pi), abs=1.5)


# ---------------------------------------------------------------------------
# constants sanity
# ---------------------------------------------------------------------------


def test_min_blob_area_matches_brief() -> None:
    assert MIN_BLOB_AREA_PX == 20


def test_roi_reset_misses_matches_brief() -> None:
    assert ROI_RESET_MISSES == 5

"""Tests for powerpath_engine.calibration.

Bar-plane calibration from a visible 450mm plate (Hough circles), with the
sanity band (0.5-3.0 mm/px), the 50mm-sleeve marker cross-check, and the
plate -> date_fallback -> manual -> CalibrationError fallback ladder. The
never-silently-wrong-scale rule: every non-plate result carries a
human-readable warning, and running out of fallbacks raises.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from powerpath_engine.calibration import (
    MAX_MARKER_DISAGREEMENT,
    PLAUSIBLE_MM_PER_PX,
    CalibrationError,
    CalibrationResult,
    calibrate,
    detect_plate_circle,
)
from powerpath_engine.geometry import PlaneScale

BG_LEVEL = 180
PLATE_LEVEL = 40


def plate_frame(
    center: tuple[int, int],
    radius: int,
    width: int = 1920,
    height: int = 1080,
) -> np.ndarray:
    frame = np.full((height, width, 3), BG_LEVEL, dtype=np.uint8)
    cv2.circle(frame, center, radius, (PLATE_LEVEL,) * 3, -1)
    return frame


def blank_frame(width: int = 1920, height: int = 1080) -> np.ndarray:
    return np.full((height, width, 3), BG_LEVEL, dtype=np.uint8)


# ---------------------------------------------------------------------------
# detect_plate_circle
# ---------------------------------------------------------------------------


def test_detect_plate_circle_finds_450px_circle_within_2pct() -> None:
    """The brief's named accuracy test: a drawn 450px-radius plate must be
    recovered within 2% radius."""
    found = detect_plate_circle(plate_frame((960, 560), 450))
    assert found is not None
    (cx, cy), radius = found
    assert radius == pytest.approx(450.0, rel=0.02)
    assert cx == pytest.approx(960.0, abs=10.0)
    assert cy == pytest.approx(560.0, abs=10.0)


def test_detect_plate_circle_none_on_blank_frame() -> None:
    assert detect_plate_circle(blank_frame()) is None


def test_detect_plate_circle_rejects_radius_below_15pct_of_height() -> None:
    """A 100px-radius circle in a 1080-high frame (9.3%) is below the
    15-45%-of-frame-height plate band and must not be reported."""
    assert detect_plate_circle(plate_frame((960, 560), 100)) is None


# ---------------------------------------------------------------------------
# calibrate: plate happy path
# ---------------------------------------------------------------------------


def test_calibrate_from_plate_happy_path() -> None:
    frames = [plate_frame((960, 560), 300) for _ in range(5)]
    result = calibrate(frames, date_fallback=None, manual=None, marker_diameter_px=None)
    assert isinstance(result, CalibrationResult)
    assert result.source == "plate"
    assert result.warning is None
    # 450mm plate spanning ~600px -> 0.075 cm/px (0.75 mm/px).
    assert result.bar_scale.cm_per_px == pytest.approx(0.075, rel=0.03)


def test_calibrate_aggregates_median_radius_and_tolerates_missed_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """calibrate must aggregate across frames (median of successful
    detections), not trust a single frame: one wild outlier detection among
    several good ones must not move the scale."""
    import powerpath_engine.calibration as calibration_module

    radii = iter([300.0, 300.0, 300.0, 60.0, None, 302.0])
    monkeypatch.setattr(
        calibration_module,
        "detect_plate_circle",
        lambda image_bgr: None if (r := next(radii)) is None else ((960.0, 560.0), r),
    )
    frames = [blank_frame(4, 4) for _ in range(6)]
    result = calibrate(frames, date_fallback=None, manual=None, marker_diameter_px=None)
    assert result.source == "plate"
    # median of [300, 300, 300, 60, 302] = 300 -> 45cm / 600px.
    assert result.bar_scale.cm_per_px == pytest.approx(45.0 / 600.0)


# ---------------------------------------------------------------------------
# calibrate: sanity band (never silently wrong)
# ---------------------------------------------------------------------------


def test_calibrate_rejects_out_of_band_background_plate_with_warning() -> None:
    """The brief's named rejection test: a 100px 'plate' (a background/rack
    plate far behind the bar) implies 4.5 mm/px -- outside the 0.5-3.0
    plausible band -- so calibrate must fall back with a warning, never
    silently return the wrong scale."""
    # 50px radius in a 240-high frame is 21% of frame height: it passes the
    # Hough geometric filter, so the mm/px sanity band is what rejects it.
    frames = [plate_frame((213, 120), 50, width=426, height=240) for _ in range(3)]
    fallback = PlaneScale(cm_per_px=0.1)

    result = calibrate(frames, date_fallback=fallback, manual=None, marker_diameter_px=None)
    assert result.source == "date_fallback"
    assert result.bar_scale is fallback
    assert result.warning is not None
    assert "outside the plausible band" in result.warning
    assert "mm/px" in result.warning


def test_calibrate_falls_back_with_warning_when_no_plate_detected() -> None:
    frames = [blank_frame() for _ in range(3)]
    fallback = PlaneScale(cm_per_px=0.1)
    result = calibrate(frames, date_fallback=fallback, manual=None, marker_diameter_px=None)
    assert result.source == "date_fallback"
    assert result.warning is not None
    assert "no plate circle detected" in result.warning


# ---------------------------------------------------------------------------
# calibrate: marker cross-check
# ---------------------------------------------------------------------------


def test_calibrate_cross_check_rejects_plate_disagreeing_over_20pct() -> None:
    """The brief's named cross-check test: plate says 0.75 mm/px but an
    87px marker (50mm sleeve) says ~0.57 mm/px -- ~30% apart, so the plate
    scale is rejected and the fallback is used, with a warning."""
    frames = [plate_frame((960, 560), 300) for _ in range(3)]
    fallback = PlaneScale(cm_per_px=0.1)

    result = calibrate(frames, date_fallback=fallback, manual=None, marker_diameter_px=87.0)
    assert result.source == "date_fallback"
    assert result.bar_scale is fallback
    assert result.warning is not None
    assert "disagrees" in result.warning
    assert "marker" in result.warning


def test_calibrate_cross_check_passes_when_marker_agrees() -> None:
    frames = [plate_frame((960, 560), 300) for _ in range(3)]
    # 50mm / 66.7px = 0.7496 mm/px, in agreement with the ~0.75 plate scale.
    result = calibrate(frames, date_fallback=None, manual=None, marker_diameter_px=66.7)
    assert result.source == "plate"
    assert result.warning is None


# ---------------------------------------------------------------------------
# calibrate: fallback ladder
# ---------------------------------------------------------------------------


def test_calibrate_fallback_ladder_reaches_manual() -> None:
    frames = [blank_frame() for _ in range(2)]
    manual = PlaneScale(cm_per_px=0.08)
    result = calibrate(frames, date_fallback=None, manual=manual, marker_diameter_px=None)
    assert result.source == "manual"
    assert result.bar_scale is manual
    assert result.warning is not None


def test_calibrate_prefers_date_fallback_over_manual() -> None:
    frames = [blank_frame() for _ in range(2)]
    date_fb = PlaneScale(cm_per_px=0.1)
    manual = PlaneScale(cm_per_px=0.08)
    result = calibrate(frames, date_fallback=date_fb, manual=manual, marker_diameter_px=None)
    assert result.source == "date_fallback"
    assert result.bar_scale is date_fb


def test_calibrate_raises_with_readable_message_when_no_fallback() -> None:
    frames = [blank_frame() for _ in range(2)]
    with pytest.raises(CalibrationError, match="no plate circle detected"):
        calibrate(frames, date_fallback=None, manual=None, marker_diameter_px=None)


def test_calibrate_error_preserves_band_rejection_reason() -> None:
    frames = [plate_frame((213, 120), 50, width=426, height=240) for _ in range(2)]
    with pytest.raises(CalibrationError, match="outside the plausible band"):
        calibrate(frames, date_fallback=None, manual=None, marker_diameter_px=None)


def test_calibrate_empty_frames_falls_back() -> None:
    fallback = PlaneScale(cm_per_px=0.1)
    result = calibrate([], date_fallback=fallback, manual=None, marker_diameter_px=None)
    assert result.source == "date_fallback"
    assert result.warning is not None


# ---------------------------------------------------------------------------
# constants sanity
# ---------------------------------------------------------------------------


def test_plausible_band_matches_brief() -> None:
    assert PLAUSIBLE_MM_PER_PX == (0.5, 3.0)


def test_max_marker_disagreement_matches_brief() -> None:
    assert MAX_MARKER_DISAGREEMENT == pytest.approx(0.20)

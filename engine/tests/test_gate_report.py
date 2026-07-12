"""Tests for tools/gate_report.py with hand-built labels/overlay JSONs.

Covers the M1 gate semantics: per-keyframe |bar - label| <= 1.0 cm, nearest
overlay frame matched within 50 ms (else UNMATCHED), null-bar overlay frames
tolerated, rep-count comparison only when both files carry a count, and the
process exit code (0 PASS / 1 FAIL).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import gate_report  # noqa: E402

# 1 px = 2 mm = 0.2 cm; the 1.0 cm gate is therefore 5 px.
SCALE_MM_PER_PX = 2.0


def labels_json(clicks: list[dict], rep_count: int | None = None) -> dict:
    data: dict = {"video": "lift.mp4", "clicks": clicks}
    if rep_count is not None:
        data["rep_count"] = rep_count
    return data


def click(name: str, t: float, x: int, y: int, frame_index: int = 0) -> dict:
    return {"name": name, "t": t, "frame_index": frame_index, "x": x, "y": y}


def overlay_json(frames: list[dict], reps: list | None = None) -> dict:
    data: dict = {"frames": frames}
    if reps is not None:
        data["reps"] = reps
    return data


def frame(t: float, bar: list | None) -> dict:
    return {"t": t, "bar": bar, "skeleton": {}}


# ------------------------------------------------------------------ evaluate


def test_evaluate_pass_within_threshold() -> None:
    labels = labels_json([click("knee", 1.0, x=100, y=200)])
    overlay = overlay_json([frame(1.0, [103.0, 204.0])])  # 5 px = exactly 1.0 cm

    result = gate_report.evaluate(labels, overlay, SCALE_MM_PER_PX)

    assert result.passed
    (row,) = result.rows
    assert row.matched and row.passed
    assert row.dx_px == pytest.approx(3.0)
    assert row.dy_px == pytest.approx(4.0)
    assert row.dist_px == pytest.approx(5.0)
    assert row.dist_cm == pytest.approx(1.0)


def test_evaluate_fails_beyond_threshold() -> None:
    labels = labels_json([click("knee", 1.0, x=100, y=200), click("hip", 2.0, x=100, y=200)])
    overlay = overlay_json(
        [frame(1.0, [100.0, 200.0]), frame(2.0, [100.0, 206.0])]  # 6 px = 1.2 cm
    )

    result = gate_report.evaluate(labels, overlay, SCALE_MM_PER_PX)

    assert not result.passed
    assert result.rows[0].passed
    assert not result.rows[1].passed
    assert result.rows[1].dist_cm == pytest.approx(1.2)


def test_evaluate_picks_nearest_frame_within_tolerance() -> None:
    labels = labels_json([click("knee", 1.000, x=0, y=0)])
    overlay = overlay_json(
        [frame(0.960, [50.0, 50.0]), frame(1.010, [1.0, 0.0]), frame(1.100, [99.0, 99.0])]
    )

    result = gate_report.evaluate(labels, overlay, SCALE_MM_PER_PX)

    (row,) = result.rows
    assert row.matched
    assert row.dist_px == pytest.approx(1.0)  # matched t=1.010, not t=0.960


def test_evaluate_unmatched_when_no_frame_within_50ms() -> None:
    labels = labels_json([click("knee", 1.0, x=0, y=0)])
    overlay = overlay_json([frame(1.051, [0.0, 0.0]), frame(2.0, [0.0, 0.0])])

    result = gate_report.evaluate(labels, overlay, SCALE_MM_PER_PX)

    assert not result.passed
    (row,) = result.rows
    assert not row.matched and not row.passed
    assert row.dist_cm is None


def test_evaluate_skips_null_bar_frames_when_matching() -> None:
    labels = labels_json([click("knee", 1.0, x=0, y=0)])
    # Nearest-t frame has no bar; a slightly farther frame within the 50 ms
    # tolerance does -- it must be used instead of reporting UNMATCHED.
    overlay = overlay_json([frame(1.001, None), frame(1.020, [3.0, 4.0])])

    result = gate_report.evaluate(labels, overlay, SCALE_MM_PER_PX)

    (row,) = result.rows
    assert row.matched
    assert row.dist_px == pytest.approx(5.0)


def test_evaluate_unmatched_when_only_null_bars_within_tolerance() -> None:
    labels = labels_json([click("knee", 1.0, x=0, y=0)])
    overlay = overlay_json([frame(1.0, None), frame(1.04, None), frame(5.0, [0.0, 0.0])])

    result = gate_report.evaluate(labels, overlay, SCALE_MM_PER_PX)

    assert not result.passed
    assert not result.rows[0].matched


def test_evaluate_rep_count_mismatch_fails_gate() -> None:
    labels = labels_json([click("knee", 1.0, x=0, y=0)], rep_count=5)
    overlay = overlay_json([frame(1.0, [0.0, 0.0])], reps=[{}, {}, {}])

    result = gate_report.evaluate(labels, overlay, SCALE_MM_PER_PX)

    assert result.rows[0].passed  # keyframe itself is fine
    assert result.rep_count_checked
    assert result.label_rep_count == 5 and result.overlay_rep_count == 3
    assert not result.rep_count_ok
    assert not result.passed


def test_evaluate_rep_count_match_passes() -> None:
    labels = labels_json([click("knee", 1.0, x=0, y=0)], rep_count=2)
    overlay = overlay_json([frame(1.0, [0.0, 0.0])], reps=[{}, {}])

    result = gate_report.evaluate(labels, overlay, SCALE_MM_PER_PX)

    assert result.rep_count_checked and result.rep_count_ok
    assert result.passed


@pytest.mark.parametrize(
    ("rep_count", "reps"),
    [(None, [{}, {}]), (2, None), (None, None)],
)
def test_evaluate_rep_count_skipped_unless_both_present(
    rep_count: int | None, reps: list | None
) -> None:
    labels = labels_json([click("knee", 1.0, x=0, y=0)], rep_count=rep_count)
    overlay = overlay_json([frame(1.0, [0.0, 0.0])], reps=reps)

    result = gate_report.evaluate(labels, overlay, SCALE_MM_PER_PX)

    assert not result.rep_count_checked
    assert result.passed


def test_evaluate_cm_conversion_uses_scale() -> None:
    labels = labels_json([click("knee", 1.0, x=0, y=0)])
    overlay = overlay_json([frame(1.0, [6.0, 8.0])])  # 10 px

    # At 0.9 mm/px, 10 px = 0.9 cm -> PASS; at 1.1 mm/px, 1.1 cm -> FAIL.
    ok = gate_report.evaluate(labels, overlay, scale_mm_per_px=0.9)
    bad = gate_report.evaluate(labels, overlay, scale_mm_per_px=1.1)

    assert ok.passed and ok.rows[0].dist_cm == pytest.approx(0.9)
    assert not bad.passed and bad.rows[0].dist_cm == pytest.approx(1.1)


def test_evaluate_no_clicks_is_a_failure() -> None:
    result = gate_report.evaluate(
        labels_json([]), overlay_json([frame(1.0, [0.0, 0.0])]), SCALE_MM_PER_PX
    )
    assert not result.passed


# ----------------------------------------------------------------- main/CLI


def write_pair(tmp_path: Path, labels: dict, overlay: dict) -> tuple[str, str]:
    labels_path = tmp_path / "lift.mp4.labels.json"
    overlay_path = tmp_path / "overlay.json"
    labels_path.write_text(json.dumps(labels))
    overlay_path.write_text(json.dumps(overlay))
    return str(labels_path), str(overlay_path)


def test_main_pass_exit_zero_and_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    labels = labels_json(
        [click("setup", 0.5, x=10, y=10), click("knee", 1.0, x=100, y=200)],
        rep_count=2,
    )
    overlay = overlay_json([frame(0.5, [10.0, 10.0]), frame(1.0, [103.0, 200.0])], reps=[{}, {}])
    labels_path, overlay_path = write_pair(tmp_path, labels, overlay)

    code = gate_report.main([labels_path, overlay_path, "--scale-mm-per-px", str(SCALE_MM_PER_PX)])
    out = capsys.readouterr().out

    assert code == 0
    assert "setup" in out and "knee" in out
    assert "PASS" in out
    assert "reps" in out.lower()


def test_main_fail_exit_one(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    labels = labels_json([click("knee", 1.0, x=0, y=0)])
    overlay = overlay_json([frame(1.0, [60.0, 0.0])])  # 60 px = 12 cm
    labels_path, overlay_path = write_pair(tmp_path, labels, overlay)

    code = gate_report.main([labels_path, overlay_path, "--scale-mm-per-px", str(SCALE_MM_PER_PX)])
    out = capsys.readouterr().out

    assert code == 1
    assert "FAIL" in out


def test_main_unmatched_keyframe_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    labels = labels_json([click("hip", 3.0, x=0, y=0)])
    overlay = overlay_json([frame(1.0, [0.0, 0.0])])
    labels_path, overlay_path = write_pair(tmp_path, labels, overlay)

    code = gate_report.main([labels_path, overlay_path, "--scale-mm-per-px", str(SCALE_MM_PER_PX)])
    out = capsys.readouterr().out

    assert code == 1
    assert "UNMATCHED" in out


def test_main_rep_count_mismatch_exit_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    labels = labels_json([click("knee", 1.0, x=0, y=0)], rep_count=4)
    overlay = overlay_json([frame(1.0, [0.0, 0.0])], reps=[{}])
    labels_path, overlay_path = write_pair(tmp_path, labels, overlay)

    code = gate_report.main([labels_path, overlay_path, "--scale-mm-per-px", str(SCALE_MM_PER_PX)])
    out = capsys.readouterr().out

    assert code == 1
    assert "4" in out and "1" in out


def test_main_rejects_nonpositive_scale(tmp_path: Path) -> None:
    labels_path, overlay_path = write_pair(
        tmp_path,
        labels_json([click("knee", 1.0, x=0, y=0)]),
        overlay_json([frame(1.0, [0.0, 0.0])]),
    )
    with pytest.raises(SystemExit) as excinfo:
        gate_report.main([labels_path, overlay_path, "--scale-mm-per-px", "0"])
    assert excinfo.value.code == 2


def test_distance_is_euclidean() -> None:
    # Guard against a Manhattan/max-axis slip: dx=3, dy=4 must be 5, not 7.
    labels = labels_json([click("knee", 1.0, x=0, y=0)])
    overlay = overlay_json([frame(1.0, [3.0, 4.0])])
    result = gate_report.evaluate(labels, overlay, scale_mm_per_px=1.0)
    assert result.rows[0].dist_px == pytest.approx(math.hypot(3.0, 4.0))

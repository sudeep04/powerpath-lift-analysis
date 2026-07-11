"""Tests for tools/label.py's pure (non-GUI) logic.

label.py is an interactive cv2 tool; its event loop cannot be integration
tested. Everything testable lives in pure functions/classes at module top:
keyframe schedule parsing, click-record bookkeeping (pending order, undo),
labels-JSON serialization, and the decode-backed FrameStepper (exercised
here against a tiny synthetic video -- no GUI involved).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import label  # noqa: E402
from video_utils import moving_square_frames, write_test_video  # noqa: E402

KEYFRAMES = ["setup", "knee", "hip", "receive", "standing"]


# ---------------------------------------------------------------- parsing


def test_parse_keyframes_happy_path() -> None:
    assert label.parse_keyframes("setup,knee,hip,receive,standing") == KEYFRAMES


def test_parse_keyframes_strips_whitespace() -> None:
    assert label.parse_keyframes(" setup , knee ,hip") == ["setup", "knee", "hip"]


def test_parse_keyframes_single_name() -> None:
    assert label.parse_keyframes("setup") == ["setup"]


@pytest.mark.parametrize("spec", ["", "  ", "setup,,knee", ",setup", "setup,"])
def test_parse_keyframes_rejects_empty_tokens(spec: str) -> None:
    with pytest.raises(ValueError):
        label.parse_keyframes(spec)


def test_parse_keyframes_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        label.parse_keyframes("setup,knee,setup")


# ------------------------------------------------------- session bookkeeping


def test_session_pending_follows_schedule_order() -> None:
    session = label.LabelSession("lift.mp4", KEYFRAMES)
    assert session.pending() == "setup"
    session.record_click(t=0.0, frame_index=0, x=10, y=20)
    assert session.pending() == "knee"
    session.record_click(t=1.23, frame_index=74, x=512, y=301)
    assert session.pending() == "hip"


def test_session_complete_only_after_all_keyframes() -> None:
    session = label.LabelSession("lift.mp4", ["setup", "knee"])
    assert not session.complete
    session.record_click(t=0.0, frame_index=0, x=1, y=2)
    assert not session.complete
    session.record_click(t=0.5, frame_index=30, x=3, y=4)
    assert session.complete
    assert session.pending() is None


def test_session_rejects_click_when_complete() -> None:
    session = label.LabelSession("lift.mp4", ["setup"])
    session.record_click(t=0.0, frame_index=0, x=1, y=2)
    with pytest.raises(ValueError):
        session.record_click(t=0.1, frame_index=6, x=3, y=4)


def test_session_undo_reopens_last_keyframe() -> None:
    session = label.LabelSession("lift.mp4", ["setup", "knee"])
    session.record_click(t=0.0, frame_index=0, x=1, y=2)
    session.record_click(t=0.5, frame_index=30, x=3, y=4)
    assert session.complete

    undone = session.undo()
    assert undone is not None and undone["name"] == "knee"
    assert not session.complete
    assert session.pending() == "knee"

    undone = session.undo()
    assert undone is not None and undone["name"] == "setup"
    assert session.pending() == "setup"


def test_session_undo_on_empty_returns_none() -> None:
    session = label.LabelSession("lift.mp4", ["setup"])
    assert session.undo() is None


# ------------------------------------------------------------- serialization


def test_to_json_matches_exact_schema() -> None:
    session = label.LabelSession("lift.mp4", ["knee"])
    session.record_click(t=1.23, frame_index=74, x=512, y=301)

    data = session.to_json()
    assert set(data.keys()) == {"video", "clicks"}
    assert data["video"] == "lift.mp4"
    assert data["clicks"] == [{"name": "knee", "t": 1.23, "frame_index": 74, "x": 512, "y": 301}]


def test_json_round_trip_through_file(tmp_path: Path) -> None:
    session = label.LabelSession("lift.mp4", ["setup", "knee"])
    session.record_click(t=0.0, frame_index=0, x=10, y=20)
    session.record_click(t=1.23, frame_index=74, x=512, y=301)

    out = tmp_path / "lift.mp4.labels.json"
    label.write_labels(session, out)

    loaded = json.loads(out.read_text())
    assert loaded == session.to_json()
    # Value types survive the round trip exactly.
    click = loaded["clicks"][1]
    assert isinstance(click["t"], float)
    assert isinstance(click["frame_index"], int)
    assert isinstance(click["x"], int) and isinstance(click["y"], int)


def test_labels_path_appends_suffix_to_video_name() -> None:
    assert label.labels_path_for(Path("/clips/lift.mp4")) == Path("/clips/lift.mp4.labels.json")


# ------------------------------------------------------------- frame stepping


@pytest.fixture(scope="module")
def tiny_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("video") / "tiny.mp4"
    write_test_video(path, moving_square_frames(12, width=64, height=48), fps=12)
    return path


def test_stepper_seeks_forward(tiny_video: Path) -> None:
    stepper = label.FrameStepper(tiny_video)
    frame = stepper.seek(5)
    assert frame is not None and frame.index == 5


def test_stepper_seeks_backward_by_restarting(tiny_video: Path) -> None:
    stepper = label.FrameStepper(tiny_video)
    stepper.seek(8)
    frame = stepper.seek(2)
    assert frame is not None and frame.index == 2


def test_stepper_seek_same_index_returns_current(tiny_video: Path) -> None:
    stepper = label.FrameStepper(tiny_video)
    first = stepper.seek(4)
    again = stepper.seek(4)
    assert again is first


def test_stepper_clamps_past_end_to_last_frame(tiny_video: Path) -> None:
    stepper = label.FrameStepper(tiny_video)
    frame = stepper.seek(10_000)
    assert frame is not None and frame.index == 11

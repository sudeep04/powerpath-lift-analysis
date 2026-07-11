"""Tests for powerpath_engine.decode.

decode.py is the engine's input boundary: `frames()` must stream (never
materialize the video), key every sample by PTS seconds (not frame index),
and normalize rotation so yielded images are always upright.
"""

from __future__ import annotations

import platform
import resource
from itertools import pairwise

import numpy as np
import pytest
from video_utils import (
    moving_square_frames,
    write_audio_only_test_file,
    write_rotated_test_video,
    write_test_video,
)

from powerpath_engine.decode import DecodeError, frames, probe

# ---------------------------------------------------------------------------
# Happy path: a 2s synthetic moving-square video
# ---------------------------------------------------------------------------

FPS = 30
DURATION_S = 2
N_FRAMES = FPS * DURATION_S


@pytest.fixture
def square_video(tmp_path):
    path = tmp_path / "square.mp4"
    write_test_video(path, moving_square_frames(N_FRAMES), fps=FPS)
    return path


def test_frames_t_is_monotonically_increasing(square_video) -> None:
    ts = [f.t for f in frames(square_video)]
    assert len(ts) > 1
    assert all(b > a for a, b in pairwise(ts))


def test_frame_count_within_10_percent_of_fps_times_duration(square_video) -> None:
    count = sum(1 for _ in frames(square_video))
    expected = FPS * DURATION_S
    assert abs(count - expected) <= 0.10 * expected


def test_decoded_frame_index_is_0_based_and_contiguous(square_video) -> None:
    indices = [f.index for f in frames(square_video)]
    assert indices == list(range(len(indices)))


def test_decoded_frame_image_is_bgr_uint8_matching_probe_dims(square_video) -> None:
    meta = probe(square_video)
    first = next(iter(frames(square_video)))
    assert first.image.dtype == np.uint8
    assert first.image.shape == (meta.height, meta.width, 3)


def test_probe_reports_fps_and_duration(square_video) -> None:
    meta = probe(square_video)
    assert meta.fps_avg == pytest.approx(FPS, rel=0.05)
    assert meta.duration_s == pytest.approx(DURATION_S, rel=0.1)
    assert meta.rotation_deg == 0
    assert meta.width == 320
    assert meta.height == 240


def test_frames_is_a_lazy_generator_not_a_list(square_video) -> None:
    gen = frames(square_video)
    # A generator object, not something that already decoded everything.
    assert hasattr(gen, "__next__")
    first = next(gen)
    assert first.index == 0
    gen.close()  # abandoning early must not raise / must close cleanly


# ---------------------------------------------------------------------------
# Rotation normalization
# ---------------------------------------------------------------------------


def test_upright_dimensions_for_portrait_stream(tmp_path) -> None:
    """Encoding portrait dimensions directly (no rotation side data) must
    round-trip as upright portrait dims -- the brief's "simulate by encoding
    portrait dimensions" case.
    """
    path = tmp_path / "portrait.mp4"
    portrait_frames = moving_square_frames(10, width=240, height=320)
    write_test_video(path, portrait_frames, fps=10)

    meta = probe(path)
    assert meta.rotation_deg == 0
    assert (meta.width, meta.height) == (240, 320)

    first = next(iter(frames(path)))
    assert first.image.shape == (320, 240, 3)


@pytest.mark.parametrize("rotation_deg", [90, 180, 270])
def test_rotation_side_data_normalizes_dimensions(tmp_path, rotation_deg) -> None:
    """A landscape-coded stream tagged with a DISPLAYMATRIX rotation must
    decode to upright dimensions matching what a viewer would expect (90/270
    swap width and height; 180 does not).
    """
    path = tmp_path / f"rot{rotation_deg}.mp4"
    coded_frames = moving_square_frames(6, width=320, height=240)
    write_rotated_test_video(path, coded_frames, fps=10, rotation_deg=rotation_deg)

    meta = probe(path)
    assert meta.rotation_deg == rotation_deg
    if rotation_deg in (90, 270):
        assert (meta.width, meta.height) == (240, 320)
    else:
        assert (meta.width, meta.height) == (320, 240)

    first = next(iter(frames(path)))
    assert first.image.shape == (meta.height, meta.width, 3)


def test_rotation_90_moves_top_left_marker_to_bottom_left(tmp_path) -> None:
    """Concrete direction check (not just dimensions): PyAV documents
    VideoFrame.rotation as the counterclockwise correction to apply. A
    marker in the coded frame's top-left corner must land in the upright
    image's bottom-left corner after a 90-degree correction -- catches an
    off-by-sign (cw vs ccw) rotation bug that a dimensions-only test can't.
    """
    path = tmp_path / "rot90_marker.mp4"
    width, height, marker = 320, 240, 40
    coded = np.zeros((height, width, 3), dtype=np.uint8)
    coded[0:marker, 0:marker] = 255
    write_rotated_test_video(path, [coded] * 4, fps=10, rotation_deg=90)

    upright = next(iter(frames(path))).image
    assert upright.shape == (width, height, 3)  # swapped: (320, 240, 3)

    # Bottom-left block should now be bright; every other corner should not.
    assert upright[-marker:, 0:marker].mean() > 200
    assert upright[0:marker, 0:marker].mean() < 50
    assert upright[0:marker, -marker:].mean() < 50
    assert upright[-marker:, -marker:].mean() < 50


# ---------------------------------------------------------------------------
# DecodeError
# ---------------------------------------------------------------------------


def test_decode_error_on_text_file(tmp_path) -> None:
    path = tmp_path / "not_a_video.txt"
    path.write_text("this is definitely not a video file, just some text.\n" * 10)

    with pytest.raises(DecodeError):
        probe(path)

    with pytest.raises(DecodeError):
        list(frames(path))


def test_decode_error_on_truncated_file(tmp_path) -> None:
    good_path = tmp_path / "good.mp4"
    write_test_video(good_path, moving_square_frames(20), fps=10)

    truncated_path = tmp_path / "truncated.mp4"
    data = good_path.read_bytes()
    truncated_path.write_bytes(data[: len(data) // 2])

    with pytest.raises(DecodeError):
        probe(truncated_path)

    with pytest.raises(DecodeError):
        list(frames(truncated_path))


def test_decode_error_on_missing_file(tmp_path) -> None:
    with pytest.raises(DecodeError):
        probe(tmp_path / "does_not_exist.mp4")


def test_decode_error_on_stream_with_no_video(tmp_path) -> None:
    path = tmp_path / "audio_only.wav"
    write_audio_only_test_file(path)

    with pytest.raises(DecodeError):
        probe(path)

    with pytest.raises(DecodeError):
        list(frames(path))


# ---------------------------------------------------------------------------
# Memory: streaming must not materialize the video
# ---------------------------------------------------------------------------


def _max_rss_bytes() -> int:
    """Peak RSS of this process so far, in bytes.

    `resource.getrusage(...).ru_maxrss` is platform-inconsistent: macOS
    reports it in bytes, Linux in kilobytes. It is also a high-water mark
    that never decreases within a process's lifetime, so a before/after
    delta measures how much the high-water mark grew during the
    intervening work, not "current usage" -- exactly what a
    never-buffer-the-whole-video assertion wants to bound.
    """
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if platform.system() == "Darwin" else rss * 1024


def test_rss_growth_bound_iterating_10s_720p_video(tmp_path) -> None:
    fps = 24
    duration_s = 10
    width, height = 1280, 720
    path = tmp_path / "long_720p.mp4"
    write_test_video(
        path,
        moving_square_frames(fps * duration_s, width=width, height=height, square=60),
        fps=fps,
    )

    before = _max_rss_bytes()
    count = 0
    for decoded_frame in frames(path):
        count += 1
        # touch the array so it's not a completely free no-op, without
        # retaining a reference across iterations
        _ = decoded_frame.image.sum()
    after = _max_rss_bytes()

    assert count > 0
    growth = after - before
    assert growth < 150 * 1024 * 1024, f"RSS grew by {growth / 1024 / 1024:.1f}MB"

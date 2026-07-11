"""Test-only helpers for encoding synthetic videos with PyAV.

Used across the decode test suite (and later task suites that need a real
decodable video file) so every test builds its fixtures the same way rather
than hand-rolling container/stream setup repeatedly.
"""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np

# libx264 (and most practical decoders) require even dimensions for 4:2:0
# chroma subsampling (yuv420p halves both axes) -- callers should pick even
# width/height, but this isn't enforced here since the encoder will simply
# raise if it isn't.
_PIX_FMT = "yuv420p"


def write_test_video(path: str | Path, frames: list[np.ndarray], fps: int) -> None:
    """Encode `frames` (BGR uint8 HxWx3 arrays, all the same shape) to an
    h264/mp4 file at `path` using PyAV (no system ffmpeg involved).
    """
    height, width = frames[0].shape[:2]
    container = av.open(str(path), mode="w")
    try:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = _PIX_FMT
        for arr in frames:
            frame = av.VideoFrame.from_ndarray(arr, format="bgr24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def write_rotated_test_video(
    path: str | Path, frames: list[np.ndarray], fps: int, rotation_deg: int
) -> None:
    """Like `write_test_video`, but tags the stream with a DISPLAYMATRIX
    rotation (`VideoStream.set_display_rotation`) so the encoded file
    round-trips through PyAV with `VideoFrame.rotation == rotation_deg` on
    decode -- used to exercise `decode.py`'s rotation-normalization path
    without needing a real rotated device recording.
    """
    height, width = frames[0].shape[:2]
    container = av.open(str(path), mode="w")
    try:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = _PIX_FMT
        stream.set_display_rotation(rotation_deg)
        for arr in frames:
            frame = av.VideoFrame.from_ndarray(arr, format="bgr24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def write_audio_only_test_file(path: str | Path) -> None:
    """A tiny valid audio-only file (no video stream at all) -- used to
    exercise `decode.py`'s "no video stream" `DecodeError` path with a file
    PyAV can genuinely open (unlike a text file, which should fail to open
    at all).

    `path` must carry an extension whose container `decode.py`'s format
    whitelist accepts (e.g. `.m4a` -- AAC in an mp4-family container);
    otherwise the test would trip the whitelist rejection in `_open()`
    instead of the `no video stream` branch it is meant to cover.
    """
    container = av.open(str(path), mode="w")
    try:
        stream = container.add_stream("aac", rate=44100)
        # 2048 interleaved stereo samples of silence (one AAC frame's worth).
        samples = np.zeros((1, 2048 * 2), dtype=np.int16)
        frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="stereo")
        frame.sample_rate = 44100
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def moving_square_frames(
    n_frames: int, width: int = 320, height: int = 240, square: int = 20
) -> list[np.ndarray]:
    """`n_frames` BGR uint8 frames of a white square moving left to right
    across a black background -- the "2s synthetic video (moving white
    square)" fixture named in the Task 3 brief.
    """
    frames = []
    max_x = max(width - square, 1)
    for i in range(n_frames):
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        x = int(round((i / max(n_frames - 1, 1)) * max_x))
        y = (height - square) // 2
        arr[y : y + square, x : x + square] = 255
        frames.append(arr)
    return frames

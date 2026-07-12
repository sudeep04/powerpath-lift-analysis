"""Streaming video decode: the engine's input boundary.

Every downstream module (bar tracking, pose, segmentation, ...) consumes the
`DecodedFrame` stream produced here and nothing else -- this is the only
place PyAV is imported. Two hard requirements flow from the global
constraints and shape everything below:

- **Streaming single-pass rule**: `frames()` is a generator that opens the
  container, decodes and yields one frame at a time, and closes the
  container deterministically (via `try`/`finally`, so this happens even if
  the consumer abandons iteration early) -- the whole video is never
  materialized in memory.
- **PTS timebase**: every yielded frame's `t` is `float(frame.pts *
  stream.time_base)` seconds, never a bare frame index -- iPhone video is
  variable frame rate, so only PTS is a meaningful clock.

Rotation is the other subtlety: phones record with a DISPLAYMATRIX side-data
rotation on the stream rather than physically rotating pixels, and PyAV
exposes this per decoded frame as `VideoFrame.rotation` -- a counterclockwise
angle in degrees (PyAV's own docs: "matching the value read back from
VideoFrame.rotation"). `frames()` reads it per frame and applies the matching
`np.rot90` correction so every yielded image is upright and BGR uint8;
`probe()` reports the resulting (post-rotation) width/height so callers never
have to reason about the raw coded orientation themselves.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import av
import av.error
import numpy as np

logger = logging.getLogger(__name__)

# FFmpeg's demuxer autodetection is over-eager for a few formats that have
# no real magic-byte signature -- most notably "tty" (an ANSI-art/terminal
# capture format), which happily decodes an arbitrary plain-text file as a
# tiny fake video instead of failing. Restricting autodetection to container
# formats real phone/camera footage actually uses means a non-video file
# reliably raises DecodeError instead of "succeeding" as bogus video.
_FORMAT_WHITELIST = "mov,mp4,m4a,3gp,3g2,mj2,matroska,webm,avi,mpegts,mpeg,flv"


class DecodeError(Exception):
    """Raised when a video cannot be decoded.

    Covers unreadable files (wrong format, truncated/corrupt data -- both
    surface as PyAV/FFmpeg errors, whether at container-open time or mid
    decode), and files whose container has no video stream at all. Callers
    only need to catch this one type regardless of which underlying
    `av.error.FFmpegError` subclass PyAV raised.
    """


@dataclass
class VideoMeta:
    """Container/stream metadata, expressed in the UPRIGHT (post-rotation)
    frame -- consistent with what `frames()` yields.

    `width`/`height` are the dimensions of the frames `frames()` yields:
    after rotation normalization, not the raw coded dimensions PyAV reports
    on the stream when the two differ (a 90/270-degree rotation swaps them).
    `rotation_deg` is the counterclockwise correction applied to the raw
    decoded frame to reach that orientation -- one of 0, 90, 180, 270 (see
    `_normalize_rotation_deg`). `fps_avg` is the stream's average frame
    rate; `duration_s` is the stream (falling back to container) duration in
    seconds.
    """

    width: int
    height: int
    rotation_deg: int
    fps_avg: float
    duration_s: float


@dataclass
class DecodedFrame:
    """One decoded, upright, BGR uint8 video frame.

    `t` is PTS seconds (see module docstring) -- the only thing downstream
    code should key on. `index` is a plain 0-based count of *yielded*
    frames (frames skipped for missing/unusable pts, see `frames()`, are
    not counted here); it exists for progress reporting and for stride
    gating (pose.StridedPose runs its backend every Nth yielded frame)
    and must never be used as a time axis.
    """

    t: float
    image: np.ndarray
    index: int


def probe(path: str | Path) -> VideoMeta:
    """Read `path`'s container/stream metadata as a `VideoMeta`.

    Decodes exactly one frame -- the first -- because PyAV only exposes the
    DISPLAYMATRIX-derived rotation on a decoded `VideoFrame`, not on the
    stream itself; everything else comes from container/stream metadata
    without decoding further. Raises `DecodeError` if the file can't be
    opened, has no video stream, or has no decodable frame.
    """
    container = _open(path)
    try:
        stream = _video_stream(container, path)
        coded_width = stream.codec_context.width
        coded_height = stream.codec_context.height

        try:
            first_frame = next(container.decode(stream))
        except StopIteration:
            raise DecodeError(f"{path}: no decodable video frames") from None
        except av.error.FFmpegError as exc:
            raise DecodeError(f"{path}: failed to decode a frame: {exc}") from exc

        rotation_deg = _normalize_rotation_deg(first_frame.rotation)
        width, height = coded_width, coded_height
        if rotation_deg in (90, 270):
            width, height = height, width

        return VideoMeta(
            width=width,
            height=height,
            rotation_deg=rotation_deg,
            fps_avg=_average_fps(stream),
            duration_s=_duration_seconds(container, stream),
        )
    finally:
        container.close()


def frames(path: str | Path) -> Iterator[DecodedFrame]:
    """Lazily decode `path` into upright BGR uint8 frames, one at a time.

    A generator: nothing happens until the caller starts iterating, at
    which point the container is opened; it is always closed before the
    generator finishes, whether iteration runs to completion, a
    `DecodeError` is raised, or the caller abandons the generator early
    (`.close()`, `break`, falling out of scope) -- the `finally` block runs
    in all of those cases. At most one decoded frame is resident at a time;
    the whole video is never materialized (the streaming single-pass global
    constraint).

    `t` is `float(frame.pts * stream.time_base)` seconds -- the PTS
    timebase global constraint means there is deliberately NO frame-index
    fallback clock in this module. A frame whose `pts` is `None` mid-stream
    is skipped (counted and logged once at the end) because its `t` cannot
    be trusted; if the very FIRST decoded frame has no pts, the stream has
    no usable timebase at all and a `DecodeError` is raised rather than
    inventing index-based timestamps for a (possibly variable frame rate)
    video. Real device footage always carries pts, so neither path is
    expected outside corrupt/exotic input.

    Rotation: each frame's own `.rotation` is read and applied via
    `np.rot90` so the yielded image is always upright; see the module
    docstring.

    Raises `DecodeError` on unreadable files, files with no video stream,
    or truncated/corrupt data encountered mid-decode.
    """
    container = _open(path)
    try:
        stream = _video_stream(container, path)

        none_pts_count = 0
        yielded = 0

        try:
            for decode_index, frame in enumerate(container.decode(stream)):
                if frame.pts is None:
                    if decode_index == 0:
                        # No pts on frame zero means the stream carries no
                        # usable timebase; refuse rather than fabricate
                        # frame-index timestamps (PTS timebase constraint).
                        raise DecodeError(
                            f"{path}: first frame has no pts; stream has no usable timebase"
                        )
                    # An isolated later frame missing pts in an otherwise-
                    # timestamped stream: its t cannot be trusted, so drop
                    # it rather than guess.
                    none_pts_count += 1
                    continue

                t = float(frame.pts * stream.time_base)
                rotation_deg = _normalize_rotation_deg(frame.rotation)
                image = _apply_rotation(frame.to_ndarray(format="bgr24"), rotation_deg)

                yield DecodedFrame(t=t, image=image, index=yielded)
                yielded += 1
        except av.error.FFmpegError as exc:
            raise DecodeError(f"{path}: decode failed after {yielded} frame(s): {exc}") from exc

        if none_pts_count:
            logger.warning("%s: skipped %d frame(s) with no pts", path, none_pts_count)
    finally:
        container.close()


def _open(path: str | Path) -> av.container.InputContainer:
    try:
        return av.open(str(path), container_options={"format_whitelist": _FORMAT_WHITELIST})
    except av.error.FFmpegError as exc:
        raise DecodeError(f"{path}: could not open as video: {exc}") from exc


def _video_stream(
    container: av.container.InputContainer, path: str | Path
) -> av.video.stream.VideoStream:
    video_streams = container.streams.video
    if not video_streams:
        raise DecodeError(f"{path}: no video stream")
    return video_streams[0]


def _normalize_rotation_deg(raw_rotation_deg: int) -> int:
    """Normalize an arbitrary CCW rotation angle to one of {0, 90, 180, 270}.

    `VideoFrame.rotation` is documented to range over [-180, 180]; real
    device/container rotations are always (effectively) multiples of 90, so
    rounding to the nearest 90 and reducing modulo 360 collapses the sign
    ambiguity at +/-180 and normalizes negative angles (e.g. -90 -> 270)
    without changing what rotation is actually being described.
    """
    return round(raw_rotation_deg / 90.0) * 90 % 360


def _apply_rotation(image: np.ndarray, rotation_deg: int) -> np.ndarray:
    """Rotate `image` counterclockwise by `rotation_deg` (a multiple of 90).

    `np.rot90(..., k=1)` rotates an (H, W, C) array counterclockwise when
    displayed with row 0 at the top (numpy's own convention) -- exactly the
    convention `VideoFrame.rotation` uses, so `k = rotation_deg // 90` is the
    correct correction with no sign flip needed. Returns a fresh contiguous
    array (`np.rot90` yields a strided view) so downstream code holding onto
    a `DecodedFrame.image` never keeps a rotated view over a bigger buffer.
    """
    if rotation_deg == 0:
        return image
    return np.ascontiguousarray(np.rot90(image, k=rotation_deg // 90))


def _average_fps(stream: av.video.stream.VideoStream) -> float:
    rate = stream.average_rate or stream.guessed_rate or stream.base_rate
    return float(rate) if rate else 0.0


def _duration_seconds(
    container: av.container.InputContainer, stream: av.video.stream.VideoStream
) -> float:
    if stream.duration is not None:
        return float(stream.duration * stream.time_base)
    if container.duration is not None:
        return container.duration / av.time_base
    return 0.0

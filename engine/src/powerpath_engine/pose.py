"""Pose estimation behind a swappable backend interface.

The `PoseBackend` protocol is the seam demanded by the global "no model
inference in the test suite" constraint: the pipeline and every test talk
to `detect(image_bgr)` and nothing else, so tests run against
`FakePoseBackend` while production picks `RTMLibBackend` (default) or
`MediaPipeBackend`. Both real backends import their libraries LAZILY
inside `__init__` -- rtmlib/onnxruntime and mediapipe are optional extras
(see `[project.optional-dependencies]` in pyproject) and the suite must
pass with neither installed; a missing library raises
`PoseUnavailableError` whose message names the exact install command.

Because the real backends cannot run in tests, every piece of their logic
that CAN be pure IS pure and module-level: COCO-17 index -> landmark-name
mapping (`coco17_to_landmarks`), MediaPipe's normalized-coordinate mapping
(`mediapipe_landmarks_to_samples`), the single-athlete-lock person
selection (`select_person` / `AthleteLock`), and the ROI crop /
coordinate-mapping helpers
(`crop_around`, `map_to_full_frame`, `points_bbox`, `keypoints_bbox`).
The backend classes are thin shells over library calls plus these
functions.

Landmark vocabulary (see `series.LANDMARK_NAMES`): COCO-17 stops at the
ankles, so `left/right_heel` and `left/right_foot_index` are
MediaPipe-only names -- downstream code must tolerate their absence and
never assume the full 17-name vocabulary is present.

Scheduling: pose is the most expensive per-frame stage, so `StridedPose`
runs the backend every `stride` frames (default 2) during the single
streaming pass and records skipped/missed frames as None. Rep windows get
a full-rate second pass via `rerun_full_rate(window_frames)`: the pipeline
supplies a SECOND decode pass restricted to the windows, because decoding
a few seconds of video again is far cheaper than pose inference and the
streaming single-pass rule forbids buffering decoded frames from the first
pass.

Coordinates: everything here is px in image space (y grows down).
`StridedPose` owns mapping crop-relative backend output back to full-frame
px (a fixed integer offset, not a scale change); geometry.py remains the
sole owner of px<->cm conversion and the y-flip.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from powerpath_engine.decode import DecodedFrame
from powerpath_engine.series import LandmarkFrame, LandmarkSeries, Sample

# Keypoints scoring below this are omitted from a detection entirely (the
# series.py convention: an undetected landmark simply has no key, and the
# per-landmark gap interpolation deals with the hole). Detectors emit
# essentially-guessed coordinates at low confidence; passing those through
# as "data" with a small visibility would poison smoothing and joint-angle
# math downstream.
MIN_LANDMARK_SCORE = 0.3

# Minimum ROI crop extent per dimension. A degenerate landmark bbox (a
# single confident landmark has zero width and height) would otherwise
# yield a ~1px crop on the next call, guaranteeing a miss; any padded
# window smaller than this is expanded to it, centered on the bbox and
# clamped to the frame.
MIN_CROP_EXTENT_PX = 32

# COCO-17 keypoint index -> our landmark name (series.LANDMARK_NAMES
# vocabulary). Indices 1-4 (eyes, ears) have no downstream use in barbell
# biomechanics and map to None (skipped). COCO-17 has NO heel or foot_index
# keypoints -- those names are MediaPipe-only.
COCO17_LANDMARK_NAMES: tuple[str | None, ...] = (
    "nose",  # 0
    None,  # 1 left_eye
    None,  # 2 right_eye
    None,  # 3 left_ear
    None,  # 4 right_ear
    "left_shoulder",  # 5
    "right_shoulder",  # 6
    "left_elbow",  # 7
    "right_elbow",  # 8
    "left_wrist",  # 9
    "right_wrist",  # 10
    "left_hip",  # 11
    "right_hip",  # 12
    "left_knee",  # 13
    "right_knee",  # 14
    "left_ankle",  # 15
    "right_ankle",  # 16
)

# Our landmark name -> MediaPipe pose_landmarker index (33-landmark model).
# MediaPipe is the only backend that provides heel/foot_index.
MEDIAPIPE_LANDMARK_INDICES: dict[str, int] = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}


class PoseUnavailableError(Exception):
    """A pose backend's library is not installed (optional extra missing)."""


@dataclass(frozen=True)
class BBox:
    """Axis-aligned float bbox (top-left + size) in image px.

    Deliberately NOT bar.Rect: that one is int-px and documented as local
    to bar tracking on purpose, and importing bar.py (marker detection,
    cv2 machinery) for a four-field record would couple two unrelated
    per-frame consumers. Pose bboxes are landmark extents, so float.
    """

    x: float
    y: float
    w: float
    h: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    @property
    def area(self) -> float:
        return self.w * self.h


class PoseBackend(Protocol):
    """One-frame pose detection: landmark name -> Sample, or None.

    Samples carry x/y in px of the IMAGE THE BACKEND WAS GIVEN (which may
    be a crop -- StridedPose maps back to full frame) and visibility in
    [0, 1]. `origin` is that image's top-left corner in FULL-FRAME px
    ((0, 0) unless the caller cropped): backends that lock onto a person
    across calls use it to keep the lock in full-frame coordinates, so
    the caller alternating full frames and crops never makes the lock
    compare across coordinate regimes. `t` is a 0.0 placeholder; the
    caller stamps PTS seconds. Returns None when no usable person is
    detected.
    """

    def detect(
        self, image_bgr: np.ndarray, *, origin: tuple[float, float] = (0.0, 0.0)
    ) -> dict[str, Sample] | None: ...


class FakePoseBackend:
    """Deterministic, call-counted test backend (the ONLY backend tests run).

    `script(call_index)` supplies the result for the call_index-th detect
    call (0-based). `calls` counts detect invocations; `seen_shapes`
    records each received image's shape and `seen_origins` each call's
    full-frame origin, so tests can assert the backend saw a crop of the
    expected size and position without buffering the images themselves.
    """

    def __init__(self, script: Callable[[int], dict[str, Sample] | None]) -> None:
        self.script = script
        self.calls = 0
        self.seen_shapes: list[tuple[int, ...]] = []
        self.seen_origins: list[tuple[float, float]] = []

    def detect(
        self, image_bgr: np.ndarray, *, origin: tuple[float, float] = (0.0, 0.0)
    ) -> dict[str, Sample] | None:
        self.seen_shapes.append(tuple(image_bgr.shape))
        self.seen_origins.append(origin)
        result = self.script(self.calls)
        self.calls += 1
        return result


def coco17_to_landmarks(keypoints: np.ndarray, scores: np.ndarray) -> dict[str, Sample]:
    """Map one person's COCO-17 keypoints to named Samples.

    `keypoints` is (17, 2) px, `scores` is (17,) confidence. Unnamed
    indices (eyes/ears) and keypoints under MIN_LANDMARK_SCORE are
    omitted; visibility is the score clamped to [0, 1] (detector scores
    can nick above 1.0). `t` is 0.0 -- the caller stamps PTS seconds.
    Returns {} when nothing is confident (callers treat that as a miss).
    """
    points: dict[str, Sample] = {}
    for index, name in enumerate(COCO17_LANDMARK_NAMES):
        if name is None:
            continue
        score = float(scores[index])
        if score < MIN_LANDMARK_SCORE:
            continue
        points[name] = Sample(
            t=0.0,
            x=float(keypoints[index, 0]),
            y=float(keypoints[index, 1]),
            visibility=min(1.0, max(0.0, score)),
        )
    return points


def mediapipe_landmarks_to_samples(
    landmarks: Sequence[tuple[float, float, float]], width: int, height: int
) -> dict[str, Sample]:
    """Map MediaPipe's 33 normalized landmarks to named px Samples.

    `landmarks` is a sequence of (x, y, visibility) with x/y normalized to
    [0, 1] (MediaPipe convention); `width`/`height` scale them to px of the
    image the backend was given. Same omission/clamp rules as the COCO
    mapper. This is the pure half of MediaPipeBackend.detect, testable
    with plain tuples.
    """
    points: dict[str, Sample] = {}
    for name, index in MEDIAPIPE_LANDMARK_INDICES.items():
        x, y, visibility = landmarks[index]
        if visibility < MIN_LANDMARK_SCORE:
            continue
        points[name] = Sample(
            t=0.0,
            x=float(x) * width,
            y=float(y) * height,
            visibility=min(1.0, max(0.0, float(visibility))),
        )
    return points


def keypoints_bbox(keypoints: np.ndarray) -> BBox:
    """Min/max bbox over an (N, 2) keypoint array (one person)."""
    xs = keypoints[:, 0]
    ys = keypoints[:, 1]
    x0 = float(xs.min())
    y0 = float(ys.min())
    return BBox(x=x0, y=y0, w=float(xs.max()) - x0, h=float(ys.max()) - y0)


def points_bbox(points: dict[str, Sample]) -> BBox:
    """Min/max bbox over a detection's landmark Samples."""
    xs = [s.x for s in points.values()]
    ys = [s.y for s in points.values()]
    x0 = min(xs)
    y0 = min(ys)
    return BBox(x=x0, y=y0, w=max(xs) - x0, h=max(ys) - y0)


def select_person(bboxes: Sequence[BBox], prev_bbox: BBox | None) -> int:
    """Single-athlete lock: index of the person to keep tracking.

    With a previous athlete bbox, the person whose bbox center is nearest
    the previous center wins -- a background person walking through frame
    must not steal the track, no matter how large or confident their
    detection is. With no previous bbox (first frame, or re-acquisition
    after a miss) the largest bbox wins: the athlete is the subject of the
    video and dominates the frame (and dominates athlete-centered crops
    even more so). Raises ValueError on an empty candidate list -- callers
    must treat "no people" as a miss before selecting.
    """
    if not bboxes:
        raise ValueError("select_person requires at least one candidate bbox")
    if prev_bbox is None:
        return max(range(len(bboxes)), key=lambda i: bboxes[i].area)
    px, py = prev_bbox.center
    return min(range(len(bboxes)), key=lambda i: math.dist(bboxes[i].center, (px, py)))


class AthleteLock:
    """Stateful single-athlete lock kept in FULL-FRAME coordinates.

    Wraps `select_person` with the cross-call state a multi-person
    backend needs: the previously chosen bbox. Candidate bboxes arrive in
    the coordinate space of whatever image the backend was given
    (possibly an ROI crop); `origin` -- that image's top-left in
    full-frame px -- translates them into full-frame space BEFORE
    selection AND storage. The invariant: the previous bbox and every
    candidate it is compared against are always full-frame px, so a
    caller alternating full frames and crops (StridedPose does, at ROI
    engagement, post-miss reset and rerun entry) can never make the lock
    compare stale coordinates from another regime -- a second person
    inside a padded crop cannot steal the track. This is the pure,
    rtmlib-free half of RTMLibBackend's person tracking.
    """

    def __init__(self) -> None:
        self.prev_bbox: BBox | None = None

    def select(self, bboxes: Sequence[BBox], origin: tuple[float, float] = (0.0, 0.0)) -> int:
        """Pick and remember the tracked person among image-space bboxes."""
        ox, oy = origin
        candidates = [BBox(x=b.x + ox, y=b.y + oy, w=b.w, h=b.h) for b in bboxes]
        chosen = select_person(candidates, self.prev_bbox)
        self.prev_bbox = candidates[chosen]
        return chosen

    def reset(self) -> None:
        """Drop the lock (miss): the next select re-acquires by area."""
        self.prev_bbox = None


def crop_around(
    prev_bbox: BBox, image: np.ndarray, pad: float = 0.3
) -> tuple[np.ndarray, int, int]:
    """Crop `image` to `prev_bbox` expanded by `pad` per side; clamp to frame.

    Returns `(crop, x0, y0)` where (x0, y0) is the crop's top-left corner
    in full-frame px -- the offset `map_to_full_frame` adds back. The pad
    is a fraction of the bbox's own width/height added on EACH side;
    0.3 covers both inter-frame athlete motion at stride 2 and the body
    extent beyond the landmark bbox (landmarks are joint centers -- the
    head, hands and feet stick out past them). A padded window narrower
    than MIN_CROP_EXTENT_PX in either dimension (degenerate bbox, e.g. a
    single confident landmark) is expanded to that minimum, centered on
    the bbox and clamped to the frame. Bounds are floored/ceiled outward
    so padding never truncates inward. A bbox that clamps to an empty
    window (track drifted off-frame) falls back to the full frame at
    offset (0, 0) rather than handing the backend an empty image.
    """
    frame_h, frame_w = image.shape[:2]
    cx, cy = prev_bbox.center
    half_w = max(prev_bbox.w * (0.5 + pad), MIN_CROP_EXTENT_PX / 2.0)
    half_h = max(prev_bbox.h * (0.5 + pad), MIN_CROP_EXTENT_PX / 2.0)
    x0 = max(0, math.floor(cx - half_w))
    y0 = max(0, math.floor(cy - half_h))
    x1 = min(frame_w, math.ceil(cx + half_w))
    y1 = min(frame_h, math.ceil(cy + half_h))
    if x1 <= x0 or y1 <= y0:
        return image, 0, 0
    return image[y0:y1, x0:x1], x0, y0


def map_to_full_frame(points: dict[str, Sample], x0: int, y0: int, t: float) -> dict[str, Sample]:
    """New Samples with the crop offset added back and PTS `t` stamped.

    Pure: builds fresh Samples (crop-relative inputs are not mutated).
    The offset is a translation only -- crops are never resized, so no
    scale factor is involved (geometry.py stays the sole owner of scaling).
    """
    return {
        name: Sample(t=t, x=s.x + x0, y=s.y + y0, visibility=s.visibility)
        for name, s in points.items()
    }


class RTMLibBackend:
    """rtmlib Body (COCO-17) backend -- the production default.

    Lazy-imports rtmlib in `__init__`; without the optional extra
    installed this raises `PoseUnavailableError` naming the install
    command. Uses `rtmlib.Body` in balanced mode on onnxruntime per the
    design brief.

    rtmlib returns EVERY detected person, so the single-athlete lock
    lives here via `AthleteLock`: the previously chosen bbox is kept in
    FULL-FRAME px, and each call's candidate bboxes are translated by
    `origin` (the given image's top-left in full-frame px) before the
    nearest-to-previous comparison. That is the invariant that makes the
    lock safe under StridedPose's regime changes -- full frame on
    bootstrap / after a miss / at rerun entry, ROI crops otherwise --
    because both sides of every comparison are always full-frame
    coordinates, a spotter inside a padded crop cannot steal the track.
    A miss (no people, or nothing confident) resets the lock and the
    next call re-acquires by largest bbox.
    """

    def __init__(self, mode: str = "balanced", device: str = "cpu") -> None:
        try:
            import rtmlib
        except ImportError as exc:
            raise PoseUnavailableError(
                "rtmlib is not installed. Pose estimation needs the optional "
                "'pose' extra: uv add rtmlib onnxruntime"
            ) from exc
        self._body = rtmlib.Body(mode=mode, backend="onnxruntime", device=device)
        self._lock = AthleteLock()

    def detect(
        self, image_bgr: np.ndarray, *, origin: tuple[float, float] = (0.0, 0.0)
    ) -> dict[str, Sample] | None:
        keypoints, scores = self._body(image_bgr)
        if keypoints is None or len(keypoints) == 0:
            self._lock.reset()
            return None
        chosen = self._lock.select([keypoints_bbox(person) for person in keypoints], origin)
        points = coco17_to_landmarks(keypoints[chosen], scores[chosen])
        if not points:
            self._lock.reset()
            return None
        return points


class MediaPipeBackend:
    """MediaPipe pose_landmarker task-API backend (optional alternative).

    Lazy-imports mediapipe in `__init__`; raises `PoseUnavailableError`
    naming the install command when missing. `model_path` is a
    pose_landmarker `.task` model asset (the tasks API does not download
    models itself). Runs in IMAGE mode with `num_poses=1` -- the
    landmarker keeps the most prominent person, which combined with
    StridedPose's athlete-centered crops is this backend's single-athlete
    story. This is also the only backend that emits heel/foot_index
    landmarks.
    """

    def __init__(self, model_path: str | Path) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError as exc:
            raise PoseUnavailableError(
                "mediapipe is not installed. This backend needs the optional "
                "'mediapipe' extra: uv add mediapipe"
            ) from exc
        self._mp = mp
        options = mp_vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    def detect(
        self, image_bgr: np.ndarray, *, origin: tuple[float, float] = (0.0, 0.0)
    ) -> dict[str, Sample] | None:
        # `origin` is accepted for protocol conformance but unused:
        # num_poses=1 keeps no cross-call person lock to translate.
        rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])  # mp.Image wants RGB
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        if not result.pose_landmarks:
            return None
        height, width = image_bgr.shape[:2]
        landmarks = [(lm.x, lm.y, lm.visibility) for lm in result.pose_landmarks[0]]
        points = mediapipe_landmarks_to_samples(landmarks, width=width, height=height)
        return points or None


class StridedPose:
    """Strided per-frame pose scheduling with an athlete-ROI fast path.

    Streaming pass: `feed(t, image, index)` runs the backend only when
    `index % stride == 0` (stride defaults to 2 -- half the pose cost for
    a segmentation-quality signal; rep windows get full rate later) and
    appends `(t, points | None)` to `results` for EVERY fed frame --
    stride-skipped frames and backend misses are both recorded as None so
    the record stays frame-aligned with the decode stream. Strictly
    per-frame per the streaming rule: the pipeline owns iteration, images
    are never retained.

    ROI: after a hit, the next backend run sees `crop_around(prev_bbox,
    image)` where prev_bbox is the last detection's landmark extent
    (full-frame px); backend output is mapped back to full-frame
    coordinates via `map_to_full_frame`, so callers only ever see
    full-frame px. A miss drops the ROI -- the next run searches the full
    frame (mirrors MarkerTracker's reset behavior). Every backend call is
    passed the given image's full-frame origin, so a backend's own person
    lock (`AthleteLock`) always compares full-frame coordinates and
    survives every regime change this scheduling induces (full->crop at
    ROI engagement, crop->full after a miss and at rerun entry/exit).

    Full-rate re-runs: `rerun_full_rate(window_frames)` runs the backend
    on EVERY frame of a rep window and returns a LandmarkSeries. The
    pipeline supplies a SECOND decode pass restricted to the windows --
    decode is cheap relative to pose inference, and re-decoding keeps the
    streaming pass from ever buffering frames (global streaming rule).
    """

    def __init__(self, backend: PoseBackend, stride: int = 2) -> None:
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        self.backend = backend
        self.stride = stride
        self.results: list[tuple[float, dict[str, Sample] | None]] = []
        self._prev_bbox: BBox | None = None

    def feed(self, t: float, image: np.ndarray, index: int) -> dict[str, Sample] | None:
        """Process one decoded frame; full-frame-mapped points on a hit.

        `index` is the decode stream's frame index (DecodedFrame.index)
        and only gates the stride phase; `t` (PTS seconds) is what gets
        recorded and stamped -- never index (PTS timebase constraint).
        """
        if index % self.stride != 0:
            self.results.append((t, None))
            return None
        points, self._prev_bbox = _detect_mapped(self.backend, image, t, self._prev_bbox)
        self.results.append((t, points))
        return points

    def series(self) -> LandmarkSeries:
        """The streaming pass's hits as a LandmarkSeries (None records
        dropped -- a missed frame contributes no LandmarkFrame, matching
        series.py's absent-means-undetected convention)."""
        return LandmarkSeries(
            [LandmarkFrame(t=t, points=points) for t, points in self.results if points is not None]
        )

    def rerun_full_rate(self, window_frames: Iterable[DecodedFrame]) -> LandmarkSeries:
        """Backend on EVERY window frame (no stride) -> LandmarkSeries.

        Uses fresh ROI state -- the window is somewhere back in the video,
        so the streaming pass's current bbox is stale for it; the first
        window frame is searched full-frame and the ROI re-locks within
        the window. Does not touch `results` or the streaming ROI.
        Consumes the iterable one frame at a time (streaming rule); misses
        simply contribute no frame, per-landmark gap handling deals with
        holes downstream.
        """
        frames: list[LandmarkFrame] = []
        prev_bbox: BBox | None = None
        for frame in window_frames:
            points, prev_bbox = _detect_mapped(self.backend, frame.image, frame.t, prev_bbox)
            if points is not None:
                frames.append(LandmarkFrame(t=frame.t, points=points))
        return LandmarkSeries(frames)


class NoOpPoseBackend:
    """A backend that never detects anyone (every ``detect`` returns None).

    This is what ``make_pose_backend("fake")`` returns: a model-free backend
    for pipeline smoke runs (e.g. ``powerpath analyze --pose fake``) where
    bar-only analysis is acceptable and no pose libraries are installed.
    Tests that need actual landmarks construct :class:`FakePoseBackend`
    directly with a script -- a name-only factory cannot supply one.
    """

    def detect(
        self, image_bgr: np.ndarray, *, origin: tuple[float, float] = (0.0, 0.0)
    ) -> dict[str, Sample] | None:
        return None


def make_pose_backend(name: str) -> PoseBackend:
    """Construct a pose backend by name: ``rtmlib`` | ``mediapipe`` | ``fake``.

    The factory the job runner and CLI resolve a backend through. Real
    backends stay lazy: their libraries are imported only inside the chosen
    backend's ``__init__``, so calling this with a name whose extra is not
    installed raises :class:`PoseUnavailableError` (naming the install
    command) and every other name costs nothing.

    ``"fake"`` returns a :class:`NoOpPoseBackend` (no landmarks ever) -- see
    its docstring; scripted fakes must be built directly. ``"mediapipe"``
    reads the required ``.task`` model asset path from the
    ``POWERPATH_MEDIAPIPE_MODEL`` environment variable (the tasks API does
    not download models itself) and raises ``ValueError`` when it is unset.
    An unknown name raises ``ValueError`` listing the options.
    """
    if name == "rtmlib":
        return RTMLibBackend()
    if name == "mediapipe":
        import os

        model_path = os.environ.get("POWERPATH_MEDIAPIPE_MODEL")
        if not model_path:
            raise ValueError(
                "the mediapipe backend needs a pose_landmarker .task model asset; "
                "set POWERPATH_MEDIAPIPE_MODEL to its path"
            )
        return MediaPipeBackend(model_path)
    if name == "fake":
        return NoOpPoseBackend()
    raise ValueError(f"unknown pose backend {name!r}; options: rtmlib, mediapipe, fake")


def _detect_mapped(
    backend: PoseBackend, image: np.ndarray, t: float, prev_bbox: BBox | None
) -> tuple[dict[str, Sample] | None, BBox | None]:
    """Run `backend` on `image` (cropped to prev_bbox when there is one,
    passing the crop's full-frame origin so backend person locks stay in
    full-frame coordinates), map the result to full-frame px with `t`
    stamped, and return it with the next ROI bbox (the detection's
    landmark extent; None on a miss -- an empty dict counts as a miss,
    it has no bbox to crop around)."""
    if prev_bbox is not None:
        crop, x0, y0 = crop_around(prev_bbox, image)
    else:
        crop, x0, y0 = image, 0, 0
    raw = backend.detect(crop, origin=(float(x0), float(y0)))
    if not raw:
        return None, None
    points = map_to_full_frame(raw, x0, y0, t)
    return points, points_bbox(points)

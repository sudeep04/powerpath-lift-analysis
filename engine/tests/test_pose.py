"""Tests for powerpath_engine.pose.

Per the global constraint there is NO model inference here: everything runs
against `FakePoseBackend` and pure mapping/selection functions fed plain
numpy arrays. rtmlib/mediapipe are optional extras that are NOT installed;
the two backend classes are only exercised on their unavailable-import
error path (forced via sys.modules so the tests are deterministic even if
someone installs the extras later).
"""

from __future__ import annotations

import math
import sys

import numpy as np
import pytest

from powerpath_engine.decode import DecodedFrame
from powerpath_engine.pose import (
    COCO17_LANDMARK_NAMES,
    MEDIAPIPE_LANDMARK_INDICES,
    MIN_LANDMARK_SCORE,
    AthleteLock,
    BBox,
    FakePoseBackend,
    MediaPipeBackend,
    PoseUnavailableError,
    RTMLibBackend,
    StridedPose,
    coco17_to_landmarks,
    crop_around,
    keypoints_bbox,
    map_to_full_frame,
    mediapipe_landmarks_to_samples,
    points_bbox,
    select_person,
)
from powerpath_engine.series import LandmarkSeries, Sample

FULL_SHAPE = (480, 640, 3)


def full_image() -> np.ndarray:
    """A 640x480 BGR frame with position-dependent values, so window
    equality checks verify WHERE a crop came from, not just its size."""
    return (np.arange(math.prod(FULL_SHAPE)) % 251).astype(np.uint8).reshape(FULL_SHAPE)


def torso_points(t: float = 0.0) -> dict[str, Sample]:
    """Four landmarks spanning the bbox (300, 200, 40, 100)."""
    return {
        "left_shoulder": Sample(t=t, x=300.0, y=200.0),
        "right_shoulder": Sample(t=t, x=340.0, y=200.0),
        "left_hip": Sample(t=t, x=300.0, y=300.0),
        "right_hip": Sample(t=t, x=340.0, y=300.0),
    }


def coco_person(cx: float, cy: float, half: float = 20.0) -> np.ndarray:
    """A (17, 2) keypoint array spread across a box centered on (cx, cy)."""
    xs = np.linspace(cx - half, cx + half, 17)
    ys = np.linspace(cy - 2.0 * half, cy + 2.0 * half, 17)
    return np.column_stack([xs, ys])


# ---------------------------------------------------------------------------
# FakePoseBackend
# ---------------------------------------------------------------------------


def test_fake_backend_counts_calls_and_passes_sequential_indices() -> None:
    received: list[int] = []

    def script(call_index: int) -> dict[str, Sample] | None:
        received.append(call_index)
        return None

    fake = FakePoseBackend(script)
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    assert fake.detect(image) is None
    assert fake.detect(image) is None
    assert fake.calls == 2
    assert received == [0, 1]
    assert fake.seen_shapes == [(48, 64, 3), (48, 64, 3)]


# ---------------------------------------------------------------------------
# Pure mapping: COCO-17 -> landmark names (RTMLib path)
# ---------------------------------------------------------------------------


def test_coco17_mapping_covers_13_names_and_skips_eyes_ears() -> None:
    keypoints = np.array([[float(i * 10), float(i * 10 + 5)] for i in range(17)])
    scores = np.ones(17)
    points = coco17_to_landmarks(keypoints, scores)

    assert set(points) == {
        "nose",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    }
    # Index -> name spot checks across the table.
    assert (points["nose"].x, points["nose"].y) == (0.0, 5.0)
    assert (points["left_shoulder"].x, points["left_shoulder"].y) == (50.0, 55.0)
    assert (points["right_ankle"].x, points["right_ankle"].y) == (160.0, 165.0)
    # t is a placeholder -- the caller (StridedPose) stamps PTS seconds.
    assert all(s.t == 0.0 for s in points.values())


def test_coco17_has_no_heel_or_foot_index_names() -> None:
    """COCO-17 stops at the ankles: heel/foot_index are MediaPipe-only
    names and downstream code must tolerate their absence."""
    mapped = {name for name in COCO17_LANDMARK_NAMES if name is not None}
    assert not any("heel" in name or "foot_index" in name for name in mapped)
    assert len(COCO17_LANDMARK_NAMES) == 17
    assert len(mapped) == 13


def test_coco17_mapping_omits_low_score_keypoints_and_clamps_visibility() -> None:
    keypoints = np.array([[float(i), float(i)] for i in range(17)])
    scores = np.ones(17)
    scores[9] = MIN_LANDMARK_SCORE - 0.05  # left_wrist: below the floor
    scores[0] = 1.4  # detector confidence can exceed 1; visibility must not
    points = coco17_to_landmarks(keypoints, scores)
    assert "left_wrist" not in points
    assert points["nose"].visibility == 1.0


def test_coco17_mapping_returns_empty_dict_when_nothing_is_confident() -> None:
    keypoints = np.zeros((17, 2))
    scores = np.zeros(17)
    assert coco17_to_landmarks(keypoints, scores) == {}


# ---------------------------------------------------------------------------
# Pure mapping: MediaPipe 33-landmark -> names
# ---------------------------------------------------------------------------


def test_mediapipe_mapping_scales_normalized_coords_and_includes_heels() -> None:
    landmarks = [(0.0, 0.0, 0.0)] * 33
    landmarks[MEDIAPIPE_LANDMARK_INDICES["nose"]] = (0.5, 0.25, 1.0)
    landmarks[MEDIAPIPE_LANDMARK_INDICES["left_heel"]] = (0.1, 0.9, 0.8)
    landmarks[MEDIAPIPE_LANDMARK_INDICES["right_foot_index"]] = (0.2, 0.95, 0.9)

    points = mediapipe_landmarks_to_samples(landmarks, width=640, height=480)

    # Normalized [0, 1] coords scale to full-frame px.
    assert (points["nose"].x, points["nose"].y) == (320.0, 120.0)
    assert (points["left_heel"].x, points["left_heel"].y) == (64.0, 432.0)
    assert points["left_heel"].visibility == pytest.approx(0.8)
    # Zero-visibility placeholder entries are omitted, not emitted at (0, 0).
    assert set(points) == {"nose", "left_heel", "right_foot_index"}


def test_mediapipe_index_table_covers_the_full_landmark_vocabulary() -> None:
    assert set(MEDIAPIPE_LANDMARK_INDICES) >= {
        "left_heel",
        "right_heel",
        "left_foot_index",
        "right_foot_index",
    }
    assert len(MEDIAPIPE_LANDMARK_INDICES) == 17


# ---------------------------------------------------------------------------
# Pure selection: single-athlete lock
# ---------------------------------------------------------------------------


def test_single_athlete_lock_picks_nearest_to_previous_bbox() -> None:
    """The brief's named test: two scripted people; the one nearest the
    previous athlete bbox wins even when the other is bigger."""
    athlete = keypoints_bbox(coco_person(100.0, 200.0, half=20.0))
    background = keypoints_bbox(coco_person(400.0, 200.0, half=60.0))
    prev = BBox(x=85.0, y=165.0, w=30.0, h=70.0)  # last known athlete bbox

    assert select_person([athlete, background], prev) == 0
    assert select_person([background, athlete], prev) == 1


def test_select_person_first_frame_picks_largest_bbox() -> None:
    small = keypoints_bbox(coco_person(100.0, 200.0, half=20.0))
    large = keypoints_bbox(coco_person(400.0, 200.0, half=60.0))
    assert select_person([small, large], None) == 1
    assert select_person([large, small], None) == 0


def test_select_person_rejects_empty_candidates() -> None:
    with pytest.raises(ValueError):
        select_person([], None)


def test_athlete_lock_holds_across_full_frame_to_crop_transition() -> None:
    """Regression: a full-frame hit followed by a cropped call whose decoy
    (spotter inside the padded crop) sits NEARER the stale full-frame
    coordinates than the athlete does in raw crop space. The lock keeps
    its bbox in full-frame px and translates crop candidates by the crop
    origin, so the true athlete must win anyway."""
    lock = AthleteLock()
    athlete_full = BBox(x=300.0, y=200.0, w=40.0, h=100.0)  # center (320, 250)
    decoy_full = BBox(x=500.0, y=60.0, w=30.0, h=80.0)  # smaller: loses frame 1
    assert lock.select([athlete_full, decoy_full], origin=(0.0, 0.0)) == 0

    # Next call is an ROI crop with origin (288, 170) (crop_around of the
    # athlete bbox at pad 0.3). In crop space the athlete is at center
    # (32, 80) -- full-frame (320, 250), dead on the previous center --
    # while the decoy at crop center (60, 150) is full-frame (348, 320).
    athlete_crop = BBox(x=12.0, y=30.0, w=40.0, h=100.0)
    decoy_crop = BBox(x=50.0, y=130.0, w=20.0, h=40.0)
    # The trap this test pins: comparing RAW crop coordinates against the
    # stale full-frame center would pick the decoy.
    stale_center = athlete_full.center
    assert math.dist(decoy_crop.center, stale_center) < math.dist(athlete_crop.center, stale_center)
    assert lock.select([athlete_crop, decoy_crop], origin=(288.0, 170.0)) == 0
    # The stored bbox is full-frame, ready for the next regime either way.
    assert lock.prev_bbox == BBox(x=300.0, y=200.0, w=40.0, h=100.0)

    # Crop -> full transition (post-miss reset / rerun entry hand the
    # backend full frames again): a larger person elsewhere still loses.
    big_decoy = BBox(x=480.0, y=40.0, w=120.0, h=200.0)
    assert lock.select([big_decoy, athlete_full], origin=(0.0, 0.0)) == 1


def test_athlete_lock_reset_clears_previous_bbox() -> None:
    lock = AthleteLock()
    anchor = BBox(x=100.0, y=100.0, w=50.0, h=50.0)
    assert lock.select([anchor]) == 0
    lock.reset()
    # After a reset the lock re-acquires by area, not by proximity to the
    # dropped bbox.
    near_prev_but_small = BBox(x=120.0, y=120.0, w=5.0, h=5.0)
    far_but_large = BBox(x=400.0, y=300.0, w=80.0, h=80.0)
    assert lock.select([near_prev_but_small, far_but_large]) == 1


def test_keypoints_bbox_spans_min_max() -> None:
    bbox = keypoints_bbox(coco_person(100.0, 200.0, half=20.0))
    assert bbox == BBox(x=80.0, y=160.0, w=40.0, h=80.0)


def test_points_bbox_spans_landmark_extent() -> None:
    assert points_bbox(torso_points()) == BBox(x=300.0, y=200.0, w=40.0, h=100.0)


# ---------------------------------------------------------------------------
# Pure crop helpers
# ---------------------------------------------------------------------------


def test_crop_around_pads_by_30_percent_per_side_and_reports_offset() -> None:
    image = full_image()
    crop, x0, y0 = crop_around(BBox(x=300.0, y=200.0, w=40.0, h=100.0), image, pad=0.3)
    # pad 0.3: 12px each side in x, 30px each side in y.
    assert (x0, y0) == (288, 170)
    assert crop.shape == (160, 64, 3)
    assert np.array_equal(crop, image[170:330, 288:352])


def test_crop_around_clamps_to_image_bounds() -> None:
    image = full_image()
    crop, x0, y0 = crop_around(BBox(x=5.0, y=5.0, w=40.0, h=40.0), image, pad=0.5)
    assert (x0, y0) == (0, 0)
    assert crop.shape == (65, 65, 3)
    assert np.array_equal(crop, image[0:65, 0:65])


def test_crop_around_enforces_minimum_extent_on_degenerate_bbox() -> None:
    """A single-landmark hit yields a zero-area bbox; the crop must still
    be at least MIN_CROP_EXTENT_PX (32) per side, centered on the bbox,
    not a ~1px sliver that guarantees the next detection misses."""
    image = full_image()
    crop, x0, y0 = crop_around(BBox(x=320.0, y=240.0, w=0.0, h=0.0), image)
    assert (x0, y0) == (304, 224)
    assert crop.shape == (32, 32, 3)
    assert np.array_equal(crop, image[224:256, 304:336])

    # Only the deficient dimension expands: a tall zero-width bbox keeps
    # its padded height and gets the 32px minimum width.
    crop, x0, y0 = crop_around(BBox(x=320.0, y=150.0, w=0.0, h=100.0), image)
    assert crop.shape == (160, 32, 3)


def test_crop_around_minimum_extent_clamps_at_frame_edge() -> None:
    image = full_image()
    crop, x0, y0 = crop_around(BBox(x=2.0, y=2.0, w=0.0, h=0.0), image)
    assert (x0, y0) == (0, 0)
    assert crop.shape == (18, 18, 3)  # 32px window clamped to the frame


def test_crop_around_off_frame_bbox_falls_back_to_full_frame() -> None:
    image = full_image()
    crop, x0, y0 = crop_around(BBox(x=1000.0, y=1000.0, w=50.0, h=50.0), image)
    assert (x0, y0) == (0, 0)
    assert crop.shape == FULL_SHAPE


def test_map_to_full_frame_offsets_coords_and_stamps_t() -> None:
    crop_points = {"nose": Sample(t=0.0, x=10.0, y=20.0, visibility=0.7)}
    mapped = map_to_full_frame(crop_points, x0=288, y0=170, t=1.25)
    assert mapped["nose"].x == 298.0
    assert mapped["nose"].y == 190.0
    assert mapped["nose"].t == 1.25
    assert mapped["nose"].visibility == 0.7
    # Pure: the input samples are not mutated.
    assert crop_points["nose"].x == 10.0
    assert crop_points["nose"].t == 0.0


# ---------------------------------------------------------------------------
# StridedPose scheduling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 5, 7, 8])
def test_stride_2_calls_backend_exactly_ceil_n_over_2_times(n: int) -> None:
    """The brief's named test: stride 2 over n frames -> ceil(n/2) calls."""
    fake = FakePoseBackend(lambda call_index: None)
    pose = StridedPose(fake, stride=2)
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    for index in range(n):
        pose.feed(index / 60.0, image, index)
    assert fake.calls == math.ceil(n / 2)


def test_strided_pose_records_skipped_frames_as_none() -> None:
    fake = FakePoseBackend(lambda call_index: torso_points())
    pose = StridedPose(fake, stride=2)
    image = full_image()
    for index in range(4):
        pose.feed(index / 60.0, image, index)

    assert [t for t, _ in pose.results] == pytest.approx([0.0, 1 / 60, 2 / 60, 3 / 60])
    assert pose.results[0][1] is not None
    assert pose.results[1][1] is None  # skipped by stride
    assert pose.results[2][1] is not None
    assert pose.results[3][1] is None


def test_strided_pose_default_stride_is_2_and_stride_below_1_rejected() -> None:
    fake = FakePoseBackend(lambda call_index: None)
    assert StridedPose(fake).stride == 2
    with pytest.raises(ValueError):
        StridedPose(fake, stride=0)


def test_stride_1_runs_backend_on_every_frame() -> None:
    fake = FakePoseBackend(lambda call_index: None)
    pose = StridedPose(fake, stride=1)
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    for index in range(5):
        pose.feed(index / 60.0, image, index)
    assert fake.calls == 5


# ---------------------------------------------------------------------------
# StridedPose ROI flow
# ---------------------------------------------------------------------------


def test_roi_mapping_returns_full_frame_coords() -> None:
    """The brief's named test: the backend sees a crop (asserted via the
    fake's recorded shapes) but feed() returns full-frame coordinates."""
    script_results = [
        torso_points(),  # call 0: full frame, bbox (300, 200, 40, 100)
        {"nose": Sample(t=0.0, x=10.0, y=20.0)},  # call 1: CROP coordinates
    ]
    fake = FakePoseBackend(lambda call_index: script_results[call_index])
    pose = StridedPose(fake, stride=2)
    image = full_image()

    first = pose.feed(0.0, image, 0)
    assert pose.feed(1 / 60.0, image, 1) is None  # skipped by stride
    second = pose.feed(2 / 60.0, image, 2)

    # Call 0 saw the full frame; call 1 saw the 30%-padded bbox crop.
    assert fake.seen_shapes[0] == FULL_SHAPE
    assert fake.seen_shapes[1] == (160, 64, 3)
    # Each call carries its full-frame origin so backend person locks can
    # stay in full-frame coordinates across the full->crop transition.
    assert fake.seen_origins == [(0.0, 0.0), (288.0, 170.0)]

    assert first is not None
    assert first["left_shoulder"].x == 300.0  # full-frame pass-through
    assert second is not None
    assert second["nose"].x == 288.0 + 10.0  # crop offset added back
    assert second["nose"].y == 170.0 + 20.0
    assert second["nose"].t == pytest.approx(2 / 60)


def test_strided_pose_miss_resets_to_full_frame_search() -> None:
    script_results = [torso_points(), None, torso_points()]
    fake = FakePoseBackend(lambda call_index: script_results[call_index])
    pose = StridedPose(fake, stride=1)
    image = full_image()

    pose.feed(0.0, image, 0)
    assert pose.feed(1 / 60.0, image, 1) is None  # backend miss
    pose.feed(2 / 60.0, image, 2)

    assert fake.seen_shapes[0] == FULL_SHAPE
    assert fake.seen_shapes[1] == (160, 64, 3)  # ROI engaged after the hit
    assert fake.seen_shapes[2] == FULL_SHAPE  # miss dropped the ROI
    assert pose.results[1][1] is None


def test_strided_pose_treats_empty_detection_as_miss() -> None:
    """An all-below-threshold detection comes back as {} -- that is a
    miss (recorded None, ROI dropped), not a zero-landmark hit that
    would crash points_bbox."""
    script_results = [torso_points(), {}, torso_points()]
    fake = FakePoseBackend(lambda call_index: script_results[call_index])
    pose = StridedPose(fake, stride=1)
    image = full_image()

    pose.feed(0.0, image, 0)
    assert pose.feed(1 / 60.0, image, 1) is None
    pose.feed(2 / 60.0, image, 2)

    assert pose.results[1][1] is None
    assert fake.seen_shapes[2] == FULL_SHAPE  # the {} miss dropped the ROI


# ---------------------------------------------------------------------------
# StridedPose series() and rerun_full_rate()
# ---------------------------------------------------------------------------


def test_series_keeps_hit_frames_and_drops_none_records() -> None:
    script_results = [torso_points(), None]
    fake = FakePoseBackend(lambda call_index: script_results[call_index])
    pose = StridedPose(fake, stride=2)
    image = full_image()
    for index in range(4):  # backend runs at 0 and 2; 2 is a miss
        pose.feed(index / 60.0, image, index)

    series = pose.series()
    assert isinstance(series, LandmarkSeries)
    assert [frame.t for frame in series.frames] == [0.0]
    assert series.frames[0].points["left_hip"].y == 300.0


def test_rerun_full_rate_runs_backend_on_every_window_frame() -> None:
    fake = FakePoseBackend(lambda call_index: torso_points())
    pose = StridedPose(fake, stride=2)
    image = full_image()
    window = [DecodedFrame(t=i / 60.0, image=image, index=i) for i in range(4)]

    series = pose.rerun_full_rate(window)

    assert fake.calls == 4  # no stride inside a rep window
    assert isinstance(series, LandmarkSeries)
    assert [frame.t for frame in series.frames] == pytest.approx([i / 60 for i in range(4)])


def test_rerun_full_rate_starts_full_frame_and_leaves_streaming_state_alone() -> None:
    fake = FakePoseBackend(lambda call_index: torso_points())
    pose = StridedPose(fake, stride=2)
    image = full_image()

    pose.feed(0.0, image, 0)  # streaming pass: sets the ROI
    assert fake.seen_shapes == [FULL_SHAPE]

    series = pose.rerun_full_rate([DecodedFrame(t=5.0, image=image, index=300)])

    # Fresh ROI state: the window's first frame is searched full-frame even
    # though the streaming pass has a live athlete bbox.
    assert fake.seen_shapes[1] == FULL_SHAPE
    # The streaming record is untouched by the window re-run.
    assert len(pose.results) == 1
    assert series.frames[0].t == 5.0
    # And the streaming ROI survives: the next strided feed still crops.
    pose.feed(2 / 60.0, image, 2)
    assert fake.seen_shapes[-1] == (160, 64, 3)


# ---------------------------------------------------------------------------
# Optional backends: unavailable-import error path only (never inference)
# ---------------------------------------------------------------------------


def test_rtmlib_backend_unavailable_error_names_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The brief's named test: the install hint must name the extra."""
    monkeypatch.setitem(sys.modules, "rtmlib", None)  # forces ImportError
    with pytest.raises(PoseUnavailableError) as excinfo:
        RTMLibBackend()
    assert "uv add rtmlib onnxruntime" in str(excinfo.value)


def test_mediapipe_backend_unavailable_error_names_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "mediapipe", None)  # forces ImportError
    with pytest.raises(PoseUnavailableError) as excinfo:
        MediaPipeBackend("pose_landmarker.task")
    assert "uv add mediapipe" in str(excinfo.value)

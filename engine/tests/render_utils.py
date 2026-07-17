"""Render a synthetic lift (tests/synthetic.py) to a real, decodable mp4.

The bridge between the pure trajectory fixtures and the pipeline's video
input: the bar trajectory is drawn as a saturated magenta marker dot (what
``bar.MarkerTracker`` tracks) and a 450mm plate disc is drawn near the floor
(what ``calibration.detect_plate_circle`` calibrates from), so a rendered
clip runs the REAL decode -> marker-tracking -> calibration path end to end
with known ground truth. Pose stays scripted (the global no-model-inference
constraint): :func:`scripted_pose_backend` wraps a ``FakePoseBackend`` whose
script answers each call with the synthetic landmark frame the pipeline is
looking at, converted to image pixels with the same rendering transform.

World -> image transform (:class:`LiftRenderSpec`): synthetic fixtures are
in centimeters, y-up, x centered on 0; the render maps ``x_px = cx + x_cm /
cm_per_px`` and ``y_px = height - y_cm / cm_per_px``. At the default 0.2
cm/px a 480x640 frame fits the whole bar path of every generator; upper-body
landmarks of a standing athlete land above the frame top (negative y_px),
which nothing in the engine minds -- pose samples are numbers, not pixels.

Frames are generated and encoded ONE AT A TIME (PyAV), so building even a
long E2E clip never buffers the frame list.
"""

from __future__ import annotations

from collections.abc import Container
from dataclasses import dataclass
from pathlib import Path

import av
import cv2
import numpy as np
from synthetic import SyntheticLift

from powerpath_engine.pose import FakePoseBackend
from powerpath_engine.series import Sample

# Grays matching the calibration test fixtures: light background, dark plate.
BG_LEVEL = 180
PLATE_LEVEL = 40

# Saturated magenta -- hue 150, dead center of bar.MarkerSpec's [140, 170].
MARKER_BGR = (255, 0, 255)

# Standard plate radius in cm (matches geometry.STANDARD_PLATE_DIAMETER_CM/2).
PLATE_RADIUS_CM = 22.5


@dataclass(frozen=True)
class LiftRenderSpec:
    """The world(cm, y-up) -> image(px, y-down) transform used by a render."""

    width: int
    height: int
    cm_per_px: float
    cx: float
    fps: int

    def x_px(self, x_cm: float) -> float:
        return self.cx + x_cm / self.cm_per_px

    def y_px(self, y_cm: float) -> float:
        return self.height - y_cm / self.cm_per_px


def render_lift_video(
    path: str | Path,
    lift: SyntheticLift,
    fps: int,
    *,
    width: int = 480,
    height: int = 640,
    cm_per_px: float = 0.2,
    marker_radius_px: int = 13,
    plate_center_cm: tuple[float, float] = (0.0, 24.0),
    draw_marker: bool = True,
    skip_marker_frames: Container[int] = (),
) -> LiftRenderSpec:
    """Encode ``lift`` to an h264/mp4 at ``path``; returns the transform used.

    One frame per bar sample (the synthetic grid is uniform at ``fps``), so
    decoded PTS line up with the fixture timestamps. Every frame carries the
    plate disc; the marker dot follows the bar trajectory except on frames
    listed in ``skip_marker_frames`` (to fabricate tracking gaps) or when
    ``draw_marker`` is False (a marker-less video).

    Defaults are chosen so the REAL calibration accepts the render: at 0.2
    cm/px the plate disc is ~112px in radius (17.6% of the 640px frame
    height -- inside the 15-45% Hough band) and the derived 2.0 mm/px scale
    sits inside calibration's 0.5-3.0 mm/px plausibility band; the 13px
    marker radius makes the 50mm-sleeve cross-check agree with the plate
    scale within a few percent.
    """
    spec = LiftRenderSpec(width=width, height=height, cm_per_px=cm_per_px, cx=width / 2.0, fps=fps)
    plate_center = (
        int(round(spec.x_px(plate_center_cm[0]))),
        int(round(spec.y_px(plate_center_cm[1]))),
    )
    plate_radius = int(round(PLATE_RADIUS_CM / cm_per_px))

    container = av.open(str(path), mode="w")
    try:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for index, sample in enumerate(lift.bar.samples):
            image = np.full((height, width, 3), BG_LEVEL, dtype=np.uint8)
            cv2.circle(image, plate_center, plate_radius, (PLATE_LEVEL,) * 3, -1)
            if draw_marker and index not in skip_marker_frames:
                marker = (int(round(spec.x_px(sample.x))), int(round(spec.y_px(sample.y))))
                cv2.circle(image, marker, marker_radius_px, MARKER_BGR, -1)
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
    return spec


def scripted_pose_backend(
    lift: SyntheticLift, spec: LiftRenderSpec, stride: int
) -> FakePoseBackend:
    """A ``FakePoseBackend`` scripted from ``lift``'s landmarks, in image px.

    Relies on the pipeline's deterministic call pattern: with no full-rate
    rerun, ``StridedPose`` invokes the backend exactly on frame indices
    ``0, stride, 2*stride, ...`` in order, so detect-call ``k`` is frame
    ``k * stride``. The script converts that frame's landmarks through
    ``spec`` and subtracts the call's crop origin (``FakePoseBackend``
    records ``seen_origins`` *before* running the script), because a
    backend must answer in the coordinates of the image it was given --
    ``StridedPose`` maps them back to full frame itself.
    """
    frames = lift.landmarks.frames
    backend: FakePoseBackend | None = None

    def script(call_index: int) -> dict[str, Sample] | None:
        frame_index = call_index * stride
        if frame_index >= len(frames):
            return None
        assert backend is not None
        ox, oy = backend.seen_origins[-1]
        return {
            name: Sample(
                t=0.0,
                x=spec.x_px(s.x) - ox,
                y=spec.y_px(s.y) - oy,
                visibility=1.0,
            )
            for name, s in frames[frame_index].points.items()
        }

    backend = FakePoseBackend(script)
    return backend

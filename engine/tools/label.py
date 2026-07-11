"""Click-to-label tool: step through a video and click the bar-end center at
each requested keyframe; writes ``VIDEO.labels.json`` next to the video.

Usage (engine venv)::

    python tools/label.py VIDEO --keyframes setup,knee,hip,receive,standing

Controls: trackbar or arrow keys (also ``.``/``,``) to step frames, left
click to record the pending keyframe, ``u`` to undo the last click, Esc to
abort without writing. The labels file is written automatically once every
keyframe has been clicked. Output schema (exact)::

    {"video": ..., "clicks": [{"name", "t", "frame_index", "x", "y"}]}

All non-GUI logic -- keyframe schedule parsing, click-record bookkeeping,
JSON serialization, decode-backed frame stepping -- lives in pure
module-level functions/classes so it can be unit tested; the cv2 event loop
(`run_gui`) is a thin untested shell around them. No engine imports beyond
`powerpath_engine.decode`.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import cv2

from powerpath_engine.decode import DecodedFrame, frames, probe

DEFAULT_KEYFRAMES = "setup,knee,hip,receive,standing"

# waitKeyEx arrow codes differ per platform (macOS/Linux/Windows); '.'/','
# always work as fallbacks.
_KEYS_NEXT = {3, 63235, 65363, 2555904, ord(".")}
_KEYS_PREV = {2, 63234, 65361, 2424832, ord(",")}
_KEY_ESC = 27


def parse_keyframes(spec: str) -> list[str]:
    """Parse a ``--keyframes`` spec ("setup,knee,...") into an ordered list.

    Whitespace around names is stripped; empty names and duplicates are
    rejected (a duplicate keyframe would make the click schedule ambiguous).
    """
    names = [token.strip() for token in spec.split(",")]
    if any(not name for name in names):
        raise ValueError(f"--keyframes contains an empty name: {spec!r}")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"--keyframes contains duplicate name(s): {duplicates}")
    return names


class LabelSession:
    """Click-record bookkeeping for one labeling run.

    Keyframes are labeled strictly in schedule order: `pending()` names the
    next keyframe awaiting a click, `record_click` attaches a click to it,
    and `undo` pops the most recent click (reopening its keyframe).
    """

    def __init__(self, video: str, keyframes: list[str]) -> None:
        self.video = video
        self.keyframes = list(keyframes)
        self.clicks: list[dict] = []

    def pending(self) -> str | None:
        """Name of the next keyframe to label, or None when complete."""
        if len(self.clicks) < len(self.keyframes):
            return self.keyframes[len(self.clicks)]
        return None

    @property
    def complete(self) -> bool:
        return self.pending() is None

    def record_click(self, t: float, frame_index: int, x: int, y: int) -> dict:
        name = self.pending()
        if name is None:
            raise ValueError("all keyframes are already labeled")
        click = {
            "name": name,
            "t": float(t),
            "frame_index": int(frame_index),
            "x": int(x),
            "y": int(y),
        }
        self.clicks.append(click)
        return click

    def undo(self) -> dict | None:
        """Remove and return the most recent click (None if there are none)."""
        if not self.clicks:
            return None
        return self.clicks.pop()

    def to_json(self) -> dict:
        return {"video": self.video, "clicks": list(self.clicks)}


def labels_path_for(video: str | Path) -> Path:
    """`VIDEO.labels.json` sits next to the video: lift.mp4 -> lift.mp4.labels.json."""
    video = Path(video)
    return video.with_name(video.name + ".labels.json")


def write_labels(session: LabelSession, path: str | Path) -> None:
    Path(path).write_text(json.dumps(session.to_json(), indent=2) + "\n")


class FrameStepper:
    """Seekable view over the streaming `decode.frames` generator.

    Forward seeks pull the generator; backward seeks close and restart it
    (decode is deliberately single-pass, so this is the only way back).
    Holds exactly one decoded frame at a time. Seeking past the end leaves
    `current` on the last frame.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = path
        self._gen: Iterator[DecodedFrame] | None = None
        self.current: DecodedFrame | None = None

    def seek(self, index: int) -> DecodedFrame | None:
        index = max(0, index)
        if self.current is not None and self.current.index == index:
            return self.current
        if self._gen is None or self.current is None or index < self.current.index:
            self._restart()
        gen = self._gen
        if gen is not None:
            for frame in gen:
                self.current = frame
                if frame.index >= index:
                    break
        return self.current

    def _restart(self) -> None:
        if self._gen is not None:
            self._gen.close()
        self._gen = frames(self._path)
        self.current = None


def _annotate(frame: DecodedFrame, session: LabelSession):
    image = frame.image.copy()
    pending = session.pending()
    prompt = f"click: {pending} (u undo, Esc abort)" if pending else "done"
    text = f"t={frame.t:.3f}s #{frame.index}  {prompt}"
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    for click in session.clicks:
        if click["frame_index"] == frame.index:
            cv2.drawMarker(image, (click["x"], click["y"]), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
    return image


def run_gui(video: Path, session: LabelSession) -> bool:
    """Interactive cv2 loop (not unit tested). True when all keyframes were
    labeled; False when the user aborted with Esc."""
    meta = probe(video)
    stepper = FrameStepper(video)
    max_index = max(int(round(meta.duration_s * meta.fps_avg)) - 1, 1)
    window = "powerpath label"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    state = {"index": 0}
    cv2.createTrackbar("frame", window, 0, max_index, lambda pos: state.update(index=pos))

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        frame = stepper.current
        if event == cv2.EVENT_LBUTTONDOWN and frame is not None and not session.complete:
            session.record_click(t=frame.t, frame_index=frame.index, x=x, y=y)

    cv2.setMouseCallback(window, on_mouse)
    try:
        while not session.complete:
            frame = stepper.seek(state["index"])
            if frame is None:
                raise SystemExit(f"{video}: no decodable frames")
            state["index"] = frame.index  # clamp after end-of-video seeks
            cv2.setTrackbarPos("frame", window, min(frame.index, max_index))
            cv2.imshow(window, _annotate(frame, session))
            key = cv2.waitKeyEx(30)
            if key == _KEY_ESC:
                return False
            if key in _KEYS_NEXT:
                state["index"] += 1
            elif key in _KEYS_PREV:
                state["index"] -= 1
            elif key == ord("u"):
                session.undo()
        return True
    finally:
        cv2.destroyWindow(window)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Label bar-end positions at keyframes.")
    parser.add_argument("video", type=Path)
    parser.add_argument(
        "--keyframes",
        default=DEFAULT_KEYFRAMES,
        help=f"comma-separated keyframe names to label in order (default: {DEFAULT_KEYFRAMES})",
    )
    args = parser.parse_args(argv)
    try:
        keyframes = parse_keyframes(args.keyframes)
    except ValueError as exc:
        parser.error(str(exc))
    session = LabelSession(str(args.video), keyframes)
    if not run_gui(args.video, session):
        print("aborted; nothing written")
        return 1
    out = labels_path_for(args.video)
    write_labels(session, out)
    print(f"wrote {out} ({len(session.clicks)} click(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

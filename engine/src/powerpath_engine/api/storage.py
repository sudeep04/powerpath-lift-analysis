"""On-disk library layout for uploaded videos and their analysis artifacts.

Layout (a global constraint):

    {library}/{YYYY-MM-DD}/{video_id}/original.<ext>
    {library}/{YYYY-MM-DD}/{video_id}/metrics.json
    {library}/{YYYY-MM-DD}/{video_id}/overlay.json

``{library}`` defaults to ``~/PowerPath/library`` and is overridable with the
``POWERPATH_LIBRARY`` environment variable (tests point it at a tmp dir). The
SQLite database lives at ``{library}/powerpath.db`` so a library directory is
fully self-contained -- copy it and you have moved the whole app's state.

This module owns *paths and bytes only*; it never opens the database.
"""

from __future__ import annotations

import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

DEFAULT_LIBRARY = Path.home() / "PowerPath" / "library"
_SAFE_EXTS = {
    ".mp4",
    ".mov",
    ".m4v",
    ".avi",
    ".mkv",
    ".webm",
}


def resolve_library_dir(library_dir: str | os.PathLike[str] | None = None) -> Path:
    """Pick the library root: explicit arg > ``POWERPATH_LIBRARY`` > default."""
    if library_dir is not None:
        return Path(library_dir).expanduser()
    env = os.environ.get("POWERPATH_LIBRARY")
    if env:
        return Path(env).expanduser()
    return DEFAULT_LIBRARY


def _safe_ext(filename: str | None) -> str:
    """Return a lowercased, dot-prefixed, whitelisted extension (default .mp4)."""
    if not filename:
        return ".mp4"
    ext = Path(filename).suffix.lower()
    if ext in _SAFE_EXTS:
        return ext
    return ".mp4"


class Storage:
    """Filesystem gateway for one library root."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)

    def db_path(self) -> str:
        return str(self.root / "powerpath.db")

    def _date_str(self, when: datetime | None = None) -> str:
        return (when or datetime.now(UTC)).strftime("%Y-%m-%d")

    def video_dir(self, video_id: str, date_str: str) -> Path:
        return self.root / date_str / video_id

    def store_upload(
        self, video_id: str, filename: str | None, source: BinaryIO
    ) -> tuple[Path, Path, str, str]:
        """Persist an uploaded stream. Returns (dir_path, file_path, ext, date_str)."""
        ext = _safe_ext(filename)
        date_str = self._date_str()
        dest_dir = self.video_dir(video_id, date_str)
        dest_dir.mkdir(parents=True, exist_ok=True)
        file_path = dest_dir / f"original{ext}"
        source.seek(0)
        with file_path.open("wb") as out:
            shutil.copyfileobj(source, out, length=1024 * 1024)
        return dest_dir, file_path, ext, date_str

    def metrics_path(self, dir_path: str | os.PathLike[str]) -> Path:
        return Path(dir_path) / "metrics.json"

    def overlay_path(self, dir_path: str | os.PathLike[str]) -> Path:
        return Path(dir_path) / "overlay.json"

    def remove_video_dir(self, dir_path: str | os.PathLike[str]) -> None:
        """Remove a video's directory tree, ignoring a missing directory."""
        shutil.rmtree(dir_path, ignore_errors=True)

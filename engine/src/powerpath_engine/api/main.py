"""FastAPI application factory for the PowerPath engine service.

``create_app(engine_runner=..., library_dir=...)`` builds the app; both
arguments are injectable so tests can supply a fake analysis runner and a tmp
library directory. The service binds ``127.0.0.1:8400`` (a global constraint --
localhost only) and speaks the exact JSON contract the Next.js UI is built
against; the field names in :func:`_video_summary` are load-bearing (getting
them wrong renders "Invalid Date"/undefined in the UI).
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from powerpath_engine import registry
from powerpath_engine.api import db
from powerpath_engine.api.jobs import JobContext, JobManager, Runner, default_runner
from powerpath_engine.api.storage import Storage, resolve_library_dir

# Only the UI dev server may call the API from a browser.
_CORS_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]

_MEDIA_TYPES = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
}


def _job_dict(row: Any | None) -> dict[str, Any]:
    """Serialize a jobs row to the {state, progress, stage, error} contract."""
    if row is None:
        return {"state": "QUEUED", "progress": 0, "stage": None, "error": None}
    return {
        "state": row["state"],
        "progress": row["progress"],
        "stage": row["stage"],
        "error": row["error"],
    }


def _display_name(movement: str) -> str:
    try:
        return registry.get(movement).display_name
    except registry.UnknownMovementError:
        return movement


def _video_summary(video: Any, job: Any | None, analysis: Any | None) -> dict[str, Any]:
    """Build one GET /api/videos item with the EXACT contract keys."""
    return {
        "video_id": video["id"],
        "movement": video["movement"],
        "display_name": _display_name(video["movement"]),
        "filmed_at": video["filmed_at"],
        "load_kg": video["load_kg"],
        "job": _job_dict(job),
        "rep_count": analysis["rep_count"] if analysis is not None else None,
        "best_score": analysis["best_score"] if analysis is not None else None,
        "job_id": job["id"] if job is not None else None,
    }


def _job_context_from_rows(video: Any, job_id: str) -> JobContext:
    return JobContext(
        job_id=job_id,
        video_id=video["id"],
        movement=video["movement"],
        load_kg=video["load_kg"],
        recalibrate=bool(video["recalibrate"]),
        video_path=video["file_path"],
        output_dir=video["dir_path"],
    )


def create_app(
    engine_runner: Runner | None = None,
    library_dir: str | os.PathLike[str] | None = None,
) -> FastAPI:
    storage = Storage(resolve_library_dir(library_dir))
    db_path = storage.db_path()

    # One-time DB setup + crash recovery, done synchronously at construction so
    # that a caller who just wants to assert the recovery happened (the restart
    # re-queue test) sees QUEUED immediately after create_app returns.
    conn = db.connect(db_path)
    try:
        db.init_db(conn)
        db.requeue_orphaned_jobs(conn)
    finally:
        conn.close()

    runner = engine_runner or default_runner
    # Default runner is picklable -> real process pool. An injected runner (a
    # test fake) may be a closure -> run it in-process on a thread pool instead.
    job_manager = JobManager(db_path, runner, use_process_pool=engine_runner is None)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        job_manager.start()
        # Re-run anything left QUEUED (including jobs just re-queued from a dead
        # process) so recovery actually completes rather than stalling.
        recover = db.connect(db_path)
        try:
            for job in db.queued_jobs(recover):
                video = db.get_video(recover, job["video_id"])
                if video is not None:
                    job_manager.submit(_job_context_from_rows(video, job["id"]))
        finally:
            recover.close()
        try:
            yield
        finally:
            job_manager.shutdown(wait=False)

    app = FastAPI(title="PowerPath API", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.storage = storage
    app.state.db_path = db_path
    app.state.job_manager = job_manager

    # --- routes -----------------------------------------------------------

    @app.post("/api/videos")
    def upload_video(
        file: UploadFile,
        movement: str = Form(...),
        load_kg: float = Form(...),
        recalibrate: bool = Form(False),
    ) -> dict[str, str]:
        if movement not in registry.all_keys():
            raise HTTPException(
                status_code=422,
                detail=f"unknown movement '{movement}'; available: {registry.all_keys()}",
            )
        video_id = db.new_id()
        dir_path, file_path, ext, _date = storage.store_upload(video_id, file.filename, file.file)
        conn = db.connect(db_path)
        try:
            db.create_video(
                conn,
                video_id=video_id,
                movement=movement,
                load_kg=load_kg,
                recalibrate=recalibrate,
                original_name=file.filename,
                ext=ext,
                dir_path=str(dir_path),
                file_path=str(file_path),
            )
            job_id = db.create_job(conn, video_id)
        finally:
            conn.close()
        job_manager.submit(
            JobContext(
                job_id=job_id,
                video_id=video_id,
                movement=movement,
                load_kg=load_kg,
                recalibrate=recalibrate,
                video_path=str(file_path),
                output_dir=str(dir_path),
            )
        )
        return {"video_id": video_id, "job_id": job_id}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        conn = db.connect(db_path)
        try:
            row = db.get_job(conn, job_id)
        finally:
            conn.close()
        if row is None:
            raise HTTPException(status_code=404, detail="job not found")
        return _job_dict(row)

    @app.get("/api/videos")
    def list_videos() -> list[dict[str, Any]]:
        conn = db.connect(db_path)
        try:
            out = []
            for video in db.all_videos(conn):
                job = db.latest_job_for_video(conn, video["id"])
                analysis = db.latest_analysis_for_video(conn, video["id"])
                out.append(_video_summary(video, job, analysis))
        finally:
            conn.close()
        return out

    @app.get("/api/videos/{video_id}/analysis")
    def get_analysis(video_id: str) -> Any:
        return _read_artifact(video_id, storage.metrics_path, "analysis")

    @app.get("/api/videos/{video_id}/overlay")
    def get_overlay(video_id: str) -> Any:
        return _read_artifact(video_id, storage.overlay_path, "overlay")

    def _read_artifact(video_id: str, path_fn: Any, label: str) -> Any:
        conn = db.connect(db_path)
        try:
            video = db.get_video(conn, video_id)
        finally:
            conn.close()
        if video is None:
            raise HTTPException(status_code=404, detail="video not found")
        artifact = path_fn(video["dir_path"])
        if not Path(artifact).exists():
            raise HTTPException(status_code=404, detail=f"{label} not ready")
        with open(artifact) as fh:
            return json.load(fh)

    @app.get("/api/videos/{video_id}/file")
    def get_file(video_id: str) -> FileResponse:
        conn = db.connect(db_path)
        try:
            video = db.get_video(conn, video_id)
        finally:
            conn.close()
        if video is None:
            raise HTTPException(status_code=404, detail="video not found")
        file_path = video["file_path"]
        if not Path(file_path).exists():
            raise HTTPException(status_code=404, detail="file not found")
        media_type = _MEDIA_TYPES.get(video["ext"], "application/octet-stream")
        # Starlette's FileResponse honours Range requests automatically.
        return FileResponse(file_path, media_type=media_type)

    @app.delete("/api/videos/{video_id}")
    def delete_video(video_id: str) -> dict[str, str]:
        conn = db.connect(db_path)
        try:
            video = db.get_video(conn, video_id)
            if video is None:
                raise HTTPException(status_code=404, detail="video not found")
            storage.remove_video_dir(video["dir_path"])
            db.delete_video(conn, video_id)
        finally:
            conn.close()
        return {"deleted": video_id}

    @app.get("/api/movements")
    def list_movements() -> list[dict[str, str]]:
        return [
            {"key": cfg.key, "display_name": cfg.display_name, "family": cfg.family}
            for cfg in registry.all_configs()
        ]

    return app


def run() -> None:
    """Console-script entrypoint (``powerpath-api``): serve on 127.0.0.1:8400."""
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=8400)

"""Tests for the FastAPI service, SQLite persistence, and job worker (Task 10).

Every test injects a *fake* analysis runner (never the real pipeline -- a global
constraint) and points the library at a tmp dir, so nothing decodes video or runs
model inference. Because an injected runner triggers the in-process thread-pool
path in :class:`~powerpath_engine.api.jobs.JobManager`, the fake can be a local
closure with no pickling worries.

Coverage mirrors the task brief: upload happy path (QUEUED -> DONE with fake
results retrievable), invalid movement -> 422, failing analysis -> FAILED with
the error surfaced, double-submit -> two distinct jobs run sequentially (pool
size 1), restart re-queue of an orphaned RUNNING job, DELETE removes rows +
files, the movements endpoint, and the EXACT GET /api/videos contract keys.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from contract_utils import assert_metrics_contract, assert_overlay_contract
from fastapi.testclient import TestClient

from powerpath_engine.api import db
from powerpath_engine.api.jobs import JobContext, RunnerResult, default_runner
from powerpath_engine.api.main import create_app
from powerpath_engine.api.storage import Storage

# A tiny, non-empty payload standing in for an uploaded video's bytes.
FAKE_VIDEO_BYTES = b"\x00\x00\x00\x18ftypmp42" + b"\x11" * 512


def _write_artifacts(ctx: JobContext, rep_count: int, best_score: float, *, overlay: bool = True):
    """Write plausible metrics.json (+ optional overlay.json) into the video dir."""
    out = Path(ctx.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    reps = [
        {
            "rep_index": i,
            "made": True,
            "score": best_score - i,
            "t_start": float(i),
            "t_end": i + 0.8,
        }
        for i in range(rep_count)
    ]
    metrics = {
        "video_id": ctx.video_id,
        "movement": ctx.movement,
        "rep_count": rep_count,
        "best_score": best_score,
        "reps": reps,
    }
    (out / "metrics.json").write_text(json.dumps(metrics))
    if overlay:
        (out / "overlay.json").write_text(
            json.dumps({"frames": [{"t": 0.0, "bar": [1.0, 2.0], "skeleton": {}}], "reps": reps})
        )
    return reps


def make_fake_runner(rep_count: int = 5, best_score: float = 92.0, *, overlay: bool = True):
    """A well-behaved runner: reports progress, writes artifacts, returns a result."""

    def runner(ctx: JobContext, progress_cb):
        for pct, stage in enumerate(("decode", "pose", "bar", "segment", "metrics")):
            progress_cb(stage, (pct + 1) * 20)
        reps = _write_artifacts(ctx, rep_count, best_score, overlay=overlay)
        return RunnerResult(
            rep_count=rep_count,
            best_score=best_score,
            extraction_version=1,
            rules_version=1,
            reps=reps,
        )

    return runner


def failing_runner(ctx: JobContext, progress_cb) -> RunnerResult:
    progress_cb("decode", 10)
    raise ValueError("corrupted video: no decodable frames")


def _upload(
    client: TestClient,
    movement="power_clean",
    load_kg="60",
    recalibrate="false",
    filename="clip.mp4",
):
    return client.post(
        "/api/videos",
        data={"movement": movement, "load_kg": load_kg, "recalibrate": recalibrate},
        files={"file": (filename, FAKE_VIDEO_BYTES, "video/mp4")},
    )


def _poll(client: TestClient, job_id: str, timeout: float = 5.0) -> dict:
    """Poll GET /api/jobs/{id} until a terminal state or timeout."""
    deadline = time.time() + timeout
    payload: dict = {}
    while time.time() < deadline:
        payload = client.get(f"/api/jobs/{job_id}").json()
        if payload["state"] in ("DONE", "FAILED"):
            return payload
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never terminated; last={payload}")


@pytest.fixture
def app(tmp_path):
    """App with a fast fake runner and a tmp library."""
    return create_app(engine_runner=make_fake_runner(), library_dir=tmp_path)


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


# --- upload happy path ----------------------------------------------------


def test_upload_then_done_with_retrievable_results(client):
    resp = _upload(client)
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"video_id", "job_id"}
    video_id, job_id = body["video_id"], body["job_id"]

    job = _poll(client, job_id)
    assert job["state"] == "DONE"
    assert job["progress"] == 100
    assert job["error"] is None

    analysis = client.get(f"/api/videos/{video_id}/analysis")
    assert analysis.status_code == 200
    assert analysis.json()["rep_count"] == 5

    overlay = client.get(f"/api/videos/{video_id}/overlay")
    assert overlay.status_code == 200
    assert "frames" in overlay.json()


def test_job_progresses_through_stages(client):
    # The QUEUED state is legal before the worker starts; assert stage vocab.
    resp = _upload(client)
    job_id = resp.json()["job_id"]
    job = _poll(client, job_id)
    assert job["state"] == "DONE"


# --- invalid movement -----------------------------------------------------


def test_invalid_movement_returns_422(client):
    resp = _upload(client, movement="not_a_real_movement")
    assert resp.status_code == 422
    assert "detail" in resp.json()
    # nothing should have been stored
    assert client.get("/api/videos").json() == []


# --- failing analysis -----------------------------------------------------


def test_failing_analysis_surfaces_error(tmp_path):
    app = create_app(engine_runner=failing_runner, library_dir=tmp_path)
    with TestClient(app) as client:
        job_id = _upload(client).json()["job_id"]
        job = _poll(client, job_id)
        assert job["state"] == "FAILED"
        assert "corrupted video" in job["error"]


# --- double submit runs sequentially (pool size 1) ------------------------


def test_double_submit_two_distinct_sequential_jobs(tmp_path):
    events: list[tuple[str, str]] = []

    def sequential_runner(ctx: JobContext, progress_cb):
        events.append(("start", ctx.job_id))
        time.sleep(0.05)
        events.append(("end", ctx.job_id))
        reps = _write_artifacts(ctx, 1, 50.0)
        return RunnerResult(rep_count=1, best_score=50.0, reps=reps)

    app = create_app(engine_runner=sequential_runner, library_dir=tmp_path)
    with TestClient(app) as client:
        first = _upload(client, filename="a.mp4").json()
        second = _upload(client, filename="b.mp4").json()
        assert first["job_id"] != second["job_id"]
        _poll(client, first["job_id"])
        _poll(client, second["job_id"])

    # With max_workers=1 the events must be start,end,start,end (no interleave).
    assert [kind for kind, _ in events] == ["start", "end", "start", "end"]
    assert events[0][1] == events[1][1]
    assert events[2][1] == events[3][1]
    assert events[0][1] != events[2][1]


# --- restart re-queue of an orphaned RUNNING job --------------------------


def test_restart_requeues_orphaned_running_job(tmp_path):
    storage = Storage(tmp_path)
    db_path = storage.db_path()
    conn = db.connect(db_path)
    db.init_db(conn)
    dir_path = storage.video_dir("v1", "2026-07-12")
    dir_path.mkdir(parents=True, exist_ok=True)
    db.create_video(
        conn,
        video_id="v1",
        movement="back_squat",
        load_kg=100.0,
        recalibrate=False,
        original_name="x.mp4",
        ext=".mp4",
        dir_path=str(dir_path),
        file_path=str(dir_path / "original.mp4"),
    )
    job_id = db.create_job(conn, "v1")
    db.mark_job_running(conn, job_id, pid=999_999)  # stale pid of a dead process
    conn.close()

    # Simulate API restart. create_app runs crash recovery synchronously.
    create_app(engine_runner=make_fake_runner(), library_dir=tmp_path)

    conn = db.connect(db_path)
    row = db.get_job(conn, job_id)
    conn.close()
    assert row["state"] == "QUEUED"
    assert row["progress"] == 0
    assert row["stage"] is None


# --- worker crash handling (hard child crash / BrokenProcessPool) ---------


def _job_manager_with_running_job(tmp_path):
    from concurrent.futures.process import BrokenProcessPool  # noqa: F401

    from powerpath_engine.api.jobs import JobContext, JobManager

    storage = Storage(tmp_path)
    db_path = storage.db_path()
    conn = db.connect(db_path)
    db.init_db(conn)
    dir_path = storage.video_dir("v1", "2026-07-12")
    dir_path.mkdir(parents=True, exist_ok=True)
    db.create_video(
        conn,
        video_id="v1",
        movement="back_squat",
        load_kg=100.0,
        recalibrate=False,
        original_name="x.mp4",
        ext=".mp4",
        dir_path=str(dir_path),
        file_path=str(dir_path / "original.mp4"),
    )
    job_id = db.create_job(conn, "v1")
    db.mark_job_running(conn, job_id, pid=999_999)
    conn.close()
    mgr = JobManager(db_path, make_fake_runner(), use_process_pool=False)
    ctx = JobContext(
        job_id=job_id,
        video_id="v1",
        movement="back_squat",
        load_kg=100.0,
        recalibrate=False,
        video_path=str(dir_path / "original.mp4"),
        output_dir=str(dir_path),
    )
    return mgr, ctx, db_path, job_id


def _job_state(db_path, job_id):
    conn = db.connect(db_path)
    try:
        return db.get_job(conn, job_id)["state"]
    finally:
        conn.close()


def test_hard_worker_crash_marks_job_failed(tmp_path):
    """A future that resolves to an exception (native child crash) -> job FAILED."""
    from concurrent.futures import Future

    mgr, ctx, db_path, job_id = _job_manager_with_running_job(tmp_path)
    fut: Future = Future()
    fut.set_exception(RuntimeError("segfault-ish child death"))
    mgr._on_future_done(fut, ctx)
    assert _job_state(db_path, job_id) == "FAILED"


def test_crash_callback_does_not_overwrite_terminal_job(tmp_path):
    from concurrent.futures import Future

    mgr, ctx, db_path, job_id = _job_manager_with_running_job(tmp_path)
    conn = db.connect(db_path)
    db.mark_job_done(conn, job_id)
    conn.close()
    fut: Future = Future()
    fut.set_exception(RuntimeError("late crash"))
    mgr._on_future_done(fut, ctx)
    assert _job_state(db_path, job_id) == "DONE"  # not clobbered


def test_broken_process_pool_is_rebuilt(tmp_path):
    from concurrent.futures import Future
    from concurrent.futures.process import BrokenProcessPool

    mgr, ctx, db_path, job_id = _job_manager_with_running_job(tmp_path)
    mgr.start()
    old_executor = mgr._executor
    fut: Future = Future()
    fut.set_exception(BrokenProcessPool("pool died"))
    mgr._on_future_done(fut, ctx)
    assert _job_state(db_path, job_id) == "FAILED"
    assert mgr._executor is not None
    assert mgr._executor is not old_executor  # fresh pool so later uploads work
    mgr.shutdown()


# --- DELETE removes rows + files ------------------------------------------


def test_delete_removes_rows_and_files(client, app):
    video_id = _upload(client).json()["video_id"]
    _poll(client, client.get("/api/videos").json()[0]["job_id"])

    conn = db.connect(app.state.db_path)
    dir_path = db.get_video(conn, video_id)["dir_path"]
    conn.close()
    assert Path(dir_path).exists()
    assert (Path(dir_path) / "metrics.json").exists()

    resp = client.delete(f"/api/videos/{video_id}")
    assert resp.status_code == 200
    assert not Path(dir_path).exists()
    assert client.get("/api/videos").json() == []
    conn = db.connect(app.state.db_path)
    assert db.get_video(conn, video_id) is None
    conn.close()


def test_delete_missing_video_404(client):
    assert client.delete("/api/videos/does-not-exist").status_code == 404


# --- movements endpoint ---------------------------------------------------


def test_movements_lists_registry_keys(client):
    resp = client.get("/api/movements")
    assert resp.status_code == 200
    items = resp.json()
    assert len(items) >= 6
    for item in items:
        assert set(item) == {"key", "display_name", "family"}
    keys = {item["key"] for item in items}
    assert {"power_clean", "back_squat", "push_press", "power_snatch", "deadlift"} <= keys


# --- GET /api/videos exact contract keys ----------------------------------


def test_videos_list_has_exact_contract_keys(client):
    video_id = _upload(client, movement="deadlift").json()["video_id"]
    _poll(client, client.get("/api/videos").json()[0]["job_id"])

    items = client.get("/api/videos").json()
    assert len(items) == 1
    item = items[0]
    assert set(item) == {
        "video_id",
        "movement",
        "display_name",
        "filmed_at",
        "load_kg",
        "job",
        "rep_count",
        "best_score",
        "job_id",
    }
    assert item["video_id"] == video_id
    assert item["movement"] == "deadlift"
    assert item["display_name"] == "Deadlift"
    assert set(item["job"]) == {"state", "progress", "stage", "error"}
    assert item["job"]["state"] == "DONE"
    assert item["job_id"] is not None
    assert item["rep_count"] == 5
    assert item["best_score"] == 92.0
    # filmed_at must be a parseable ISO 8601 string (UI renders it as a date).
    from datetime import datetime

    datetime.fromisoformat(item["filmed_at"])


# --- 404s for artifacts ---------------------------------------------------


def test_overlay_404_when_not_written(tmp_path):
    app = create_app(engine_runner=make_fake_runner(overlay=False), library_dir=tmp_path)
    with TestClient(app) as client:
        video_id = _upload(client).json()["video_id"]
        _poll(client, client.get("/api/videos").json()[0]["job_id"])
        assert client.get(f"/api/videos/{video_id}/analysis").status_code == 200
        assert client.get(f"/api/videos/{video_id}/overlay").status_code == 404


def test_analysis_404_for_unknown_video(client):
    assert client.get("/api/videos/nope/analysis").status_code == 404


def test_job_404_for_unknown_id(client):
    assert client.get("/api/jobs/nope").status_code == 404


# --- file endpoint + range support ----------------------------------------


def test_file_endpoint_serves_bytes_with_range(client):
    video_id = _upload(client).json()["video_id"]
    full = client.get(f"/api/videos/{video_id}/file")
    assert full.status_code == 200
    assert full.content == FAKE_VIDEO_BYTES

    ranged = client.get(f"/api/videos/{video_id}/file", headers={"Range": "bytes=0-9"})
    assert ranged.status_code == 206
    assert ranged.content == FAKE_VIDEO_BYTES[:10]
    assert ranged.headers["content-range"].startswith("bytes 0-9/")


# --- CORS -----------------------------------------------------------------


def test_cors_allows_ui_origin(client):
    resp = client.get("/api/movements", headers={"Origin": "http://localhost:3000"})
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"


# --- fake-engine mode (POWERPATH_FAKE_ENGINE=1) ---------------------------


def test_fake_engine_default_runner_writes_canned_artifacts(tmp_path, monkeypatch):
    """The default runner in fake mode emits instant canned results (Task 12 E2E)
    -- and those artifacts must match the FROZEN overlay/metrics contract, since
    the player renders them exactly like real engine output."""
    monkeypatch.setenv("POWERPATH_FAKE_ENGINE", "1")
    out = tmp_path / "out"
    ctx = JobContext(
        job_id="j",
        video_id="v",
        movement="power_clean",
        load_kg=60.0,
        recalibrate=False,
        video_path=str(tmp_path / "v.mp4"),
        output_dir=str(out),
    )
    stages: list[str] = []
    result = default_runner(ctx, lambda stage, pct: stages.append(stage))
    assert result.rep_count == 5
    assert stages == ["decode", "pose", "bar", "segment", "metrics"]

    metrics = json.loads((out / "metrics.json").read_text())
    overlay = json.loads((out / "overlay.json").read_text())
    assert_metrics_contract(metrics)
    assert_overlay_contract(overlay)
    assert metrics["movement"] == "power_clean"
    assert len(metrics["reps"]) >= 1
    assert len(overlay["frames"]) >= 1

    # At least one canned rep carries BOTH severities so the Task 12 E2E can
    # render a fault chip and exercise the informational-muted styling.
    severity_pairs = {
        tuple(sorted(f["severity"] for f in rep["faults"])) for rep in overlay["reps"]
    }
    assert ("fault", "informational") in severity_pairs


def test_process_pool_fake_engine_end_to_end(tmp_path, monkeypatch):
    """The REAL ProcessPoolExecutor path: no injected runner, so the app uses
    the picklable default runner in a single-worker process pool (fake-engine
    mode -- the env var is inherited by the spawned worker). Upload -> poll ->
    DONE -> analysis + overlay retrievable and contract-valid, summary rows
    persisted through the pool boundary."""
    monkeypatch.setenv("POWERPATH_FAKE_ENGINE", "1")
    app = create_app(library_dir=tmp_path)  # no engine_runner -> process pool
    with TestClient(app) as client:
        resp = _upload(client)
        assert resp.status_code == 200
        body = resp.json()

        job = _poll(client, body["job_id"], timeout=30.0)  # allow subprocess spawn
        assert job["state"] == "DONE"
        assert job["progress"] == 100
        assert job["error"] is None

        analysis = client.get(f"/api/videos/{body['video_id']}/analysis")
        overlay = client.get(f"/api/videos/{body['video_id']}/overlay")
        assert analysis.status_code == 200 and overlay.status_code == 200
        assert_metrics_contract(analysis.json())
        assert_overlay_contract(overlay.json())
        assert len(analysis.json()["reps"]) >= 1
        assert len(overlay.json()["frames"]) >= 1

        videos = client.get("/api/videos").json()
        assert videos[0]["rep_count"] == 5
        assert videos[0]["best_score"] == 92.0
        assert videos[0]["job"]["state"] == "DONE"


# --- schema / db sanity ---------------------------------------------------


def test_schema_version_stamped(tmp_path):
    conn = db.connect(str(tmp_path / "x.db"))
    db.init_db(conn)
    assert db.schema_version(conn) == db.SCHEMA_VERSION
    tables = {
        row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"videos", "calibrations", "jobs", "analyses", "reps", "templates", "settings"} <= tables
    conn.close()

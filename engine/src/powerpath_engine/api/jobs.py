"""Background analysis jobs: a single-worker executor plus the worker body.

Design (from the brief + global constraints):

* Analysis is CPU-heavy (pose inference, decoding) so production runs it in a
  ``ProcessPoolExecutor(max_workers=1)`` -- one job at a time, isolated from the
  event loop, and a crash cannot take down the API.
* The analysis runner is **injectable**. The default runner lazily imports
  ``powerpath_engine.pipeline.analyze`` *inside* the worker so this module (and
  the whole test suite) imports fine before Task 8 exists. Tests inject a fake
  runner that writes canned results.
* Pickling boundary: a ``ProcessPoolExecutor`` must pickle the callable and its
  args to ship them to the child. The default runner is a module-level function
  (picklable) and :class:`JobContext` / :class:`RunnerResult` are plain
  dataclasses of primitives (picklable). An *injected* runner (a test fake,
  often a closure or a fixture-bound function) usually is **not** picklable, so
  when a custom runner is supplied :class:`JobManager` transparently uses a
  ``ThreadPoolExecutor(max_workers=1)`` instead -- same single-worker, sequential
  semantics, but in-process so nothing needs pickling. This is the resolution to
  the escalation note about sqlite/runner pickling across the pool boundary.
* The worker opens its **own** sqlite connection and writes progress rows back
  through it; the API's request handlers read those rows on their own
  connections (WAL mode makes the concurrent reader/writer safe).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from powerpath_engine.api import db

# Stage vocabulary the pipeline reports through (kept in sync with the UI's
# JobStage type). Progress is an int 0-100.
STAGES = ("decode", "pose", "bar", "segment", "metrics")

# Default athlete height used by the real pipeline path when the upload did not
# carry one (the M1 upload form does not collect it). Overridable via env so the
# integration in Task 8 can tune it without touching this module.
_DEFAULT_HEIGHT_CM = 175.0


@dataclass
class JobContext:
    """Everything a runner needs, as picklable primitives (safe for a subprocess)."""

    job_id: str
    video_id: str
    movement: str
    load_kg: float
    recalibrate: bool
    video_path: str
    output_dir: str
    athlete_height_cm: float = _DEFAULT_HEIGHT_CM


@dataclass
class RunnerResult:
    """What a runner returns; the worker persists it into the analyses table.

    The runner is responsible for having written ``metrics.json`` and
    ``overlay.json`` into ``ctx.output_dir`` before returning.
    """

    rep_count: int | None = None
    best_score: float | None = None
    extraction_version: int | None = None
    rules_version: int | None = None
    reps: list[dict[str, Any]] = field(default_factory=list)


# A runner takes the job context and a progress callback ``(stage, pct)`` and
# returns a RunnerResult. Must have written the JSON artifacts to output_dir.
Runner = Callable[[JobContext, Callable[[str, int], None]], RunnerResult]


def _metrics_file(ctx: JobContext) -> str:
    return str(os.path.join(ctx.output_dir, "metrics.json"))


def _overlay_file(ctx: JobContext) -> str:
    return str(os.path.join(ctx.output_dir, "overlay.json"))


def _write_canned_artifacts(ctx: JobContext) -> RunnerResult:
    """Write plausible metrics.json + overlay.json with 5 made reps.

    Used by the fake-engine mode (``POWERPATH_FAKE_ENGINE=1``) so the whole
    stack -- upload, poll, player -- can be exercised end to end (Task 12's
    Playwright E2E) with no model inference and no real decoding.
    """
    reps = []
    frames = []
    bar_path_by_rep: dict[str, list[list[float]]] = {}
    for i in range(5):
        t0 = float(i)
        t1 = t0 + 0.8
        score = 80.0 + i * 3.0
        reps.append(
            {
                "rep_index": i,
                "made": True,
                "score": score,
                "t_start": t0,
                "t_end": t1,
                "faults": [],
                "phases": [],
                "unanalyzed_reason": None,
            }
        )
        path = [[100.0, 400.0 - j * 20.0] for j in range(5)]
        bar_path_by_rep[str(i)] = path
        for j in range(5):
            t = round(t0 + j * 0.1, 3)
            y = 400.0 - j * 20.0
            frames.append(
                {
                    "t": t,
                    "bar": [100.0, y],
                    "skeleton": {
                        "left_hip": [90.0, y + 10.0],
                        "right_hip": [110.0, y + 10.0],
                        "left_knee": [88.0, y + 60.0],
                        "right_knee": [112.0, y + 60.0],
                    },
                }
            )

    best_score = max(r["score"] for r in reps)
    metrics = {
        "video_id": ctx.video_id,
        "movement": ctx.movement,
        "load_kg": ctx.load_kg,
        "extraction_version": 1,
        "rules_version": 1,
        "rep_count": len(reps),
        "best_score": best_score,
        "reps": reps,
        "unanalyzed": [],
        "fake_engine": True,
    }
    overlay = {"frames": frames, "reps": reps, "bar_path_by_rep": bar_path_by_rep}

    os.makedirs(ctx.output_dir, exist_ok=True)
    with open(_metrics_file(ctx), "w") as fh:
        json.dump(metrics, fh)
    with open(_overlay_file(ctx), "w") as fh:
        json.dump(overlay, fh)

    return RunnerResult(
        rep_count=len(reps),
        best_score=best_score,
        extraction_version=1,
        rules_version=1,
        reps=reps,
    )


def default_runner(ctx: JobContext, progress_cb: Callable[[str, int], None]) -> RunnerResult:
    """Production runner. Module-level so it is picklable for the process pool.

    In ``POWERPATH_FAKE_ENGINE=1`` mode it emits instant canned results (for the
    E2E harness). Otherwise it *lazily* imports the real pipeline -- the import
    is deferred to here so importing this module never requires Task 8's
    ``pipeline``/``overlay`` modules to exist.
    """
    if os.environ.get("POWERPATH_FAKE_ENGINE") == "1":
        for pct, stage in enumerate(STAGES):
            progress_cb(stage, int((pct + 1) / len(STAGES) * 100))
        return _write_canned_artifacts(ctx)

    # --- real pipeline path (finalised when Task 8 lands) -----------------
    from powerpath_engine import overlay, pipeline, versions  # lazy: not needed for tests
    from powerpath_engine.pose import make_pose_backend  # lazy

    pose_backend = make_pose_backend(os.environ.get("POWERPATH_POSE", "rtmlib"))

    def _cb(stage: str, pct: int) -> None:
        progress_cb(stage, int(pct))

    result = pipeline.analyze(
        ctx.video_path,
        movement_key=ctx.movement,
        load_kg=ctx.load_kg,
        athlete_height_cm=ctx.athlete_height_cm,
        pose_backend=pose_backend,
        progress_cb=_cb,
    )
    overlay.write_metrics_json(result, _metrics_file(ctx))
    overlay.write_overlay_json(result, _overlay_file(ctx))

    made_scores = [
        r.score for r in result.reps if getattr(r, "made", False) and r.score is not None
    ]
    reps = [
        {
            "rep_index": i,
            "made": bool(getattr(r, "made", False)),
            "score": getattr(r, "score", None),
            "t_start": getattr(getattr(r, "window", None), "t_start", None),
            "t_end": getattr(getattr(r, "window", None), "t_end", None),
            "unanalyzed_reason": None,
        }
        for i, r in enumerate(result.reps)
    ]
    return RunnerResult(
        rep_count=len(result.reps),
        best_score=max(made_scores) if made_scores else None,
        extraction_version=getattr(versions, "EXTRACTION_VERSION", None),
        rules_version=getattr(versions, "RULES_VERSION", None),
        reps=reps,
    )


def _execute_job(runner: Runner, db_path: str, ctx: JobContext) -> None:
    """Worker body: run one job start-to-finish, owning its own DB connection.

    Runs either in a subprocess (default runner) or in a worker thread (injected
    runner). Either way it opens a fresh sqlite connection, marks the job
    RUNNING, streams progress, persists the analysis, and marks DONE -- or
    records the failure so ``GET /api/jobs/{id}`` can surface the error string.
    """
    conn = db.connect(db_path)
    try:
        db.mark_job_running(conn, ctx.job_id, os.getpid())

        def progress_cb(stage: str, pct: int) -> None:
            db.update_job_progress(conn, ctx.job_id, stage=stage, progress=int(pct))

        result = runner(ctx, progress_cb)
        db.write_analysis(
            conn,
            video_id=ctx.video_id,
            job_id=ctx.job_id,
            rep_count=result.rep_count,
            best_score=result.best_score,
            extraction_version=result.extraction_version,
            rules_version=result.rules_version,
            metrics_path=_metrics_file(ctx),
            overlay_path=_overlay_file(ctx),
            reps=result.reps,
        )
        db.mark_job_done(conn, ctx.job_id)
    except Exception as exc:
        try:
            db.mark_job_failed(conn, ctx.job_id, f"{type(exc).__name__}: {exc}")
        except Exception:
            pass
    finally:
        conn.close()


class JobManager:
    """Owns the single-worker executor and submits jobs to it.

    ``use_process_pool`` is chosen by the app factory: process pool for the
    default (picklable) runner, thread pool when a custom runner is injected.
    """

    def __init__(self, db_path: str, runner: Runner, *, use_process_pool: bool) -> None:
        self._db_path = db_path
        self._runner = runner
        self._use_process_pool = use_process_pool
        self._executor: Executor | None = None
        self._futures: list[Future[None]] = []

    def start(self) -> None:
        if self._executor is not None:
            return
        if self._use_process_pool:
            self._executor = ProcessPoolExecutor(max_workers=1)
        else:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ppjob")

    def submit(self, ctx: JobContext) -> None:
        if self._executor is None:
            raise RuntimeError("JobManager not started")
        # Prune completed futures so the list does not grow unbounded.
        self._futures = [f for f in self._futures if not f.done()]
        future = self._executor.submit(_execute_job, self._runner, self._db_path, ctx)
        self._futures.append(future)

    def shutdown(self, wait: bool = False) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=wait, cancel_futures=not wait)
            self._executor = None

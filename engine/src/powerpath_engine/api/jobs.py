"""Background analysis jobs: a single-worker executor plus the worker body.

Design (from the brief + global constraints):

* Analysis is CPU-heavy (pose inference, decoding) so production runs it in a
  ``ProcessPoolExecutor(max_workers=1)`` -- one job at a time, isolated from the
  event loop, and a crash cannot take down the API.
* The analysis runner is **injectable**. The default runner lazily imports
  ``powerpath_engine.pipeline.analyze`` *inside* the worker so the API process
  never pays the heavy cv2/av/scipy import cost. Tests inject a fake runner
  that writes canned results.
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
    """Write metrics.json + overlay.json (5 made reps) in the FROZEN contract shape.

    Used by the fake-engine mode (``POWERPATH_FAKE_ENGINE=1``) so the whole
    stack -- upload, poll, player -- can be exercised end to end (Task 12's
    Playwright E2E renders a real overlay from this) with no model inference
    and no real decoding. Both payloads follow
    ``.superpowers/sdd/overlay-metrics-contract.md`` exactly, same as the
    real writers in ``powerpath_engine.overlay`` (the engine tests validate
    this file's output against the shared contract assertions).
    """
    video = {"width": 1920, "height": 1080, "fps_avg": 30.0, "duration_s": 5.0}
    frames: list[dict[str, Any]] = []
    overlay_reps: list[dict[str, Any]] = []
    metrics_reps: list[dict[str, Any]] = []
    for i in range(5):
        t0, t1 = float(i), float(i) + 0.8
        score = 80 + i * 3
        bar_path: list[list[float]] = []
        for j in range(5):
            t = round(t0 + j * 0.1, 3)
            x, y = 960.0, 700.0 - j * 80.0
            bar_path.append([x, y])
            frames.append(
                {
                    "t": t,
                    "bar": [x, y],
                    "skeleton": {
                        "left_hip": [930.0, y + 60.0],
                        "right_hip": [990.0, y + 60.0],
                        "left_knee": [925.0, y + 200.0],
                        "right_knee": [995.0, y + 200.0],
                    },
                }
            )
        phases = {"knee_pass": round(t0 + 0.2, 3), "catch": round(t0 + 0.5, 3)}
        overlay_reps.append(
            {
                "rep_index": i,
                "t_start": t0,
                "t_end": t1,
                "made": True,
                "score": score,
                "bar_path": bar_path,
                "phases": phases,
                "faults": [],
                "unanalyzed_reason": None,
            }
        )
        metrics_reps.append(
            {
                "rep_index": i,
                "made": True,
                "score": score,
                "excluded_from_templates": False,
                "metrics": {
                    "bar_drift_cm": 2.0,
                    "peak_concentric_velocity_ms": 1.5,
                    "path_length_ratio": 1.05,
                    "smoothness": 0.8,
                    "hip_angle_at_phase": {"catch": 120.0},
                    "knee_angle_at_phase": {"catch": 110.0},
                    "elbow_angle_at_phase": {"catch": 60.0},
                },
                "faults": [],
                "phases": phases,
            }
        )

    best_score = max(r["score"] for r in metrics_reps)
    metrics = {
        "video": video,
        "movement": ctx.movement,
        "load_kg": ctx.load_kg,
        "extraction_version": 1,
        "rules_version": 1,
        "calibration": {
            "source": "manual",
            "bar_scale_cm_per_px": 0.2,
            "warning": "fake engine (POWERPATH_FAKE_ENGINE=1): canned analysis",
        },
        "reps": metrics_reps,
    }
    overlay = {"video": video, "movement": ctx.movement, "frames": frames, "reps": overlay_reps}

    os.makedirs(ctx.output_dir, exist_ok=True)
    with open(_metrics_file(ctx), "w") as fh:
        json.dump(metrics, fh)
    with open(_overlay_file(ctx), "w") as fh:
        json.dump(overlay, fh)

    return RunnerResult(
        rep_count=len(overlay_reps),
        best_score=float(best_score),
        extraction_version=1,
        rules_version=1,
        reps=overlay_reps,
    )


def default_runner(ctx: JobContext, progress_cb: Callable[[str, int], None]) -> RunnerResult:
    """Production runner. Module-level so it is picklable for the process pool.

    In ``POWERPATH_FAKE_ENGINE=1`` mode it emits instant canned results (for the
    E2E harness). Otherwise it runs ``pipeline.analyze`` and the ``overlay``
    writers -- imported *lazily* here so the API process never pays the
    cv2/av/scipy import cost (the pool's worker subprocess is what actually
    executes this) and fake-mode runs stay dependency-light.
    """
    if os.environ.get("POWERPATH_FAKE_ENGINE") == "1":
        for pct, stage in enumerate(STAGES):
            progress_cb(stage, int((pct + 1) / len(STAGES) * 100))
        return _write_canned_artifacts(ctx)

    # --- real pipeline path ------------------------------------------------
    from powerpath_engine import overlay, pipeline  # lazy: keep API import light
    from powerpath_engine.pose import make_pose_backend  # lazy

    pose_backend = make_pose_backend(os.environ.get("POWERPATH_POSE", "rtmlib"))

    result = pipeline.analyze(
        ctx.video_path,
        ctx.movement,
        ctx.load_kg,
        ctx.athlete_height_cm,
        pose_backend,
        progress_cb=lambda stage, pct: progress_cb(stage, int(pct)),
    )
    overlay.write_metrics_json(result, _metrics_file(ctx))
    overlay.write_overlay_json(result, result.bar_px, result.landmarks_px, _overlay_file(ctx))

    reps = [
        {
            "rep_index": r.window.rep_index,
            "made": r.made,
            "score": None if r.score is None else int(round(r.score)),
            "t_start": r.window.t_start,
            "t_end": r.window.t_end,
            "unanalyzed_reason": r.unanalyzed_reason,
        }
        for r in result.reps
    ]
    made_scores = [r["score"] for r in reps if r["made"] and r["score"] is not None]
    return RunnerResult(
        rep_count=len(result.reps),
        best_score=float(max(made_scores)) if made_scores else None,
        extraction_version=result.extraction_version,
        rules_version=result.rules_version,
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

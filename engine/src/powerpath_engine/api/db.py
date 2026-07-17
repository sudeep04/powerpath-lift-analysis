"""SQLite persistence for the PowerPath API layer.

The FastAPI layer is the *sole* owner of this database (a global constraint):
the UI reads everything over HTTP, never by touching SQLite directly. Every
connection is opened in WAL mode so the single background worker (which owns its
own connection) can write job progress rows while request handlers read them
concurrently.

The schema is deliberately explicit -- one ``CREATE TABLE`` per entity in the
design's data model (videos, calibrations, jobs, analyses, reps, templates,
settings) -- and versioned with the ``user_version`` pragma so a future
migration can detect an old file. This module contains *only* data access: no
FastAPI, no job execution, no filesystem layout decisions (those live in
``storage.py`` / ``jobs.py``).
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import UTC, datetime
from typing import Any

# Bump when the DDL below changes in an incompatible way. Read back via
# ``PRAGMA user_version``; a mismatch is where a future migration would hook in.
SCHEMA_VERSION = 1

JobState = str  # one of: QUEUED, RUNNING, DONE, FAILED

_SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id              TEXT PRIMARY KEY,
    movement        TEXT NOT NULL,
    load_kg         REAL NOT NULL,
    recalibrate     INTEGER NOT NULL DEFAULT 0,
    original_name   TEXT,
    ext             TEXT NOT NULL,
    dir_path        TEXT NOT NULL,
    file_path       TEXT NOT NULL,
    filmed_at       TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS calibrations (
    id              TEXT PRIMARY KEY,
    video_id        TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    source          TEXT,
    bar_cm_per_px   REAL,
    body_cm_per_px  REAL,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id              TEXT PRIMARY KEY,
    video_id        TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    state           TEXT NOT NULL,
    progress        INTEGER NOT NULL DEFAULT 0,
    stage           TEXT,
    error           TEXT,
    pid             INTEGER,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS analyses (
    id                  TEXT PRIMARY KEY,
    video_id            TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    job_id              TEXT REFERENCES jobs(id) ON DELETE SET NULL,
    extraction_version  INTEGER,
    rules_version       INTEGER,
    rep_count           INTEGER,
    best_score          REAL,
    metrics_path        TEXT,
    overlay_path        TEXT,
    created_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reps (
    id                  TEXT PRIMARY KEY,
    analysis_id         TEXT NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    rep_index           INTEGER NOT NULL,
    made                INTEGER NOT NULL DEFAULT 0,
    score               REAL,
    t_start             REAL,
    t_end               REAL,
    unanalyzed_reason   TEXT
);

CREATE TABLE IF NOT EXISTS templates (
    id              TEXT PRIMARY KEY,
    movement        TEXT NOT NULL,
    load_kg         REAL,
    data_json       TEXT,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key             TEXT PRIMARY KEY,
    value           TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_video ON jobs(video_id);
CREATE INDEX IF NOT EXISTS idx_analyses_video ON analyses(video_id);
CREATE INDEX IF NOT EXISTS idx_reps_analysis ON reps(analysis_id);
"""


def _now() -> str:
    """Current UTC time as an ISO 8601 string (what the UI renders)."""
    return datetime.now(UTC).isoformat()


def new_id() -> str:
    return uuid.uuid4().hex


def connect(db_path: str) -> sqlite3.Connection:
    """Open a WAL-mode connection with row access by column name.

    Each caller (every request handler, and the worker process/thread) opens
    its own connection; sqlite connections are not shared across threads.
    ``busy_timeout`` lets a writer wait briefly rather than raising
    ``database is locked`` when the single worker is mid-write.
    """
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create tables (idempotent) and stamp the schema version."""
    conn.executescript(_SCHEMA)
    conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
    conn.commit()


def schema_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


# --- videos ---------------------------------------------------------------


def create_video(
    conn: sqlite3.Connection,
    *,
    video_id: str,
    movement: str,
    load_kg: float,
    recalibrate: bool,
    original_name: str | None,
    ext: str,
    dir_path: str,
    file_path: str,
    filmed_at: str | None = None,
) -> None:
    created = _now()
    conn.execute(
        """
        INSERT INTO videos
            (id, movement, load_kg, recalibrate, original_name, ext,
             dir_path, file_path, filmed_at, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            video_id,
            movement,
            float(load_kg),
            1 if recalibrate else 0,
            original_name,
            ext,
            dir_path,
            file_path,
            filmed_at or created,
            created,
        ),
    )
    conn.commit()


def get_video(conn: sqlite3.Connection, video_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()


def delete_video(conn: sqlite3.Connection, video_id: str) -> bool:
    """Delete a video and all rows that reference it. Returns True if it existed.

    Foreign keys with ``ON DELETE CASCADE`` remove jobs/analyses/reps/
    calibrations; we delete explicitly too so the behaviour does not depend on
    the ``foreign_keys`` pragma being on for this connection.
    """
    if get_video(conn, video_id) is None:
        return False
    analysis_ids = [
        row["id"] for row in conn.execute("SELECT id FROM analyses WHERE video_id = ?", (video_id,))
    ]
    for analysis_id in analysis_ids:
        conn.execute("DELETE FROM reps WHERE analysis_id = ?", (analysis_id,))
    conn.execute("DELETE FROM analyses WHERE video_id = ?", (video_id,))
    conn.execute("DELETE FROM calibrations WHERE video_id = ?", (video_id,))
    conn.execute("DELETE FROM jobs WHERE video_id = ?", (video_id,))
    conn.execute("DELETE FROM videos WHERE id = ?", (video_id,))
    conn.commit()
    return True


# --- jobs -----------------------------------------------------------------


def create_job(conn: sqlite3.Connection, video_id: str) -> str:
    job_id = new_id()
    now = _now()
    conn.execute(
        """
        INSERT INTO jobs (id, video_id, state, progress, stage, error, pid,
                          created_at, updated_at)
        VALUES (?, ?, 'QUEUED', 0, NULL, NULL, NULL, ?, ?)
        """,
        (job_id, video_id, now, now),
    )
    conn.commit()
    return job_id


def get_job(conn: sqlite3.Connection, job_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()


def latest_job_for_video(conn: sqlite3.Connection, video_id: str) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT * FROM jobs WHERE video_id = ?
        ORDER BY created_at DESC, rowid DESC LIMIT 1
        """,
        (video_id,),
    ).fetchone()


def mark_job_running(conn: sqlite3.Connection, job_id: str, pid: int) -> None:
    conn.execute(
        "UPDATE jobs SET state='RUNNING', pid=?, error=NULL, updated_at=? WHERE id=?",
        (pid, _now(), job_id),
    )
    conn.commit()


def update_job_progress(
    conn: sqlite3.Connection, job_id: str, *, stage: str | None, progress: int
) -> None:
    conn.execute(
        "UPDATE jobs SET stage=?, progress=?, updated_at=? WHERE id=?",
        (stage, int(progress), _now(), job_id),
    )
    conn.commit()


def mark_job_done(conn: sqlite3.Connection, job_id: str) -> None:
    conn.execute(
        "UPDATE jobs SET state='DONE', progress=100, error=NULL, updated_at=? WHERE id=?",
        (_now(), job_id),
    )
    conn.commit()


def mark_job_failed(conn: sqlite3.Connection, job_id: str, error: str) -> None:
    conn.execute(
        "UPDATE jobs SET state='FAILED', error=?, updated_at=? WHERE id=?",
        (error, _now(), job_id),
    )
    conn.commit()


def requeue_orphaned_jobs(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Reset every RUNNING job back to QUEUED and return the affected jobs.

    Called once at app startup. Because the FastAPI process is the sole owner
    of the database, any job still marked RUNNING when a fresh process boots was
    orphaned by a previous process that died mid-analysis -- it can never make
    progress again, so we put it back in the queue (progress/stage cleared).
    """
    orphaned = list(conn.execute("SELECT * FROM jobs WHERE state='RUNNING'"))
    conn.execute(
        "UPDATE jobs SET state='QUEUED', progress=0, stage=NULL, pid=NULL, updated_at=? "
        "WHERE state='RUNNING'",
        (_now(),),
    )
    conn.commit()
    return orphaned


def queued_jobs(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(conn.execute("SELECT * FROM jobs WHERE state='QUEUED' ORDER BY created_at"))


# --- analyses / reps ------------------------------------------------------


def write_analysis(
    conn: sqlite3.Connection,
    *,
    video_id: str,
    job_id: str,
    rep_count: int | None,
    best_score: float | None,
    extraction_version: int | None,
    rules_version: int | None,
    metrics_path: str | None,
    overlay_path: str | None,
    reps: list[dict[str, Any]] | None = None,
) -> str:
    """Persist analysis summary + per-rep rows. Latest analysis wins for a video.

    Any prior analysis for the video is removed first so re-runs do not stack.
    """
    old = [
        row["id"] for row in conn.execute("SELECT id FROM analyses WHERE video_id=?", (video_id,))
    ]
    for old_id in old:
        conn.execute("DELETE FROM reps WHERE analysis_id=?", (old_id,))
    conn.execute("DELETE FROM analyses WHERE video_id=?", (video_id,))

    analysis_id = new_id()
    conn.execute(
        """
        INSERT INTO analyses
            (id, video_id, job_id, extraction_version, rules_version,
             rep_count, best_score, metrics_path, overlay_path, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            analysis_id,
            video_id,
            job_id,
            extraction_version,
            rules_version,
            rep_count,
            best_score,
            metrics_path,
            overlay_path,
            _now(),
        ),
    )
    for i, rep in enumerate(reps or []):
        conn.execute(
            """
            INSERT INTO reps
                (id, analysis_id, rep_index, made, score, t_start, t_end, unanalyzed_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_id(),
                analysis_id,
                int(rep.get("rep_index", i)),
                1 if rep.get("made") else 0,
                rep.get("score"),
                rep.get("t_start"),
                rep.get("t_end"),
                rep.get("unanalyzed_reason"),
            ),
        )
    conn.commit()
    return analysis_id


def latest_analysis_for_video(conn: sqlite3.Connection, video_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM analyses WHERE video_id=? ORDER BY created_at DESC, rowid DESC LIMIT 1",
        (video_id,),
    ).fetchone()


def all_videos(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(conn.execute("SELECT * FROM videos ORDER BY created_at DESC, rowid DESC"))

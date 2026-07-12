"""PowerPath HTTP API: FastAPI service, SQLite persistence, job worker.

Public surface:

* :func:`powerpath_engine.api.main.create_app` -- the app factory (injectable
  analysis runner + library directory for tests).
* :func:`powerpath_engine.api.main.run` -- the ``powerpath-api`` console script,
  binding ``127.0.0.1:8400``.
"""

from __future__ import annotations

from powerpath_engine.api.main import create_app, run

__all__ = ["create_app", "run"]

"""Tests for versions.py and the pose backend factory (Task 8).

The version constants are the two axes stored on every analysis; the factory
is the seam the job runner and CLI resolve a backend name through. Neither
test touches a real pose library (the global no-model-inference constraint):
the real-backend factory paths are exercised only for their *unavailable*
error behavior, mirroring test_pose.py.
"""

from __future__ import annotations

import builtins

import numpy as np
import pytest

from powerpath_engine import faults, versions
from powerpath_engine.pose import NoOpPoseBackend, PoseUnavailableError, make_pose_backend

# --- versions.py -----------------------------------------------------------


def test_extraction_version_is_1() -> None:
    assert versions.EXTRACTION_VERSION == 1


def test_rules_version_reexported_from_faults() -> None:
    """One source of truth: versions.RULES_VERSION IS faults.RULES_VERSION."""
    assert versions.RULES_VERSION is faults.RULES_VERSION
    assert versions.RULES_VERSION == 1


# --- make_pose_backend -----------------------------------------------------


def test_factory_unknown_name_lists_options() -> None:
    with pytest.raises(ValueError, match="rtmlib, mediapipe, fake"):
        make_pose_backend("openpose")


def test_factory_fake_returns_noop_backend() -> None:
    backend = make_pose_backend("fake")
    assert isinstance(backend, NoOpPoseBackend)
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    assert backend.detect(image) is None
    assert backend.detect(image, origin=(5.0, 5.0)) is None


def test_factory_rtmlib_without_library_raises_unavailable(monkeypatch) -> None:
    """Lazy import: asking for rtmlib without the extra raises the install hint."""
    real_import = builtins.__import__

    def no_rtmlib(name: str, *args: object, **kwargs: object):
        if name == "rtmlib":
            raise ImportError("No module named 'rtmlib'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_rtmlib)
    with pytest.raises(PoseUnavailableError, match="uv add rtmlib onnxruntime"):
        make_pose_backend("rtmlib")


def test_factory_mediapipe_without_model_env_raises(monkeypatch) -> None:
    """The mediapipe path needs POWERPATH_MEDIAPIPE_MODEL before any import."""
    monkeypatch.delenv("POWERPATH_MEDIAPIPE_MODEL", raising=False)
    with pytest.raises(ValueError, match="POWERPATH_MEDIAPIPE_MODEL"):
        make_pose_backend("mediapipe")

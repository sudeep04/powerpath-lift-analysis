"""Smoke tests for the ``powerpath`` CLI (cli.main called in-process).

``analyze`` runs against a real rendered single-rep video with the no-op
``fake`` pose backend (bar-only analysis -- no model inference, and the
clean's made criteria are bar-only, so the rep still lands). The scripted-
pose path through the pipeline is covered by test_pipeline/test_e2e; here
the point is the command wiring: artifacts written, table printed, exit
codes right.
"""

from __future__ import annotations

import json

import pytest
from contract_utils import assert_metrics_contract, assert_overlay_contract
from render_utils import render_lift_video
from synthetic import single_rep

from powerpath_engine import decode, registry
from powerpath_engine.cli import main

FPS = 60


@pytest.fixture(scope="module")
def rep_video(tmp_path_factory) -> str:
    lift = single_rep(fps=FPS)
    path = tmp_path_factory.mktemp("cli") / "rep.mp4"
    render_lift_video(path, lift, FPS)
    return str(path)


def test_analyze_writes_artifacts_and_prints_rep_table(rep_video, tmp_path, capsys) -> None:
    out_dir = tmp_path / "out"
    code = main(
        [
            "analyze",
            rep_video,
            "--movement",
            "power_clean",
            "--load",
            "60",
            "--height",
            "157",
            "--pose",
            "fake",
            "--out",
            str(out_dir),
        ]
    )
    assert code == 0

    metrics = json.loads((out_dir / "metrics.json").read_text())
    overlay_data = json.loads((out_dir / "overlay.json").read_text())
    assert_metrics_contract(metrics)
    assert_overlay_contract(overlay_data)
    assert len(metrics["reps"]) == 1
    annotated = out_dir / "annotated.mp4"
    assert annotated.exists()
    assert sum(1 for _ in decode.frames(annotated)) >= 1

    captured = capsys.readouterr()
    assert "power_clean @ 60kg -- 1 rep(s)" in captured.out
    assert "rep" in captured.out and "top fault" in captured.out
    assert "  1  yes " in captured.out  # rep 1, made, in the table
    # Progress went to stderr, not into the result table.
    assert "decode..." in captured.err


def test_analyze_defaults_out_dir_next_to_video(rep_video, capsys) -> None:
    from pathlib import Path

    code = main(
        ["analyze", rep_video, "--movement", "power_clean", "--load", "60", "--pose", "fake"]
    )
    assert code == 0
    assert (Path(rep_video).parent / "metrics.json").exists()


def test_analyze_unknown_movement_exits_2(rep_video, tmp_path, capsys) -> None:
    code = main(
        [
            "analyze",
            rep_video,
            "--movement",
            "bench_press",
            "--load",
            "60",
            "--pose",
            "fake",
            "--out",
            str(tmp_path),
        ]
    )
    assert code == 2
    assert "unknown movement" in capsys.readouterr().err


def test_analyze_undecodable_video_exits_2(tmp_path, capsys) -> None:
    bad = tmp_path / "not_a_video.mp4"
    bad.write_bytes(b"definitely not video data")
    code = main(
        [
            "analyze",
            str(bad),
            "--movement",
            "power_clean",
            "--load",
            "60",
            "--pose",
            "fake",
            "--out",
            str(tmp_path),
        ]
    )
    assert code == 2
    assert "error:" in capsys.readouterr().err


def test_extract_fixtures_freezes_bar_and_landmark_series(rep_video, tmp_path, capsys) -> None:
    out_dir = tmp_path / "fixtures"
    code = main(
        [
            "extract-fixtures",
            rep_video,
            "--movement",
            "power_clean",
            "--out",
            str(out_dir),
            "--pose",
            "fake",
        ]
    )
    assert code == 0
    bar = json.loads((out_dir / "bar_series.json").read_text())
    landmarks = json.loads((out_dir / "landmark_series.json").read_text())
    assert bar["movement"] == "power_clean"
    assert len(bar["samples"]) > 100  # the marker was tracked across the clip
    sample = bar["samples"][0]
    assert set(sample) == {"t", "x", "y", "visibility"}
    assert landmarks["frames"] == []  # the no-op fake backend never detects


def test_movements_lists_registry_keys(capsys) -> None:
    assert main(["movements"]) == 0
    out = capsys.readouterr().out
    for key in registry.all_keys():
        assert key in out


def test_console_script_registered() -> None:
    from importlib.metadata import entry_points

    scripts = {ep.name: ep.value for ep in entry_points(group="console_scripts")}
    assert scripts.get("powerpath") == "powerpath_engine.cli:main"

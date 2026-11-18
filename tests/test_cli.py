"""CLI 冒烟测试。"""

from __future__ import annotations

import pytest

from kite_drp.cli import main


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert out.startswith("kite-drp ")


def test_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 1
    assert "usage: kite-drp" in capsys.readouterr().out


def test_paths_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["paths"]) == 0
    out = capsys.readouterr().out
    assert "REPO_ROOT" in out
    assert "DATA_DIR" in out


def test_data_help_lists_pipeline_commands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["data", "--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    for command in (
        "inventory",
        "canonical",
        "splits",
        "fit-stats",
        "build",
        "validate",
        "preflight",
    ):
        assert command in out


def test_eval_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["eval", "--help"])
    assert excinfo.value.code == 0
    assert "naive" in capsys.readouterr().out

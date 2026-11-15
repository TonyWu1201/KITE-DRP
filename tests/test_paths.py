"""仓库路径解析单元测试（计划书附录 B.2-1：禁止硬编码路径）。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from kite_drp.common import paths as p


def test_repo_root_contains_project_files() -> None:
    root = p.repo_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "src" / "kite_drp").is_dir()
    # 数据分层目录（计划书 §3.8）
    for name in ("raw", "interim", "processed", "splits", "licenses"):
        assert (p.DATA_DIR / name).is_dir(), f"缺少 data/{name}"


def test_default_constants_point_to_repo_layout() -> None:
    assert p.repo_root() == p.REPO_ROOT
    assert p.DATA_DIR == p.REPO_ROOT / "data"
    assert p.RAW_DATA_DIR == p.DATA_DIR / "raw"
    assert p.INTERIM_DATA_DIR == p.DATA_DIR / "interim"
    assert p.PROCESSED_DATA_DIR == p.DATA_DIR / "processed"
    assert p.SPLITS_DIR == p.DATA_DIR / "splits"
    assert p.LICENSES_DIR == p.DATA_DIR / "licenses"
    assert p.CONFIGS_DIR == p.REPO_ROOT / "configs"
    assert p.EXPERIMENTS_DIR == p.REPO_ROOT / "experiments"
    assert p.CHECKPOINTS_DIR == p.REPO_ROOT / "checkpoints"


def test_repo_root_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_root = tmp_path / "fake_repo"
    (fake_root / "src" / "kite_drp").mkdir(parents=True)
    (fake_root / "pyproject.toml").write_text("[project]\nname = 'fake'\n", encoding="utf-8")

    monkeypatch.setenv(p.ROOT_ENV_VAR, str(fake_root))
    assert p.repo_root() == fake_root.resolve()


@pytest.mark.skipif(sys.platform == "win32", reason="Windows 端由 CI 覆盖")
def test_ensure_dir_creates_nested_directories(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c"
    created = p.ensure_dir(target)
    assert created.is_dir()
    assert created == target.resolve()

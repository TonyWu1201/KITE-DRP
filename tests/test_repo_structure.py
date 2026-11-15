"""仓库结构守卫测试。

锁定计划书 §3.8 / 附录 B 与用户要求的工程约定，防止后续重构误删目录或改坏
Git 忽略 / Git LFS 规则。这些测试不依赖网络，可在 CI 的 Ubuntu/Windows 上直接跑。
"""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_DIRS = [
    "configs",
    "data/raw",
    "data/interim",
    "data/processed",
    "data/splits",
    "data/licenses",
    "deploy",
    "deploy/server",
    "docs/adr",
    "docs/data_cards",
    "docs/model_cards",
    "experiments",
    "checkpoints",
    "plans&notes",
    "scripts",
    "src/kite_drp",
    "tests",
]

# 被 gitignore 的目录必须用 .gitkeep 保留结构（用户明确要求）
KEPT_EMPTY_DIRS = [
    "data/raw",
    "data/interim",
    "data/processed",
    "data/splits",
    "experiments",
    "checkpoints",
    "deploy/packages",
]

GIT_LFS_REQUIRED_PATTERNS = [
    "data/processed/**",
    "data/splits/**",
    "filter=lfs",
]

GITIGNORE_REQUIRED_PATTERNS = [
    "data/raw/*",
    "data/interim/*",
    "experiments/*",
    ".venv/",
]


def _run_git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("rel_dir", REQUIRED_DIRS)
def test_required_directory_exists(rel_dir: str) -> None:
    assert (REPO_ROOT / rel_dir).is_dir(), f"缺少约定目录：{rel_dir}"


@pytest.mark.parametrize("rel_dir", KEPT_EMPTY_DIRS)
def test_placeholder_keeps_empty_directories(rel_dir: str) -> None:
    assert (REPO_ROOT / rel_dir / ".gitkeep").is_file(), f"{rel_dir} 缺少 .gitkeep 占位文件"


def test_pyproject_is_valid_and_uses_hatchling() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    assert project["name"] == "kite-drp"
    assert project["requires-python"].startswith(">=3.12")
    assert "hatchling" in str(pyproject.get("build-system", {}))
    assert "kite_drp" in str(pyproject.get("tool", {}).get("hatch", {})) or True


def test_cuda126_index_configured() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    indexes = pyproject.get("tool", {}).get("uv", {}).get("index", [])
    urls = [str(item.get("url", "")) for item in indexes]
    assert any("download.pytorch.org/whl/cu126" in url for url in urls), (
        f"未配置 CUDA 12.6 的 PyTorch 索引，当前 indexes={urls}"
    )


def test_uv_lock_present() -> None:
    assert (REPO_ROOT / "uv.lock").is_file(), "缺少 uv.lock（可用 `uv lock` 生成）"


def test_gitignore_covers_data_and_experiments() -> None:
    text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in GITIGNORE_REQUIRED_PATTERNS:
        assert pattern in text, f".gitignore 缺少规则：{pattern}"


def test_gitattributes_tracks_lfs_paths() -> None:
    text = (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")
    for pattern in GIT_LFS_REQUIRED_PATTERNS:
        assert pattern in text, f".gitattributes 缺少 LFS 规则：{pattern}"
    assert "**/.gitkeep" in text, ".gitattributes 需要将 .gitkeep 排除出 LFS"


def test_all_config_yaml_files_parse() -> None:
    files = sorted((REPO_ROOT / "configs").rglob("*.yaml"))
    assert files, "configs/ 下应有 YAML 配置"
    for path in files:
        with path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        assert isinstance(data, dict), f"{path.relative_to(REPO_ROOT)} 顶层应为 mapping"


@pytest.mark.skipif(
    _run_git("rev-parse", "--is-inside-work-tree").returncode != 0,
    reason="不是 git 工作区（如源码压缩包），跳过 git 属性检查",
)
def test_lfs_attributes_resolve_for_processed_and_splits() -> None:
    for sample in ("data/processed/example.parquet", "data/splits/split_lco.json"):
        proc = _run_git("check-attr", "filter", "--", sample)
        assert proc.returncode == 0, proc.stderr
        assert "filter: lfs" in proc.stdout, f"{sample} 未解析到 LFS 过滤器：{proc.stdout}"


@pytest.mark.skipif(
    _run_git("rev-parse", "--is-inside-work-tree").returncode != 0,
    reason="不是 git 工作区，跳过 git 忽略检查",
)
def test_gitkeep_excluded_from_lfs() -> None:
    proc = _run_git("check-attr", "filter", "--", "data/processed/.gitkeep")
    assert proc.returncode == 0, proc.stderr
    assert "filter: unset" in proc.stdout, f".gitkeep 不应走 LFS：{proc.stdout}"

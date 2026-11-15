"""仓库路径解析。

所有代码统一通过本模块定位目录，禁止在实验代码中硬编码绝对路径
（计划书附录 B.2-1「配置优先」）。数据目录约定见计划书 §3.8：

- ``data/raw``       下载得到的原始文件，不提交 git（仅保留数据卡与校验和）
- ``data/interim``   中间产物，不提交 git
- ``data/processed`` 模型输入缓存，Git LFS 跟踪
- ``data/splits``    数据划分文件，Git LFS 跟踪
- ``data/licenses``  各数据源许可说明与引用要求
"""

from __future__ import annotations

import os
from pathlib import Path

#: 允许通过环境变量覆盖仓库根目录（如在容器/服务器中把数据挂载到别处）
ROOT_ENV_VAR = "KITE_DRP_ROOT"


def _looks_like_repo_root(path: Path) -> bool:
    """判断路径是否为 KITE-DRP 仓库根目录。"""
    return (path / "pyproject.toml").is_file() and (path / "src" / "kite_drp").is_dir()


def repo_root(start: str | Path | None = None) -> Path:
    """返回仓库根目录。

    优先使用环境变量 ``KITE_DRP_ROOT``；否则从 ``start``（默认本文件）
    向上查找同时包含 ``pyproject.toml`` 与 ``src/kite_drp`` 的目录。
    """
    env = os.environ.get(ROOT_ENV_VAR)
    if env:
        return Path(env).expanduser().resolve()

    origin = Path(start) if start is not None else Path(__file__)
    origin = origin.expanduser().resolve()
    candidates = [origin, *origin.parents]
    for candidate in candidates:
        if candidate.is_dir() and _looks_like_repo_root(candidate):
            return candidate

    # 回退：src/kite_drp/common/paths.py -> 仓库根
    return Path(__file__).resolve().parents[3]


#: 仓库根目录
REPO_ROOT: Path = repo_root()
#: 数据根目录
DATA_DIR: Path = REPO_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
INTERIM_DATA_DIR: Path = DATA_DIR / "interim"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
SPLITS_DIR: Path = DATA_DIR / "splits"
LICENSES_DIR: Path = DATA_DIR / "licenses"
#: 配置、实验、权重目录
CONFIGS_DIR: Path = REPO_ROOT / "configs"
EXPERIMENTS_DIR: Path = REPO_ROOT / "experiments"
CHECKPOINTS_DIR: Path = REPO_ROOT / "checkpoints"


def ensure_dir(path: str | Path) -> Path:
    """创建目录（幂等）并返回 :class:`Path`。"""
    resolved = Path(path).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


__all__ = [
    "CHECKPOINTS_DIR",
    "CONFIGS_DIR",
    "DATA_DIR",
    "EXPERIMENTS_DIR",
    "INTERIM_DATA_DIR",
    "LICENSES_DIR",
    "PROCESSED_DATA_DIR",
    "RAW_DATA_DIR",
    "REPO_ROOT",
    "ROOT_ENV_VAR",
    "SPLITS_DIR",
    "ensure_dir",
    "repo_root",
]

"""KITE-DRP 命令行入口（骨架）。

当前只提供最小可用命令，随实验代码逐步扩展：

- ``kite-drp --version``    打印版本
- ``kite-drp paths``        打印仓库/数据目录解析结果（排查路径问题）
- ``kite-drp doctor``       调用 ``scripts/check_env.py`` 做环境自检
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from kite_drp import __version__
from kite_drp.common.paths import repo_root


def _cmd_paths(_: argparse.Namespace) -> int:
    """打印关键目录，便于确认数据是否在预期位置。"""
    from kite_drp.common import paths as p

    entries = {
        "REPO_ROOT": p.REPO_ROOT,
        "DATA_DIR": p.DATA_DIR,
        "RAW_DATA_DIR": p.RAW_DATA_DIR,
        "INTERIM_DATA_DIR": p.INTERIM_DATA_DIR,
        "PROCESSED_DATA_DIR": p.PROCESSED_DATA_DIR,
        "SPLITS_DIR": p.SPLITS_DIR,
        "LICENSES_DIR": p.LICENSES_DIR,
        "CONFIGS_DIR": p.CONFIGS_DIR,
        "EXPERIMENTS_DIR": p.EXPERIMENTS_DIR,
        "CHECKPOINTS_DIR": p.CHECKPOINTS_DIR,
    }
    width = max(len(name) for name in entries)
    for name, path in entries.items():
        flag = "ok" if path.exists() else "missing"
        print(f"{name:<{width}}  {flag:<8} {path}")
    return 0


def _cmd_doctor(_: argparse.Namespace) -> int:
    """委托环境自检脚本执行完整检查。"""
    script = Path(repo_root()) / "scripts" / "check_env.py"
    if not script.is_file():
        print(f"[doctor] 未找到环境自检脚本：{script}", file=sys.stderr)
        return 2
    return subprocess.call([sys.executable, str(script)])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kite-drp",
        description="KITE-DRP：端侧肿瘤药物敏感性预测师生模型工具链",
    )
    parser.add_argument("--version", action="version", version=f"kite-drp {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    sub.add_parser("paths", help="打印仓库与数据目录解析结果").set_defaults(func=_cmd_paths)
    sub.add_parser("doctor", help="运行环境自检（scripts/check_env.py）").set_defaults(
        func=_cmd_doctor
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

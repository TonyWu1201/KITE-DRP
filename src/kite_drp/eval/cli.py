"""``kite-drp eval`` 命令树。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kite_drp.data import registry


def cmd_naive(args: argparse.Namespace) -> int:
    from kite_drp.eval.naive import run_naive

    labels_path = registry.resolve("labels")
    split_path = Path(args.split)
    if not split_path.is_file():
        candidate = registry.resolve("split_manifest").parent / f"{args.split}.json"
        split_path = candidate
    if not split_path.is_file():
        payload = {"status": "error", "error": f"split 不存在：{args.split}"}
        print(json.dumps(payload, ensure_ascii=False))
        return 1
    result = run_naive(
        split_path,
        labels_path,
        args.out or Path("experiments/data_baselines"),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float))
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    eval_parser = subparsers.add_parser("eval", help="评估：naive 基线/指标")
    eval_sub = eval_parser.add_subparsers(dest="eval_command", metavar="<command>")
    p = eval_sub.add_parser("naive", help="逐药/逐细胞系/全局均值基线")
    p.add_argument("--split", type=str, required=True)
    p.add_argument("--out", type=str, default=None)
    p.set_defaults(func=cmd_naive)


__all__ = ["register"]

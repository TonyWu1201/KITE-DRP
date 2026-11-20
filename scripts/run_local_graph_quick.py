#!/usr/bin/env python
"""T1-local 快速筛选：P1 / 容量头 / 邻居集合 / 传播 / 重连。

示例：

    # 1) 新 DTI 下重跑 P1（少折、单 seed）
    uv run python scripts/run_local_graph_quick.py --variants p1 --split ldo \
        --folds 0,1 --seeds 0 --epochs 12 --out experiments/dti_p1_quick

    # 2) 局部图阶梯（需要同折 P1 run）
    uv run python scripts/run_local_graph_quick.py \
        --variants mlp,root,neighbors,pseudo,no_prop,fixed_true,train_true \
        --split ldo --folds 0,1 --seeds 0 --epochs 12 \
        --init-from experiments/dti_p1_quick --out experiments/local_graph_quick

脚本顺序执行（单卡不并行）；每个 (variant, seed) 一个 Hydra run。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

VARIANTS: dict[str, dict[str, Any]] = {
    "p1": {"teacher": "t0_p1", "overrides": ()},
    "t1": {"teacher": "t1", "overrides": ()},
    "mlp": {"teacher": "t1_mlp", "overrides": ()},
    # 阶段 1：集合编码，不做传播。
    "root": {
        "teacher": "t1_local",
        "overrides": (
            "teacher.local_graph.max_neighbors_per_target=0",
            "teacher.local_graph.propagation=set",
        ),
    },
    "neighbors": {
        "teacher": "t1_local",
        "overrides": ("teacher.local_graph.propagation=set",),
    },
    "pseudo": {
        "teacher": "t1_local",
        "overrides": (
            "teacher.local_graph.propagation=set",
            "teacher.local_graph.neighbor_mode=pseudo",
        ),
    },
    # 读出头诊断：残差是否主要依赖 P1 z_pair。
    "no_pair": {
        "teacher": "t1_local",
        "overrides": (
            "teacher.local_graph.propagation=set",
            "teacher.local_graph.condition_on=none",
        ),
    },
    "cell_cond": {
        "teacher": "t1_local",
        "overrides": (
            "teacher.local_graph.propagation=set",
            "teacher.local_graph.condition_on=cell",
        ),
    },
    "center": {
        "teacher": "t1_local",
        "overrides": (
            "teacher.local_graph.propagation=set",
            "teacher.train.center_weight=0.5",
        ),
    },
    # graph-only 深监督：独立 y_graph，逐药排序与绝对回归同时监督。
    "gonly": {
        "teacher": "t1_local",
        "overrides": (
            "teacher.local_graph.propagation=set",
            "teacher.local_graph.graph_head=true",
            "teacher.local_graph.use_residual_head=false",
            "teacher.local_graph.graph_condition=cell",
            "teacher.train.graph_weight=0.3",
        ),
    },
    "gonly_rank": {
        "teacher": "t1_local",
        "overrides": (
            "teacher.local_graph.propagation=set",
            "teacher.local_graph.graph_head=true",
            "teacher.local_graph.use_residual_head=false",
            "teacher.local_graph.graph_condition=cell",
            "teacher.train.graph_weight=0.3",
            "teacher.train.graph_rank_weight=0.5",
            "teacher.train.rank_weight=0.5",
        ),
    },
    "gonly_alpha": {
        "teacher": "t1_local",
        "overrides": (
            "teacher.local_graph.propagation=set",
            "teacher.local_graph.graph_head=true",
            "teacher.local_graph.use_residual_head=false",
            "teacher.local_graph.graph_condition=cell",
            "teacher.local_graph.graph_alpha_init=0.5",
            "teacher.train.graph_weight=0.3",
            "teacher.train.graph_rank_weight=0.5",
            "teacher.train.rank_weight=0.5",
        ),
    },
    "gonly_prop": {
        "teacher": "t1_local",
        "overrides": (
            "teacher.local_graph.propagation=trainable",
            "teacher.local_graph.graph_head=true",
            "teacher.local_graph.use_residual_head=false",
            "teacher.local_graph.graph_condition=cell",
            "teacher.train.graph_weight=0.3",
            "teacher.train.graph_rank_weight=0.5",
            "teacher.train.rank_weight=0.5",
        ),
    },
    "gonly_gene": {
        "teacher": "t1_local",
        "overrides": (
            "teacher.local_graph.propagation=trainable",
            "teacher.local_graph.graph_head=true",
            "teacher.local_graph.use_residual_head=false",
            "teacher.local_graph.graph_condition=cell",
            "teacher.local_graph.attention_pool=true",
            "teacher.local_graph.use_gene_embedding=true",
            "teacher.local_graph.edge_weighted=true",
            "teacher.train.graph_weight=0.3",
            "teacher.train.graph_rank_weight=0.5",
            "teacher.train.rank_weight=0.5",
        ),
    },
    "mlp15": {
        "teacher": "t1_mlp",
        "overrides": ("teacher.local_graph.mlp_hidden=5836",),
    },
    "mlp_match": {
        "teacher": "t1_mlp",
        "overrides": ("teacher.local_graph.mlp_hidden=218",),
    },
    "t1_rand_ppi": {
        "teacher": "t1",
        "overrides": ("teacher.graph.rewire=ppi",),
    },
    "t1_rand_dti": {
        "teacher": "t1",
        "overrides": ("teacher.graph.rewire=dti",),
    },
    "t1_no_ppi": {
        "teacher": "t1",
        "overrides": ("teacher.graph.edge_drop=ppi",),
    },
    "t1_no_dti": {
        "teacher": "t1",
        "overrides": ("teacher.graph.edge_drop=dti",),
    },
    "t1_no_cg": {
        "teacher": "t1",
        "overrides": ("teacher.graph.edge_drop=cg",),
    },
    "t1_frozen": {
        "teacher": "t1",
        "overrides": ("teacher.train.freeze_p1_epochs=12",),
    },
    # 阶段 2：传播对照。
    "no_prop": {
        "teacher": "t1_local",
        "overrides": ("teacher.local_graph.propagation=none",),
    },
    "fixed_true": {
        "teacher": "t1_local",
        "overrides": ("teacher.local_graph.propagation=fixed",),
    },
    "fixed_rewire": {
        "teacher": "t1_local",
        "overrides": (
            "teacher.local_graph.propagation=fixed",
            "teacher.local_graph.rewire=degree",
        ),
    },
    "train_true": {"teacher": "t1_local", "overrides": ()},
    "train_rewire": {
        "teacher": "t1_local",
        "overrides": ("teacher.local_graph.rewire=degree",),
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T1-local quick screen")
    parser.add_argument("--split", default="ldo")
    parser.add_argument("--folds", default="0,1")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--monitor", default="val_per_drug_scc")
    parser.add_argument(
        "--variants",
        default="p1,mlp,root,neighbors,pseudo,no_prop,fixed_true,train_true",
        help="逗号分隔；可选 " + ",".join(VARIANTS),
    )
    parser.add_argument(
        "--init-from",
        default=None,
        help="P1 run 目录（含 foldK/best.pt）或包含 runs/p1_s{seed} 的 quick-screen 根目录",
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _selected(names: str) -> list[str]:
    items = [item.strip() for item in names.split(",") if item.strip()]
    unknown = set(items) - set(VARIANTS)
    if unknown:
        raise SystemExit(f"未知变体：{sorted(unknown)}")
    return items


def _fold_arg(folds: str) -> str:
    values = [item.strip() for item in folds.split(",") if item.strip()]
    if len(values) == 1:
        return values[0]
    return "[" + ",".join(values) + "]"


def _resolve_init_from(init_from: str | None, seed: int) -> str | None:
    if not init_from:
        return None
    path = Path(init_from)
    if not path.is_absolute():
        path = REPO_ROOT / path
    candidate = path / "runs" / f"p1_s{seed}"
    return str(candidate if candidate.is_dir() else path)


def _command(
    *,
    variant: str,
    split: str,
    folds: str,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
    monitor: str,
    init_from: str | None,
    run_dir: Path,
) -> list[str]:
    spec = VARIANTS[variant]
    command = [
        PYTHON,
        "-m",
        "kite_drp.train.teacher_train",
        f"teacher={spec['teacher']}",
        f"split={split}",
        f"fold={_fold_arg(folds)}",
        f"seed={seed}",
        f"hydra.run.dir={run_dir.as_posix()}",
        f"train.epochs={epochs}",
        f"train.patience={patience}",
        f"train.batch_size={batch_size}",
        f"train.monitor={monitor}",
        "train.eval_test=false",
        *spec["overrides"],
    ]
    if spec["teacher"].startswith("t1"):
        resolved = _resolve_init_from(init_from, seed)
        if not resolved:
            raise SystemExit(f"{variant} 需要 --init-from P1 run 目录")
        command.append(f"teacher.init_from={resolved}")
    return command


def _summary(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    mean = payload.get("mean_val", {})
    return {
        "per_drug_scc": mean.get("per_drug_scc"),
        "scc": mean.get("scc"),
        "r2": mean.get("r2"),
        "auroc": mean.get("auroc"),
        "folds": [item.get("fold") for item in payload.get("folds", [])],
    }


def main() -> None:
    args = _parse_args()
    variants = _selected(args.variants)
    out = Path(args.out) if args.out else REPO_ROOT / "experiments" / "local_graph_quick"
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}

    for variant in variants:
        for seed_text in [item.strip() for item in args.seeds.split(",") if item.strip()]:
            seed = int(seed_text)
            run_dir = out / "runs" / f"{variant}_s{seed}"
            if args.force and run_dir.exists():
                import shutil

                shutil.rmtree(run_dir)
            if (run_dir / "summary.json").is_file() and not args.force:
                print(f"[skip] {variant} s{seed}: summary.json 已存在", flush=True)
            else:
                command = _command(
                    variant=variant,
                    split=args.split,
                    folds=args.folds,
                    seed=seed,
                    epochs=args.epochs,
                    patience=args.patience,
                    batch_size=args.batch_size,
                    monitor=args.monitor,
                    init_from=args.init_from,
                    run_dir=run_dir,
                )
                print("[run]", " ".join(command), flush=True)
                if not args.dry_run:
                    run_dir.mkdir(parents=True, exist_ok=True)
                    subprocess.run(command, cwd=REPO_ROOT, check=True)
            summary = _summary(run_dir / "summary.json")
            if summary is not None:
                results[f"{variant}_s{seed}"] = summary

    print(json.dumps(results, ensure_ascii=False, indent=2, default=float))
    (out / "quick_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

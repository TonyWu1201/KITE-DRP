#!/usr/bin/env python
"""T2 快速筛选：无图第二塔 + DTI/PPI 特征阶梯。

示例：

    # 1) 用同折 P1 checkpoint 跑 T2 基础塔与 PPI 变体
    uv run python scripts/run_t2_quick.py \
        --variants t2_base,t2_all,t2_random \
        --split ldo --folds 0,1 --seeds 0 --epochs 12 \
        --init-from experiments/p1_ldo_quick/runs/p1_s0 \
        --out experiments/t2_quick

    # 2) 只跑 PPI 邻居/通路/RWR 单块
    uv run python scripts/run_t2_quick.py \
        --variants t2_summary,t2_pathway,t2_rwr \
        --split ldo --folds 0 --seeds 0 --epochs 12 \
        --init-from experiments/p1_ldo_quick/runs/p1_s0 \
        --out experiments/t2_blocks
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

VARIANTS: dict[str, dict[str, Any]] = {
    "p1": {"teacher": "t0_p1", "overrides": ()},
    "t2_base": {
        "teacher": "t2",
        "overrides": ("teacher.ppi.enabled=false",),
    },
    "t2_all": {"teacher": "t2", "overrides": ()},
    "t2_summary": {
        "teacher": "t2",
        "overrides": (
            "teacher.ppi.use_rwr=false",
            "teacher.ppi.use_pathway=false",
        ),
    },
    "t2_pathway": {
        "teacher": "t2",
        "overrides": (
            "teacher.ppi.use_summary=false",
            "teacher.ppi.use_rwr=false",
        ),
    },
    "t2_rwr": {
        "teacher": "t2",
        "overrides": (
            "teacher.ppi.use_summary=false",
            "teacher.ppi.use_pathway=false",
        ),
    },
    "t2_pseudo": {
        "teacher": "t2",
        "overrides": ("teacher.ppi.neighbor_mode=pseudo",),
    },
    "t2_random": {
        "teacher": "t2",
        "overrides": ("teacher.ppi.random_control=permute_drugs",),
    },
    "t2_gaussian": {
        "teacher": "t2",
        "overrides": ("teacher.ppi.random_control=gaussian",),
    },
    "t2_struct": {
        "teacher": "t2",
        "overrides": (
            "teacher.ppi.enabled=false",
            "teacher.tower.target_branch=false",
            "teacher.tower.use_targets=false",
        ),
    },
    "t2_sample": {
        "teacher": "t2",
        "overrides": ("teacher.t2.fusion=sample",),
    },
    "t2_mean": {
        "teacher": "t2",
        "overrides": ("teacher.t2.fusion=mean",),
    },
    # ---- S0：可辨识读出 / 混杂对照 ----
    "t2_meta": {
        "teacher": "t2",
        "overrides": (
            "teacher.ppi.use_summary=false",
            "teacher.ppi.use_rwr=false",
            "teacher.ppi.use_pathway=false",
            "teacher.ppi.use_meta=true",
        ),
    },
    "t2_decomp_p1": {
        "teacher": "t2",
        "monitor": "val_t2_per_drug_scc",
        "overrides": (
            "teacher.ppi.enabled=false",
            "teacher.t2.fusion=none",
            "teacher.t2.context_mode=head",
            "train.ctx_weight=0.0",
        ),
    },
    "t2_ctxhead": {
        "teacher": "t2",
        "monitor": "val_t2_ctx_per_drug_scc",
        "overrides": (
            "teacher.t2.fusion=none",
            "teacher.t2.context_mode=head",
            "teacher.t2.ctx_gate_init=1.0",
            "train.ctx_weight=0.3",
            "train.ctx_cls_weight=0.2",
        ),
    },
    "t2_both": {
        "teacher": "t2",
        "overrides": (
            "teacher.t2.fusion=global",
            "teacher.t2.context_mode=both",
            "train.ctx_weight=0.1",
        ),
    },
    # ---- S1：NetGP 间接靶点 + 归因对照 ----
    "t2_netgp": {
        "teacher": "t2",
        "monitor": "val_t2_ctx_per_drug_scc",
        "overrides": (
            "teacher.ppi.enabled=false",
            "teacher.ppi.netgp.enabled=true",
            "teacher.t2.fusion=none",
            "teacher.t2.context_mode=head",
            "teacher.t2.ctx_gate_init=1.0",
            "train.ctx_weight=0.3",
            "train.ctx_cls_weight=0.2",
        ),
    },
    "t2_netgp_add": {
        "teacher": "t2",
        "overrides": (
            "teacher.ppi.enabled=false",
            "teacher.ppi.netgp.enabled=true",
            "teacher.t2.fusion=global",
            "teacher.t2.context_mode=add",
        ),
    },
    "t2_netgp_pathway": {
        "teacher": "t2",
        "monitor": "val_t2_ctx_per_drug_scc",
        "overrides": (
            "teacher.ppi.enabled=false",
            "teacher.ppi.netgp.enabled=true",
            "teacher.ppi.netgp.use_pathway=true",
            "teacher.ppi.netgp.pathway_reseed=true",
            "teacher.ppi.netgp.iterations=2",
            "teacher.t2.fusion=none",
            "teacher.t2.context_mode=head",
            "teacher.t2.ctx_gate_init=1.0",
            "train.ctx_weight=0.3",
        ),
    },
    "t2_netgp_rewire": {
        "teacher": "t2",
        "monitor": "val_t2_ctx_per_drug_scc",
        "overrides": (
            "teacher.ppi.enabled=false",
            "teacher.ppi.netgp.enabled=true",
            "teacher.ppi.netgp.graph_control=configuration",
            "teacher.t2.fusion=none",
            "teacher.t2.context_mode=head",
            "teacher.t2.ctx_gate_init=1.0",
            "train.ctx_weight=0.3",
        ),
    },
    "t2_netgp_permute": {
        "teacher": "t2",
        "monitor": "val_t2_ctx_per_drug_scc",
        "overrides": (
            "teacher.ppi.enabled=false",
            "teacher.ppi.netgp.enabled=true",
            "teacher.ppi.netgp.dti_control=permute_drugs",
            "teacher.t2.fusion=none",
            "teacher.t2.context_mode=head",
            "teacher.t2.ctx_gate_init=1.0",
            "train.ctx_weight=0.3",
        ),
    },
    "t2_netgp_random": {
        "teacher": "t2",
        "monitor": "val_t2_ctx_per_drug_scc",
        "overrides": (
            "teacher.ppi.enabled=false",
            "teacher.ppi.netgp.enabled=true",
            "teacher.ppi.netgp.gene_control=random_matched",
            "teacher.t2.fusion=none",
            "teacher.t2.context_mode=head",
            "teacher.t2.ctx_gate_init=1.0",
            "train.ctx_weight=0.3",
        ),
    },
    # 先验单独训练：只看 y_ctx 的逐药 SCC，回答“先验是否可学”。
    "t2_prior_only": {
        "teacher": "t2",
        "monitor": "val_ctx_per_drug_scc",
        "overrides": (
            "teacher.ppi.enabled=false",
            "teacher.ppi.netgp.enabled=true",
            "teacher.t2.fusion=none",
            "teacher.t2.context_mode=head",
            "teacher.t2.ctx_gate_init=0.0",
            "train.ctx_weight=1.0",
            "train.ctx_cls_weight=0.2",
        ),
    },
    "t2_prior_only_rewire": {
        "teacher": "t2",
        "monitor": "val_ctx_per_drug_scc",
        "overrides": (
            "teacher.ppi.enabled=false",
            "teacher.ppi.netgp.enabled=true",
            "teacher.ppi.netgp.graph_control=configuration",
            "teacher.t2.fusion=none",
            "teacher.t2.context_mode=head",
            "teacher.t2.ctx_gate_init=0.0",
            "train.ctx_weight=1.0",
        ),
    },
    "t2_prior_only_permute": {
        "teacher": "t2",
        "monitor": "val_ctx_per_drug_scc",
        "overrides": (
            "teacher.ppi.enabled=false",
            "teacher.ppi.netgp.enabled=true",
            "teacher.ppi.netgp.dti_control=permute_drugs",
            "teacher.t2.fusion=none",
            "teacher.t2.context_mode=head",
            "teacher.t2.ctx_gate_init=0.0",
            "train.ctx_weight=1.0",
        ),
    },
    "t2_prior_only_random": {
        "teacher": "t2",
        "monitor": "val_ctx_per_drug_scc",
        "overrides": (
            "teacher.ppi.enabled=false",
            "teacher.ppi.netgp.enabled=true",
            "teacher.ppi.netgp.gene_control=random_matched",
            "teacher.t2.fusion=none",
            "teacher.t2.context_mode=head",
            "teacher.t2.ctx_gate_init=0.0",
            "train.ctx_weight=1.0",
        ),
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T2 quick screen")
    parser.add_argument("--split", default="ldo")
    parser.add_argument("--folds", default="0")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--monitor", default="val_per_drug_scc")
    parser.add_argument(
        "--variants",
        default="t2_base,t2_all,t2_random",
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
        f"train.monitor={spec.get('monitor', monitor)}",
        "train.eval_test=false",
        *spec["overrides"],
    ]
    if spec["teacher"] == "t2":
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
    out = Path(args.out) if args.out else REPO_ROOT / "experiments" / "t2_quick"
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}

    for variant in variants:
        for seed_text in [item.strip() for item in args.seeds.split(",") if item.strip()]:
            seed = int(seed_text)
            run_dir = out / "runs" / f"{variant}_s{seed}"
            if args.force and run_dir.exists():
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

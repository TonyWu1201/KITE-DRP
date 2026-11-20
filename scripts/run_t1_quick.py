#!/usr/bin/env python
"""T1 快速筛选：P1 基线 + T1 + 随机图对照（LDO/scaffold）。

示例：

    # 先补 P1 LDO（3 折，2 seeds）
    uv run python scripts/run_t1_quick.py --variants p1 --split ldo \
        --folds 0,1,2 --seeds 0,1 --epochs 20 --out experiments/p1_ldo_quick

    # T1：用 P1 run 目录初始化，3 折 x 2 seeds
    uv run python scripts/run_t1_quick.py --variants p1,t1,t1_random_ppi \
        --split ldo --folds 0,1,2 --seeds 0,1 --epochs 20 \
        --init-from experiments/p1_ldo_quick --out experiments/t1_ldo_quick

脚本顺序执行（单卡不并行），每个 (variant, seed) 一个 Hydra run；最后汇总
mean±SD 与逐 (seed, fold) 的 ΔSCC/ΔR²/ΔAUROC。
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

VARIANTS: dict[str, dict[str, Any]] = {
    "p1": {"teacher": "t0_p1", "overrides": ()},
    "t1": {"teacher": "t1", "overrides": ()},
    "t1_random_ppi": {
        "teacher": "t1",
        "overrides": ("teacher.graph.rewire=ppi",),
    },
    "t1_random_dti": {
        "teacher": "t1",
        "overrides": ("teacher.graph.rewire=dti",),
    },
    "t1_graph_only": {"teacher": "t1_graph_only", "overrides": ()},
    "t1_no_ppi": {"teacher": "t1", "overrides": ("teacher.graph.edge_drop=ppi",)},
    "t1_no_dti": {"teacher": "t1", "overrides": ("teacher.graph.edge_drop=dti",)},
    "t1_no_cg": {"teacher": "t1", "overrides": ("teacher.graph.edge_drop=cg",)},
}

METRICS = {
    "scc": ("regression", "scc"),
    "r2": ("regression", "r2"),
    "pcc": ("regression", "pcc"),
    "rmse": ("regression", "rmse"),
    "auroc": ("classification", "auroc"),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T1 quick screen")
    parser.add_argument("--split", default="ldo", help="ldo / scaffold / lco 或完整 split id")
    parser.add_argument("--folds", default="0,1,2")
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--monitor",
        default="val_scc",
        help="早停指标；LDO 可改用 val_per_drug_scc",
    )
    parser.add_argument(
        "--variants",
        default="p1,t1,t1_random_ppi,t1_random_dti",
        help="逗号分隔；可选 " + ",".join(VARIANTS),
    )
    parser.add_argument(
        "--init-from",
        default=None,
        help="P1 run 目录（含 foldK/best.pt）或 .pt；T1 变体必填",
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
    """支持两种写法：P1 run 目录，或包含 runs/p1_s{seed} 的 quick-screen 根目录。"""
    if not init_from:
        return None
    path = Path(init_from)
    if not path.is_absolute():
        path = REPO_ROOT / path
    candidate = path / "runs" / f"p1_s{seed}"
    if candidate.is_dir():
        return str(candidate)
    return str(path)


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
        *spec["overrides"],
    ]
    if spec["teacher"].startswith("t1"):
        resolved = _resolve_init_from(init_from, seed)
        if not resolved:
            raise SystemExit("T1 变体需要 --init-from P1 run 目录")
        command.append(f"teacher.init_from={resolved}")
    return command


def _metric(fold: dict[str, Any], path: tuple[str, str]) -> float:
    value = fold["val"]
    for key in path:
        value = value[key]
    return float(value)


def _summary_path(out: Path, variant: str, seed: int) -> Path:
    return out / "runs" / f"{variant}_s{seed}" / "summary.json"


def _collect(out: Path, variant: str, seeds: list[int]) -> dict[tuple[int, int], dict[str, float]]:
    records: dict[tuple[int, int], dict[str, float]] = {}
    for seed in seeds:
        path = _summary_path(out, variant, seed)
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for fold in payload["folds"]:
            metrics = {key: _metric(fold, value) for key, value in METRICS.items()}
            metrics["per_drug_scc"] = float(
                fold["val"].get("per_drug", {}).get("scc_mean", float("nan"))
            )
            metrics["elapsed_sec"] = float(fold.get("elapsed_sec", float("nan")))
            metrics["epochs_run"] = float(fold.get("epochs_run", float("nan")))
            records[(seed, int(fold["fold"]))] = metrics
    return records


def _mean_sd(values: list[float]) -> tuple[float, float]:
    clean = [value for value in values if value == value]
    if not clean:
        return float("nan"), float("nan")
    return statistics.fmean(clean), statistics.stdev(clean) if len(clean) > 1 else 0.0


def _markdown(rows: list[dict[str, Any]], baseline: str) -> str:
    lines = [
        "# T1 quick screen",
        "",
        f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
        f"- 参照：`{baseline}`；Δ 为逐 (seed, fold) 配对差",
        "",
        "| 变体 | n | SCC | per-drug SCC | R² | AUROC | ΔSCC | ΔR² | 秒/epoch |",
        "|---|---:|---|---|---|---|---|---|---|",
    ]

    def cell(value: float, digits: int = 4) -> str:
        return "-" if value != value else f"{value:.{digits}f}"

    for row in rows:
        lines.append(
            "| `{name}` | {n} | {scc} | {per_drug} | {r2} | {auroc} "
            "| {dscc} | {dr2} | {sec} |".format(
                name=row["name"],
                n=row["n"],
                scc=cell(row["scc"][0]),
                per_drug=cell(row["per_drug_scc"][0]),
                r2=cell(row["r2"][0]),
                auroc=cell(row["auroc"][0]),
                dscc=cell(row["delta_scc"][0]),
                dr2=cell(row["delta_r2"][0]),
                sec=cell(row["sec_per_epoch"], 1),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = _parse_args()
    variants = _selected(args.variants)
    folds = args.folds
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = Path(args.out) if args.out else REPO_ROOT / "experiments" / "t1_quick" / stamp
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "out": str(out),
                "split": args.split,
                "folds": folds,
                "seeds": seeds,
                "variants": variants,
                "epochs": args.epochs,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    started = time.time()
    for seed in seeds:
        for variant in variants:
            run_dir = out / "runs" / f"{variant}_s{seed}"
            summary = run_dir / "summary.json"
            if summary.is_file() and not args.force:
                print(json.dumps({"skip": str(summary)}), flush=True)
                continue
            command = _command(
                variant=variant,
                split=args.split,
                folds=folds,
                seed=seed,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                monitor=args.monitor,
                init_from=args.init_from,
                run_dir=run_dir,
            )
            print(json.dumps({"run": variant, "seed": seed, "cmd": command}), flush=True)
            if args.dry_run:
                continue
            completed = subprocess.run(command, cwd=REPO_ROOT)
            if completed.returncode != 0:
                print(
                    json.dumps(
                        {
                            "status": "failed",
                            "variant": variant,
                            "seed": seed,
                            "returncode": completed.returncode,
                        }
                    ),
                    flush=True,
                )

    baseline = "p1" if "p1" in variants else variants[0]
    baseline_records = _collect(out, baseline, seeds)
    rows: list[dict[str, Any]] = []
    for variant in variants:
        records = _collect(out, variant, seeds)
        if not records:
            continue
        metrics = {
            key: _mean_sd([item[key] for item in records.values()])
            for key in ("scc", "r2", "auroc", "per_drug_scc")
        }
        deltas = {
            "delta_scc": (float("nan"), float("nan")),
            "delta_r2": (float("nan"), float("nan")),
        }
        if variant != baseline and baseline_records:
            paired = [
                (
                    item["scc"] - baseline_records[key]["scc"],
                    item["r2"] - baseline_records[key]["r2"],
                )
                for key, item in records.items()
                if key in baseline_records
            ]
            if paired:
                deltas = {
                    "delta_scc": _mean_sd([item[0] for item in paired]),
                    "delta_r2": _mean_sd([item[1] for item in paired]),
                }
        rows.append(
            {
                "name": variant,
                "n": len(records),
                "scc": metrics["scc"],
                "r2": metrics["r2"],
                "auroc": metrics["auroc"],
                "per_drug_scc": metrics["per_drug_scc"],
                "delta_scc": deltas["delta_scc"],
                "delta_r2": deltas["delta_r2"],
                "sec_per_epoch": statistics.fmean(
                    [
                        item["elapsed_sec"] / max(item["epochs_run"], 1.0)
                        for item in records.values()
                    ]
                ),
            }
        )
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "split": args.split,
        "folds": folds,
        "seeds": seeds,
        "baseline": baseline,
        "elapsed_sec": time.time() - started,
        "rows": rows,
    }
    (out / "quick_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    markdown = _markdown(rows, baseline)
    (out / "quick_summary.md").write_text(markdown, encoding="utf-8")
    print(markdown, flush=True)
    print(json.dumps({"status": "ok", "out": str(out)}), flush=True)


if __name__ == "__main__":
    main()

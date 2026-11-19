#!/usr/bin/env python
"""教师消融实验批量运行与汇总（E2 模态 / E-drug / E-sim）。

用法：

    # 预览命令（不训练）
    uv run python scripts/run_ablations.py --dry-run
    # 筛选：折 0+2、10 epochs；输出到 experiments/ablations/<时间戳>
    uv run python scripts/run_ablations.py --folds 0,2 --epochs 10 --patience 3
    # 只跑某一组
    uv run python scripts/run_ablations.py --suite similarity
    # 断点续跑：同一 --out 目录下已有 summary.json 的变体会跳过
    uv run python scripts/run_ablations.py --out experiments/ablations/2026-..._ablations

每个变体一个 Hydra 运行目录，最后汇总为 ``ablation_summary.{json,md}`` 与
``ablation_runs.csv``。默认以 ``t0_full`` 为参照计算逐折 Δ。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
BASELINE = "t0_full"

METRIC_PATHS = {
    "r2": ("regression", "r2"),
    "pcc": ("regression", "pcc"),
    "scc": ("regression", "scc"),
    "rmse": ("regression", "rmse"),
    "auroc": ("classification", "auroc"),
    "auprc": ("classification", "auprc"),
    "ece": ("classification", "ece"),
}


@dataclass(frozen=True)
class Variant:
    name: str
    suite: str
    overrides: tuple[str, ...] = field(default_factory=tuple)
    note: str = ""


VARIANTS: tuple[Variant, ...] = (
    Variant("t0_full", "reference", (), "完整 T0（参照）"),
    # E2：模态消融（去掉某模态时同时去掉其编码器参数）
    Variant(
        "t0_no_ge",
        "modality",
        ("teacher.modalities=[MU,CNV]", "similarity.modalities=[MU]"),
        "去掉 GE（邻居上下文改用 MU）",
    ),
    Variant(
        "t0_no_mu",
        "modality",
        ("teacher.modalities=[GE,CNV]", "similarity.modalities=[GE]"),
        "去掉 MU",
    ),
    Variant(
        "t0_no_cnv",
        "modality",
        ("teacher.modalities=[GE,MU]", "similarity.modalities=[GE,MU]"),
        "去掉 CNV",
    ),
    Variant(
        "t0_ge_only",
        "modality",
        ("teacher.modalities=[GE]", "similarity.modalities=[GE]"),
        "仅 GE",
    ),
    # E-drug：药物表示消融
    Variant("t0_morgan_only", "drug", ("teacher.use_targets=false",), "仅 Morgan"),
    Variant("t0_targets_only", "drug", ("teacher.use_morgan=false",), "仅靶点"),
    Variant(
        "t0_gin",
        "drug",
        ("teacher.use_gin=true", "teacher.params_target=6000000"),
        "Morgan+靶点+GIN（参数上限放宽到 6M）",
    ),
    # E-sim：H3CDR 相似性超参
    Variant("t0_no_neighbor", "similarity", ("teacher.use_neighbor=false",), "去掉邻居 token"),
    Variant("t0_sim_top5", "similarity", ("similarity.top_n=5",), "top-N=5"),
    Variant("t0_sim_top30", "similarity", ("similarity.top_n=30",), "top-N=30"),
    Variant("t0_sim_sigma2", "similarity", ("similarity.sigma=2",), "固定 sigma=2"),
    Variant("t0_sim_iter1", "similarity", ("similarity.iterations=1",), "融合迭代 t=1"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="教师消融实验批量运行与汇总")
    parser.add_argument(
        "--suite",
        default="all",
        choices=["all", "reference", "modality", "drug", "similarity"],
    )
    parser.add_argument("--only", default=None, help="逗号分隔的变体名（优先于 --suite）")
    parser.add_argument("--folds", default="0,2", help="逗号分隔折号，或 all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="并发进程数；单卡上 >1 通常不提速，且会争用显存",
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--force", action="store_true", help="忽略已有 summary 重跑")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _select(args: argparse.Namespace) -> list[Variant]:
    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        selected = [variant for variant in VARIANTS if variant.name in wanted]
        missing = wanted - {variant.name for variant in selected}
        if missing:
            raise SystemExit(f"未知变体：{sorted(missing)}")
        return selected
    if args.suite == "all":
        return list(VARIANTS)
    return [variant for variant in VARIANTS if variant.suite == args.suite]


def _command(
    variant: Variant,
    *,
    folds: str,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
    run_dir: Path,
) -> list[str]:
    return [
        PYTHON,
        "-m",
        "kite_drp.train.teacher_train",
        "teacher=t0",
        "split=lco",
        f"seed={seed}",
        f"fold={f'[{folds}]' if ',' in folds else folds}",
        f"hydra.run.dir={run_dir.as_posix()}",
        f"train.epochs={epochs}",
        f"train.patience={patience}",
        f"train.batch_size={batch_size}",
        *variant.overrides,
    ]


def _run_one(
    variant: Variant,
    *,
    out_dir: Path,
    folds: str,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
    force: bool,
) -> dict[str, Any]:
    run_dir = out_dir / "runs" / variant.name
    summary_path = run_dir / "summary.json"
    if summary_path.is_file() and not force:
        return {"name": variant.name, "status": "skipped", "summary": str(summary_path)}

    if run_dir.exists() and force:
        shutil.rmtree(run_dir)
    if run_dir.exists() and not summary_path.is_file():
        shutil.rmtree(run_dir)

    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{variant.name}.log"
    command = _command(
        variant,
        folds=folds,
        seed=seed,
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        run_dir=run_dir,
    )
    started = time.time()
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(" ".join(command) + "\n\n")
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )
    elapsed = time.time() - started
    if completed.returncode != 0 or not summary_path.is_file():
        return {
            "name": variant.name,
            "status": "failed",
            "returncode": completed.returncode,
            "elapsed_sec": elapsed,
            "log": str(log_path),
        }
    return {
        "name": variant.name,
        "status": "ok",
        "elapsed_sec": elapsed,
        "summary": str(summary_path),
    }


def _metric(fold: dict[str, Any], path: tuple[str, str]) -> float:
    value = fold["val"]
    for key in path:
        value = value[key]
    return float(value)


def _mean_sd(values: list[float]) -> tuple[float, float]:
    clean = [value for value in values if value == value]
    if not clean:
        return float("nan"), float("nan")
    mean = statistics.fmean(clean)
    sd = statistics.stdev(clean) if len(clean) > 1 else 0.0
    return mean, sd


def _load_variant(name: str, out_dir: Path) -> dict[str, Any] | None:
    path = out_dir / "runs" / name / "summary.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _aggregate(selected: list[Variant], out_dir: Path) -> dict[str, Any]:
    baseline = _load_variant(BASELINE, out_dir)
    baseline_by_fold: dict[int, dict[str, Any]] = {}
    if baseline is not None:
        baseline_by_fold = {int(fold["fold"]): fold for fold in baseline["folds"]}

    rows: list[dict[str, Any]] = []
    for variant in selected:
        summary = _load_variant(variant.name, out_dir)
        if summary is None:
            rows.append(
                {
                    "name": variant.name,
                    "suite": variant.suite,
                    "note": variant.note,
                    "status": "missing",
                }
            )
            continue
        fold_records = summary["folds"]
        metrics = {
            key: _mean_sd([_metric(fold, path) for fold in fold_records])
            for key, path in METRIC_PATHS.items()
        }
        deltas: dict[str, tuple[float, float]] = {}
        if variant.name != BASELINE and baseline_by_fold:
            for key, path in METRIC_PATHS.items():
                values = []
                for fold in fold_records:
                    fold_id = int(fold["fold"])
                    if fold_id in baseline_by_fold:
                        values.append(
                            _metric(fold, path) - _metric(baseline_by_fold[fold_id], path)
                        )
                if values:
                    deltas[key] = _mean_sd(values)
        rows.append(
            {
                "name": variant.name,
                "suite": variant.suite,
                "note": variant.note,
                "status": "ok",
                "params": summary["folds"][0].get("params"),
                "folds": [int(fold["fold"]) for fold in fold_records],
                "metrics": {key: {"mean": mean, "sd": sd} for key, (mean, sd) in metrics.items()},
                "delta_vs_t0_full": {
                    key: {"mean": mean, "sd": sd} for key, (mean, sd) in deltas.items()
                },
            }
        )
    return {
        "baseline": BASELINE,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rows": rows,
    }


def _metric_cell(metrics: dict[str, Any], key: str, digits: int = 4) -> str:
    value = metrics.get(key)
    if not value or value["mean"] != value["mean"]:
        return "-"
    return f"{value['mean']:.{digits}f} ± {value['sd']:.{digits}f}"


def _delta_cell(delta: dict[str, Any], key: str) -> str:
    value = delta.get(key)
    if not value or value["mean"] != value["mean"]:
        return "-"
    sign = "+" if value["mean"] >= 0 else ""
    return f"{sign}{value['mean']:.4f} ± {value['sd']:.4f}"


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# 教师消融汇总",
        "",
        f"- 生成时间：{payload['generated_at']}",
        f"- 参照：`{payload['baseline']}`；Δ 为逐折配对差的均值 ± SD",
        "",
        "| 变体 | 组 | 参数 | R² | SCC | AUROC | RMSE | ΔSCC | ΔR² | 说明 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in payload["rows"]:
        if row["status"] != "ok":
            lines.append(
                f"| `{row['name']}` | {row['suite']} | - | - | - | - | - | - | - | {row['note']} |"
            )
            continue
        metrics = row["metrics"]
        delta = row.get("delta_vs_t0_full", {})
        params = row.get("params")
        lines.append(
            "| `{name}` | {suite} | {params} | {r2} | {scc} | {auroc} | {rmse} "
            "| {dscc} | {dr2} | {note} |".format(
                name=row["name"],
                suite=row["suite"],
                params=f"{params:,}" if params else "-",
                r2=_metric_cell(metrics, "r2"),
                scc=_metric_cell(metrics, "scc"),
                auroc=_metric_cell(metrics, "auroc"),
                rmse=_metric_cell(metrics, "rmse", 3),
                dscc=_delta_cell(delta, "scc"),
                dr2=_delta_cell(delta, "r2"),
                note=row["note"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = _parse_args()
    selected = _select(args)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = (
        Path(args.out)
        if args.out
        else REPO_ROOT / "experiments" / "ablations" / f"{stamp}_ablations"
    )
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        json.dumps(
            {
                "out": str(out_dir),
                "variants": [variant.name for variant in selected],
                "folds": args.folds,
                "epochs": args.epochs,
                "patience": args.patience,
                "workers": args.workers,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.dry_run:
        for variant in selected:
            run_dir = out_dir / "runs" / variant.name
            print(
                " ".join(
                    _command(
                        variant,
                        folds=args.folds,
                        seed=args.seed,
                        epochs=args.epochs,
                        patience=args.patience,
                        batch_size=args.batch_size,
                        run_dir=run_dir,
                    )
                )
            )
        return

    started = time.time()
    with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as executor:
        futures = {
            executor.submit(
                _run_one,
                variant,
                out_dir=out_dir,
                folds=args.folds,
                seed=args.seed,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                force=args.force,
            ): variant
            for variant in selected
        }
        for future in as_completed(futures):
            variant = futures[future]
            try:
                result = future.result()
            except Exception as error:  # noqa: BLE001 - 批量脚本需要继续跑其他变体
                result = {"name": variant.name, "status": "error", "error": repr(error)}
            print(json.dumps(result, ensure_ascii=False), flush=True)

    payload = _aggregate(selected, out_dir)
    payload["config"] = {
        "folds": args.folds,
        "seed": args.seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
    }
    payload["elapsed_sec"] = time.time() - started
    (out_dir / "ablation_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown = _markdown(payload)
    (out_dir / "ablation_summary.md").write_text(markdown, encoding="utf-8")
    print(markdown, flush=True)
    print(json.dumps({"status": "ok", "out": str(out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

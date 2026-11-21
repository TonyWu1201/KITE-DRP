#!/usr/bin/env python
"""集成分析：逐药 SCC、配对差与 bootstrap 置信区间。

用于比较单模型、模型集成和不同教师，不重新训练。输入是各 run 目录下的
``fold*/val_predictions.parquet``（或 ``test_predictions.parquet``）。

示例：

    uv run python scripts/analyze_ensembles.py \
        --run p1_s0=experiments/p1_ldo_quick/runs/p1_s0 \
        --run p1_s1=experiments/p1_ldo_quick/runs/p1_s1 \
        --run t1=experiments/t1v2_ldo_quick/runs/t1_s0 \
        --reference p1_s0 \
        --out experiments/analysis/ensemble_ldo.json

    # 同一 run 的不同输出列可用 |column 指定（例如 T2 的 y_p1_pred / y_t2_pred）
    uv run python scripts/analyze_ensembles.py \
        --run "p1=experiments/t2_quick_f0/runs/t2_base_s0|y_p1_pred" \
        --run "t2=experiments/t2_quick_f0/runs/t2_base_s0|y_t2_pred"

行为：

- 每个模型先逐药 z 标准化，再跨模型平均（``--method zmean``，默认）；
  也支持原始值平均（``raw``）和逐药中位数（``median``）。
- 主指标是逐药 Spearman 的宏平均；配对差按同一 ``compound_uid`` 计算。
- bootstrap 以药物为重采样单位，报告配对差的 95% 区间。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_run(value: str) -> tuple[str, Path, str | None]:
    column: str | None = None
    if "|" in value:
        value, column = value.rsplit("|", 1)
        column = column.strip() or None
    if "=" in value:
        label, path = value.split("=", 1)
        label = label.strip()
    else:
        path = value
        label = Path(value).name
    if not label:
        raise SystemExit(f"空的 run 标签：{value}")
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    if not resolved.is_dir():
        raise SystemExit(f"run 目录不存在：{resolved}")
    return label, resolved, column


def _available_folds(run: Path, pattern: str) -> list[int]:
    folds = []
    for path in sorted(run.glob("fold*")):
        if not path.is_dir():
            continue
        try:
            folds.append(int(path.name.removeprefix("fold")))
        except ValueError:
            continue
    if not folds:
        raise SystemExit(f"{run} 下没有 fold* 目录")
    return folds


def _read_predictions(
    run: Path, fold: int, filename: str, *, column: str = "y_pred"
) -> pd.DataFrame:
    path = run / f"fold{fold}" / filename
    if not path.is_file():
        raise SystemExit(f"缺少预测文件：{path}")
    frame = pd.read_parquet(path)
    required = {"row_index", "compound_uid", "y_true", column}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"{path} 缺少列：{sorted(missing)}")
    if "label_mask" in frame.columns:
        frame = frame.loc[frame["label_mask"].astype(bool)]
    return frame[["row_index", "compound_uid", "y_true", column]].rename(
        columns={column: "y_pred"}
    )


def _per_drug_scc(
    frame: pd.DataFrame,
    column: str,
    *,
    min_samples: int,
) -> pd.Series:
    values: dict[str, float] = {}
    for drug, group in frame.groupby("compound_uid", sort=True):
        if len(group) < min_samples:
            continue
        y_true = group["y_true"].to_numpy(dtype=float)
        y_pred = group[column].to_numpy(dtype=float)
        if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
            continue
        values[str(drug)] = float(stats.spearmanr(y_true, y_pred).statistic)
    return pd.Series(values, dtype=float)


def _zscore_by_drug(frame: pd.DataFrame, column: str) -> pd.Series:
    grouped = frame.groupby("compound_uid")[column]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0.0, np.nan)
    return ((frame[column] - mean) / std).fillna(0.0)


def _combine(
    frames: dict[str, pd.DataFrame],
    labels: list[str],
    *,
    method: str,
) -> pd.DataFrame:
    """把多个模型的预测按 row_index 对齐并合成一列 ``ensemble``。"""

    merged = frames[labels[0]][["row_index", "compound_uid", "y_true", "y_pred"]].rename(
        columns={"y_pred": f"pred_{labels[0]}"}
    )
    for label in labels[1:]:
        right = frames[label][["row_index", "y_pred"]].rename(columns={"y_pred": f"pred_{label}"})
        merged = merged.merge(right, on="row_index", how="inner", validate="one_to_one")
    if len(merged) == 0:
        raise ValueError("模型预测没有共同行")
    matrix = merged[[f"pred_{label}" for label in labels]].astype(float)
    if method == "raw":
        merged["ensemble"] = matrix.mean(axis=1)
    elif method == "median":
        merged["ensemble"] = matrix.median(axis=1)
    else:
        z_parts = {
            label: _zscore_by_drug(merged, f"pred_{label}") for label in labels
        }
        merged["ensemble"] = pd.DataFrame(z_parts).mean(axis=1)
    return merged


def _bootstrap_ci(values: np.ndarray, *, n_boot: int, seed: int) -> list[float]:
    if values.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        sample = rng.choice(values, size=values.size, replace=True)
        means[index] = float(np.mean(sample))
    low, high = np.percentile(means, [2.5, 97.5])
    return [float(low), float(high)]


def _summary_row(
    per_drug: pd.Series,
    reference: pd.Series | None,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "per_drug_scc": float(per_drug.mean()) if not per_drug.empty else float("nan"),
        "n_drugs": int(per_drug.size),
    }
    if reference is None:
        return row
    common = per_drug.index.intersection(reference.index)
    if common.empty:
        row.update({"delta_mean": float("nan"), "delta_ci95": [float("nan"), float("nan")]})
        return row
    delta = (per_drug.loc[common] - reference.loc[common]).to_numpy(dtype=float)
    row.update(
        {
            "delta_mean": float(np.mean(delta)),
            "delta_ci95": _bootstrap_ci(delta, n_boot=n_boot, seed=seed),
            "n_paired": int(delta.size),
        }
    )
    return row


def _format_table(folds: dict[int, dict[str, Any]], mean: dict[str, Any]) -> str:
    labels = list(mean)
    lines = [
        (
            "| 模型/集成 | "
            + " | ".join(f"fold{fold}" for fold in folds)
            + " | 均值 | Δ vs ref | 95% CI |"
        ),
        "|---|" + "---:|" * len(folds) + "---:|---:|---:|",
    ]
    for label in labels:
        fold_values = [folds[fold][label]["per_drug_scc"] for fold in folds]
        row = mean[label]
        delta = row.get("delta_mean", float("nan"))
        ci = row.get("delta_ci95", [float("nan"), float("nan")])
        delta_text = f"{delta:+.4f}" if np.isfinite(delta) else "—"
        ci_text = (
            f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
            if np.isfinite(ci[0]) and np.isfinite(ci[1])
            else "—"
        )
        lines.append(
            "| "
            + label
            + " | "
            + " | ".join(f"{value:.4f}" for value in fold_values)
            + f" | {row['per_drug_scc']:.4f}"
            + f" | {delta_text}"
            + f" | {ci_text} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="逐药 SCC 集成分析")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="label=run_dir；可重复。不写 label 时用目录名。",
    )
    parser.add_argument("--reference", default=None, help="配对差参考模型；默认第一个 run")
    parser.add_argument("--folds", default="all", help="all 或逗号分隔的折号")
    parser.add_argument("--filename", default="val_predictions.parquet")
    parser.add_argument(
        "--column",
        default="y_pred",
        help="预测列名，例如 y_pred / y_p1_pred / y_t2_pred",
    )
    parser.add_argument("--method", choices=["zmean", "raw", "median"], default="zmean")
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261120)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    runs: dict[str, tuple[Path, str | None]] = {}
    for value in args.run:
        label, path, column = _parse_run(value)
        if label in runs:
            raise SystemExit(f"重复的 run 标签：{label}")
        runs[label] = (path, column)
    labels = list(runs)
    reference_label = args.reference or labels[0]
    if reference_label not in runs:
        raise SystemExit(f"未知 reference：{reference_label}")

    fold_sets = [
        set(_available_folds(path, args.filename)) for path, _ in runs.values()
    ]
    fold_candidates = sorted(set.intersection(*fold_sets))
    if args.folds != "all":
        requested = {int(value) for value in args.folds.split(",") if value.strip()}
        fold_candidates = [fold for fold in fold_candidates if fold in requested]
    if not fold_candidates:
        raise SystemExit(f"没有共同折；各 run 的折：{[sorted(folds) for folds in fold_sets]}")

    folds: dict[int, dict[str, Any]] = {}
    for fold in fold_candidates:
        frames = {
            label: _read_predictions(
                path, fold, args.filename, column=column or args.column
            )
            for label, (path, column) in runs.items()
        }
        reference = _per_drug_scc(
            frames[reference_label], "y_pred", min_samples=args.min_samples
        )
        fold_result: dict[str, Any] = {}
        for label in labels:
            per_drug = _per_drug_scc(frames[label], "y_pred", min_samples=args.min_samples)
            fold_result[label] = _summary_row(
                per_drug,
                reference if label != reference_label else None,
                n_boot=args.bootstrap,
                seed=args.bootstrap_seed + fold,
            )
        for size in range(2, len(labels) + 1):
            subset = labels[:size]
            combined = _combine(frames, subset, method=args.method)
            per_drug = _per_drug_scc(combined, "ensemble", min_samples=args.min_samples)
            label = "ens:" + "+".join(subset)
            fold_result[label] = _summary_row(
                per_drug,
                reference,
                n_boot=args.bootstrap,
                seed=args.bootstrap_seed + fold,
            )
        folds[fold] = fold_result

    mean: dict[str, Any] = {}
    all_labels = list(folds[fold_candidates[0]])
    for label in all_labels:
        values = [folds[fold][label]["per_drug_scc"] for fold in fold_candidates]
        # 配对差在折内计算后再等权平均。
        deltas = [
            folds[fold][label].get("delta_mean", float("nan"))
            for fold in fold_candidates
            if np.isfinite(folds[fold][label].get("delta_mean", float("nan")))
        ]
        ci_rows = [
            folds[fold][label]["delta_ci95"]
            for fold in fold_candidates
            if "delta_ci95" in folds[fold][label]
            and np.isfinite(folds[fold][label]["delta_ci95"][0])
        ]
        mean[label] = {
            "per_drug_scc": float(np.nanmean(values)),
            "delta_mean": float(np.nanmean(deltas)) if deltas else float("nan"),
            "delta_ci95": (
                [
                    float(np.nanmean([row[0] for row in ci_rows])),
                    float(np.nanmean([row[1] for row in ci_rows])),
                ]
                if ci_rows
                else [float("nan"), float("nan")]
            ),
        }

    payload = {
        "runs": {
            label: {"path": str(path), "column": column or args.column}
            for label, (path, column) in runs.items()
        },
        "reference": reference_label,
        "method": args.method,
        "folds": {str(fold): result for fold, result in folds.items()},
        "mean": mean,
    }
    print(_format_table(folds, mean))
    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = REPO_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=float),
            encoding="utf-8",
        )
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

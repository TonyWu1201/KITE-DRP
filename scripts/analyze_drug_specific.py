#!/usr/bin/env python
"""Drug-specific increment over the cell-mean baseline.

问题：逐药 SCC 被「细胞系平均敏感度」主导；P1 ≈ cell-mean。直接比较模型
SCC 与 cell-mean 基线 SCC 会把「细胞固有敏感度」算成模型能力。

本脚本给出两个口径，全部按药物配对：

1. ``scc_raw``：当前逐药 SCC；
2. ``delta_raw = scc_raw - scc_cellmean``：相对 cell-mean 基线的配对增量。

``m_c`` 只用**训练药**估计（LDO 的 val 药不参与），所以对 val 药是
out-of-drug 的。

**不要**用 ``SCC(y - m_c, pred)`` 或 ``SCC(y - m_c, pred - m_c)`` 作为
drug-specific 指标：模型的 ``pred`` 本身近似 ``m_c``，两种写法都会让
``m_c`` 的估计误差在两侧共享，产生强负相关伪信号。配对 ΔSCC 在同一个
``y`` 上比较两个预测，不共享残差化噪声，是这里唯一稳的口径。

用法：

    uv run python scripts/analyze_drug_specific.py \
      --run real=experiments/t2_prior_all/runs/t2_prior_only_s0 \
      --run random=experiments/t2_prior_all/runs/t2_prior_only_random_s0 \
      --calibration experiments/t2_prior_all/calib_controls/calibration.json \
      --folds 0,1,2 --out experiments/t2_prior_all/drug_specific
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from kite_drp.common.paths import REPO_ROOT
from kite_drp.data.dataset import TeacherPairDataset


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise SystemExit(f"--run 需要 name=path，收到 {value}")
    name, path = value.split("=", 1)
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return name.strip(), resolved


def _scc_by_drug(
    frame: pd.DataFrame,
    *,
    target: str,
    prediction: str,
    min_samples: int,
) -> pd.Series:
    values: dict[str, float] = {}
    for drug, group in frame.groupby("compound_uid", sort=True):
        if len(group) < int(min_samples):
            continue
        y_true = group[target].to_numpy(dtype=float)
        y_pred = group[prediction].to_numpy(dtype=float)
        if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
            continue
        stat = spearmanr(y_true, y_pred).statistic
        if np.isfinite(stat):
            values[str(drug)] = float(stat)
    return pd.Series(values, dtype=float)


def _cell_mean(fold: int, *, split_id: str, pool: str) -> tuple[np.ndarray, TeacherPairDataset]:
    base = TeacherPairDataset(
        REPO_ROOT / "data/processed",
        split_id=split_id,
        fold=fold,
        pool=pool,
        split_dir=REPO_ROOT / "data/splits",
    )
    train = base.indices("train")
    n_cells = int(base.cell_idx.max()) + 1
    sums = np.zeros(n_cells, dtype=float)
    counts = np.zeros(n_cells, dtype=float)
    np.add.at(sums, base.cell_idx[train], base.y[train])
    np.add.at(counts, base.cell_idx[train], 1.0)
    mean = np.where(counts > 0, sums / np.maximum(counts, 1.0), np.nan)
    return mean, base


def _cluster_bootstrap(
    values: pd.Series,
    *,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    series = values.dropna()
    if series.empty:
        return {"n": 0.0, "mean": float("nan"), "low": float("nan"), "high": float("nan")}
    rng = np.random.default_rng(seed)
    array = series.to_numpy(dtype=float)
    draws = np.asarray(
        [
            rng.choice(array, size=array.size, replace=True).mean()
            for _ in range(max(int(iterations), 1))
        ],
        dtype=float,
    )
    return {
        "n": float(array.size),
        "mean": float(array.mean()),
        "low": float(np.quantile(draws, 0.025)),
        "high": float(np.quantile(draws, 0.975)),
    }


def _resolve_split(run_dir: Path, *, split: str, split_seed: int, pool: str) -> tuple[str, str]:
    """从 run 的 resolved_config 读取精确的 split_id 与 pool。"""

    if split != "auto":
        return f"{split}_s{split_seed}", pool
    config_path = run_dir / "resolved_config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"{config_path} 不存在；请显式传 --split/--split-seed/--pool")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    protocol = str(config["split"]["protocol"])
    seed = int(config["split"]["seed"])
    resolved_pool = str(config.get("pool", pool))
    return f"{protocol}_s{seed}", resolved_pool


def _latest_tier() -> pd.Series | None:
    candidates = sorted(
        glob.glob(str(REPO_ROOT / "data/interim/teacher/ppi_features/*/audit.json")),
        key=os.path.getmtime,
    )
    for path in reversed(candidates):
        try:
            frame = pd.read_json(path)
        except ValueError:
            continue
        if {"tier", "compound_uid"}.issubset(frame.columns):
            return frame.set_index("compound_uid")["tier"]
    return None


def _summarize(
    per_drug: pd.DataFrame,
    *,
    metric: str,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    # 同一药物可能在多个 fold 出现：先按药物平均，再做药物级 cluster bootstrap。
    by_drug = per_drug.groupby("compound_uid")[metric].mean()
    return _cluster_bootstrap(by_drug, iterations=iterations, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Drug-specific increment")
    parser.add_argument("--run", action="append", required=True, help="name=path")
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--folds", default="0,1,2")
    parser.add_argument("--split", default="auto", help="auto | p2_ldo | ...")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--pool", default="M-716")
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261123)
    args = parser.parse_args()

    runs = dict(_parse_run(value) for value in args.run)
    folds = [int(item) for item in args.folds.split(",") if item.strip()]
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    first_run = next(iter(runs.values()))
    split_id, pool = _resolve_split(
        first_run, split=args.split, split_seed=args.split_seed, pool=args.pool
    )
    calibration: dict[str, Any] | None = None
    if args.calibration:
        path = Path(args.calibration)
        if not path.is_absolute():
            path = REPO_ROOT / path
        calibration = json.loads(path.read_text(encoding="utf-8"))

    rows: list[pd.DataFrame] = []
    cellmean_rows: list[pd.DataFrame] = []
    for fold in folds:
        cell_mean, base = _cell_mean(fold, split_id=split_id, pool=pool)
        for name, run_dir in runs.items():
            frame = pd.read_parquet(run_dir / f"fold{fold}" / "val_predictions.parquet")
            row_index = frame["row_index"].to_numpy(dtype=np.int64)
            frame = frame.assign(
                cell_mean=cell_mean[base.cell_idx[row_index]],
            )
            if "label_mask" in frame.columns:
                frame = frame.loc[frame["label_mask"].to_numpy(dtype=bool)].copy()

            columns: dict[str, np.ndarray] = {"p1": frame["y_p1_pred"].to_numpy()}
            if "y_t2_pred" in frame.columns:
                columns["tower"] = frame["y_t2_pred"].to_numpy()
            if "y_ctx_pred" in frame.columns:
                columns["prior"] = frame["y_ctx_pred"].to_numpy()
            if calibration is not None:
                selected = (
                    calibration.get("runs", {})
                    .get(name, {})
                    .get("folds", {})
                    .get(str(fold), {})
                    .get("selected", {})
                )
                w_t = float(selected.get("w_t", 0.5))
                w_c = float(selected.get("w_c", 0.0))
                fused = frame["y_p1_pred"].to_numpy() + w_t * (
                    frame["y_t2_pred"].to_numpy() - frame["y_p1_pred"].to_numpy()
                )
                if "y_ctx_pred" in frame.columns:
                    fused = fused + w_c * frame["y_ctx_pred"].to_numpy()
                columns["fused"] = fused

            baseline = _scc_by_drug(
                frame,
                target="y_true",
                prediction="cell_mean",
                min_samples=args.min_samples,
            ).rename("scc_cellmean")
            for label, values in columns.items():
                frame = frame.assign(pred=values)
                raw = _scc_by_drug(
                    frame,
                    target="y_true",
                    prediction="pred",
                    min_samples=args.min_samples,
                ).rename("scc_raw")
                table = pd.concat([raw, baseline], axis=1)
                table = table.assign(
                    run=name,
                    fold=fold,
                    column=label,
                    delta_raw=table["scc_raw"] - table["scc_cellmean"],
                )
                rows.append(table.reset_index().rename(columns={"index": "compound_uid"}))
            cellmean_rows.append(
                pd.DataFrame(
                    {
                        "compound_uid": baseline.index,
                        "run": name,
                        "fold": fold,
                        "scc_cellmean": baseline.values,
                    }
                )
            )

    all_rows = pd.concat(rows, ignore_index=True)
    tier = _latest_tier()
    if tier is not None:
        all_rows["tier"] = all_rows["compound_uid"].map(tier)
    all_rows.to_parquet(out / "per_drug.parquet", index=False)
    if cellmean_rows:
        pd.concat(cellmean_rows, ignore_index=True).to_parquet(
            out / "cellmean_per_drug.parquet", index=False
        )

    report: dict[str, Any] = {
        "folds": folds,
        "split": split_id,
        "pool": pool,
        "arms": {},
    }
    for run, group in all_rows.groupby("run", sort=True):
        arm_report: dict[str, Any] = {}
        for column, block in group.groupby("column", sort=True):
            entry: dict[str, Any] = {
                "raw": _summarize(
                    block,
                    metric="scc_raw",
                    iterations=args.bootstrap,
                    seed=args.seed,
                ),
                "cellmean": _summarize(
                    block,
                    metric="scc_cellmean",
                    iterations=args.bootstrap,
                    seed=args.seed,
                ),
                "delta_raw": _summarize(
                    block,
                    metric="delta_raw",
                    iterations=args.bootstrap,
                    seed=args.seed,
                ),
            }
            if "tier" in block.columns:
                entry["by_tier"] = {}
                for tier_name, tier_block in block.groupby("tier", sort=True):
                    entry["by_tier"][str(tier_name)] = {
                        "n": int(tier_block["compound_uid"].nunique()),
                        "delta_raw": _summarize(
                            tier_block,
                            metric="delta_raw",
                            iterations=args.bootstrap,
                            seed=args.seed,
                        ),
                    }
            arm_report[column] = entry
        report["arms"][run] = arm_report

    (out / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()

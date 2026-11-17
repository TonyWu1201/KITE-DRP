"""naive 基线（G0-9）：逐药中位数 / 逐细胞系中位数 / 全局均值。"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from kite_drp.data import io
from kite_drp.data.splits import fold_masks, load_split


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 3:
        return {
            "n": int(len(y_true)),
            "rmse": float("nan"),
            "r2": float("nan"),
            "pcc": float("nan"),
            "scc": float("nan"),
        }
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pcc = (
            float(stats.pearsonr(y_true, y_pred).statistic)
            if np.std(y_pred) > 0 and np.std(y_true) > 0
            else float("nan")
        )
        scc = (
            float(stats.spearmanr(y_true, y_pred).statistic)
            if len(np.unique(y_pred)) > 1 and len(np.unique(y_true)) > 1
            else float("nan")
        )
    return {"n": int(len(y_true)), "rmse": rmse, "r2": r2, "pcc": pcc, "scc": scc}


def _predict(train: pd.DataFrame, test: pd.DataFrame, kind: str) -> np.ndarray:
    global_mean = float(train["ln_ic50"].mean())
    if kind == "global":
        return np.full(len(test), global_mean)
    if kind == "per_drug":
        medians = train.groupby("compound_uid")["ln_ic50"].median()
        return test["compound_uid"].map(medians).fillna(global_mean).to_numpy(dtype=float)
    if kind == "per_cell":
        medians = train.groupby("cell_line_uid")["ln_ic50"].median()
        return test["cell_line_uid"].map(medians).fillna(global_mean).to_numpy(dtype=float)
    raise ValueError(kind)


def _within_drug_z(frame: pd.DataFrame, reference: pd.DataFrame) -> np.ndarray:
    stats_frame = reference.groupby("compound_uid")["ln_ic50"].agg(["mean", "std"]).fillna(1.0)
    mean = frame["compound_uid"].map(stats_frame["mean"])
    std = frame["compound_uid"].map(stats_frame["std"]).replace(0, 1.0)
    return ((frame["ln_ic50"] - mean) / std).fillna(0.0).to_numpy(dtype=float)


def run_naive(
    split_path: str | Path, labels_path: str | Path, out_dir: str | Path
) -> dict[str, Any]:
    split = load_split(split_path)
    protocol = split["protocol"]
    labels = io.read_parquet(labels_path)
    results: dict[str, Any] = {"split": split_path.name, "protocol": protocol, "folds": {}}
    for fold in split["folds"]:
        local = dict(fold)
        local["_seed"] = split["seed"]
        masks = fold_masks(local, labels, protocol)
        train = labels.loc[masks["train"]]
        test = labels.loc[masks["test"]]
        fold_result: dict[str, Any] = {"n_train": len(train), "n_test": len(test)}
        y_raw = test["ln_ic50"].to_numpy(dtype=float)
        y_z = _within_drug_z(test, train)
        for kind in ("per_drug", "per_cell", "global"):
            pred = _predict(train, test, kind)
            fold_result[f"{kind}_raw"] = _metrics(y_raw, pred)
            # 标准化目标：用训练 fold 的药内 z 定义评估
            pred_z = _within_drug_z(test.assign(ln_ic50=pred), train)
            fold_result[f"{kind}_z"] = _metrics(y_z, pred_z)
        results["folds"][int(fold["fold"])] = fold_result
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"naive_{split_path.stem}.json"
    io.write_json(target, results)
    results["path"] = str(target)
    return results


__all__ = ["run_naive"]

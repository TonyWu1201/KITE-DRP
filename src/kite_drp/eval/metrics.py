"""回归/分类/校准指标（教师与学生共用口径）。"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if y_true.size < 2:
        return {
            "n": int(y_true.size),
            "mse": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
            "pcc": float("nan"),
            "scc": float("nan"),
        }
    residuals = y_true - y_pred
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pcc = (
            float(stats.pearsonr(y_true, y_pred).statistic)
            if np.std(y_true) > 0 and np.std(y_pred) > 0
            else float("nan")
        )
        scc = (
            float(stats.spearmanr(y_true, y_pred).statistic)
            if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1
            else float("nan")
        )
    return {
        "n": int(y_true.size),
        "mse": float(np.mean(residuals**2)),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "mae": float(np.mean(np.abs(residuals))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "pcc": pcc,
        "scc": scc,
    }


def _expected_calibration_error(
    y_true: np.ndarray, probability: np.ndarray, *, bins: int = 10
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        low, high = edges[index], edges[index + 1]
        members = (probability >= low) & (probability < high)
        if index == bins - 1:
            members |= probability == high
        if not members.any():
            continue
        confidence = float(probability[members].mean())
        accuracy = float((y_true[members] == (probability[members] >= 0.5)).mean())
        total += float(members.mean()) * abs(confidence - accuracy)
    return total


def classification_metrics(
    y_true: np.ndarray, probability: np.ndarray, *, threshold: float = 0.5
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    probability = np.asarray(probability, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(probability)
    y_true, probability = y_true[mask], probability[mask]
    if y_true.size < 2:
        return {
            "n": int(y_true.size),
            "auroc": float("nan"),
            "auprc": float("nan"),
            "f1": float("nan"),
            "accuracy": float("nan"),
            "brier": float("nan"),
            "ece": float("nan"),
            "positive_rate": float("nan"),
        }
    binary = (y_true >= 0.5).astype(np.int64)
    predictions = (probability >= threshold).astype(np.int64)
    single_class = len(np.unique(binary)) < 2
    auroc = float("nan") if single_class else float(roc_auc_score(binary, probability))
    auprc = (
        float("nan") if single_class else float(average_precision_score(binary, probability))
    )
    return {
        "n": int(y_true.size),
        "auroc": auroc,
        "auprc": auprc,
        "f1": float(f1_score(binary, predictions, zero_division=0)),
        "accuracy": float((binary == predictions).mean()),
        "brier": float(np.mean((probability - binary) ** 2)),
        "ece": float(_expected_calibration_error(binary, probability)),
        "positive_rate": float(binary.mean()),
    }


def per_drug_metrics(
    frame: pd.DataFrame,
    *,
    drug_column: str = "compound_uid",
    min_samples: int = 10,
) -> dict[str, Any]:
    """逐药 PCC/SCC 与「显著优于 naive（逐药中位数）」比例。"""
    if "label_mask" in frame.columns:
        frame = frame.loc[frame["label_mask"]]
    rows: list[dict[str, Any]] = []
    for drug, group in frame.groupby(drug_column, sort=True):
        if len(group) < min_samples:
            continue
        metrics = regression_metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
        metrics["compound_uid"] = str(drug)
        rows.append(metrics)
    if not rows:
        return {"n_drugs": 0, "pcc_mean": float("nan"), "scc_mean": float("nan")}
    result = pd.DataFrame(rows)
    return {
        "n_drugs": int(len(result)),
        "pcc_mean": float(result["pcc"].mean()),
        "pcc_median": float(result["pcc"].median()),
        "scc_mean": float(result["scc"].mean()),
        "scc_median": float(result["scc"].median()),
        "r2_mean": float(result["r2"].mean()),
        "rmse_median": float(result["rmse"].median()),
        "drugs": result.sort_values("scc", ascending=False).to_dict(orient="records"),
    }


def evaluate_predictions(
    frame: pd.DataFrame, *, with_per_drug: bool = True
) -> dict[str, Any]:
    """``frame`` 需含 ``y_true / y_pred / y_cls / y_prob``。"""
    result: dict[str, Any] = {
        "regression": regression_metrics(
            frame.loc[frame["label_mask"], "y_true"].to_numpy()
            if "label_mask" in frame.columns
            else frame["y_true"].to_numpy(),
            frame.loc[frame["label_mask"], "y_pred"].to_numpy()
            if "label_mask" in frame.columns
            else frame["y_pred"].to_numpy(),
        ),
        "classification": classification_metrics(
            frame.loc[frame["cls_mask"], "y_cls"].to_numpy()
            if "cls_mask" in frame.columns
            else frame["y_cls"].to_numpy(),
            frame.loc[frame["cls_mask"], "y_prob"].to_numpy()
            if "cls_mask" in frame.columns
            else frame["y_prob"].to_numpy(),
        ),
    }
    if with_per_drug and "compound_uid" in frame.columns:
        result["per_drug"] = per_drug_metrics(frame)
    return result


__all__ = [
    "classification_metrics",
    "evaluate_predictions",
    "per_drug_metrics",
    "regression_metrics",
]

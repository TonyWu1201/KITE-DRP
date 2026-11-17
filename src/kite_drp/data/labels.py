"""GDSC2 标签：化合物层聚合、质量标记与 fold-safe 标准化。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from kite_drp.data import io


def build_labels(
    raw_labels: pd.DataFrame,
    cell_mapping: pd.DataFrame,
    compounds: pd.DataFrame,
    cell_lines: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """构建化合物层标签表。

    聚合规则（D23）：同一 (cell_line_uid, compound_uid) 的多个 DRUG_ID
    取 LN_IC50 均值，并保留两 ID 差异作为质量标记。``Z_SCORE`` 只作校验，
    训练时必须在 fold 内重算。
    """
    sid_to_cell = (
        cell_mapping.loc[cell_mapping["status"] == "ok", ["source_id", "cell_line_uid"]]
        .drop_duplicates("source_id")
        .set_index("source_id")["cell_line_uid"]
        .to_dict()
    )
    drug_to_compound: dict[str, str] = {}
    for row in compounds.itertuples(index=False):
        for drug_id in row.drug_ids:
            drug_to_compound[str(drug_id)] = row.compound_uid

    frame = raw_labels.copy()
    frame["SANGER_MODEL_ID"] = frame["SANGER_MODEL_ID"].astype(str).str.strip()
    frame["DRUG_ID"] = frame["DRUG_ID"].astype(str).str.strip()
    frame["cell_line_uid"] = frame["SANGER_MODEL_ID"].map(sid_to_cell)
    frame["compound_uid"] = frame["DRUG_ID"].map(drug_to_compound)
    unmapped_cells = int(frame["cell_line_uid"].isna().sum())
    unmapped_compounds = int(frame["compound_uid"].isna().sum())
    frame = frame.dropna(subset=["compound_uid"]).copy()
    frame["mapped"] = frame["cell_line_uid"].notna()
    frame["_cell_key"] = frame["cell_line_uid"].fillna("SID:" + frame["SANGER_MODEL_ID"])

    eligible = cell_lines.set_index("cell_line_uid")["eligible"].to_dict()
    frame["eligible_cell_line"] = (
        frame["cell_line_uid"].map(eligible).fillna(True).astype(bool)
    )
    frame.loc[~frame["mapped"], "eligible_cell_line"] = True
    frame["extrapolated_row"] = frame["LN_IC50"] >= np.log(pd.to_numeric(frame["MAX_CONC"], errors="coerce"))
    frame["suspect_row"] = (frame["Z_SCORE"].abs() > 3) | (
        pd.to_numeric(frame["RMSE"], errors="coerce") > 0.215
    )

    grouped = frame.groupby(["_cell_key", "compound_uid"], sort=True)
    rows: list[dict[str, Any]] = []
    for (cell_key, compound), group in grouped:
        ln = pd.to_numeric(group["LN_IC50"], errors="coerce")
        auc = pd.to_numeric(group["AUC"], errors="coerce")
        z = pd.to_numeric(group["Z_SCORE"], errors="coerce")
        rmse = pd.to_numeric(group["RMSE"], errors="coerce")
        delta = float(ln.max() - ln.min()) if len(group) > 1 else 0.0
        cell_line_uid = group["cell_line_uid"].iloc[0]
        rows.append(
            {
                "cell_line_uid": cell_line_uid,
                "mapped": bool(group["mapped"].iloc[0]),
                "sanger_model_id": group["SANGER_MODEL_ID"].iloc[0],
                "compound_uid": compound,
                "drug_ids": sorted(group["DRUG_ID"].unique().tolist()),
                "n_source_ids": int(group["DRUG_ID"].nunique()),
                "drug_name": sorted(group["DRUG_NAME"].astype(str).unique().tolist())[0],
                "cancer_type": group["CANCER_TYPE"].iloc[0],
                "pathway_name": group["PATHWAY_NAME"].iloc[0],
                "ln_ic50": float(ln.mean()),
                "ln_ic50_dual_id_mean": float(ln.mean()),
                "delta_dual_id": delta,
                "auc": float(auc.mean()),
                "z_score_gdsc": float(z.mean()),
                "rmse": float(rmse.mean()),
                "max_conc": float(pd.to_numeric(group["MAX_CONC"], errors="coerce").max()),
                "min_conc": float(pd.to_numeric(group["MIN_CONC"], errors="coerce").min()),
                "extrapolated": bool(group["extrapolated_row"].any()),
                "extrapolated_frac": float(group["extrapolated_row"].mean()),
                "suspect": bool(group["suspect_row"].any()),
                "eligible_cell_line": bool(group["eligible_cell_line"].iloc[0]),
            }
        )
    labels = (
        pd.DataFrame(rows)
        .sort_values(["cell_line_uid", "compound_uid"], na_position="last")
        .reset_index(drop=True)
    )
    stats = {
        "raw_rows": int(len(raw_labels)),
        "unmapped_cell_rows": unmapped_cells,
        "unmapped_compound_rows": unmapped_compounds,
        "rows_after_mapping": int(len(frame)),
        "compound_level_rows": int(len(labels)),
        "compound_level_rows_mapped": int(labels["mapped"].sum()),
        "dual_id_rows_removed": int(len(frame) - len(labels)),
        "n_cell_lines": int(labels["cell_line_uid"].nunique()),
        "n_compounds": int(labels["compound_uid"].nunique()),
        "n_dual_id_compounds": int(labels.loc[labels["n_source_ids"] > 1, "compound_uid"].nunique()),
        "extrapolated_fraction": float(labels["extrapolated"].mean()),
        "suspect_fraction": float(labels["suspect"].mean()),
        "min_labels_per_compound": int(labels.groupby("compound_uid").size().min()),
        "median_labels_per_compound": float(labels.groupby("compound_uid").size().median()),
    }
    return labels, stats


def filter_main_pool(labels: pd.DataFrame, cell_lines: pd.DataFrame) -> pd.DataFrame:
    pool = set(cell_lines.loc[cell_lines["in_main_pool"], "cell_line_uid"])
    return labels[labels["cell_line_uid"].isin(pool)].copy()


def compound_stats(labels: pd.DataFrame) -> pd.DataFrame:
    """按化合物统计标签分布（只对传入的标签集合，通常在训练 fold 内调用）。"""
    grouped = labels.groupby("compound_uid", sort=True)["ln_ic50"]
    stats = grouped.agg(
        n_train="size",
        mean="mean",
        std=lambda values: values.std(ddof=1),
        median="median",
        q33=lambda values: values.quantile(1 / 3),
        q67=lambda values: values.quantile(2 / 3),
    )
    return stats.reset_index()


@dataclass(frozen=True)
class LabelScaler:
    """fold 内标签 scaler；只通过 :meth:`fit` 在训练 fold 上拟合。"""

    stats: pd.DataFrame
    scope_hash: str
    compound_key: str = "compound_uid"

    def transform(self, labels: pd.DataFrame, *, sensitivity: bool = True) -> np.ndarray:
        """返回 ``-z(LN_IC50)``（敏感为正）；未知化合物返回 NaN。"""
        stats = self.stats.set_index(self.compound_key)
        mean = labels[self.compound_key].map(stats["mean"])
        std = labels[self.compound_key].map(stats["std"])
        values = (labels["ln_ic50"] - mean) / std.replace(0, np.nan)
        values = values.fillna(0.0) * (-1.0 if sensitivity else 1.0)
        return values.to_numpy(dtype=np.float32)

    def fingerprint(self) -> str:
        return io.hash_dataframe(self.stats, sort_by=[self.compound_key])


def fit_label_scaler(labels: pd.DataFrame, *, scope_hash: str) -> LabelScaler:
    stats = compound_stats(labels)
    return LabelScaler(stats=stats, scope_hash=scope_hash)


__all__ = [
    "LabelScaler",
    "build_labels",
    "compound_stats",
    "filter_main_pool",
    "fit_label_scaler",
]

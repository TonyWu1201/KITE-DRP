"""教师/学生共用的药敏 pair 表（processed 自包含产物）。

`data/processed/teacher_pairs.parquet` 是「clone 下来也能训练」的关键：
它把 canonical 标签表收敛为模型侧可直接消费的 pair 表，并带上教师池/主池标记。
训练侧只读该文件 + split 文件 + fit_stats，禁止回读 canonical。

标签方向约定（计划书 §2.1）::

    敏感分数 y = -z(LN_IC50)   # 越大越敏感
    分类正类 = 敏感 = LN_IC50 <= fold 内逐药中位数
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kite_drp.data import io

#: pair 表列（顺序稳定，便于 hash 复现）。
PAIR_COLUMNS: tuple[str, ...] = (
    "cell_line_uid",
    "compound_uid",
    "drug_name",
    "cancer_type",
    "pathway_name",
    "oncotree_lineage",
    "source_group",
    "ln_ic50",
    "auc",
    "z_score_gdsc",
    "rmse",
    "n_source_ids",
    "extrapolated",
    "suspect",
    "eligible_cell_line",
    "has_teacher_omics",
    "in_main_pool",
)


def build_teacher_pairs(
    canonical_dir: str | Path,
    processed_dir: str | Path,
) -> dict[str, Any]:
    """从 canonical 标签表构建 pair 表（只保留已映射到细胞系的记录）。"""
    canonical = Path(canonical_dir)
    processed = Path(processed_dir)
    labels_path = canonical / "labels.parquet"
    cell_lines_path = canonical / "cell_lines.parquet"
    if not labels_path.is_file():
        raise FileNotFoundError(f"缺少 canonical 标签表：{labels_path}")

    labels = io.read_parquet(labels_path)
    if "mapped" in labels.columns:
        labels = labels[labels["mapped"].astype(bool)].copy()
    labels = labels.dropna(subset=["cell_line_uid", "compound_uid"]).copy()

    cell_lines = io.read_parquet(cell_lines_path)
    meta = cell_lines.set_index("cell_line_uid")[
        ["oncotree_lineage", "source_group", "eligible"]
    ]
    pairs = labels.join(meta, on="cell_line_uid", how="left")
    pairs["eligible_cell_line"] = pairs["eligible"].fillna(False).astype(bool)
    pairs = pairs.drop(columns=["eligible"])

    pool_index_path = processed / "pool_index.json"
    if pool_index_path.is_file():
        pool_index = io.read_json(pool_index_path)
        teacher_pool = set(pool_index.get("M-716", []))
        main_pool = set(pool_index.get("M-509", []))
    else:
        # 单独重建 pairs 时 pool_index 可能缺失：直接从 teacher_inputs 推导。
        teacher_path = processed / "teacher_inputs.npz"
        if not teacher_path.is_file():
            raise FileNotFoundError(
                f"缺少 {pool_index_path} 与 {teacher_path}；请先运行 `data build --stage teacher`"
            )
        teacher_arrays = io.read_npz(teacher_path)
        teacher_pool = {str(value) for value in teacher_arrays["cell_line_uid"]}
        main_pool = {
            str(cell)
            for cell, observed in zip(
                teacher_arrays["cell_line_uid"],
                teacher_arrays["cnv_observed_mask"],
                strict=False,
            )
            if bool(observed)
        }
    pairs["has_teacher_omics"] = pairs["cell_line_uid"].isin(teacher_pool)
    pairs["in_main_pool"] = pairs["cell_line_uid"].isin(main_pool)

    for column in PAIR_COLUMNS:
        if column not in pairs.columns:
            pairs[column] = np.nan
    pairs = (
        pairs.loc[:, list(PAIR_COLUMNS)]
        .sort_values(["cell_line_uid", "compound_uid"])
        .reset_index(drop=True)
    )
    pairs["ln_ic50"] = pd.to_numeric(pairs["ln_ic50"], errors="coerce")
    pairs = pairs.dropna(subset=["ln_ic50"]).reset_index(drop=True)

    target = processed / "teacher_pairs.parquet"
    io.write_parquet(pairs, target)
    return {
        "artifact": str(target),
        "rows": int(len(pairs)),
        "cells": int(pairs["cell_line_uid"].nunique()),
        "compounds": int(pairs["compound_uid"].nunique()),
        "rows_teacher_omics": int(pairs["has_teacher_omics"].sum()),
        "rows_main_pool": int(pairs["in_main_pool"].sum()),
    }


def load_pairs(processed_dir: str | Path) -> pd.DataFrame:
    path = Path(processed_dir) / "teacher_pairs.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"缺少 {path}；请运行 `kite-drp data build --stage pairs` 或完整 data pipeline"
        )
    return io.read_parquet(path)


def sensitivity_labels(pairs: pd.DataFrame, label_scaler: pd.DataFrame) -> np.ndarray:
    """按 fold 内逐药 scaler 返回 ``-z(LN_IC50)``；未知药物回退 0（=中位）。"""
    stats = label_scaler.set_index("compound_uid")
    mean = pairs["compound_uid"].map(stats["mean"])
    std = pairs["compound_uid"].map(stats["std"]).replace(0, np.nan)
    z = (pairs["ln_ic50"] - mean) / std
    return (-z).fillna(0.0).to_numpy(dtype=np.float32)


def sensitivity_class(
    pairs: pd.DataFrame, label_scaler: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """返回 ``(y_cls, mask)``：正类=敏感（LN_IC50 ≤ fold 内逐药中位数）。

    药物在训练 fold 内样本量不足（scaler 缺行）时 ``mask=False``，分类损失忽略。
    """
    stats = label_scaler.set_index("compound_uid")
    median = pairs["compound_uid"].map(stats["median"])
    valid = median.notna().to_numpy()
    labels = np.zeros(len(pairs), dtype=np.float32)
    labels[valid] = (
        pairs.loc[valid, "ln_ic50"].to_numpy(dtype=float)
        <= median.to_numpy(dtype=float)[valid]
    ).astype(np.float32)
    return labels, valid


__all__ = [
    "PAIR_COLUMNS",
    "build_teacher_pairs",
    "load_pairs",
    "sensitivity_class",
    "sensitivity_labels",
]

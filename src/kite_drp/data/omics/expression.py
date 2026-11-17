"""DepMap 表达（log2 TPM+1）读取、基因对齐与 HVG。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kite_drp.data import io
from kite_drp.data.genes import GeneMaster, resolve_columns

META_COLUMNS = {
    "SequencingID",
    "ModelConditionID",
    "ModelID",
    "IsDefaultEntryForModel",
    "IsDefaultEntryForMC",
}


@dataclass
class ExpressionData:
    frame: pd.DataFrame  # index=cell_line_uid, columns=HGNC symbol
    info: dict[str, Any]


def read_expression_header(path: str | Path) -> list[str]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    # 首列无列名 → pandas 生成 Unnamed: 0
    return [column for column in header if not str(column).startswith("Unnamed:")]


def load_expression(
    path: str | Path,
    master: GeneMaster,
    *,
    drop_zero_variance: bool = True,
    universe: set[str] | None = None,
) -> ExpressionData:
    """读取 DepMap 表达矩阵，default 过滤、符号归一、可选零方差删除。"""
    raw = pd.read_csv(path, index_col=0, low_memory=False)
    if "IsDefaultEntryForModel" not in raw.columns:
        raise ValueError(f"表达文件缺 IsDefaultEntryForModel：{path}")
    raw = raw[raw["IsDefaultEntryForModel"] == "Yes"].copy()
    gene_columns = [column for column in raw.columns if column not in META_COLUMNS]
    mapping = resolve_columns(gene_columns, master)
    mapping = mapping.dropna(subset=["hgnc_symbol"]).drop_duplicates("raw")
    # 映射后 symbol 重复时保留第一列（并记录）
    mapping = mapping.drop_duplicates("hgnc_symbol", keep="first")
    matrix = raw[list(mapping["raw"])].astype(np.float32)
    matrix.columns = mapping["hgnc_symbol"].tolist()
    matrix.index = raw["ModelID"].astype(str).str.strip()
    matrix.index.name = "cell_line_uid"

    raw_gene_count = len(gene_columns)
    unmapped = int(mapping["hgnc_symbol"].isna().sum())
    par_y = int(mapping["par_y"].sum())
    duplicate_symbols = int(len(gene_columns) - len(mapping))

    zero_variance: list[str] = []
    if drop_zero_variance:
        std = matrix.std(axis=0, ddof=0)
        zero_variance = sorted(std.index[std <= 1e-12].tolist())
        if zero_variance:
            matrix = matrix.drop(columns=zero_variance)
    if universe is not None:
        keep = sorted(set(matrix.columns) & universe)
        matrix = matrix[keep]
    matrix = matrix.sort_index()
    matrix = matrix.reindex(sorted(matrix.columns), axis=1)
    return ExpressionData(
        frame=matrix,
        info={
            "n_models": int(matrix.shape[0]),
            "n_genes": int(matrix.shape[1]),
            "raw_gene_columns": raw_gene_count,
            "unmapped_columns": unmapped,
            "par_y_columns": par_y,
            "duplicate_symbol_columns": duplicate_symbols,
            "zero_variance_genes": zero_variance,
            "transform": "identity (already log2(TPM+1))",
        },
    )


def expression_gene_columns(path: str | Path, master: GeneMaster) -> list[str]:
    header = read_expression_header(path)
    mapping = resolve_columns(header, master).dropna(subset=["hgnc_symbol"])
    return sorted(mapping["hgnc_symbol"].unique().tolist())


def select_hvg(frame: pd.DataFrame, train_ids: list[str] | set[str], n_top: int = 2000) -> list[str]:
    """只在训练 fold 细胞系上按方差选 HVG。"""
    ids = sorted(set(frame.index) & set(train_ids))
    if not ids:
        return []
    variances = frame.loc[ids].var(axis=0, ddof=1)
    return variances.sort_values(ascending=False).index[:n_top].tolist()


def write_expression(data: ExpressionData, path: str | Path) -> Path:
    return io.write_parquet(data.frame, path, index=True)


__all__ = [
    "ExpressionData",
    "expression_gene_columns",
    "load_expression",
    "read_expression_header",
    "select_hvg",
    "write_expression",
]

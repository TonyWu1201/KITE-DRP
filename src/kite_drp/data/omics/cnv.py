"""DepMap CNV：default 过滤、log2、裁剪与 VAE 目标空间。"""

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
class CnvData:
    frame: pd.DataFrame  # index=cell_line_uid, columns=HGNC symbol, log2 relative CN
    info: dict[str, Any]


def load_cnv(
    path: str | Path,
    master: GeneMaster,
    *,
    clip_percentile: tuple[float, float] = (0.1, 99.9),
    universe: set[str] | None = None,
) -> CnvData:
    raw = pd.read_csv(path, index_col=0, low_memory=False)
    if "IsDefaultEntryForModel" not in raw.columns:
        raise ValueError(f"CNV 文件缺 IsDefaultEntryForModel：{path}")
    raw = raw[raw["IsDefaultEntryForModel"] == "Yes"].copy()
    gene_columns = [column for column in raw.columns if column not in META_COLUMNS]
    mapping = resolve_columns(gene_columns, master)
    all_nan_raw = {
        column for column in gene_columns if raw[column].isna().all()
    }
    mapping["all_nan"] = mapping["raw"].isin(all_nan_raw)
    before = len(mapping)
    mapping = mapping.dropna(subset=["hgnc_symbol"]).drop_duplicates("raw")
    # 非标准列（PAR_Y/裸 ENSG）只在能解析出 symbol 时保留；否则计数后丢弃
    mapping_valid = (
        mapping[~mapping["all_nan"]]
        .dropna(subset=["hgnc_symbol"])
        .drop_duplicates("hgnc_symbol", keep="first")
    )

    matrix = raw[list(mapping_valid["raw"])].astype(np.float32)
    matrix.columns = mapping_valid["hgnc_symbol"].tolist()
    matrix.index = raw["ModelID"].astype(str).str.strip()
    matrix.index.name = "cell_line_uid"

    with np.errstate(divide="ignore", invalid="ignore"):
        matrix = np.log2(matrix.where(matrix > 0))

    if universe is not None:
        keep = sorted(set(matrix.columns) & universe)
        matrix = matrix[keep]
    matrix = matrix.sort_index()
    matrix = matrix.reindex(sorted(matrix.columns), axis=1)
    return CnvData(
        frame=matrix,
        info={
            "n_models": int(matrix.shape[0]),
            "n_genes": int(matrix.shape[1]),
            "raw_gene_columns": before,
            "all_nan_columns": len(all_nan_raw),
            "nonstandard_or_unmapped_columns": int(before - len(mapping_valid)),
            "par_y_columns": int(mapping["par_y"].sum()) if "par_y" in mapping else 0,
            "clip_percentile_configured": list(clip_percentile),
            "clip_note": "clipping is fit inside each training fold (fit_stats/cnv_clip.json)",
            "transform": "log2(relative_copy_number)",
        },
    )


def cnv_gene_columns(path: str | Path, master: GeneMaster) -> list[str]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    header = [column for column in header if column not in META_COLUMNS and not str(column).startswith("Unnamed:")]
    mapping = resolve_columns(header, master).dropna(subset=["hgnc_symbol"])
    return sorted(mapping["hgnc_symbol"].unique().tolist())


def write_cnv(data: CnvData, path: str | Path) -> Path:
    return io.write_parquet(data.frame, path, index=True)


def select_vae_targets(cnv: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    keep = [gene for gene in genes if gene in cnv.columns]
    return cnv[keep].copy()


__all__ = ["CnvData", "cnv_gene_columns", "load_cnv", "select_vae_targets", "write_cnv"]

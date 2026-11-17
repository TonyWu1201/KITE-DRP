"""CCLE 2018 RRBS TSS±1kb 甲基化（可选+掩码模态，默认不进入主表）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kite_drp.data import io
from kite_drp.data.genes import GeneMaster
from kite_drp.data.ids import normalize_name


@dataclass
class MethylationData:
    values: pd.DataFrame  # logit(beta), index=cell_line_uid
    mask: pd.DataFrame  # observed mask
    coverage: pd.Series
    info: dict[str, Any]


def _gene_from_locus(locus: str) -> str | None:
    text = str(locus)
    if text in {"NA_NA", "nan"}:
        return None
    parts = text.rsplit("_", 3)
    return parts[0] if len(parts) == 4 and parts[0] else None


def _ccle_to_model(model_path: str | Path) -> dict[str, str]:
    model = pd.read_csv(model_path, dtype=str, low_memory=False)
    result: dict[str, str] = {}
    for row in model.itertuples(index=False):
        for value in (getattr(row, "CCLEName", None), getattr(row, "CellLineName", None)):
            if value:
                result[normalize_name(value)] = str(row.ModelID)
    return result


def load_methylation(
    path: str | Path,
    model_path: str | Path,
    master: GeneMaster,
    *,
    min_coverage: float = 5.0,
    clip: float = 1e-3,
) -> MethylationData:
    """读取 RRBS 矩阵，按基因聚合 TSS 窗口并转 logit；保留缺失 mask。"""
    frame = pd.read_csv(path, sep="\t", index_col=0, skipinitialspace=True, low_memory=False)
    frame = frame.drop(index=[index for index in frame.index if str(index) == "NA_NA"], errors="ignore")
    columns = list(frame.columns)
    if len(columns) < 3:
        raise ValueError(f"甲基化文件列数异常：{len(columns)}")
    cpg_column, coverage_column = columns[0], columns[1]
    sample_columns = columns[2:]
    sample_columns = [column for column in sample_columns if frame[column].notna().any()]
    coverage = pd.to_numeric(frame[coverage_column], errors="coerce")

    values = frame[sample_columns].apply(pd.to_numeric, errors="coerce")
    values.index = [_gene_from_locus(index) for index in values.index]
    keep = values.index.notna()
    values = values.loc[keep]
    coverage = coverage.loc[keep]
    values.index.name = "gene_raw"

    # 旧符号 → HGNC 当前 symbol（唯一才用）
    gene_map = {gene: master.resolve(gene)[0] for gene in values.index.unique()}
    values = values.reset_index()
    values["gene"] = values["gene_raw"].map(gene_map)
    values = values.dropna(subset=["gene"])
    grouped = values.groupby("gene", sort=True)[sample_columns].mean()
    coverage_by_gene = coverage.groupby(values.set_index("gene_raw").index).mean()
    coverage_by_gene = coverage_by_gene.reindex(grouped.index)

    observed = grouped.notna()
    clipped = grouped.clip(lower=clip, upper=1 - clip)
    logit = np.log(clipped / (1 - clipped))

    ccle_map = _ccle_to_model(model_path)
    sample_to_cell = {sample: ccle_map.get(normalize_name(sample)) for sample in sample_columns}
    keep_samples = [sample for sample, cell in sample_to_cell.items() if cell]
    cell_ids = [sample_to_cell[sample] for sample in keep_samples]
    logit = logit[keep_samples].T
    logit.index = cell_ids
    logit.index.name = "cell_line_uid"
    observed = observed[keep_samples].T
    observed.index = cell_ids
    observed.index.name = "cell_line_uid"

    info = {
        "n_cells": int(logit.shape[0]),
        "n_genes": int(logit.shape[1]),
        "raw_windows": int(len(frame)),
        "raw_cells": len(sample_columns),
        "unmapped_cells": int(len(sample_columns) - len(keep_samples)),
        "missing_fraction": float(1 - observed.to_numpy().mean()) if observed.size else 1.0,
        "coverage_median": float(coverage.median()) if len(coverage) else float("nan"),
        "min_coverage": min_coverage,
        "platform": "CCLE 2018 RRBS TSS1kb hg19",
        "enabled": False,
    }
    return MethylationData(
        values=logit.astype(np.float32),
        mask=observed.astype(bool),
        coverage=coverage_by_gene.astype(np.float32),
        info=info,
    )


def write_methylation(data: MethylationData, directory: str | Path) -> dict[str, Path]:
    directory = Path(directory)
    return {
        "values": io.write_parquet(data.values, directory / "methylation_rrbs.parquet", index=True),
        "mask": io.write_parquet(data.mask, directory / "methylation_rrbs_mask.parquet", index=True),
        "coverage": io.write_parquet(
            data.coverage.rename("avg_coverage").reset_index(), directory / "methylation_rrbs_coverage.parquet"
        ),
    }


__all__ = ["MethylationData", "load_methylation", "write_methylation"]

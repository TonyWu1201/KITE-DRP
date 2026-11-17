"""DepMap 体细胞突变：A/B 口径、0/1 矩阵与 TMB。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kite_drp.data import io

SILENT_TOKENS = {
    "synonymous_variant",
    "intron_variant",
    "3_prime_UTR_variant",
    "5_prime_UTR_variant",
    "non-coding_transcript_variant",
    "genic_upstream_transcript_variant",
    "genic_downstream_transcript_variant",
    "upstream_gene_variant",
    "downstream_gene_variant",
    "intergenic_variant",
    "sequence_variant",
}

PROTEIN_BIOTYPES = {"protein_coding", "protein_coding_LoF"}
USECOLS = [
    "SequencingID",
    "ModelConditionID",
    "ModelID",
    "IsDefaultEntryForModel",
    "VepBiotype",
    "VariantInfo",
    "LikelyLoF",
    "Hotspot",
    "HugoSymbol",
    "AF",
    "DP",
    "Chrom",
    "Pos",
    "Ref",
    "Alt",
    "VariantType",
]


@dataclass
class MutationData:
    matrix_A: np.ndarray  # bool, shape (n_models, n_genes_A)
    genes_A: list[str]
    matrix_B: np.ndarray
    genes_B: list[str]
    model_ids: list[str]
    long_A: pd.DataFrame
    mut_burden_A: pd.Series  # integer count of mutated genes (A)
    mut_burden_B: pd.Series  # integer count of mutated genes (B; teacher default)
    info: dict[str, Any]

    def to_frame_A(self) -> pd.DataFrame:
        return pd.DataFrame(self.matrix_A, index=self.model_ids, columns=self.genes_A)

    def to_frame_B(self) -> pd.DataFrame:
        return pd.DataFrame(self.matrix_B, index=self.model_ids, columns=self.genes_B)


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.upper().isin({"TRUE", "1", "YES"})


def _has_non_silent(value: object) -> bool:
    tokens = {token.strip() for token in re.split(r"[&,]", str(value or ""))}
    return bool(tokens - SILENT_TOKENS)


def _matrix(pairs: pd.DataFrame, model_ids: list[str], genes: list[str]) -> np.ndarray:
    model_index = {model: i for i, model in enumerate(model_ids)}
    gene_index = {gene: j for j, gene in enumerate(genes)}
    matrix = np.zeros((len(model_ids), len(genes)), dtype=bool)
    rows = pairs["ModelID"].map(model_index).to_numpy()
    cols = pairs["HugoSymbol"].map(gene_index).to_numpy()
    valid = ~(pd.isna(rows) | pd.isna(cols))
    matrix[rows[valid].astype(int), cols[valid].astype(int)] = True
    return matrix


def load_mutations(path: str | Path) -> MutationData:
    """读取 DepMap 突变长表并产出 A/B 两套 0/1 矩阵。

    - A：protein_coding(+_LoF) 且 ``VariantInfo`` 含非沉默 token（教师主口径）
    - B：A 的编码过滤 + (LikelyLoF 或 Hotspot)（driver/消融口径）
    """
    raw = pd.read_csv(path, usecols=lambda column: column in USECOLS, dtype=str, low_memory=False)
    default = raw[raw["IsDefaultEntryForModel"] == "Yes"].copy()
    default["ModelID"] = default["ModelID"].astype(str).str.strip()
    default["HugoSymbol"] = default["HugoSymbol"].astype("string").str.strip()
    protein_coding = default["VepBiotype"].isin(PROTEIN_BIOTYPES)
    non_silent = default["VariantInfo"].map(_has_non_silent)
    mask_A = protein_coding & non_silent
    mask_B = protein_coding & (_truthy(default["LikelyLoF"]) | _truthy(default["Hotspot"]))

    frame_A = default[mask_A].copy()
    frame_B = default[mask_B].copy()
    pairs_A = frame_A[["ModelID", "HugoSymbol"]].dropna().drop_duplicates()
    pairs_B = frame_B[["ModelID", "HugoSymbol"]].dropna().drop_duplicates()
    model_ids = sorted(default["ModelID"].unique().tolist())
    genes_A = sorted(pairs_A["HugoSymbol"].unique().tolist())
    genes_B = sorted(pairs_B["HugoSymbol"].unique().tolist())

    matrix_A = _matrix(pairs_A, model_ids, genes_A)
    matrix_B = _matrix(pairs_B, model_ids, genes_B)
    counts_A = frame_A.groupby("ModelID").size().reindex(model_ids, fill_value=0)
    counts_B = frame_B.groupby("ModelID").size().reindex(model_ids, fill_value=0)
    mut_burden_A = counts_A.astype(int)
    mut_burden_B = counts_B.astype(int)

    long_columns = [
        column
        for column in [
            "ModelID",
            "HugoSymbol",
            "VepBiotype",
            "VariantInfo",
            "LikelyLoF",
            "Hotspot",
            "AF",
            "DP",
            "Chrom",
            "Pos",
            "Ref",
            "Alt",
            "VariantType",
        ]
        if column in frame_A.columns
    ]
    long_A = frame_A[long_columns].sort_values(["ModelID", "HugoSymbol"]).reset_index(drop=True)
    info = {
        "raw_rows": int(len(raw)),
        "default_rows": int(len(default)),
        "n_models": len(model_ids),
        "rows_A": int(len(frame_A)),
        "genes_A": len(genes_A),
        "pairs_A": int(len(pairs_A)),
        "rows_B": int(len(frame_B)),
        "genes_B": len(genes_B),
        "pairs_B": int(len(pairs_B)),
        "median_mutations_A": float(counts_A.median()),
        "min_mutations_A": int(counts_A.min()),
        "mut_burden_type": "integer mutated-gene count on officially pre-filtered somatic calls; NOT TMB",
        "filter_A": "protein_coding + VariantInfo non-silent",
        "filter_B": "protein_coding + (LikelyLoF or Hotspot)",
    }
    return MutationData(
        matrix_A=matrix_A,
        genes_A=genes_A,
        matrix_B=matrix_B,
        genes_B=genes_B,
        model_ids=model_ids,
        long_A=long_A,
        mut_burden_A=mut_burden_A,
        mut_burden_B=mut_burden_B,
        info=info,
    )


def write_mutations(data: MutationData, canonical_dir: str | Path) -> dict[str, Path]:
    canonical = Path(canonical_dir)
    outputs = {
        "A": io.save_packed_matrix(
            canonical / "mutations_A.npz", data.matrix_A, data.model_ids, data.genes_A
        ),
        "B": io.save_packed_matrix(
            canonical / "mutations_B.npz", data.matrix_B, data.model_ids, data.genes_B
        ),
        "long": io.write_parquet(data.long_A, canonical / "mutation_long.parquet"),
    }
    io.write_parquet(
        pd.DataFrame(
            {
                "cell_line_uid": data.mut_burden_A.index,
                "mut_burden_A": data.mut_burden_A.to_numpy(),
                "mut_burden_B": data.mut_burden_B.to_numpy(),
            }
        ),
        canonical / "mut_burden.parquet",
    )
    return outputs


__all__ = ["MutationData", "load_mutations", "write_mutations"]

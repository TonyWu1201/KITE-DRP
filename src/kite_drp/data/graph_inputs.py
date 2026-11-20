"""教师图输入（processed 自包含）：PPI / DTI 边表 + 基因/化合物节点表。

`data/processed/graph_inputs.npz` 让 clone 后的仓库在没有 ``data/interim/canonical``
的情况下也能构建 T1 异构图。只保留进入 ``U_teach_ge`` 的基因，边数可控。

键约定::

    genes               (G,)   HGNC symbol，节点索引 = 行号
    ppi_edges           (2,E)  int32，基因索引
    ppi_score           (E,)   float32，STRING combined_score / 1000
    dti_edges           (2,D)  int32，[compound_idx, gene_idx]
    dti_score           (D,)   float32
    dti_source          (D,)   str
    compounds           (C,)   compound_uid，与 drug_features.npz 行对齐
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kite_drp.data import io


def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"缺少 canonical 先验产物：{path}")
    return path


def build_graph_inputs(
    canonical_dir: str | Path,
    processed_dir: str | Path,
) -> dict[str, Any]:
    canonical = Path(canonical_dir)
    processed = Path(processed_dir)
    ppi = io.read_parquet(_require(canonical / "ppi.parquet"))
    dti_path = processed / "dti.parquet"
    if not dti_path.is_file():
        dti_path = canonical / "dti.parquet"
    dti = io.read_parquet(_require(dti_path))
    universes = io.read_json(canonical / "gene_universes.json")

    teach_ge = set(universes.get("teach_ge", []))
    if not teach_ge:
        raise ValueError("gene_universes.json 缺少 teach_ge，无法限定图基因空间")

    ppi = ppi[ppi["gene_a"].isin(teach_ge) & ppi["gene_b"].isin(teach_ge)].copy()
    dti = dti[dti["hgnc_symbol"].isin(teach_ge)].copy()

    genes = sorted(
        set(ppi["gene_a"]) | set(ppi["gene_b"]) | set(dti["hgnc_symbol"])
    )
    gene_index = {gene: index for index, gene in enumerate(genes)}

    # 化合物顺序以 drug_features.npz 为准（processed 已存在时），否则回退 canonical。
    drug_features_path = processed / "drug_features.npz"
    if not drug_features_path.is_file():
        drug_features_path = canonical / "drug_features.npz"
    drug_arrays = io.read_npz(_require(drug_features_path))
    compounds = [str(value) for value in drug_arrays["compound_uid"]]
    compound_index = {uid: index for index, uid in enumerate(compounds)}

    ppi_edges = np.asarray(
        [
            [gene_index[a], gene_index[b]]
            for a, b in zip(ppi["gene_a"], ppi["gene_b"], strict=False)
        ],
        dtype=np.int32,
    ).T
    ppi_score = (
        ppi["combined_score"].to_numpy(dtype=np.float32) / 1000.0
        if "combined_score" in ppi.columns
        else np.ones(ppi.shape[0], dtype=np.float32)
    )

    dti = dti[dti["compound_uid"].isin(compound_index)].copy()
    dti_edges = np.asarray(
        [
            [compound_index[uid], gene_index[gene]]
            for uid, gene in zip(dti["compound_uid"], dti["hgnc_symbol"], strict=False)
        ],
        dtype=np.int32,
    ).T
    dti_score = (
        pd.to_numeric(dti.get("score"), errors="coerce").fillna(1.0).to_numpy(dtype=np.float32)
        if "score" in dti.columns
        else np.ones(dti.shape[0], dtype=np.float32)
    )
    dti_source = (
        dti["source"].fillna("unknown").astype(str).to_numpy(dtype=np.str_)
        if "source" in dti.columns
        else np.full(dti.shape[0], "unknown", dtype=np.str_)
    )

    out = processed / "graph_inputs.npz"
    io.write_npz(
        out,
        {
            "genes": np.asarray(genes, dtype=np.str_),
            "ppi_edges": ppi_edges,
            "ppi_score": ppi_score,
            "dti_edges": dti_edges,
            "dti_score": dti_score,
            "dti_source": dti_source,
            "compounds": np.asarray(compounds, dtype=np.str_),
        },
    )
    return {
        "artifact": str(out),
        "genes": len(genes),
        "ppi_edges": int(ppi_edges.shape[1]),
        "dti_edges": int(dti_edges.shape[1]),
        "compounds_with_dti": int(len(set(dti["compound_uid"]))),
        "compounds": len(compounds),
    }


def load_graph_inputs(processed_dir: str | Path) -> dict[str, np.ndarray]:
    path = Path(processed_dir) / "graph_inputs.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"缺少 {path}；请运行 `kite-drp data build --stage graph` 或完整 data pipeline"
        )
    return io.read_npz(path)


def validate_graph_inputs(arrays: dict[str, np.ndarray]) -> list[str]:
    """返回结构问题列表（空 = 通过）。供 `data validate` / `data preflight` 复用。"""
    problems: list[str] = []
    required = {"genes", "ppi_edges", "ppi_score", "dti_edges", "dti_score", "compounds"}
    missing = required - set(arrays)
    if missing:
        return [f"缺少键：{sorted(missing)}"]
    genes = arrays["genes"]
    ppi_edges = arrays["ppi_edges"]
    dti_edges = arrays["dti_edges"]
    if ppi_edges.ndim != 2 or ppi_edges.shape[0] != 2:
        problems.append(f"ppi_edges 形状应为 (2,E)，实际 {ppi_edges.shape}")
    elif ppi_edges.size and (ppi_edges.min() < 0 or ppi_edges.max() >= len(genes)):
        problems.append("ppi_edges 存在越界基因索引")
    if dti_edges.ndim != 2 or dti_edges.shape[0] != 2:
        problems.append(f"dti_edges 形状应为 (2,D)，实际 {dti_edges.shape}")
    elif dti_edges.size:
        if dti_edges[0].min() < 0 or dti_edges[0].max() >= len(arrays["compounds"]):
            problems.append("dti_edges 存在越界化合物索引")
        if dti_edges[1].min() < 0 or dti_edges[1].max() >= len(genes):
            problems.append("dti_edges 存在越界基因索引")
    if len(arrays["ppi_score"]) != ppi_edges.shape[1]:
        problems.append("ppi_score 长度与 ppi_edges 不一致")
    if len(arrays["dti_score"]) != dti_edges.shape[1]:
        problems.append("dti_score 长度与 dti_edges 不一致")
    return problems


__all__ = ["build_graph_inputs", "load_graph_inputs", "validate_graph_inputs"]

"""教师训练特征装配：通路分数、药物特征、fold-safe 相似性邻居。

输入只读 `data/processed` + `data/splits` + `fit_stats`（ADR-0005），不碰 canonical。
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kite_drp.data import io
from kite_drp.data.dataset import FitStats
from kite_drp.data.similarity import (
    SimilarityConfig,
    fit_cell_similarity,
    save_similarity,
)


@dataclass
class CellFeatureSet:
    cell_ids: list[str]
    ge: np.ndarray  # (n, P) pathway scores (fold z)
    mu: np.ndarray  # (n, P) pathway mutation fraction
    cnv: np.ndarray  # (n, P) pathway CNV score (0 where unobserved)
    cnv_mask: np.ndarray  # (n,) bool
    ge_mask: np.ndarray  # (n,) bool
    mu_mask: np.ndarray  # (n,) bool
    index: dict[str, int]


@dataclass
class DrugFeatureSet:
    compound_uid: list[str]
    morgan: np.ndarray  # (n, 2048) float32
    target_indices: list[np.ndarray]
    target_dim: int
    graph_atom_ptr: np.ndarray | None
    graph_edge_ptr: np.ndarray | None
    graph_node_feat: np.ndarray | None
    graph_edge_index: np.ndarray | None
    graph_edge_feat: np.ndarray | None
    graph_has_graph: np.ndarray | None
    index: dict[str, int]


def _align_weights(
    weights: np.ndarray, weight_genes: list[str], target_genes: list[str]
) -> np.ndarray:
    """把通路权重矩阵的列重排到目标基因顺序；缺失基因权重记 0。"""
    if weight_genes == target_genes:
        return weights.astype(np.float32)
    lookup = {gene: position for position, gene in enumerate(weight_genes)}
    columns = [lookup.get(gene) for gene in target_genes]
    aligned = np.zeros((weights.shape[0], len(target_genes)), dtype=np.float32)
    for target, source in enumerate(columns):
        if source is not None:
            aligned[:, target] = weights[:, source]
    return aligned


def build_cell_features(
    processed_dir: str | Path,
    *,
    split_id: str,
    fold: int,
    fit_stats_root: str | Path | None = None,
) -> CellFeatureSet:
    processed = Path(processed_dir)
    teacher = io.read_npz(processed / "teacher_inputs.npz")
    index = io.read_json(processed / "teacher_index.json")
    pathway_arrays = io.read_npz(processed / "student_inputs" / "pathway_weights.npz")

    cell_ids = [str(value) for value in teacher["cell_line_uid"]]
    ge_genes = [str(value) for value in index["ge_genes"]]
    cnv_genes = [str(value) for value in index["cnv_genes"]]
    if "mu_B_genes" not in index:
        raise KeyError(
            "teacher_index.json 缺少 mu_B_genes；请重跑 `kite-drp data build --stage teacher`"
        )
    mu_genes = [str(value) for value in index["mu_B_genes"]]
    weight_genes = [str(value) for value in pathway_arrays["genes"]]
    weights = pathway_arrays["weights"].astype(np.float32)  # (P, G)

    root = Path(fit_stats_root or processed / "fit_stats")
    fit = FitStats.load(root / split_id / f"fold{fold}")

    # GE：fold 内 z → 通路平均 z
    ge_z = fit.standardize(teacher["ge"], ge_genes, kind="expr")
    w_ge = _align_weights(weights, weight_genes, ge_genes)
    ge_scores = ge_z @ w_ge.T

    # CNV：fold 内 z（含裁剪）→ 通路平均 z；缺失细胞整行 0
    cnv_mask = teacher["cnv_observed_mask"].astype(bool)
    cnv_z = fit.standardize(teacher["cnv"], cnv_genes, kind="cnv")
    cnv_z = cnv_z * cnv_mask[:, None]
    w_cnv = _align_weights(weights, weight_genes, cnv_genes)
    cnv_scores = cnv_z @ w_cnv.T

    # MU：pathway 内 driver 突变基因占比
    mu_b = teacher["mu_B"].astype(np.float32)
    mu_lookup = {gene: position for position, gene in enumerate(mu_genes)}
    pairs = [
        (weight_position, mu_lookup[gene])
        for weight_position, gene in enumerate(weight_genes)
        if gene in mu_lookup
    ]
    weight_positions = [pair[0] for pair in pairs]
    mu_positions = [pair[1] for pair in pairs]
    membership = (weights[:, weight_positions] > 0).astype(np.float32)
    mu_sub = mu_b[:, mu_positions]
    mu_counts = np.maximum(membership.sum(axis=1), 1.0)
    mu_scores = (mu_sub @ membership.T) / mu_counts

    return CellFeatureSet(
        cell_ids=cell_ids,
        ge=ge_scores.astype(np.float32),
        mu=mu_scores.astype(np.float32),
        cnv=cnv_scores.astype(np.float32),
        cnv_mask=cnv_mask,
        ge_mask=teacher["ge_observed_mask"].astype(bool),
        mu_mask=teacher["mu_observed_mask"].astype(bool),
        index={cell: position for position, cell in enumerate(cell_ids)},
    )


def cell_scores(features: CellFeatureSet) -> dict[str, np.ndarray]:
    return {"GE": features.ge, "MU": features.mu, "CNV": features.cnv}


def build_drug_features(processed_dir: str | Path) -> DrugFeatureSet:
    processed = Path(processed_dir)
    arrays = io.read_npz(processed / "drug_features.npz")
    compound_uid = [str(value) for value in arrays["compound_uid"]]
    targets = arrays["targets"].astype(bool)
    target_indices = [np.flatnonzero(row).astype(np.int64) for row in targets]
    return DrugFeatureSet(
        compound_uid=compound_uid,
        morgan=arrays["morgan"].astype(np.float32),
        target_indices=target_indices,
        target_dim=int(targets.shape[1]),
        graph_atom_ptr=arrays.get("graph_atom_ptr"),
        graph_edge_ptr=arrays.get("graph_edge_ptr"),
        graph_node_feat=arrays.get("graph_node_feat"),
        graph_edge_index=arrays.get("graph_edge_index"),
        graph_edge_feat=arrays.get("graph_edge_feat"),
        graph_has_graph=(
            arrays["graph_has_graph"].astype(bool) if "graph_has_graph" in arrays else None
        ),
        index={uid: position for position, uid in enumerate(compound_uid)},
    )


def neighbor_scores(scores: np.ndarray, neighbors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``sum_k w_ik * scores[neighbors_ik]``。"""
    return np.einsum("nk,nkp->np", weights, scores[neighbors]).astype(np.float32)


def resolve_similarity_config(
    config: dict[str, Any], active_modalities: list[str] | tuple[str, ...]
) -> SimilarityConfig:
    """相似性只用当前启用的组学模态；CNV 缺失时自动退化为 GE/MU 可用项。"""
    payload = dict(config.get("similarity", {}))
    requested = list(payload.get("modalities", ["GE", "MU"]))
    names = [name for name in requested if name in set(active_modalities)]
    if not names:
        names = [name for name in active_modalities if name != "CNV"]
    if not names:
        raise ValueError(f"相似性没有可用模态：active={active_modalities}")
    payload["modalities"] = names
    return SimilarityConfig.from_dict(payload)


def resolve_neighbor_source(
    config: dict[str, Any], active_modalities: list[str] | tuple[str, ...]
) -> str:
    """邻居上下文取自哪个模态；``auto`` 用第一个启用模态。"""
    value = str(config.get("neighbor_source", "auto"))
    if value == "auto":
        for name in ("GE", "MU", "CNV"):
            if name in active_modalities:
                return name
    if value not in active_modalities:
        raise ValueError(f"neighbor_source={value} 不在启用模态 {active_modalities} 中")
    return value


def build_fold_similarity(
    features: CellFeatureSet,
    *,
    train_cells: list[str],
    config: SimilarityConfig,
    processed_dir: str | Path,
    split_id: str,
    fold: int,
    cache_dir: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """返回全 716 行的 ``(neighbors, weights, is_train, meta)``。"""
    train_idx = np.asarray(
        sorted(features.index[cell] for cell in train_cells if cell in features.index),
        dtype=np.int64,
    )
    if train_idx.size == 0:
        raise ValueError(f"fold {fold} 的 train 细胞与教师池无交集")
    if train_idx.size <= config.top_n:
        raise ValueError(f"train 细胞数 {train_idx.size} 不足以取 top_n={config.top_n} 个邻居")
    is_train = np.zeros(len(features.cell_ids), dtype=bool)
    is_train[train_idx] = True
    query_idx = np.flatnonzero(~is_train)

    scores = cell_scores(features)
    names = [name for name in config.modality_names if name in scores]
    if "CNV" in names and not features.cnv_mask.all():
        raise ValueError(
            "相似性包含 CNV，但教师池存在 CNV 缺失细胞；请用 pool=M-509 或改 modalities=[GE, MU]"
        )
    train_scores = {name: scores[name][train_idx] for name in names}
    query_scores = {name: scores[name][query_idx] for name in names} if query_idx.size else None
    fit = fit_cell_similarity(train_scores, query_scores, config=config)

    n_rows = len(features.cell_ids)
    top_n = config.top_n
    neighbors = np.zeros((n_rows, top_n), dtype=np.int32)
    weights = np.zeros((n_rows, top_n), dtype=np.float32)
    neighbors[train_idx] = train_idx[fit["train_neighbors"]]
    weights[train_idx] = fit["train_weights"]
    if query_idx.size:
        neighbors[query_idx] = train_idx[fit["query_neighbors"]]
        weights[query_idx] = fit["query_weights"]

    meta = {
        "split_id": split_id,
        "fold": int(fold),
        "n_train": int(train_idx.size),
        "n_query": int(query_idx.size),
        "n_cells": n_rows,
        "sigma": fit["sigma"],
        "config": fit["config"],
        "diagnostics": fit["diagnostics"],
        "fold_safe": True,
        "note": "fit 只在 train 细胞；query 只连到 train 锚点",
    }
    if cache_dir is not None:
        target_dir = Path(cache_dir) / split_id / f"fold{fold}"
        target_dir.mkdir(parents=True, exist_ok=True)
        # 缓存非必需；并发 worker 同时写同一 fold 时 Windows 可能拒绝 replace。
        with contextlib.suppress(OSError):
            save_similarity(
                target_dir / "cell_similarity.npz",
                cell_ids=features.cell_ids,
                is_train=is_train,
                neighbors=neighbors,
                weights=weights,
                meta=meta,
            )
    return neighbors, weights, is_train, meta


__all__ = [
    "CellFeatureSet",
    "DrugFeatureSet",
    "build_cell_features",
    "build_drug_features",
    "build_fold_similarity",
    "cell_scores",
    "neighbor_scores",
    "resolve_neighbor_source",
    "resolve_similarity_config",
]

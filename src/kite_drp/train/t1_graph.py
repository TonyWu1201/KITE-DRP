"""T1 异构图构建：基因空间裁剪、PPI/DTI/cell-gene 边、随机对照。

只读 ``data/processed`` + ``data/splits`` + ``fit_stats``（ADR-0005），不碰 canonical。
图按 ``(split_id, fold)`` 缓存到 ``data/interim/teacher/t1_graph``；缓存键包含
基因空间、边阈值、随机重连/删边配置，避免不同消融互相覆盖。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from kite_drp.data import io
from kite_drp.data.graph_inputs import load_graph_inputs
from kite_drp.train.features import (
    CellFeatureSet,
    DrugFeatureSet,
    build_cell_features,
    build_drug_features,
)

DEFAULT_GRAPH_CONFIG: dict[str, Any] = {
    "version": 2,
    "gene_space": {
        "hvg_top": 1000,
        "mutation_top": 500,
        "include_targets": True,
    },
    "ppi_min_score": 0.7,
    "cell_gene_top_q": 0.9,
    "cell_gene_mutation": True,
    "max_targets_per_drug": 64,
    "rewire": "none",  # none | ppi | dti | cg | ppi+dti
    "edge_drop": "none",  # none | ppi | dti | cg
    "seed": 20261120,
}


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    payload = json.loads(json.dumps(DEFAULT_GRAPH_CONFIG))
    for key, value in (config or {}).items():
        if key not in payload:
            # hidden/layers/dropout 等模型超参不改变图结构，不进入缓存键。
            continue
        if isinstance(value, dict) and isinstance(payload.get(key), dict):
            payload[key].update(value)
        else:
            payload[key] = value
    return payload


def _hash_payload(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _train_rows(features: CellFeatureSet, train_cells: list[str]) -> np.ndarray:
    rows = [features.index[cell] for cell in train_cells if cell in features.index]
    if not rows:
        raise ValueError("train cells 与教师池无交集")
    return np.asarray(sorted(rows), dtype=np.int64)


def _select_genes(
    features: CellFeatureSet,
    drugs: DrugFeatureSet,
    hvg: list[str],
    train_rows: np.ndarray,
    graph_genes: list[str],
    config: dict[str, Any],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    gene_space = config["gene_space"]
    hvg_top = int(gene_space.get("hvg_top", 1000))
    mutation_top = int(gene_space.get("mutation_top", 500))

    target_mask = np.zeros(len(drugs.target_genes), dtype=bool)
    for indices in drugs.target_indices:
        target_mask[np.asarray(indices, dtype=np.int64)] = True
    target_names = {drugs.target_genes[i] for i in np.flatnonzero(target_mask)}
    if not gene_space.get("include_targets", True):
        target_names = set()

    mutation_freq = features.gene_mu[train_rows].mean(axis=0)
    top_mut = {features.gene_ids[i] for i in np.argsort(-mutation_freq)[: max(mutation_top, 0)]}
    hvg_names = set(hvg[: max(hvg_top, 0)])

    candidates = (target_names | top_mut | hvg_names) & set(features.gene_ids)
    candidates &= set(graph_genes)
    gene_ids = sorted(candidates)
    if not gene_ids:
        raise ValueError("T1 基因空间为空；检查 targets/HVG/mutation 配置")

    gene_full_idx = np.asarray([features.gene_ids.index(gene) for gene in gene_ids], dtype=np.int64)
    graph_gene_index = {gene: i for i, gene in enumerate(graph_genes)}
    gene_graph_idx = np.asarray([graph_gene_index[gene] for gene in gene_ids], dtype=np.int64)
    return gene_ids, gene_full_idx, gene_graph_idx


def _ppi_edges(
    graph: dict[str, np.ndarray],
    local_of_graph: np.ndarray,
    *,
    min_score: float,
) -> tuple[np.ndarray, np.ndarray]:
    edges = graph["ppi_edges"]
    score = graph["ppi_score"]
    keep = score >= float(min_score)
    edges = edges[:, keep]
    score = score[keep]
    local = local_of_graph[edges]
    keep = (local[0] >= 0) & (local[1] >= 0)
    local = local[:, keep]
    score = score[keep]
    # 无向 PPI：双向边，方便 HeteroConv 传播。
    edge_index = np.concatenate([local, local[::-1]], axis=1)
    edge_attr = np.concatenate([score, score])[:, None].astype(np.float32)
    return edge_index.astype(np.int64), edge_attr


def _dti_edges(
    graph: dict[str, np.ndarray],
    local_of_graph: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    edges = graph["dti_edges"]  # (2, D) = [compound_idx, graph_gene_idx]
    score = graph["dti_score"]
    gene_local = local_of_graph[edges[1]]
    keep = gene_local >= 0
    # HeteroConv 边方向：source gene -> target drug。
    edge_index = np.vstack([gene_local[keep], edges[0][keep]]).astype(np.int64)
    edge_attr = score[keep][:, None].astype(np.float32)
    return edge_index, edge_attr


def _cell_gene_edges(
    features: CellFeatureSet,
    gene_full_idx: np.ndarray,
    *,
    top_q: float,
    include_mutation: bool,
) -> tuple[np.ndarray, np.ndarray]:
    expr = np.clip(features.gene_ge[:, gene_full_idx], -3.0, 3.0)
    mut = features.gene_mu[:, gene_full_idx] > 0
    cnv = np.clip(features.gene_cnv[:, gene_full_idx], -3.0, 3.0)
    cnv_obs = features.cnv_mask[:, None].astype(np.float32)

    threshold = np.quantile(expr, float(top_q), axis=1, keepdims=True)
    selected = expr >= threshold
    if include_mutation:
        selected |= mut
    cell_idx, gene_idx = np.nonzero(selected)
    edge_index = np.vstack([gene_idx, cell_idx]).astype(np.int64)
    edge_attr = np.stack(
        [
            expr[cell_idx, gene_idx],
            mut[cell_idx, gene_idx].astype(np.float32),
            cnv[cell_idx, gene_idx],
            np.broadcast_to(cnv_obs, expr.shape)[cell_idx, 0],
        ],
        axis=1,
    ).astype(np.float32)
    return edge_index, edge_attr


def _gene_stats(
    features: CellFeatureSet,
    train_rows: np.ndarray,
    gene_full_idx: np.ndarray,
    ppi_edge_index: np.ndarray,
    n_genes: int,
    raw_ge: np.ndarray,
) -> np.ndarray:
    degree = np.bincount(ppi_edge_index[0], minlength=n_genes).astype(np.float32)
    mutation_freq = features.gene_mu[train_rows][:, gene_full_idx].mean(axis=0)
    expr = raw_ge[train_rows][:, gene_full_idx]
    mean_expr = expr.mean(axis=0)
    std_expr = expr.std(axis=0)
    stats = np.stack([np.log1p(degree), mutation_freq, mean_expr, std_expr], axis=1).astype(
        np.float32
    )
    mean = stats.mean(axis=0, keepdims=True)
    std = stats.std(axis=0, keepdims=True)
    stats = (stats - mean) / np.where(std == 0, 1.0, std)
    return np.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _drug_target_table(
    drugs: DrugFeatureSet,
    local_of_full: np.ndarray,
    max_targets: int,
) -> tuple[np.ndarray, np.ndarray]:
    local_targets: list[np.ndarray] = []
    for indices in drugs.target_indices:
        local = local_of_full[np.asarray(indices, dtype=np.int64)]
        local_targets.append(np.unique(local[local >= 0]))
    max_size = max((item.size for item in local_targets), default=0)
    width = min(max_size, int(max_targets)) if int(max_targets) > 0 else max_size
    ids = np.zeros((len(drugs.compound_uid), width), dtype=np.int64)
    mask = np.zeros((len(drugs.compound_uid), width), dtype=bool)
    for row, local in enumerate(local_targets):
        if local.size > width:
            local = local[:width]
        ids[row, : local.size] = local
        mask[row, : local.size] = True
    return ids, mask


def _rewire_undirected(edge_index: np.ndarray, *, seed: int, swaps: int) -> np.ndarray:
    """度分布保持的无向图双边交换；输入可含双向边，先折叠为无向边。"""
    edges = {
        (int(min(a, b)), int(max(a, b)))
        for a, b in zip(edge_index[0], edge_index[1], strict=False)
        if int(a) != int(b)
    }
    if len(edges) < 2:
        return edge_index
    rng = np.random.default_rng(seed)
    edge_list = list(edges)
    max_tries = max(int(swaps) * 10, 1000)
    done = 0
    for _ in range(max_tries):
        if done >= swaps:
            break
        i, j = rng.integers(0, len(edge_list), size=2)
        if i == j:
            continue
        a, b = edge_list[i]
        c, d = edge_list[j]
        if len({a, b, c, d}) < 4:
            continue
        first = (min(a, d), max(a, d))
        second = (min(c, b), max(c, b))
        if first in edges or second in edges:
            continue
        edges.remove((a, b))
        edges.remove((c, d))
        edges.add(first)
        edges.add(second)
        edge_list[i] = first
        edge_list[j] = second
        done += 1
    result = np.asarray(sorted(edges), dtype=np.int64).T
    return np.concatenate([result, result[::-1]], axis=1)


def _rewire_bipartite(edge_index: np.ndarray, *, seed: int, swaps: int) -> np.ndarray:
    """保持 source/target 度分布的二分图双边交换。"""
    pairs = {(int(a), int(b)) for a, b in zip(edge_index[0], edge_index[1], strict=False)}
    if len(pairs) < 2:
        return edge_index
    rng = np.random.default_rng(seed)
    pair_list = list(pairs)
    max_tries = max(int(swaps) * 10, 1000)
    done = 0
    for _ in range(max_tries):
        if done >= swaps:
            break
        i, j = rng.integers(0, len(pair_list), size=2)
        if i == j:
            continue
        s1, d1 = pair_list[i]
        s2, d2 = pair_list[j]
        if s1 == s2 or d1 == d2:
            continue
        first = (s1, d2)
        second = (s2, d1)
        if first in pairs or second in pairs:
            continue
        pairs.remove((s1, d1))
        pairs.remove((s2, d2))
        pairs.add(first)
        pairs.add(second)
        pair_list[i] = first
        pair_list[j] = second
        done += 1
    return np.asarray(sorted(pairs), dtype=np.int64).T


def _apply_controls(
    graph: dict[str, np.ndarray],
    config: dict[str, Any],
    *,
    n_genes: int,
    n_drugs: int,
    n_cells: int,
) -> dict[str, np.ndarray]:
    rewire = str(config.get("rewire", "none"))
    edge_drop = str(config.get("edge_drop", "none"))
    seed = int(config.get("seed", 20261120))
    if rewire in {"ppi", "ppi+dti"}:
        swaps = max(int(graph["ppi_edge_index"].shape[1] // 2), 1)
        graph["ppi_edge_index"] = _rewire_undirected(
            graph["ppi_edge_index"], seed=seed, swaps=swaps
        )
    if rewire in {"dti", "ppi+dti"}:
        graph["dti_edge_index"] = _rewire_bipartite(
            graph["dti_edge_index"],
            seed=seed + 1,
            swaps=graph["dti_edge_index"].shape[1],
        )
    if rewire == "cg":
        graph["cg_edge_index"] = _rewire_bipartite(
            graph["cg_edge_index"],
            seed=seed + 2,
            swaps=graph["cg_edge_index"].shape[1],
        )
    for name in ("ppi", "dti", "cg"):
        if edge_drop == name:
            key = f"{name}_edge_index"
            graph[key] = np.zeros((2, 0), dtype=np.int64)
            attr_dim = graph[f"{name}_edge_attr"].shape[1]
            graph[f"{name}_edge_attr"] = np.zeros((0, attr_dim), dtype=np.float32)
    return graph


def build_t1_graph(
    processed_dir: str | Path,
    *,
    split_id: str,
    fold: int,
    config: dict[str, Any] | None = None,
    split_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """构建/加载单个 fold 的 T1 图；返回 numpy 数组 + spec。"""
    processed = Path(processed_dir)
    splits = Path(split_dir or processed.parent / "splits")
    cfg = _resolve_config(config)
    cache_root = Path(cache_dir or processed.parent / "interim" / "teacher" / "t1_graph")
    cache_path = cache_root / split_id / f"fold{fold}"
    cache_file = cache_path / "graph.npz"
    spec_file = cache_path / "graph_spec.json"

    features = build_cell_features(processed, split_id=split_id, fold=fold)
    drugs = build_drug_features(processed)
    graph = load_graph_inputs(processed)
    teacher = io.read_npz(processed / "teacher_inputs.npz")
    split = io.read_json(splits / f"{split_id}.json")
    fold_def = next(item for item in split["folds"] if int(item["fold"]) == int(fold))
    train_rows = _train_rows(features, list(fold_def["train"]))
    hvg = list(io.read_json(processed / "fit_stats" / split_id / f"fold{fold}" / "hvg.json")["hvg"])

    gene_ids, gene_full_idx, gene_graph_idx = _select_genes(
        features,
        drugs,
        hvg,
        train_rows,
        [str(gene) for gene in graph["genes"]],
        cfg,
    )
    n_genes = len(gene_ids)
    n_drugs = len(drugs.compound_uid)
    n_cells = len(features.cell_ids)

    local_of_graph = np.full(len(graph["genes"]), -1, dtype=np.int64)
    local_of_graph[gene_graph_idx] = np.arange(n_genes)
    local_of_full = np.full(len(features.gene_ids), -1, dtype=np.int64)
    local_of_full[gene_full_idx] = np.arange(n_genes)

    ppi_edge_index, ppi_edge_attr = _ppi_edges(
        graph, local_of_graph, min_score=float(cfg["ppi_min_score"])
    )
    dti_edge_index, dti_edge_attr = _dti_edges(graph, local_of_graph)
    cg_edge_index, cg_edge_attr = _cell_gene_edges(
        features,
        gene_full_idx,
        top_q=float(cfg["cell_gene_top_q"]),
        include_mutation=bool(cfg["cell_gene_mutation"]),
    )
    gene_stats = _gene_stats(
        features,
        train_rows,
        gene_full_idx,
        ppi_edge_index,
        n_genes,
        teacher["ge"],
    )
    cell_features = np.concatenate(
        [
            features.ge,
            features.mu,
            features.cnv,
            features.ge_mask[:, None].astype(np.float32),
            features.mu_mask[:, None].astype(np.float32),
            features.cnv_mask[:, None].astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    drug_target_ids, drug_target_mask = _drug_target_table(
        drugs, local_of_full, int(cfg["max_targets_per_drug"])
    )
    drug_target_ids_full, drug_target_mask_full = _drug_target_table(drugs, local_of_full, 0)

    arrays: dict[str, np.ndarray] = {
        "gene_ids": np.asarray(gene_ids, dtype=np.str_),
        "gene_full_idx": gene_full_idx,
        "cell_ids": np.asarray(features.cell_ids, dtype=np.str_),
        "drug_uids": np.asarray(drugs.compound_uid, dtype=np.str_),
        "ppi_edge_index": ppi_edge_index,
        "ppi_edge_attr": ppi_edge_attr,
        "dti_edge_index": dti_edge_index,
        "dti_edge_attr": dti_edge_attr,
        "cg_edge_index": cg_edge_index,
        "cg_edge_attr": cg_edge_attr,
        "gene_stats": gene_stats,
        "cell_features": cell_features,
        "drug_morgan": drugs.morgan.astype(np.float32),
        "drug_target_ids": drug_target_ids,
        "drug_target_mask": drug_target_mask,
        "drug_target_ids_full": drug_target_ids_full,
        "drug_target_mask_full": drug_target_mask_full,
    }
    arrays = _apply_controls(
        arrays,
        cfg,
        n_genes=n_genes,
        n_drugs=n_drugs,
        n_cells=n_cells,
    )

    spec = {
        "split_id": split_id,
        "fold": int(fold),
        "config": cfg,
        "n_genes": n_genes,
        "n_cells": n_cells,
        "n_drugs": n_drugs,
        "n_ppi_edges": int(arrays["ppi_edge_index"].shape[1]),
        "n_dti_edges": int(arrays["dti_edge_index"].shape[1]),
        "n_cg_edges": int(arrays["cg_edge_index"].shape[1]),
        "gene_hash": hashlib.sha256("\n".join(gene_ids).encode()).hexdigest()[:16],
        "cache": str(cache_file),
    }
    spec["spec_hash"] = _hash_payload(spec)
    if cache_file.is_file() and spec_file.is_file() and not force:
        cached_spec = io.read_json(spec_file)
        if cached_spec.get("spec_hash") == spec["spec_hash"]:
            cached = io.read_npz(cache_file)
            return {"arrays": cached, "spec": cached_spec, "cache_path": str(cache_path)}
    cache_path.mkdir(parents=True, exist_ok=True)
    io.write_npz(cache_file, arrays)
    io.write_json(spec_file, spec)
    return {"arrays": arrays, "spec": spec, "cache_path": str(cache_path)}


def graph_tensors(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """把 numpy 图数组转成 torch 张量（模型侧 set_graph 用）。"""
    import torch

    mapping = {
        "gene_full_idx": torch.long,
        "ppi_edge_index": torch.long,
        "ppi_edge_attr": torch.float32,
        "dti_edge_index": torch.long,
        "dti_edge_attr": torch.float32,
        "cg_edge_index": torch.long,
        "cg_edge_attr": torch.float32,
        "gene_stats": torch.float32,
        "cell_features": torch.float32,
        "drug_morgan": torch.float32,
        "drug_target_ids": torch.long,
        "drug_target_mask": torch.bool,
        "drug_target_ids_full": torch.long,
        "drug_target_mask_full": torch.bool,
    }
    tensors = {
        key: torch.as_tensor(np.asarray(arrays[key]), dtype=dtype)
        for key, dtype in mapping.items()
        if key in arrays
    }
    tensors["gene_ids"] = [str(value) for value in arrays["gene_ids"]]
    tensors["cell_ids"] = [str(value) for value in arrays["cell_ids"]]
    tensors["drug_uids"] = [str(value) for value in arrays["drug_uids"]]
    return tensors


__all__ = [
    "DEFAULT_GRAPH_CONFIG",
    "build_t1_graph",
    "graph_tensors",
]

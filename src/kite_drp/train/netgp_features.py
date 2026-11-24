"""NetGP 式“间接靶点”传播特征（无标签、无细胞依赖）。

与 ``ppi_features.py`` 的区别：

- ``ppi_features`` 的 RWR 从靶点出发后**保留根**，实测根权重约 0.50，
  与 P1 target branch 读取的靶点表达高度相关（GE 0.963 / MU 0.991 / CNV 0.984）；
- 本模块显式**排除直接靶点**，只保留传播得到的“间接靶点”，再按 top-k 截断，
  让 PPI 提供 target branch 看不到的 gene-level 基。

流程（对齐 NetGP / DGDRP 的 L3 用法）：

1. 每个药物的 DTI 根（OT + GDSC target，复用 ``build_local_graphs`` 的
   canonical tier 规则）；
2. 在 STRING 上做 personalized PageRank（对称或行归一化，restart 可配）；
3. 排除根，取 top-k 间接靶点；
4. 可选：对 top-N 基因做通路 ORA，把显著通路成员并回种子后再传播
   （``pathway_reseed``，默认关闭）；
5. 输出 gene index / weight / pathway profile / reliability，供
   ``T2DualTeacher`` 的 netgp token 使用。

归因对照（同一模块内实现，缓存键区分）：

- ``graph_control="configuration"``：保度随机重连（stub matching），只换拓扑；
- ``dti_control="permute_drugs"``：在 tier 内打乱药物-靶点对应，只换“谁结合谁”；
- ``gene_control="random_matched"``：保留根，用匹配度分位/可观测性的随机基因
  替换间接靶点，只换“选了哪些基因”。

所有输出只读 DTI/PPI/通路与基因空间，不读标签；因此 LDO 的 test 药也可以
直接构图。特征本身是药物级的，细胞侧的值在 ``data.py`` collate 中按
``(drug, cell)`` 读取。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import hypergeom

from kite_drp.data import io
from kite_drp.data.graph_inputs import load_graph_inputs
from kite_drp.train.features import CellFeatureSet, DrugFeatureSet
from kite_drp.train.local_graph import build_local_graphs
from kite_drp.train.ppi_features import (
    TIER_CODES,
    _align_pathway_weights,
    build_meta_features,
)

DEFAULT_NETGP_CONFIG: dict[str, Any] = {
    "version": 1,
    "ppi_min_score": 0.7,
    "restart": 0.5,
    "steps": 20,
    "top_k": 64,
    "exclude_roots": True,
    # symmetric | row
    "normalization": "symmetric",
    # 可选：对 top-N 间接靶点做通路 ORA，把通路成员并回种子后再传播
    "pathway_reseed": False,
    "pathway_top_n": 10,
    "pathway_min_overlap": 5,
    "pathway_ora_top_genes": 200,
    "reseed_weight": 0.5,
    "iterations": 1,
    # none | configuration
    "graph_control": "none",
    # none | permute_drugs
    "dti_control": "none",
    # none | random_matched
    "gene_control": "none",
    "seed": 20261123,
}

RELIABILITY_NAMES = (
    "coverage",
    "tier_norm",
    "selected_fraction",
    "weight_entropy_norm",
    "top1_mass",
    "mean_weight",
    "max_score_log",
    "n_roots_norm",
)


@dataclass
class NetGPFeatureBundle:
    """NetGP 间接靶点特征的 numpy 容器（全部为药物级）。"""

    gene_ids: np.ndarray  # (n_drugs, K) int32，索引到 CellFeatureSet.gene_ids
    gene_weight: np.ndarray  # (n_drugs, K) float32，和为 1；未覆盖药全 0
    pathway_profile: np.ndarray  # (n_drugs, P) float32，-log10(ORA p)
    reliability: np.ndarray  # (n_drugs, R) float32
    meta: np.ndarray  # (n_drugs, 8) float32，与 ppi_features.meta 同定义
    coverage: np.ndarray  # (n_drugs,) bool
    tier: np.ndarray  # (n_drugs,) int8
    audit: pd.DataFrame
    spec: dict[str, Any]

    @property
    def top_k(self) -> int:
        return int(self.gene_ids.shape[1])

    @property
    def pathway_dim(self) -> int:
        return int(self.pathway_profile.shape[1])

    @property
    def reliability_dim(self) -> int:
        return int(self.reliability.shape[1])

    @property
    def meta_dim(self) -> int:
        return int(self.meta.shape[1])


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    payload = json.loads(json.dumps(DEFAULT_NETGP_CONFIG))
    for key, value in (config or {}).items():
        if key in payload:
            payload[key] = value
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_payload(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _row_normalize(adjacency: sparse.csr_matrix) -> sparse.csr_matrix:
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inv = np.zeros_like(degree)
    positive = degree > 0
    inv[positive] = 1.0 / degree[positive]
    return (sparse.diags(inv) @ adjacency).tocsr()


def _symmetric_normalize(adjacency: sparse.csr_matrix) -> sparse.csr_matrix:
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(degree)
    positive = degree > 0
    inv_sqrt[positive] = 1.0 / np.sqrt(degree[positive])
    scale = sparse.diags(inv_sqrt)
    return (scale @ adjacency @ scale).tocsr()


def _build_adjacency(
    edges: np.ndarray,
    score: np.ndarray,
    *,
    n_nodes: int,
    min_score: float,
    normalization: str,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """构建无向加权邻接；返回 ``(S, degree, undirected_edges)``。"""

    keep = score >= float(min_score)
    edges = np.asarray(edges, dtype=np.int64)[:, keep]
    score = np.asarray(score, dtype=np.float32)[keep]
    rows = np.concatenate([edges[0], edges[1]])
    cols = np.concatenate([edges[1], edges[0]])
    values = np.concatenate([score, score]).astype(np.float32)
    adjacency = sparse.coo_matrix((values, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    # 同一无向边只保留最大置信度。
    adjacency = adjacency.maximum(adjacency.T).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    isolated = np.flatnonzero(degree <= 0)
    if isolated.size:
        adjacency = (
            adjacency
            + sparse.coo_matrix(
                (np.ones(isolated.size, dtype=np.float32), (isolated, isolated)),
                shape=adjacency.shape,
            ).tocsr()
        )
        degree = np.asarray(adjacency.sum(axis=1)).ravel()
    if normalization == "symmetric":
        transition = _symmetric_normalize(adjacency)
    elif normalization == "row":
        transition = _row_normalize(adjacency)
    else:
        raise ValueError(f"未知 normalization：{normalization}")
    undirected = np.sort(edges, axis=0)
    return transition.tocsr(), degree, undirected


def _configuration_edges(
    edges: np.ndarray,
    *,
    n_nodes: int,
    rng: np.random.Generator,
    score_pool: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """保度随机图：stub matching + 去自环/重边。

    保留节点集与度序列（去重前精确）；去重会略少边，审计里记录数量。
    """

    pair = np.sort(np.asarray(edges, dtype=np.int64), axis=0)
    degree = np.bincount(pair.ravel(), minlength=n_nodes)
    stubs = np.repeat(np.arange(n_nodes, dtype=np.int64), degree)
    rng.shuffle(stubs)
    left = stubs[0::2]
    right = stubs[1::2]
    keep = left != right
    left = left[keep]
    right = right[keep]
    candidate = np.sort(np.stack([left, right], axis=1), axis=1)
    unique = np.unique(candidate, axis=0)
    weight = rng.choice(
        np.asarray(score_pool, dtype=np.float32),
        size=unique.shape[0],
        replace=True,
    ).astype(np.float32)
    stats = {
        "edges_before": int(pair.shape[1]),
        "stub_edges": int(left.size),
        "edges_after": int(unique.shape[0]),
    }
    return unique.T.astype(np.int64), weight, stats


def _propagate(
    transition: sparse.csr_matrix,
    restart: np.ndarray,
    *,
    alpha: float,
    steps: int,
) -> np.ndarray:
    current = restart.astype(np.float32, copy=True)
    for _ in range(max(int(steps), 0)):
        current = (1.0 - float(alpha)) * restart + float(alpha) * (transition @ current)
    current = np.asarray(current).ravel().astype(np.float32)
    current[current < 0] = 0.0
    total = float(current.sum())
    if total > 0:
        current /= total
    return current


def _pathway_ora_for_genes(
    membership: sparse.csr_matrix,
    pathway_sizes: np.ndarray,
    genes: np.ndarray,
    *,
    n_background: int,
    min_overlap: int,
) -> np.ndarray:
    """对给定基因集合做超几何 ORA，返回 ``-log10(p)``。"""

    genes = np.asarray(genes, dtype=np.int64)
    genes = genes[(genes >= 0) & (genes < membership.shape[1])]
    if genes.size == 0:
        return np.zeros(membership.shape[0], dtype=np.float32)
    overlap = np.asarray(membership[:, genes].sum(axis=1)).ravel()
    pvals = hypergeom.sf(
        overlap - 1,
        int(n_background),
        pathway_sizes.astype(np.int64),
        int(genes.size),
    )
    score = -np.log10(np.clip(pvals, 1e-300, 1.0)).astype(np.float32)
    score[overlap < int(min_overlap)] = 0.0
    return score


def _top_genes(
    weights: np.ndarray,
    gene_symbols: list[str],
    *,
    top_n: int = 10,
) -> list[tuple[str, float]]:
    if weights.size == 0:
        return []
    order = np.argsort(-weights)[:top_n]
    return [
        (gene_symbols[int(index)], float(weights[int(index)]))
        for index in order
        if weights[int(index)] > 0
    ]


def _entropy(weights: np.ndarray) -> float:
    positive = weights[weights > 0]
    if positive.size == 0:
        return 0.0
    return float(-(positive * np.log(positive)).sum())


def _build_reliability(
    *,
    selected: np.ndarray,
    weights: np.ndarray,
    top_k: int,
    roots: np.ndarray,
    tier: int,
    coverage: bool,
) -> np.ndarray:
    values = np.zeros(len(RELIABILITY_NAMES), dtype=np.float32)
    if not coverage:
        return values
    values[0] = 1.0
    values[1] = float(tier) / 2.0
    values[2] = float(selected.size) / max(int(top_k), 1)
    entropy = _entropy(weights)
    denom = np.log(max(int(top_k), 2))
    values[3] = entropy / denom if denom > 0 else 0.0
    values[4] = float(weights.max()) if weights.size else 0.0
    values[5] = float(weights.mean()) if weights.size else 0.0
    values[6] = float(np.log1p(weights.max())) if weights.size else 0.0
    values[7] = float(roots.size) / 64.0
    return values


def _cache_key(
    processed: Path,
    features: CellFeatureSet,
    config: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    core = {
        "version": int(config["version"]),
        "ppi_min_score": float(config["ppi_min_score"]),
        "restart": float(config["restart"]),
        "steps": int(config["steps"]),
        "top_k": int(config["top_k"]),
        "exclude_roots": bool(config["exclude_roots"]),
        "normalization": str(config["normalization"]),
        "pathway_reseed": bool(config["pathway_reseed"]),
        "pathway_top_n": int(config["pathway_top_n"]),
        "pathway_min_overlap": int(config["pathway_min_overlap"]),
        "pathway_ora_top_genes": int(config["pathway_ora_top_genes"]),
        "reseed_weight": float(config["reseed_weight"]),
        "iterations": int(config["iterations"]),
        "graph_control": str(config["graph_control"]),
        "dti_control": str(config["dti_control"]),
        "gene_control": str(config["gene_control"]),
        "seed": int(config["seed"]),
        "dti_sha256": _sha256_file(processed / "dti.parquet"),
        "graph_inputs_sha256": _sha256_file(processed / "graph_inputs.npz"),
        "pathway_weights_sha256": _sha256_file(
            processed / "student_inputs" / "pathway_weights.npz"
        ),
        "gene_space_sha256": hashlib.sha256(
            "\n".join(features.gene_ids).encode("utf-8")
        ).hexdigest(),
        "code_version": "netgp_v1",
    }
    return _sha256_payload(core), core


def build_netgp_features(
    processed_dir: str | Path,
    features: CellFeatureSet,
    drugs: DrugFeatureSet,
    *,
    config: dict[str, Any] | None = None,
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> NetGPFeatureBundle:
    """构建/加载 NetGP 间接靶点特征。"""

    processed = Path(processed_dir)
    cfg = _resolve_config(config)
    top_k = max(int(cfg["top_k"]), 1)
    cache_hash, cache_core = _cache_key(processed, features, cfg)
    default_root = processed.parent / "interim" / "teacher" / "netgp_features"
    cache_root = (Path(cache_dir) if cache_dir is not None else default_root) / cache_hash[:16]
    cache_features = cache_root / "features.npz"
    cache_spec = cache_root / "spec.json"
    cache_audit = cache_root / "audit.json"
    if not force and cache_features.is_file() and cache_spec.is_file():
        with np.load(cache_features, allow_pickle=False) as payload:
            arrays = {key: payload[key] for key in payload.files}
        spec_payload = json.loads(cache_spec.read_text(encoding="utf-8"))
        if spec_payload.get("hash") == cache_hash:
            audit = (
                pd.read_json(cache_audit, orient="records")
                if cache_audit.is_file()
                else pd.DataFrame()
            )
            return NetGPFeatureBundle(
                gene_ids=arrays["gene_ids"].astype(np.int32),
                gene_weight=arrays["gene_weight"].astype(np.float32),
                pathway_profile=arrays["pathway_profile"].astype(np.float32),
                reliability=arrays["reliability"].astype(np.float32),
                meta=arrays["meta"].astype(np.float32),
                coverage=arrays["coverage"].astype(bool),
                tier=arrays["tier"].astype(np.int8),
                audit=audit,
                spec=spec_payload,
            )

    local_cfg = {
        "ppi_min_score": float(cfg["ppi_min_score"]),
        "neighbor_mode": "ppi",
        "max_neighbors_per_target": 8,
        "max_nodes_per_drug": 64,
        "rewire": "none",
        "coverage": "any",
        "seed": int(cfg["seed"]),
    }
    local_bundle = build_local_graphs(
        processed,
        features,
        drugs,
        config=local_cfg,
        cache_dir=(Path(cache_dir) / "local_graph" if cache_dir is not None else None),
    )
    node_ptr = np.asarray(local_bundle.arrays["node_ptr"], dtype=np.int64)
    node_gene = np.asarray(local_bundle.arrays["node_gene"], dtype=np.int64)
    node_root = np.asarray(local_bundle.arrays["node_root"], dtype=bool)
    n_roots = np.asarray(local_bundle.arrays["n_roots"], dtype=np.int64)
    n_neighbors = np.asarray(local_bundle.arrays["n_neighbors"], dtype=np.int64)
    tier_raw = np.asarray(local_bundle.arrays["tier"])
    tier = np.asarray([TIER_CODES.get(str(value), 0) for value in tier_raw], dtype=np.int8)
    coverage = np.asarray(local_bundle.arrays["coverage"], dtype=bool)

    graph = load_graph_inputs(processed)
    graph_genes = [str(value) for value in graph["genes"]]
    feature_index = {gene: index for index, gene in enumerate(features.gene_ids)}
    graph_to_feature = np.asarray(
        [feature_index.get(gene, -1) for gene in graph_genes], dtype=np.int64
    )
    valid_graph = graph_to_feature >= 0
    n_graph = len(graph_genes)
    feature_to_graph = np.full(len(features.gene_ids), -1, dtype=np.int64)
    feature_to_graph[graph_to_feature[valid_graph]] = np.flatnonzero(valid_graph)
    n_background = int(valid_graph.sum())

    pathway_arrays = io.read_npz(processed / "student_inputs" / "pathway_weights.npz")
    pathway_weights = _align_pathway_weights(
        np.asarray(pathway_arrays["weights"], dtype=np.float32),
        [str(value) for value in pathway_arrays["genes"]],
        features.gene_ids,
    )
    pathway_ids = [str(value) for value in pathway_arrays["pathway_ids"]]
    n_pathways = pathway_weights.shape[0]
    membership = sparse.csr_matrix((pathway_weights > 0).astype(np.float32))
    pathway_sizes = np.asarray(membership.sum(axis=1)).ravel().astype(np.int64)

    rng = np.random.default_rng(int(cfg["seed"]))
    graph_stats: dict[str, Any] = {}
    if str(cfg["graph_control"]) == "configuration":
        edges, weights, graph_stats = _configuration_edges(
            np.asarray(graph["ppi_edges"], dtype=np.int64),
            n_nodes=n_graph,
            rng=rng,
            score_pool=np.asarray(graph["ppi_score"], dtype=np.float32),
        )
        transition, degree, _ = _build_adjacency(
            edges,
            weights,
            n_nodes=n_graph,
            min_score=-1.0,
            normalization=str(cfg["normalization"]),
        )
    else:
        transition, degree, _ = _build_adjacency(
            np.asarray(graph["ppi_edges"], dtype=np.int64),
            np.asarray(graph["ppi_score"], dtype=np.float32),
            n_nodes=n_graph,
            min_score=float(cfg["ppi_min_score"]),
            normalization=str(cfg["normalization"]),
        )

    # 度分位池（random_matched 对照用）。
    quantiles = np.quantile(degree[valid_graph], np.linspace(0.0, 1.0, 11))
    bins = np.searchsorted(quantiles, degree, side="right") - 1
    bins = np.clip(bins, 0, 9)
    degree_pools: dict[int, list[int]] = {}
    for graph_index in np.flatnonzero(valid_graph):
        degree_pools.setdefault(int(bins[graph_index]), []).append(int(graph_index))

    n_drugs = len(drugs.compound_uid)
    gene_ids = np.zeros((n_drugs, top_k), dtype=np.int32)
    gene_weight = np.zeros((n_drugs, top_k), dtype=np.float32)
    pathway_profile = np.zeros((n_drugs, n_pathways), dtype=np.float32)
    reliability = np.zeros((n_drugs, len(RELIABILITY_NAMES)), dtype=np.float32)
    audit_rows: list[dict[str, Any]] = []

    for drug_index, compound_uid in enumerate(drugs.compound_uid):
        start = int(node_ptr[drug_index])
        end = int(node_ptr[drug_index + 1])
        nodes = node_gene[start:end]
        is_root = node_root[start:end]
        roots = nodes[is_root]
        if not coverage[drug_index] or roots.size == 0:
            audit_rows.append(
                {
                    "compound_uid": compound_uid,
                    "tier": str(tier_raw[drug_index]),
                    "coverage": False,
                    "n_roots": int(n_roots[drug_index]),
                    "n_selected": 0,
                    "top_genes": "",
                    "top_pathways": "",
                    "reseed_used": False,
                }
            )
            continue

        graph_roots = feature_to_graph[roots]
        graph_roots = graph_roots[graph_roots >= 0]
        restart = np.zeros(n_graph, dtype=np.float32)
        if graph_roots.size:
            restart[graph_roots] = 1.0 / graph_roots.size

        reseed_used = False
        scores = np.zeros(n_graph, dtype=np.float32)
        iterations = max(int(cfg["iterations"]), 1)
        for iteration in range(iterations):
            scores = _propagate(
                transition,
                restart,
                alpha=float(cfg["restart"]),
                steps=int(cfg["steps"]),
            )
            if bool(cfg["exclude_roots"]):
                scores[graph_roots] = 0.0
                total = float(scores.sum())
                if total > 0:
                    scores /= total
            if bool(cfg["pathway_reseed"]) and iteration < iterations - 1:
                ora_genes = np.argsort(-scores)[: int(cfg["pathway_ora_top_genes"])]
                ora_genes = ora_genes[scores[ora_genes] > 0]
                ora = _pathway_ora_for_genes(
                    membership,
                    pathway_sizes,
                    graph_to_feature[ora_genes],
                    n_background=n_background,
                    min_overlap=int(cfg["pathway_min_overlap"]),
                )
                top_pathways = np.argsort(-ora)[: int(cfg["pathway_top_n"])]
                member_feature: list[int] = []
                for pathway_index in top_pathways:
                    if ora[pathway_index] <= 0:
                        continue
                    row = membership.getrow(int(pathway_index))
                    member_feature.extend(row.indices.tolist())
                if member_feature:
                    member_graph = feature_to_graph[np.asarray(member_feature, dtype=np.int64)]
                    member_graph = member_graph[member_graph >= 0]
                    member_graph = member_graph[~np.isin(member_graph, graph_roots)]
                    if member_graph.size:
                        seed = restart.copy()
                        seed[member_graph] += float(cfg["reseed_weight"]) / max(
                            member_graph.size, 1
                        )
                        total = float(seed.sum())
                        if total > 0:
                            seed /= total
                        restart = seed
                        reseed_used = True

        order = np.argsort(-scores)[:top_k]
        order = order[scores[order] > 0]
        weights = scores[order].astype(np.float32)
        total = float(weights.sum())
        if total > 0:
            weights /= total
        selected_gene = graph_to_feature[order]
        valid = selected_gene >= 0
        selected_gene = selected_gene[valid].astype(np.int64)
        weights = weights[valid].astype(np.float32)
        if selected_gene.size:
            gene_ids[drug_index, : selected_gene.size] = selected_gene
            gene_weight[drug_index, : selected_gene.size] = weights
            pathway_profile[drug_index] = _pathway_ora_for_genes(
                membership,
                pathway_sizes,
                selected_gene,
                n_background=n_background,
                min_overlap=int(cfg["pathway_min_overlap"]),
            )
        reliability[drug_index] = _build_reliability(
            selected=selected_gene,
            weights=weights,
            top_k=top_k,
            roots=roots,
            tier=int(tier[drug_index]),
            coverage=True,
        )
        top_pathways = np.argsort(-pathway_profile[drug_index])[:5]
        top_pathways = [
            int(index) for index in top_pathways if pathway_profile[drug_index, int(index)] > 0
        ]
        audit_rows.append(
            {
                "compound_uid": compound_uid,
                "tier": str(tier_raw[drug_index]),
                "coverage": True,
                "n_roots": int(n_roots[drug_index]),
                "n_selected": int(selected_gene.size),
                "top_genes": ";".join(
                    f"{gene}:{weight:.4f}"
                    for gene, weight in _top_genes(
                        weights,
                        [features.gene_ids[int(index)] for index in selected_gene],
                    )
                ),
                "top_pathways": ";".join(pathway_ids[int(index)] for index in top_pathways),
                "reseed_used": reseed_used,
            }
        )

    meta = build_meta_features(
        n_roots=n_roots,
        n_neighbors=n_neighbors,
        tier=tier,
        coverage=coverage,
    )

    # 事后对照：只改“谁结合谁”，不改每个药自己的根/覆盖/tier。
    if str(cfg["dti_control"]) == "permute_drugs":
        covered = np.flatnonzero(coverage)
        for tier_code in np.unique(tier[covered]):
            group = covered[tier[covered] == tier_code]
            if group.size < 2:
                continue
            permutation = rng.permutation(group)
            gene_ids[group] = gene_ids[permutation]
            gene_weight[group] = gene_weight[permutation]
            pathway_profile[group] = pathway_profile[permutation]
    elif str(cfg["dti_control"]) != "none":
        raise ValueError(f"未知 dti_control：{cfg['dti_control']}")

    # 事后对照：保留根，用匹配度分位的随机基因替换间接靶点。
    if str(cfg["gene_control"]) == "random_matched":
        random_gene = np.zeros_like(gene_ids)
        random_weight = np.zeros_like(gene_weight)
        random_pathway = np.zeros_like(pathway_profile)
        for drug_index in range(n_drugs):
            if not coverage[drug_index]:
                continue
            start = int(node_ptr[drug_index])
            end = int(node_ptr[drug_index + 1])
            roots = node_gene[start:end][node_root[start:end]]
            used = {int(value) for value in feature_to_graph[roots] if value >= 0}
            count = int((gene_weight[drug_index] > 0).sum())
            if count == 0:
                continue
            chosen: list[int] = []
            for position in range(count):
                graph_index = int(feature_to_graph[gene_ids[drug_index, position]])
                pool = [
                    value
                    for value in degree_pools.get(int(bins[graph_index]), [])
                    if value not in used
                ]
                if not pool:
                    pool = [
                        value
                        for values in degree_pools.values()
                        for value in values
                        if value not in used
                    ]
                if not pool:
                    break
                pick = int(rng.choice(pool))
                used.add(pick)
                chosen.append(pick)
            if not chosen:
                continue
            feature_chosen = graph_to_feature[np.asarray(chosen, dtype=np.int64)]
            random_gene[drug_index, : len(chosen)] = feature_chosen
            random_weight[drug_index, : len(chosen)] = rng.permutation(
                gene_weight[drug_index, :count]
            )
            random_pathway[drug_index] = _pathway_ora_for_genes(
                membership,
                pathway_sizes,
                feature_chosen,
                n_background=n_background,
                min_overlap=int(cfg["pathway_min_overlap"]),
            )
        gene_ids = random_gene
        gene_weight = random_weight
        pathway_profile = random_pathway
    elif str(cfg["gene_control"]) != "none":
        raise ValueError(f"未知 gene_control：{cfg['gene_control']}")

    audit = pd.DataFrame(audit_rows)
    spec = {
        **cache_core,
        "hash": cache_hash,
        "cache_dir": str(cache_root),
        "n_drugs": n_drugs,
        "top_k": top_k,
        "n_pathways": n_pathways,
        "n_covered": int(coverage.sum()),
        "coverage_counts": {
            "ot": int((tier == TIER_CODES["ot"]).sum()),
            "gdsc_only": int((tier == TIER_CODES["gdsc_only"]).sum()),
            "none": int((tier == TIER_CODES["none"]).sum()),
        },
        "reliability_names": list(RELIABILITY_NAMES),
        "graph_stats": graph_stats,
        "n_selected_mean": float(
            (gene_weight > 0).sum(axis=1)[coverage].mean() if coverage.any() else 0.0
        ),
        "reseed_used": int(audit["reseed_used"].sum()) if "reseed_used" in audit else 0,
    }
    cache_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_features,
        gene_ids=gene_ids.astype(np.int32),
        gene_weight=gene_weight.astype(np.float32),
        pathway_profile=pathway_profile.astype(np.float32),
        reliability=reliability.astype(np.float32),
        meta=meta.astype(np.float32),
        coverage=coverage,
        tier=tier,
    )
    cache_spec.write_text(
        json.dumps(spec, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    audit.to_json(cache_audit, orient="records", force_ascii=False, indent=2)
    return NetGPFeatureBundle(
        gene_ids=gene_ids.astype(np.int32),
        gene_weight=gene_weight.astype(np.float32),
        pathway_profile=pathway_profile.astype(np.float32),
        reliability=reliability.astype(np.float32),
        meta=meta.astype(np.float32),
        coverage=coverage,
        tier=tier,
        audit=audit,
        spec=spec,
    )


__all__ = [
    "DEFAULT_NETGP_CONFIG",
    "NetGPFeatureBundle",
    "RELIABILITY_NAMES",
    "build_netgp_features",
]

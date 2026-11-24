"""无图 DTI/PPI 特征：靶点/邻居集合统计、通路 profile、RWR 权重。

设计口径见 ``plans&notes/2026-11-20_T1局部图实施.md`` 的后续方案：

- 不建图、不做消息传递；DTI/PPI 只用于生成**固定特征**。
- 每个药物取一跳局部节点（根 = 直接靶点，邻居 = STRING 一跳），
  在当前细胞上读取 GE/MU/CNV，计算集合统计与通路投影。
- RWR（personalized PageRank）从靶点出发在 STRING 上扩散，得到每药
  固定基因权重，再与细胞组学做加权和。
- 只读 DTI/PPI/组学与 fold 内统计，不读标签；缓存键覆盖所有上游内容。

特征分三块，供 ``T2DualTeacher`` 的 context 分支使用：

``summary``
    (n_drugs, n_cells, F_summary)：根/邻居 × GE/MU/CNV 的均值、极值、
    标准差、观测率、异常率，以及节点数。
``rwr``
    (n_drugs, n_cells, F_rwr)：RWR 加权 GE/MU/CNV、异常率、熵、
    最大权重、有效集合大小。
``pathway``
    (n_drugs, P)：靶点/邻居基因在 1000 条通路上的权重 profile；
    模型里与当前细胞通路分数逐元素相乘。
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

from kite_drp.data import io
from kite_drp.data.graph_inputs import load_graph_inputs
from kite_drp.train.features import CellFeatureSet, DrugFeatureSet
from kite_drp.train.local_graph import LocalGraphBundle, build_local_graphs

DEFAULT_PPI_FEATURE_CONFIG: dict[str, Any] = {
    "version": 2,
    "ppi_min_score": 0.7,
    "neighbor_mode": "ppi",  # ppi | pseudo
    "max_neighbors_per_target": 8,
    "max_nodes_per_drug": 64,
    "rwr_alpha": 0.5,
    "rwr_steps": 10,
    "rwr_min_score": 0.7,
    "random_control": "none",  # none | permute_drugs | gaussian
    "seed": 20261120,
}

TIER_CODES = {"none": 0, "gdsc_only": 1, "ot": 2}
SUMMARY_STATS = ("mean", "max", "min", "std", "frac_observed", "frac_altered")
META_NAMES = (
    "coverage",
    "tier_ot",
    "tier_gdsc_only",
    "tier_none",
    "n_roots_norm",
    "n_neighbors_norm",
    "log_n_roots",
    "log_n_neighbors",
)


def build_meta_features(
    *,
    n_roots: np.ndarray,
    n_neighbors: np.ndarray,
    tier: np.ndarray,
    coverage: np.ndarray,
) -> np.ndarray:
    """药物级元特征：覆盖、tier、根/邻居数。

    这些特征与细胞无关，逐药常数。它们不能改变逐药 SCC（层内排序对
    药物级常数不敏感），主要用于：(a) ``t2_meta`` 混杂对照；
    (b) reliability gate 的输入；(c) 全局指标诊断。
    """

    coverage = np.asarray(coverage, dtype=bool)
    tier = np.asarray(tier)
    n_roots = np.asarray(n_roots, dtype=np.float32)
    n_neighbors = np.asarray(n_neighbors, dtype=np.float32)
    meta = np.zeros((len(coverage), len(META_NAMES)), dtype=np.float32)
    meta[:, 0] = coverage.astype(np.float32)
    meta[:, 1] = (tier == TIER_CODES["ot"]).astype(np.float32)
    meta[:, 2] = (tier == TIER_CODES["gdsc_only"]).astype(np.float32)
    meta[:, 3] = (tier == TIER_CODES["none"]).astype(np.float32)
    denom = 64.0
    meta[:, 4] = np.clip(n_roots / denom, 0.0, 1.0)
    meta[:, 5] = np.clip(n_neighbors / denom, 0.0, 1.0)
    meta[:, 6] = np.log1p(n_roots) / np.log1p(denom)
    meta[:, 7] = np.log1p(n_neighbors) / np.log1p(denom)
    return meta


@dataclass
class PPIFeatureBundle:
    """无图 DTI/PPI 特征的 numpy 容器。"""

    summary: np.ndarray  # (n_drugs, n_cells, F_summary)
    rwr: np.ndarray  # (n_drugs, n_cells, F_rwr)
    target_pathway: np.ndarray  # (n_drugs, P)
    neighbor_pathway: np.ndarray  # (n_drugs, P)
    coverage: np.ndarray  # (n_drugs,) bool
    tier: np.ndarray  # (n_drugs,) int8
    meta: np.ndarray  # (n_drugs, F_meta) float32，药物级元特征
    summary_names: list[str]
    rwr_names: list[str]
    audit: pd.DataFrame
    spec: dict[str, Any]

    @property
    def summary_dim(self) -> int:
        return int(self.summary.shape[2])

    @property
    def rwr_dim(self) -> int:
        return int(self.rwr.shape[2])

    @property
    def meta_dim(self) -> int:
        return int(self.meta.shape[1])


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    payload = json.loads(json.dumps(DEFAULT_PPI_FEATURE_CONFIG))
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


def _align_pathway_weights(
    weights: np.ndarray,
    weight_genes: list[str],
    target_genes: list[str],
) -> np.ndarray:
    """把 (P, G_weight) 通路权重对齐到目标基因顺序。"""

    if weight_genes == target_genes:
        return weights.astype(np.float32)
    lookup = {gene: position for position, gene in enumerate(weight_genes)}
    columns = np.asarray([lookup.get(gene, -1) for gene in target_genes], dtype=np.int64)
    aligned = np.zeros((weights.shape[0], len(target_genes)), dtype=np.float32)
    valid = columns >= 0
    aligned[:, valid] = weights[:, columns[valid]]
    return aligned


def _masked_stats(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    altered: np.ndarray,
) -> dict[str, np.ndarray]:
    """对 (n_cells, n_genes) 的集合做 masked 统计；空集合返回 0。"""

    weights = valid.astype(np.float32)
    count = weights.sum(axis=1)
    safe = np.maximum(count, 1.0)
    mean = (values * weights).sum(axis=1) / safe
    centered = np.where(valid, values - mean[:, None], 0.0)
    std = np.sqrt((centered**2 * weights).sum(axis=1) / safe)
    masked_max = np.where(valid, values, -np.inf).max(axis=1)
    masked_min = np.where(valid, values, np.inf).min(axis=1)
    masked_max = np.where(count > 0, masked_max, 0.0)
    masked_min = np.where(count > 0, masked_min, 0.0)
    frac_observed = count / max(values.shape[1], 1)
    frac_altered = (altered & valid).sum(axis=1) / safe
    return {
        "mean": mean.astype(np.float32),
        "max": masked_max.astype(np.float32),
        "min": masked_min.astype(np.float32),
        "std": std.astype(np.float32),
        "frac_observed": frac_observed.astype(np.float32),
        "frac_altered": np.where(count > 0, frac_altered, 0.0).astype(np.float32),
    }


def _summary_names() -> list[str]:
    names = [
        f"{group}_{modality}_{stat}"
        for group in ("root", "neighbor")
        for modality in ("GE", "MU", "CNV")
        for stat in SUMMARY_STATS
    ]
    names.extend(["n_roots_norm", "n_neighbors_norm"])
    return names


def _rwr_names() -> list[str]:
    return [
        "rwr_GE_mean",
        "rwr_MU_mean",
        "rwr_CNV_mean",
        "rwr_GE_altered",
        "rwr_CNV_altered",
        "rwr_entropy",
        "rwr_max_weight",
        "rwr_effective_size",
    ]


def _build_rwr_adjacency(
    graph: dict[str, np.ndarray],
    *,
    min_score: float,
) -> tuple[sparse.csr_matrix, list[str]]:
    """构建列随机 PPI 邻接（A[target, source]），孤立节点自环。"""

    genes = [str(gene) for gene in graph["genes"]]
    edges = np.asarray(graph["ppi_edges"], dtype=np.int64)
    score = np.asarray(graph["ppi_score"], dtype=np.float32)
    keep = score >= float(min_score)
    edges = edges[:, keep]
    score = score[keep]
    if edges.size == 0:
        return sparse.eye(len(genes), format="csr", dtype=np.float32), genes
    rows = np.concatenate([edges[1], edges[0]])
    cols = np.concatenate([edges[0], edges[1]])
    values = np.concatenate([score, score]).astype(np.float32)
    adjacency = sparse.coo_matrix((values, (rows, cols)), shape=(len(genes), len(genes))).tocsr()
    degree = np.asarray(adjacency.sum(axis=0)).ravel()
    inv = np.zeros_like(degree)
    inv[degree > 0] = 1.0 / degree[degree > 0]
    adjacency = adjacency.multiply(inv[None, :]).tocsr()
    # 孤立节点用自环，避免 RWR 质量消失。
    isolated = np.flatnonzero(degree <= 0)
    if isolated.size:
        adjacency = (
            adjacency
            + sparse.coo_matrix(
                (np.ones(isolated.size, dtype=np.float32), (isolated, isolated)),
                shape=adjacency.shape,
            ).tocsr()
        )
    return adjacency, genes


def _rwr_weights(
    adjacency: sparse.csr_matrix,
    roots: np.ndarray,
    *,
    alpha: float,
    steps: int,
) -> np.ndarray:
    n_graph = adjacency.shape[0]
    weights = np.zeros(n_graph, dtype=np.float32)
    valid_roots = roots[(roots >= 0) & (roots < n_graph)]
    if valid_roots.size == 0:
        return weights
    restart = np.zeros(n_graph, dtype=np.float32)
    restart[valid_roots] = 1.0 / valid_roots.size
    current = restart.copy()
    for _ in range(max(int(steps), 0)):
        current = (1.0 - float(alpha)) * restart + float(alpha) * (adjacency @ current)
    total = float(current.sum())
    if total > 0:
        current = current / total
    return current.astype(np.float32)


def _pathway_profile(
    weights: np.ndarray,
    genes: np.ndarray,
) -> np.ndarray:
    if genes.size == 0:
        return np.zeros(weights.shape[0], dtype=np.float32)
    profile = weights[:, genes].sum(axis=1).astype(np.float32)
    total = float(profile.sum())
    if total > 0:
        profile /= total
    return profile


def _top_genes(
    weights: np.ndarray,
    graph_genes: list[str],
    *,
    top_k: int = 20,
) -> list[tuple[str, float]]:
    if weights.size == 0:
        return []
    order = np.argsort(-weights)[:top_k]
    return [(graph_genes[int(index)], float(weights[int(index)])) for index in order]


def _feature_fingerprint(features: CellFeatureSet) -> str:
    """fold 内组学值 + 观测掩码的指纹，防止跨 fold 误用 PPI 特征缓存。"""

    digest = hashlib.sha256()
    for array in (features.gene_ge, features.gene_mu, features.gene_cnv):
        digest.update(np.ascontiguousarray(array, dtype=np.float32).tobytes())
    for array in (
        features.gene_ge_valid,
        features.gene_cnv_valid,
        features.gene_mu_available,
        features.ge_mask,
        features.mu_mask,
        features.cnv_mask,
    ):
        digest.update(np.ascontiguousarray(array, dtype=np.bool_).tobytes())
    return digest.hexdigest()


def _build_cache_key(
    processed: Path,
    features: CellFeatureSet,
    local_bundle: LocalGraphBundle,
    config: dict[str, Any],
    *,
    fold_key: str = "",
) -> tuple[str, dict[str, Any]]:
    dti_path = processed / "dti.parquet"
    graph_path = processed / "graph_inputs.npz"
    pathway_path = processed / "student_inputs" / "pathway_weights.npz"
    core = {
        "fold_key": str(fold_key),
        "feature_fingerprint": _feature_fingerprint(features),
        "version": int(config["version"]),
        "ppi_min_score": float(config["ppi_min_score"]),
        "neighbor_mode": str(config["neighbor_mode"]),
        "max_neighbors_per_target": int(config["max_neighbors_per_target"]),
        "max_nodes_per_drug": int(config["max_nodes_per_drug"]),
        "rwr_alpha": float(config["rwr_alpha"]),
        "rwr_steps": int(config["rwr_steps"]),
        "rwr_min_score": float(config["rwr_min_score"]),
        "random_control": str(config["random_control"]),
        "seed": int(config["seed"]),
        "dti_sha256": _sha256_file(dti_path),
        "graph_inputs_sha256": _sha256_file(graph_path),
        "pathway_weights_sha256": _sha256_file(pathway_path),
        "local_graph_hash": str(local_bundle.spec.get("hash", "")),
        "gene_space_sha256": hashlib.sha256(
            "\n".join(features.gene_ids).encode("utf-8")
        ).hexdigest(),
        "code_version": "ppi_features_v2",
    }
    return _sha256_payload(core), core


def _empty_features(
    *,
    n_drugs: int,
    n_cells: int,
    n_pathways: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((n_drugs, n_cells, len(_summary_names())), dtype=np.float32),
        np.zeros((n_drugs, n_cells, len(_rwr_names())), dtype=np.float32),
        np.zeros((n_drugs, n_pathways), dtype=np.float32),
        np.zeros((n_drugs, n_pathways), dtype=np.float32),
    )


def build_ppi_features(
    processed_dir: str | Path,
    features: CellFeatureSet,
    drugs: DrugFeatureSet,
    *,
    config: dict[str, Any] | None = None,
    cache_dir: str | Path | None = None,
    fold_key: str = "",
    force: bool = False,
) -> PPIFeatureBundle:
    """构建/加载无图 DTI/PPI 特征。"""

    processed = Path(processed_dir)
    cfg = _resolve_config(config)
    local_cfg = {
        "ppi_min_score": cfg["ppi_min_score"],
        "neighbor_mode": cfg["neighbor_mode"],
        "max_neighbors_per_target": cfg["max_neighbors_per_target"],
        "max_nodes_per_drug": cfg["max_nodes_per_drug"],
        "seed": cfg["seed"],
    }
    local_bundle = build_local_graphs(
        processed,
        features,
        drugs,
        config=local_cfg,
        cache_dir=(Path(cache_dir) / "local_graph" if cache_dir is not None else None),
    )
    cache_hash, cache_core = _build_cache_key(
        processed, features, local_bundle, cfg, fold_key=fold_key
    )
    default_root = processed.parent / "interim" / "teacher" / "ppi_features"
    cache_root = (Path(cache_dir) if cache_dir is not None else default_root) / cache_hash[:16]
    cache_features = cache_root / "features.npz"
    cache_spec = cache_root / "spec.json"
    cache_audit = cache_root / "audit.json"
    if not force and cache_features.is_file() and cache_spec.is_file():
        with np.load(cache_features, allow_pickle=False) as payload:
            arrays = {key: payload[key] for key in payload.files}
        spec = json.loads(cache_spec.read_text(encoding="utf-8"))
        if spec.get("hash") == cache_hash:
            audit = (
                pd.read_json(cache_audit, orient="records")
                if cache_audit.is_file()
                else pd.DataFrame()
            )
            return PPIFeatureBundle(
                summary=arrays["summary"],
                rwr=arrays["rwr"],
                target_pathway=arrays["target_pathway"],
                neighbor_pathway=arrays["neighbor_pathway"],
                coverage=arrays["coverage"].astype(bool),
                tier=arrays["tier"].astype(np.int8),
                meta=arrays["meta"].astype(np.float32),
                summary_names=[str(value) for value in arrays["summary_names"]],
                rwr_names=[str(value) for value in arrays["rwr_names"]],
                audit=audit,
                spec=spec,
            )

    pathway_arrays = io.read_npz(processed / "student_inputs" / "pathway_weights.npz")
    pathway_weights = _align_pathway_weights(
        np.asarray(pathway_arrays["weights"], dtype=np.float32),
        [str(value) for value in pathway_arrays["genes"]],
        features.gene_ids,
    )
    pathway_ids = [str(value) for value in pathway_arrays["pathway_ids"]]
    graph = load_graph_inputs(processed)
    graph_genes = [str(value) for value in graph["genes"]]
    graph_index = {gene: index for index, gene in enumerate(graph_genes)}
    feature_index = {gene: index for index, gene in enumerate(features.gene_ids)}
    adjacency, _ = _build_rwr_adjacency(graph, min_score=float(cfg["rwr_min_score"]))

    n_drugs = len(drugs.compound_uid)
    n_cells = len(features.cell_ids)
    n_pathways = pathway_weights.shape[0]
    summary, rwr, target_pathway, neighbor_pathway = _empty_features(
        n_drugs=n_drugs, n_cells=n_cells, n_pathways=n_pathways
    )
    coverage = np.asarray(local_bundle.arrays["coverage"], dtype=bool).copy()
    tier_raw = np.asarray(local_bundle.arrays["tier"])
    tier = np.asarray([TIER_CODES.get(str(value), 0) for value in tier_raw], dtype=np.int8)
    node_ptr = np.asarray(local_bundle.arrays["node_ptr"], dtype=np.int64)
    node_gene = np.asarray(local_bundle.arrays["node_gene"], dtype=np.int64)
    node_root = np.asarray(local_bundle.arrays["node_root"], dtype=bool)
    n_roots = np.asarray(local_bundle.arrays["n_roots"], dtype=np.int64)
    n_neighbors = np.asarray(local_bundle.arrays["n_neighbors"], dtype=np.int64)

    ge = features.gene_ge.astype(np.float32)
    mu = features.gene_mu.astype(np.float32)
    cnv = features.gene_cnv.astype(np.float32)
    ge_valid = features.gene_ge_valid[None, :] & features.ge_mask[:, None]
    mu_valid = features.gene_mu_available[None, :] & features.mu_mask[:, None]
    cnv_valid = features.gene_cnv_valid[None, :] & features.cnv_mask[:, None]
    ge_altered = np.abs(ge) > 1.0
    mu_altered = mu > 0.0
    cnv_altered = np.abs(cnv) > 1.0

    audit_rows: list[dict[str, Any]] = []
    max_nodes = max(int(cfg["max_nodes_per_drug"]), 1)
    for drug_index, compound_uid in enumerate(drugs.compound_uid):
        start = int(node_ptr[drug_index])
        end = int(node_ptr[drug_index + 1])
        nodes = node_gene[start:end]
        is_root = node_root[start:end]
        roots = nodes[is_root]
        neighbors = nodes[~is_root]
        summary[drug_index, :, -2] = n_roots[drug_index] / max_nodes
        summary[drug_index, :, -1] = n_neighbors[drug_index] / max_nodes
        if not coverage[drug_index] or nodes.size == 0:
            audit_rows.append(
                {
                    "compound_uid": compound_uid,
                    "tier": str(tier_raw[drug_index]),
                    "coverage": False,
                    "n_roots": int(n_roots[drug_index]),
                    "n_neighbors": int(n_neighbors[drug_index]),
                    "rwr_top_genes": "",
                    "top_target_pathways": "",
                    "top_neighbor_pathways": "",
                }
            )
            continue

        for group_index, genes in enumerate((roots, neighbors)):
            for modality_index, (values, valid, altered) in enumerate(
                (
                    (ge, ge_valid, ge_altered),
                    (mu, mu_valid, mu_altered),
                    (cnv, cnv_valid, cnv_altered),
                )
            ):
                if genes.size == 0:
                    continue
                stats = _masked_stats(
                    values[:, genes],
                    valid[:, genes],
                    altered=altered[:, genes],
                )
                offset = (group_index * 3 + modality_index) * len(SUMMARY_STATS)
                for stat_index, stat in enumerate(SUMMARY_STATS):
                    summary[drug_index, :, offset + stat_index] = stats[stat]

        target_pathway[drug_index] = _pathway_profile(pathway_weights, roots)
        neighbor_pathway[drug_index] = _pathway_profile(pathway_weights, neighbors)

        graph_roots = np.asarray(
            [graph_index.get(features.gene_ids[index], -1) for index in roots],
            dtype=np.int64,
        )
        weights = _rwr_weights(
            adjacency,
            graph_roots,
            alpha=float(cfg["rwr_alpha"]),
            steps=int(cfg["rwr_steps"]),
        )
        weights_feature = np.zeros(len(features.gene_ids), dtype=np.float32)
        for graph_position, gene in enumerate(graph_genes):
            feature_position = feature_index.get(gene)
            if feature_position is not None and weights[graph_position] != 0:
                weights_feature[feature_position] = weights[graph_position]
        total = float(weights_feature.sum())
        if total > 0:
            weights_feature /= total
        rwr[drug_index, :, 0] = ge @ weights_feature
        rwr[drug_index, :, 1] = mu @ weights_feature
        rwr[drug_index, :, 2] = cnv @ weights_feature
        rwr[drug_index, :, 3] = (ge_altered & ge_valid) @ weights_feature
        rwr[drug_index, :, 4] = (cnv_altered & cnv_valid) @ weights_feature
        positive = weights_feature[weights_feature > 0]
        if positive.size:
            rwr[drug_index, :, 5] = float(-(positive * np.log(positive)).sum())
            rwr[drug_index, :, 6] = float(positive.max())
            rwr[drug_index, :, 7] = float(1.0 / np.sum(positive**2))
        top_genes = _top_genes(weights_feature, features.gene_ids)
        top_target_pathways = np.argsort(-target_pathway[drug_index])[:5]
        top_neighbor_pathways = np.argsort(-neighbor_pathway[drug_index])[:5]
        audit_rows.append(
            {
                "compound_uid": compound_uid,
                "tier": str(tier_raw[drug_index]),
                "coverage": True,
                "n_roots": int(n_roots[drug_index]),
                "n_neighbors": int(n_neighbors[drug_index]),
                "rwr_top_genes": ";".join(f"{gene}:{weight:.4f}" for gene, weight in top_genes),
                "top_target_pathways": ";".join(
                    pathway_ids[int(index)] for index in top_target_pathways
                ),
                "top_neighbor_pathways": ";".join(
                    pathway_ids[int(index)] for index in top_neighbor_pathways
                ),
            }
        )

    if str(cfg["random_control"]) == "permute_drugs":
        rng = np.random.default_rng(int(cfg["seed"]) + 31)
        covered = np.flatnonzero(coverage)
        permutation = rng.permutation(covered)
        summary[covered] = summary[permutation]
        rwr[covered] = rwr[permutation]
        target_pathway[covered] = target_pathway[permutation]
        neighbor_pathway[covered] = neighbor_pathway[permutation]
    elif str(cfg["random_control"]) == "gaussian":
        rng = np.random.default_rng(int(cfg["seed"]) + 37)
        covered = np.flatnonzero(coverage)
        summary[covered] = rng.normal(0.0, 1.0, size=summary[covered].shape).astype(np.float32)
        rwr[covered] = rng.normal(0.0, 1.0, size=rwr[covered].shape).astype(np.float32)
        target_pathway[covered] = rng.random(target_pathway[covered].shape).astype(np.float32)
        neighbor_pathway[covered] = rng.random(neighbor_pathway[covered].shape).astype(np.float32)
    elif str(cfg["random_control"]) != "none":
        raise ValueError(f"未知 random_control：{cfg['random_control']}")

    meta = build_meta_features(
        n_roots=n_roots,
        n_neighbors=n_neighbors,
        tier=tier,
        coverage=coverage,
    )
    summary_names = _summary_names()
    rwr_names = _rwr_names()
    audit = pd.DataFrame(audit_rows)
    spec = {
        **cache_core,
        "hash": cache_hash,
        "cache_dir": str(cache_root),
        "n_drugs": n_drugs,
        "n_cells": n_cells,
        "n_pathways": n_pathways,
        "n_covered": int(coverage.sum()),
        "summary_dim": len(summary_names),
        "rwr_dim": len(rwr_names),
        "meta_dim": len(META_NAMES),
        "summary_names": summary_names,
        "rwr_names": rwr_names,
        "meta_names": list(META_NAMES),
        "coverage_counts": {
            "ot": int((tier == TIER_CODES["ot"]).sum()),
            "gdsc_only": int((tier == TIER_CODES["gdsc_only"]).sum()),
            "none": int((tier == TIER_CODES["none"]).sum()),
        },
    }
    cache_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_features,
        summary=summary,
        rwr=rwr,
        target_pathway=target_pathway,
        neighbor_pathway=neighbor_pathway,
        coverage=coverage,
        tier=tier,
        meta=meta,
        summary_names=np.asarray(summary_names, dtype=np.str_),
        rwr_names=np.asarray(rwr_names, dtype=np.str_),
    )
    cache_spec.write_text(
        json.dumps(spec, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    audit.to_json(cache_audit, orient="records", force_ascii=False, indent=2)
    return PPIFeatureBundle(
        summary=summary,
        rwr=rwr,
        target_pathway=target_pathway,
        neighbor_pathway=neighbor_pathway,
        coverage=coverage,
        tier=tier,
        meta=meta,
        summary_names=summary_names,
        rwr_names=rwr_names,
        audit=audit,
        spec=spec,
    )


__all__ = [
    "DEFAULT_PPI_FEATURE_CONFIG",
    "META_NAMES",
    "PPIFeatureBundle",
    "TIER_CODES",
    "build_meta_features",
    "build_ppi_features",
]

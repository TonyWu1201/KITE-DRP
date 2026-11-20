"""每药一张一跳局部图：DTI 根节点 + STRING 邻居 + 审计/缓存。

设计口径见 ``plans&notes/2026-11-20_教师模型复盘与图教师重构方案.md`` §5：

- 根：``dti.parquet`` 中该药的可测直接靶点；Open Targets MoA 优先，
  GDSC ``TARGET`` 文本作为第二层证据；仅名称匹配的低置信边不进入主图。
- 邻居：STRING ``combined_score >= ppi_min_score`` 的一跳邻居；每个根最多
  ``max_neighbors_per_target`` 个，每药最多 ``max_nodes_per_drug`` 个节点。
- 拓扑只由 DTI + PPI + 版本化先验决定，不读标签、不读同批其他样本；
  因此 LDO 的 test 药也可以直接构图。
- 缓存键覆盖 DTI/PPI 内容、基因空间、裁剪参数和构图代码版本；旧 T1 那种
  只依赖配置和计数的 ``spec_hash`` 不足以区分上游内容替换。

局部图不包含细胞特征；细胞侧 GE/MU/CNV 在 ``data.py`` 的 collate 中按
``(drug, cell)`` 读取，保证同一邻接在不同细胞上产生不同消息。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from kite_drp.data import io
from kite_drp.data.graph_inputs import load_graph_inputs
from kite_drp.train.features import CellFeatureSet, DrugFeatureSet

DEFAULT_LOCAL_GRAPH_CONFIG: dict[str, Any] = {
    "version": 1,
    "ppi_min_score": 0.7,
    "max_neighbors_per_target": 8,
    "max_nodes_per_drug": 64,
    # ppi | pseudo：pseudo 保留根，用匹配度/可观测性的随机基因替换邻居。
    "neighbor_mode": "ppi",
    # none | degree：保度双边交换；只改变拓扑，不改变节点集。
    "rewire": "none",
    "seed": 20261120,
    # 主门控使用 any（OT+GDSC）；OT 只用于分层报告。
    "coverage": "any",
}

DTI_COLUMNS = (
    "compound_uid",
    "hgnc_symbol",
    "source",
    "score",
    "action_type",
    "match_method",
)


@dataclass
class LocalGraphBundle:
    """局部图的 numpy 数组 + 审计信息。"""

    arrays: dict[str, np.ndarray]
    spec: dict[str, Any]
    audit: pd.DataFrame


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    payload = json.loads(json.dumps(DEFAULT_LOCAL_GRAPH_CONFIG))
    for key, value in (config or {}).items():
        if key not in payload:
            continue
        payload[key] = value
    return payload


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_payload(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return _sha256_bytes(text.encode("utf-8"))


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def _split_tags(value: Any) -> set[str]:
    text = _clean(value).replace(";", "+")
    return {part.strip() for part in text.split("+") if part.strip()}


def _root_tier(source: str, match_method: str, action_type: str) -> str:
    """把一条 DTI 记录分成 ot / gdsc / weak。

    OT 需要可追溯的 ChEMBL/InChIKey 匹配和非空作用类型；仅名称匹配的边
    仍保留在审计里，但不进入主图，避免同名药引入假靶点。
    """

    sources = _split_tags(source)
    methods = _split_tags(match_method)
    actions = _split_tags(action_type)
    high_confidence = bool(methods & {"chembl_id", "inchikey"})
    if "opentargets" in sources and high_confidence and actions:
        return "ot"
    if "gdsc_target" in sources:
        return "gdsc"
    return "weak"


def _load_dti(processed: Path) -> pd.DataFrame:
    path = processed / "dti.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"缺少 {path}；请先跑 DTI 构建")
    dti = io.read_parquet(path)
    missing = [column for column in DTI_COLUMNS if column not in dti.columns]
    if missing:
        raise ValueError(f"DTI 缺少列：{missing}")
    return dti


def _root_table(
    dti: pd.DataFrame,
    *,
    compound_order: list[str],
    gene_to_feature: dict[str, int],
) -> tuple[dict[str, list[int]], dict[str, str], dict[str, list[str]], dict[str, Any]]:
    """按化合物聚合根基因，返回 feature 索引、覆盖层级和审计信息。"""

    roots: dict[str, list[int]] = {}
    tiers: dict[str, str] = {}
    root_names: dict[str, list[str]] = {}
    rows = dti.copy()
    rows["source"] = rows["source"].map(_clean)
    rows["match_method"] = rows["match_method"].map(_clean)
    rows["action_type"] = rows["action_type"].map(_clean)
    rows["score"] = pd.to_numeric(rows["score"], errors="coerce").fillna(1.0)

    stats = {
        "dti_rows": int(len(rows)),
        "dti_compounds": int(rows["compound_uid"].nunique()),
        "roots_ot": 0,
        "roots_gdsc": 0,
        "roots_weak": 0,
        "roots_unmapped": 0,
    }

    for compound_uid, group in rows.groupby("compound_uid", sort=False):
        if compound_uid not in compound_order:
            continue
        by_gene: dict[str, list[tuple[str, str, str]]] = {}
        for row in group.itertuples(index=False):
            gene = _clean(row.hgnc_symbol)
            if not gene:
                continue
            by_gene.setdefault(gene, []).append(
                (_clean(row.source), _clean(row.match_method), _clean(row.action_type))
            )
        drug_roots: list[tuple[int, str, str]] = []
        for gene, records in by_gene.items():
            gene_tier = "weak"
            for source, method, action in records:
                tier = _root_tier(source, method, action)
                if tier == "ot":
                    gene_tier = "ot"
                    break
                if tier == "gdsc" and gene_tier != "ot":
                    gene_tier = "gdsc"
            stats[f"roots_{gene_tier}"] += 1
            if gene_tier == "weak":
                continue
            feature_index = gene_to_feature.get(gene)
            if feature_index is None:
                stats["roots_unmapped"] += 1
                continue
            drug_roots.append((feature_index, gene, gene_tier))

        # 根按证据层级 + HGNC symbol 确定性排序；根优先于邻居。
        drug_roots.sort(key=lambda item: (0 if item[2] == "ot" else 1, item[1]))
        roots[compound_uid] = [item[0] for item in drug_roots]
        root_names[compound_uid] = [item[1] for item in drug_roots]
        if any(item[2] == "ot" for item in drug_roots):
            tiers[compound_uid] = "ot"
        elif drug_roots:
            tiers[compound_uid] = "gdsc_only"
        else:
            tiers[compound_uid] = "none"
    return roots, tiers, root_names, stats


def _build_adjacency(
    graph: dict[str, np.ndarray],
    *,
    min_score: float,
) -> tuple[list[str], list[dict[int, float]], np.ndarray]:
    genes = [str(gene) for gene in graph["genes"]]
    adjacency: list[dict[int, float]] = [{} for _ in genes]
    edge_index = graph["ppi_edges"]
    edge_score = graph["ppi_score"]
    for (left, right), score in zip(edge_index.T, edge_score, strict=False):
        a, b = int(left), int(right)
        if a == b:
            continue
        value = float(score)
        if value < float(min_score):
            continue
        adjacency[a][b] = max(adjacency[a].get(b, 0.0), value)
        adjacency[b][a] = max(adjacency[b].get(a, 0.0), value)
    degree = np.asarray([len(item) for item in adjacency], dtype=np.int64)
    return genes, adjacency, degree


def _select_nodes(
    root_indices: list[int],
    *,
    feature_genes: list[str],
    gene_to_feature: dict[str, int],
    gene_to_graph: dict[str, int],
    adjacency: list[dict[int, float]],
    graph_genes: list[str],
    max_neighbors_per_target: int,
    max_nodes_per_drug: int,
) -> tuple[list[int], list[int], bool]:
    """根 + top-k 一跳邻居；返回 ``(roots, neighbors, trimmed)``。"""

    root_set = set(root_indices)
    neighbor_scores: dict[int, float] = {}
    root_limit = max(0, int(max_nodes_per_drug))
    if len(root_indices) > root_limit:
        root_indices = root_indices[:root_limit]
        root_set = set(root_indices)
    available = max(0, root_limit - len(root_indices))

    for root in root_indices:
        graph_index = gene_to_graph.get(feature_genes[root])
        if graph_index is None:
            continue
        candidates = sorted(
            adjacency[graph_index].items(),
            key=lambda item: (-item[1], graph_genes[item[0]]),
        )
        count = 0
        if int(max_neighbors_per_target) <= 0:
            continue
        for neighbor, score in candidates:
            feature_index = gene_to_feature.get(graph_genes[neighbor], -1)
            if feature_index < 0 or feature_index in root_set:
                continue
            neighbor_scores[feature_index] = max(
                neighbor_scores.get(feature_index, 0.0), float(score)
            )
            count += 1
            if count >= int(max_neighbors_per_target):
                break

    ordered = sorted(
        neighbor_scores.items(),
        key=lambda item: (-item[1], feature_genes[item[0]]),
    )
    neighbors = [item[0] for item in ordered[:available]]
    trimmed = len(ordered) > available
    return root_indices, neighbors, trimmed


def _induced_edges(
    nodes: list[int],
    *,
    feature_genes: list[str],
    gene_to_graph: dict[str, int],
    adjacency: list[dict[int, float]],
    graph_genes: list[str],
) -> tuple[np.ndarray, np.ndarray, int]:
    """在选中节点上取 PPI 诱导子图；返回双向边、边权和唯一边数。"""

    local_of_graph: dict[int, int] = {}
    graph_nodes: set[int] = set()
    for local_index, feature_index in enumerate(nodes):
        graph_index = gene_to_graph.get(feature_genes[feature_index])
        if graph_index is None:
            continue
        local_of_graph[graph_index] = local_index
        graph_nodes.add(graph_index)
    weighted: dict[tuple[int, int], float] = {}
    for graph_index in graph_nodes:
        for neighbor, weight in adjacency[graph_index].items():
            if neighbor in graph_nodes and graph_index < neighbor:
                pair = (local_of_graph[graph_index], local_of_graph[neighbor])
                weighted[pair] = max(weighted.get(pair, 0.0), float(weight))
    if not weighted:
        return (
            np.zeros((2, 0), dtype=np.int64),
            np.zeros(0, dtype=np.float32),
            0,
        )
    ordered = sorted(weighted.items())
    edge_index = np.asarray(
        [[left for (left, _), _ in ordered], [right for (_, right), _ in ordered]],
        dtype=np.int64,
    )
    edge_weight = np.asarray([weight for _, weight in ordered], dtype=np.float32)
    return (
        np.concatenate([edge_index, edge_index[::-1]], axis=1),
        np.concatenate([edge_weight, edge_weight]),
        len(ordered),
    )


def _empty_arrays() -> dict[str, np.ndarray]:
    return {
        "node_ptr": np.zeros(1, dtype=np.int64),
        "node_gene": np.zeros(0, dtype=np.int64),
        "node_root": np.zeros(0, dtype=bool),
        "edge_ptr": np.zeros(1, dtype=np.int64),
        "edge_index": np.zeros((2, 0), dtype=np.int64),
        "edge_weight": np.zeros(0, dtype=np.float32),
        "coverage": np.zeros(0, dtype=bool),
        "coverage_ot": np.zeros(0, dtype=bool),
        "tier": np.zeros(0, dtype=np.str_),
        "n_roots": np.zeros(0, dtype=np.int64),
        "n_neighbors": np.zeros(0, dtype=np.int64),
        "rewire_effective": np.zeros(0, dtype=bool),
        "pseudo_exact": np.zeros(0, dtype=bool),
        "trimmed": np.zeros(0, dtype=bool),
    }


def _assemble(
    per_drug: list[dict[str, Any]],
    *,
    compound_order: list[str],
    feature_genes: list[str],
    gene_to_graph: dict[str, int],
    adjacency: list[dict[int, float]],
    graph_genes: list[str],
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    if len(per_drug) != len(compound_order):
        raise ValueError(f"per_drug 长度 {len(per_drug)} != 药物数 {len(compound_order)}")
    arrays = _empty_arrays()
    node_ptr = [0]
    edge_ptr = [0]
    node_gene: list[int] = []
    node_root: list[bool] = []
    edge_index: list[list[int]] = [[], []]
    edge_weight: list[float] = []
    audit_rows: list[dict[str, Any]] = []

    for compound_uid, item in zip(compound_order, per_drug, strict=False):
        roots = [int(value) for value in item.get("roots", [])]
        neighbors = [int(value) for value in item.get("neighbors", [])]
        # 去重且根在前；保持确定性。
        seen: set[int] = set()
        nodes: list[int] = []
        for value in [*roots, *neighbors]:
            if value not in seen:
                seen.add(value)
                nodes.append(value)
        edges, weights, n_edges = _induced_edges(
            nodes,
            feature_genes=feature_genes,
            gene_to_graph=gene_to_graph,
            adjacency=adjacency,
            graph_genes=graph_genes,
        )
        node_start = node_ptr[-1]
        node_gene.extend(nodes)
        node_root.extend([index < len(roots) for index in range(len(nodes))])
        node_ptr.append(node_start + len(nodes))
        edge_index[0].extend(edges[0].tolist())
        edge_index[1].extend(edges[1].tolist())
        edge_weight.extend(weights.tolist())
        edge_ptr.append(edge_ptr[-1] + int(edges.shape[1]))
        tier = str(item.get("tier", "none"))
        arrays["coverage"] = np.append(arrays["coverage"], tier in {"ot", "gdsc_only"})
        arrays["coverage_ot"] = np.append(arrays["coverage_ot"], tier == "ot")
        arrays["tier"] = np.append(arrays["tier"], np.asarray(tier, dtype=np.str_))
        arrays["n_roots"] = np.append(arrays["n_roots"], len(roots))
        arrays["n_neighbors"] = np.append(arrays["n_neighbors"], len(nodes) - len(roots))
        arrays["pseudo_exact"] = np.append(
            arrays["pseudo_exact"], bool(item.get("pseudo_exact", False))
        )
        arrays["trimmed"] = np.append(arrays["trimmed"], bool(item.get("trimmed", False)))
        audit_rows.append(
            {
                "compound_uid": compound_uid,
                "tier": tier,
                "n_roots": len(roots),
                "n_neighbors": len(nodes) - len(roots),
                "n_nodes": len(nodes),
                "n_edges": n_edges,
                "edge_weight_mean": float(weights.mean()) if weights.size else 0.0,
                "edge_weight_max": float(weights.max()) if weights.size else 0.0,
                "root_genes": ";".join(feature_genes[index] for index in roots),
                "neighbor_genes": ";".join(feature_genes[index] for index in nodes[len(roots) :]),
                "trimmed": bool(item.get("trimmed", False)),
                "pseudo_exact": bool(item.get("pseudo_exact", False)),
            }
        )

    arrays["node_ptr"] = np.asarray(node_ptr, dtype=np.int64)
    arrays["node_gene"] = np.asarray(node_gene, dtype=np.int64)
    arrays["node_root"] = np.asarray(node_root, dtype=bool)
    arrays["edge_ptr"] = np.asarray(edge_ptr, dtype=np.int64)
    arrays["edge_index"] = (
        np.asarray(edge_index, dtype=np.int64)
        if edge_index[0]
        else np.zeros((2, 0), dtype=np.int64)
    )
    arrays["edge_weight"] = np.asarray(edge_weight, dtype=np.float32)
    arrays["rewire_effective"] = np.zeros(len(compound_order), dtype=bool)
    return arrays, pd.DataFrame(audit_rows)


def _build_true_nodes(
    *,
    compound_order: list[str],
    roots: dict[str, list[int]],
    tiers: dict[str, str],
    feature_genes: list[str],
    gene_to_feature: dict[str, int],
    gene_to_graph: dict[str, int],
    adjacency: list[dict[int, float]],
    graph_genes: list[str],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    per_drug: list[dict[str, Any]] = []
    for compound_uid in compound_order:
        root_indices = roots.get(compound_uid, [])
        selected_roots, neighbors, trimmed = _select_nodes(
            root_indices,
            feature_genes=feature_genes,
            gene_to_feature=gene_to_feature,
            gene_to_graph=gene_to_graph,
            adjacency=adjacency,
            graph_genes=graph_genes,
            max_neighbors_per_target=int(config["max_neighbors_per_target"]),
            max_nodes_per_drug=int(config["max_nodes_per_drug"]),
        )
        per_drug.append(
            {
                "roots": selected_roots,
                "neighbors": neighbors,
                "tier": tiers.get(compound_uid, "none"),
                "trimmed": trimmed,
            }
        )
    return per_drug


def _pseudo_neighbor_matches(
    true_arrays: dict[str, np.ndarray],
    *,
    compound_order: list[str],
    feature_genes: list[str],
    gene_to_graph: dict[str, int],
    graph_genes: list[str],
    adjacency: list[dict[int, float]],
    degree: np.ndarray,
    feature_valid: dict[str, np.ndarray],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """保留根，用匹配全局度数与可观测性的随机基因替换邻居。"""

    rng = np.random.default_rng(int(config["seed"]) + 17)
    # 只在图基因空间里抽样；所有图基因都在 feature 空间内。
    graph_feature: list[int] = [i for i, gene in enumerate(feature_genes) if gene in gene_to_graph]
    graph_feature = np.asarray(graph_feature, dtype=np.int64)
    if graph_feature.size == 0:
        return []
    degree_values = degree[np.asarray([gene_to_graph[feature_genes[i]] for i in graph_feature])]
    quantiles = np.quantile(degree_values, np.linspace(0.0, 1.0, 11))
    bins = np.searchsorted(quantiles, degree_values, side="right") - 1
    bins = np.clip(bins, 0, 9)
    availability = np.stack(
        [
            feature_valid["ge"][graph_feature],
            feature_valid["cnv"][graph_feature],
            feature_valid["mu"][graph_feature],
        ],
        axis=1,
    )
    # key = (ge, cnv, mu, degree_bin) -> candidate feature indices
    pools: dict[tuple[int, int, int, int], list[int]] = {}
    for feature_index, bin_index, avail in zip(graph_feature, bins, availability, strict=False):
        key = (int(avail[0]), int(avail[1]), int(avail[2]), int(bin_index))
        pools.setdefault(key, []).append(int(feature_index))

    per_drug: list[dict[str, Any]] = []
    for drug_index in range(len(compound_order)):
        node_start = int(true_arrays["node_ptr"][drug_index])
        node_end = int(true_arrays["node_ptr"][drug_index + 1])
        nodes = true_arrays["node_gene"][node_start:node_end].tolist()
        is_root = true_arrays["node_root"][node_start:node_end].tolist()
        roots = [node for node, flag in zip(nodes, is_root, strict=False) if flag]
        true_neighbors = [node for node, flag in zip(nodes, is_root, strict=False) if not flag]
        if not true_neighbors:
            per_drug.append(
                {
                    "roots": roots,
                    "neighbors": [],
                    "tier": str(true_arrays["tier"][drug_index]),
                    "trimmed": bool(true_arrays["trimmed"][drug_index]),
                    "pseudo_exact": True,
                }
            )
            continue

        selected = set(roots)
        exact = True
        for old in true_neighbors:
            key_base = (
                int(feature_valid["ge"][old]),
                int(feature_valid["cnv"][old]),
                int(feature_valid["mu"][old]),
            )
            old_graph = gene_to_graph.get(feature_genes[old])
            old_bin = 0
            if old_graph is not None:
                old_bin = int(np.searchsorted(quantiles, degree[old_graph], side="right") - 1)
                old_bin = int(np.clip(old_bin, 0, 9))
            candidates = [
                value
                for value in pools.get((*key_base, old_bin), [])
                if value not in selected and value not in true_neighbors
            ]
            if not candidates:
                # 先放宽度数 bin，再放宽可观测性，但记录为不完全匹配。
                for delta in range(1, 10):
                    for candidate_bin in (old_bin - delta, old_bin + delta):
                        if 0 <= candidate_bin <= 9:
                            candidates = [
                                value
                                for value in pools.get((*key_base, candidate_bin), [])
                                if value not in selected and value not in true_neighbors
                            ]
                        if candidates:
                            break
                    if candidates:
                        break
                exact = False
            if not candidates:
                candidates = [
                    int(value)
                    for value in graph_feature
                    if int(value) not in selected and int(value) not in true_neighbors
                ]
                exact = False
            if not candidates:
                continue
            chosen = int(rng.choice(candidates))
            selected.add(chosen)
        per_drug.append(
            {
                "roots": roots,
                "neighbors": [value for value in sorted(selected) if value not in set(roots)],
                "tier": str(true_arrays["tier"][drug_index]),
                "trimmed": bool(true_arrays["trimmed"][drug_index]),
                "pseudo_exact": exact,
            }
        )
    return per_drug


def _rewire_degree(
    arrays: dict[str, np.ndarray],
    *,
    compound_order: list[str],
    seed: int,
) -> dict[str, np.ndarray]:
    """逐药保度双边交换；只改边，不改节点集和边数。"""

    result = {key: value.copy() for key, value in arrays.items()}
    result["rewire_effective"] = np.zeros(len(compound_order), dtype=bool)
    edge_index = result["edge_index"]
    edge_weight = result["edge_weight"]
    for drug_index in range(len(compound_order)):
        start = int(result["edge_ptr"][drug_index])
        end = int(result["edge_ptr"][drug_index + 1])
        if end - start < 4:
            continue
        block = edge_index[:, start:end]
        block_weight = edge_weight[start:end]
        pairs = {tuple(sorted((int(a), int(b)))) for a, b in block.T}
        pairs = {pair for pair in pairs if pair[0] != pair[1]}
        if len(pairs) < 2:
            continue
        pair_weight: dict[tuple[int, int], float] = {}
        for (a, b), weight in zip(block.T.tolist(), block_weight.tolist(), strict=False):
            pair = tuple(sorted((int(a), int(b))))
            pair_weight.setdefault(pair, float(weight))
        edge_list = sorted(pairs)
        rng = np.random.default_rng(int(seed) + drug_index)
        done = 0
        max_tries = max(len(edge_list) * 20, 100)
        for _ in range(max_tries):
            i, j = rng.integers(0, len(edge_list), size=2)
            if i == j:
                continue
            a, b = edge_list[i]
            c, d = edge_list[j]
            if len({a, b, c, d}) < 4:
                continue
            first = tuple(sorted((a, d)))
            second = tuple(sorted((c, b)))
            if first in pairs or second in pairs:
                first = tuple(sorted((a, c)))
                second = tuple(sorted((b, d)))
            if first in pairs or second in pairs:
                continue
            pairs.remove((a, b))
            pairs.remove((c, d))
            pairs.add(first)
            pairs.add(second)
            edge_list[i] = first
            edge_list[j] = second
            done += 1
        if done == 0:
            continue
        ordered = sorted(pairs)
        local = np.asarray(
            [[left for left, _ in ordered], [right for _, right in ordered]], dtype=np.int64
        )
        edge_index[:, start:end] = np.concatenate([local, local[::-1]], axis=1)
        # 重连保留节点集、度数和边权多重集；权重与具体连边的对应关系被打乱，
        # 避免真实 STRING 分数继续附着在新拓扑上。
        weight_values = np.asarray(
            [pair_weight.get(pair, 0.0) for pair in sorted(pair_weight)], dtype=np.float32
        )
        shuffled = np.random.default_rng(int(seed) + drug_index + 7919).permutation(weight_values)
        edge_weight[start:end] = np.concatenate([shuffled, shuffled])
        result["rewire_effective"][drug_index] = True
    return result


def _cache_root(processed: Path, cache_dir: str | Path | None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir)
    return processed.parent / "interim" / "teacher" / "local_graph"


def build_local_graphs(
    processed_dir: str | Path,
    features: CellFeatureSet,
    drugs: DrugFeatureSet,
    *,
    config: dict[str, Any] | None = None,
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> LocalGraphBundle:
    """构建/加载每药一跳局部图。

    返回的 ``arrays`` 以药物为单位扁平存储；``node_ptr`` / ``edge_ptr``
    给出每药节点和边的区间。细胞特征不在这里读取，避免把图拓扑和某一折
    的训练统计绑定。
    """

    processed = Path(processed_dir)
    cfg = _resolve_config(config)
    dti_path = processed / "dti.parquet"
    graph_path = processed / "graph_inputs.npz"
    if not dti_path.is_file():
        raise FileNotFoundError(f"缺少 {dti_path}")
    if not graph_path.is_file():
        raise FileNotFoundError(f"缺少 {graph_path}")

    dti_sha = _sha256_file(dti_path)
    graph_sha = _sha256_file(graph_path)
    gene_space_sha = _sha256_bytes("\n".join(features.gene_ids).encode("utf-8"))
    gene_is_target = np.zeros(len(features.gene_ids), dtype=bool)
    for indices in drugs.target_indices:
        gene_is_target[np.asarray(indices, dtype=np.int64)] = True
    target_mask_sha = _sha256_bytes(np.ascontiguousarray(gene_is_target, dtype=np.bool_).tobytes())
    feature_valid_sha = _sha256_bytes(
        b"".join(
            np.ascontiguousarray(array, dtype=np.bool_).tobytes()
            for array in (
                features.gene_ge_valid,
                features.gene_cnv_valid,
                features.gene_mu_available,
            )
        )
    )
    spec_core = {
        "version": int(cfg["version"]),
        "ppi_min_score": float(cfg["ppi_min_score"]),
        "max_neighbors_per_target": int(cfg["max_neighbors_per_target"]),
        "max_nodes_per_drug": int(cfg["max_nodes_per_drug"]),
        "neighbor_mode": str(cfg["neighbor_mode"]),
        "rewire": str(cfg["rewire"]),
        "seed": int(cfg["seed"]),
        "coverage": str(cfg["coverage"]),
        "dti_sha256": dti_sha,
        "graph_inputs_sha256": graph_sha,
        "gene_space_sha256": gene_space_sha,
        "target_mask_sha256": target_mask_sha,
        "feature_valid_sha256": feature_valid_sha,
        "code_version": "local_graph_v3",
    }
    cache_hash = _sha256_payload(spec_core)
    cache_root = _cache_root(processed, cache_dir) / cache_hash[:16]
    cache_graph = cache_root / "graphs.npz"
    cache_spec = cache_root / "spec.json"
    cache_audit = cache_root / "audit.json"
    if not force and cache_graph.is_file() and cache_spec.is_file():
        with np.load(cache_graph, allow_pickle=False) as payload:
            arrays = {key: payload[key] for key in payload.files}
        spec_payload = json.loads(cache_spec.read_text(encoding="utf-8"))
        if spec_payload.get("hash") == cache_hash:
            audit = (
                pd.read_json(cache_audit, orient="records")
                if cache_audit.is_file()
                else pd.DataFrame()
            )
            return LocalGraphBundle(arrays=arrays, spec=spec_payload, audit=audit)

    graph = load_graph_inputs(processed)
    dti = _load_dti(processed)
    graph_genes, adjacency, degree = _build_adjacency(graph, min_score=float(cfg["ppi_min_score"]))
    gene_to_graph = {gene: index for index, gene in enumerate(graph_genes)}
    gene_to_feature = {gene: index for index, gene in enumerate(features.gene_ids)}
    roots, tiers, root_names, root_stats = _root_table(
        dti, compound_order=drugs.compound_uid, gene_to_feature=gene_to_feature
    )
    per_drug = _build_true_nodes(
        compound_order=drugs.compound_uid,
        roots=roots,
        tiers=tiers,
        feature_genes=features.gene_ids,
        gene_to_feature=gene_to_feature,
        gene_to_graph=gene_to_graph,
        adjacency=adjacency,
        graph_genes=graph_genes,
        config=cfg,
    )
    arrays, audit = _assemble(
        per_drug,
        compound_order=drugs.compound_uid,
        feature_genes=features.gene_ids,
        gene_to_graph=gene_to_graph,
        adjacency=adjacency,
        graph_genes=graph_genes,
    )
    if str(cfg["neighbor_mode"]) == "pseudo":
        feature_valid = {
            "ge": features.gene_ge_valid,
            "cnv": features.gene_cnv_valid,
            "mu": features.gene_mu_available,
        }
        pseudo = _pseudo_neighbor_matches(
            arrays,
            compound_order=drugs.compound_uid,
            feature_genes=features.gene_ids,
            gene_to_graph=gene_to_graph,
            graph_genes=graph_genes,
            adjacency=adjacency,
            degree=degree,
            feature_valid=feature_valid,
            config=cfg,
        )
        arrays, audit = _assemble(
            pseudo,
            compound_order=drugs.compound_uid,
            feature_genes=features.gene_ids,
            gene_to_graph=gene_to_graph,
            adjacency=adjacency,
            graph_genes=graph_genes,
        )
    if str(cfg["rewire"]) == "degree":
        arrays = _rewire_degree(arrays, compound_order=drugs.compound_uid, seed=int(cfg["seed"]))
    if str(cfg["coverage"]) == "ot":
        arrays["coverage"] = arrays["coverage_ot"].copy()
    arrays["gene_is_target"] = gene_is_target

    spec = {
        **spec_core,
        "hash": cache_hash,
        "cache_dir": str(cache_root),
        "n_drugs": len(drugs.compound_uid),
        "n_covered": int(arrays["coverage"].sum()),
        "n_covered_ot": int(arrays["coverage_ot"].sum()),
        "n_target_genes": int(gene_is_target.sum()),
        "n_nodes": int(arrays["node_gene"].size),
        "n_edges": int(arrays["edge_index"].shape[1] // 2),
        "coverage_counts": {
            "ot": int((arrays["tier"] == "ot").sum()),
            "gdsc_only": int((arrays["tier"] == "gdsc_only").sum()),
            "none": int((arrays["tier"] == "none").sum()),
        },
        "root_stats": root_stats,
        "rewire_effective": int(arrays["rewire_effective"].sum()),
        "pseudo_exact": int(arrays["pseudo_exact"].sum()),
        "trimmed": int(arrays["trimmed"].sum()),
        "root_names": root_names,
    }
    cache_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_graph, **arrays)
    cache_spec.write_text(
        json.dumps(spec, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    audit.to_json(cache_audit, orient="records", force_ascii=False, indent=2)
    return LocalGraphBundle(arrays=arrays, spec=spec, audit=audit)


def local_graph_tensors(
    bundle: LocalGraphBundle | dict[str, np.ndarray],
    *,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """把 numpy 数组转成模型可缓存的张量。"""

    arrays = bundle.arrays if isinstance(bundle, LocalGraphBundle) else bundle
    tensors: dict[str, torch.Tensor] = {}
    for key, dtype in (
        ("node_ptr", torch.long),
        ("node_gene", torch.long),
        ("node_root", torch.bool),
        ("edge_ptr", torch.long),
        ("edge_index", torch.long),
        ("edge_weight", torch.float32),
        ("coverage", torch.bool),
        ("coverage_ot", torch.bool),
        ("n_roots", torch.long),
        ("n_neighbors", torch.long),
        ("rewire_effective", torch.bool),
        ("pseudo_exact", torch.bool),
        ("trimmed", torch.bool),
        ("gene_is_target", torch.bool),
    ):
        value = np.asarray(arrays[key])
        tensors[key] = torch.as_tensor(value, dtype=dtype, device=device)
    return tensors


__all__ = [
    "DEFAULT_LOCAL_GRAPH_CONFIG",
    "LocalGraphBundle",
    "build_local_graphs",
    "local_graph_tensors",
]

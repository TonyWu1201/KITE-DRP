"""H3CDR 式细胞系/药物相似性（fold-safe 改造）。

原文（Gu et al., IEEE/ACM TCBB 2026, §III-B）：

- 每个组学模态算高斯核 ``E_ij = exp(-||x_i-x_j||^2 / (2 sigma^2))``；
- 归一化全核 ``F_ij = E_ij / (2 sum_j E_ij)``，对角 1/2；
- top-N 稀疏 ``S_ij = E_ij / sum_{j in N_i} E_ij``（N=15）；
- 三模态交叉迭代 ``F_t^q = S^q · mean_{q'!=q} F_{t-1}^{q'} · (S^q)^T``（t=3），
  再取 ``C = mean_q F_t^q``；
- 药物相似性 = 子结构指纹 Tanimoto。

我们的改造（ADR-0005 / 计划书 D13、E-sim）：

- **fit 只在训练细胞系**，val/test 只算到 train 锚点的核；
- sigma 默认 median heuristic（1000 维通路分数下固定 sigma=2 会数值下溢）；
- 只输出 top-N 邻居索引/权重，不落 n×n 稠密矩阵；
- 药物相似性用 Morgan Tanimoto（与教师/学生药物编码一致）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kite_drp.data import io


@dataclass(frozen=True)
class SimilarityConfig:
    modality_names: tuple[str, ...] = ("GE", "MU")
    sigma: float | str = "auto"  # auto = median pairwise distance
    top_n: int = 15
    iterations: int = 3
    eps: float = 1e-8

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> SimilarityConfig:
        payload = payload or {}
        sigma = payload.get("sigma", "auto")
        if isinstance(sigma, str) and sigma != "auto":
            sigma = float(sigma)
        return cls(
            modality_names=tuple(payload.get("modalities", ["GE", "MU"])),
            sigma=sigma,
            top_n=int(payload.get("top_n", 15)),
            iterations=int(payload.get("iterations", 3)),
            eps=float(payload.get("eps", 1e-8)),
        )


def _pairwise_sq_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left_norm = np.sum(left * left, axis=1, keepdims=True)
    right_norm = np.sum(right * right, axis=1, keepdims=True).T
    distances = left_norm + right_norm - 2.0 * left @ right.T
    return np.maximum(distances, 0.0)


def _auto_sigma(distances: np.ndarray, eps: float) -> float:
    if distances.size == 0:
        return 1.0
    positive = distances[distances > 0]
    if positive.size == 0:
        return 1.0
    value = float(np.median(positive))
    return max(value, eps)


def gaussian_kernel(
    left: np.ndarray, right: np.ndarray, *, sigma: float | str = "auto", eps: float = 1e-8
) -> np.ndarray:
    distances = _pairwise_sq_distances(left, right)
    value = _auto_sigma(distances, eps) if sigma == "auto" else float(sigma)
    return np.exp(-distances / (2.0 * value * value)).astype(np.float32)


def normalized_full_kernel(kernel: np.ndarray) -> np.ndarray:
    """H3CDR 式 (2)：行归一化、对角 1/2。"""
    kernel = np.asarray(kernel, dtype=np.float32)
    row_sum = kernel.sum(axis=1, keepdims=True)
    result = kernel / (2.0 * np.maximum(row_sum, 1e-12))
    np.fill_diagonal(result, 0.5)
    return result


def top_n_sparse(kernel: np.ndarray, top_n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 ``(S, indices, weights)``；S 行内 top-N 归一化，排除自身。"""
    n = kernel.shape[0]
    k = min(int(top_n), max(n - 1, 1))
    work = np.asarray(kernel, dtype=np.float32).copy()
    np.fill_diagonal(work, -np.inf)
    indices = np.argpartition(-work, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(n)[:, None]
    values = work[rows, indices]
    values = np.maximum(values, 0.0)
    denom = values.sum(axis=1, keepdims=True)
    weights = np.divide(values, denom, out=np.zeros_like(values), where=denom > 0)
    sparse = np.zeros_like(work, dtype=np.float32)
    sparse[rows, indices] = weights
    return sparse, indices.astype(np.int32), weights.astype(np.float32)


def top_n_anchors(
    kernel: np.ndarray, top_n: int
) -> tuple[np.ndarray, np.ndarray]:
    """矩形核（query × anchors）的 top-N 邻居，不做对角排除。"""
    kernel = np.asarray(kernel, dtype=np.float32)
    n_rows, n_cols = kernel.shape
    if n_rows == 0 or n_cols == 0:
        return (
            np.zeros((n_rows, 0), dtype=np.int32),
            np.zeros((n_rows, 0), dtype=np.float32),
        )
    k = min(int(top_n), n_cols)
    indices = np.argpartition(-kernel, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(n_rows)[:, None]
    values = np.maximum(kernel[rows, indices], 0.0)
    denom = values.sum(axis=1, keepdims=True)
    weights = np.divide(values, denom, out=np.zeros_like(values), where=denom > 0)
    return indices.astype(np.int32), weights.astype(np.float32)


def fuse_similarity(
    kernels: list[np.ndarray], *, top_n: int, iterations: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """H3CDR 式 (4)–(7)：多模态交叉迭代融合，返回 ``(C, neighbors, weights)``。"""
    if not kernels:
        raise ValueError("至少需要一个模态的核矩阵")
    full = [normalized_full_kernel(kernel) for kernel in kernels]
    sparse = [top_n_sparse(kernel, top_n)[0] for kernel in kernels]
    for _ in range(max(int(iterations), 1)):
        updated = []
        for index, sparse_q in enumerate(sparse):
            others = [
                full[other] for other in range(len(full)) if other != index
            ]
            mean_others = np.mean(others, axis=0) if others else full[index]
            updated.append(sparse_q @ mean_others @ sparse_q.T)
        full = updated
    fused = np.mean(full, axis=0)
    fused = (fused + fused.T) / 2.0
    np.fill_diagonal(fused, 0.0)
    _, neighbors, weights = top_n_sparse(fused, top_n)
    return fused.astype(np.float32), neighbors, weights


def fit_cell_similarity(
    train_scores: dict[str, np.ndarray],
    query_scores: dict[str, np.ndarray] | None = None,
    *,
    config: SimilarityConfig,
) -> dict[str, Any]:
    """fold-safe 细胞系相似性。

    ``train_scores[q]``: (n_train, n_features)；``query_scores[q]``: (n_query, n_features)。
    返回：
    - ``train_neighbors/train_weights``: (n_train, N)，排除自身；
    - ``query_neighbors/query_weights``: (n_query, N)，锚点只来自 train；
    - ``sigma`` / ``diagnostics``。
    """
    names = [name for name in config.modality_names if name in train_scores]
    if not names:
        raise KeyError(f"缺少相似性模态：{config.modality_names}")
    sigmas: dict[str, float] = {}
    train_kernels = []
    for name in names:
        distances = _pairwise_sq_distances(train_scores[name], train_scores[name])
        sigma = _auto_sigma(distances, config.eps) if config.sigma == "auto" else float(config.sigma)
        sigmas[name] = sigma
        train_kernels.append(
            np.exp(-distances / (2.0 * sigma * sigma)).astype(np.float32)
        )
    fused, neighbors, weights = fuse_similarity(
        train_kernels, top_n=config.top_n, iterations=config.iterations
    )

    result: dict[str, Any] = {
        "train_neighbors": neighbors,
        "train_weights": weights,
        "sigma": sigmas,
        "config": {
            "modalities": names,
            "sigma": config.sigma,
            "top_n": config.top_n,
            "iterations": config.iterations,
        },
        "diagnostics": {
            "fused_mean": float(fused.mean()),
            "fused_max": float(fused.max()),
            "neighbor_weight_mean": float(weights.mean()),
        },
    }

    if query_scores is not None:
        query_kernels = []
        for name in names:
            distances = _pairwise_sq_distances(query_scores[name], train_scores[name])
            sigma = sigmas[name]
            query_kernels.append(
                np.exp(-distances / (2.0 * sigma * sigma)).astype(np.float32)
            )
        anchor_kernel = np.mean(query_kernels, axis=0)
        query_neighbors, query_weights = top_n_anchors(anchor_kernel, config.top_n)
        result["query_neighbors"] = query_neighbors
        result["query_weights"] = query_weights
    return result


def tanimoto_similarity(fingerprints: np.ndarray) -> np.ndarray:
    """Morgan 0/1 指纹的 Tanimoto（H3CDR 式 (8)）。"""
    matrix = np.asarray(fingerprints, dtype=np.float32)
    intersection = matrix @ matrix.T
    row_sum = matrix.sum(axis=1, keepdims=True)
    union = row_sum + row_sum.T - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def drug_similarity_neighbors(
    fingerprints: np.ndarray, *, top_n: int = 15
) -> dict[str, np.ndarray]:
    similarity = tanimoto_similarity(fingerprints)
    _, neighbors, weights = top_n_sparse(similarity, top_n)
    return {
        "similarity": similarity.astype(np.float32),
        "neighbors": neighbors,
        "weights": weights,
    }


def save_similarity(
    path: str | Path,
    *,
    cell_ids: list[str],
    is_train: np.ndarray,
    neighbors: np.ndarray,
    weights: np.ndarray,
    meta: dict[str, Any],
) -> Path:
    target = Path(path)
    io.write_npz(
        target,
        {
            "cell_ids": np.asarray(cell_ids, dtype=np.str_),
            "is_train": np.asarray(is_train, dtype=bool),
            "neighbors": np.asarray(neighbors, dtype=np.int32),
            "weights": np.asarray(weights, dtype=np.float32),
        },
    )
    io.write_json(target.with_suffix(".json"), meta)
    return target


def load_similarity(
    path: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    target = Path(path)
    arrays = io.read_npz(target)
    meta_path = target.with_suffix(".json")
    meta = io.read_json(meta_path) if meta_path.is_file() else {}
    return arrays, meta


__all__ = [
    "SimilarityConfig",
    "drug_similarity_neighbors",
    "fit_cell_similarity",
    "fuse_similarity",
    "gaussian_kernel",
    "load_similarity",
    "normalized_full_kernel",
    "save_similarity",
    "tanimoto_similarity",
    "top_n_sparse",
]

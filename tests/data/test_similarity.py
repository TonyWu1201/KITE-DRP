"""H3CDR 式相似性的正确性与 fold-safe 测试。"""

from __future__ import annotations

import numpy as np

from kite_drp.data.similarity import (
    SimilarityConfig,
    fit_cell_similarity,
    normalized_full_kernel,
    tanimoto_similarity,
    top_n_sparse,
)
from kite_drp.train.features import resolve_neighbor_source, resolve_similarity_config


def _data(seed: int = 0, n: int = 24, d: int = 32):
    rng = np.random.default_rng(seed)
    return {
        "GE": rng.normal(size=(n, d)).astype(np.float32),
        "MU": rng.normal(size=(n, d)).astype(np.float32),
    }


def test_normalized_kernel_diagonal_and_rows() -> None:
    rng = np.random.default_rng(0)
    raw = rng.random((6, 6)).astype(np.float32)
    normalized = normalized_full_kernel(raw)
    assert np.allclose(np.diag(normalized), 0.5)
    assert np.all(normalized >= 0)


def test_top_n_sparse_excludes_self_and_normalizes() -> None:
    rng = np.random.default_rng(1)
    kernel = rng.random((10, 10)).astype(np.float32)
    sparse, indices, weights = top_n_sparse(kernel, 4)
    assert indices.shape == (10, 4)
    for row in range(10):
        assert row not in indices[row]
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert np.allclose(sparse.sum(axis=1), 1.0)


def test_fit_cell_similarity_shapes_and_query_anchors() -> None:
    train = _data(0, n=20)
    query = _data(1, n=5)
    config = SimilarityConfig(modality_names=("GE", "MU"), top_n=5, iterations=3)
    result = fit_cell_similarity(train, query, config=config)
    assert result["train_neighbors"].shape == (20, 5)
    assert result["query_neighbors"].shape == (5, 5)
    assert np.all(result["query_neighbors"] >= 0)
    assert np.all(result["query_neighbors"] < 20)
    assert np.allclose(result["train_weights"].sum(axis=1), 1.0)
    assert np.allclose(result["query_weights"].sum(axis=1), 1.0)


def test_fit_is_independent_of_query_data() -> None:
    """fold-safe：改 query 组学不得改变 train 邻居。"""
    train = _data(0, n=20)
    query_a = _data(1, n=5)
    query_b = _data(2, n=5)
    config = SimilarityConfig(top_n=5)
    first = fit_cell_similarity(train, query_a, config=config)
    second = fit_cell_similarity(train, query_b, config=config)
    assert np.array_equal(first["train_neighbors"], second["train_neighbors"])
    assert np.allclose(first["train_weights"], second["train_weights"])


def test_tanimoto_similarity_diagonal() -> None:
    rng = np.random.default_rng(3)
    fingerprints = rng.integers(0, 2, size=(8, 64)).astype(np.float32)
    similarity = tanimoto_similarity(fingerprints)
    assert np.allclose(np.diag(similarity), 1.0)
    assert np.allclose(similarity, similarity.T)


def test_similarity_follows_active_modalities() -> None:
    config = {"similarity": {"modalities": ["GE", "MU"], "top_n": 15}}
    assert resolve_similarity_config(config, ["GE", "MU", "CNV"]).modality_names == (
        "GE",
        "MU",
    )
    assert resolve_similarity_config(config, ["MU", "CNV"]).modality_names == ("MU",)
    assert resolve_similarity_config(config, ["GE"]).modality_names == ("GE",)


def test_neighbor_source_auto_uses_first_active_modality() -> None:
    config = {"neighbor_source": "auto"}
    assert resolve_neighbor_source(config, ["GE", "MU", "CNV"]) == "GE"
    assert resolve_neighbor_source(config, ["MU", "CNV"]) == "MU"
    assert resolve_neighbor_source({"neighbor_source": "MU"}, ["GE", "MU"]) == "MU"

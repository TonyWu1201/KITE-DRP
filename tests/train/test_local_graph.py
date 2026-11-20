"""每药一跳局部图的构图、裁剪、重连与伪邻居测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from kite_drp.train.features import build_cell_features, build_drug_features
from kite_drp.train.local_graph import (
    DEFAULT_LOCAL_GRAPH_CONFIG,
    build_local_graphs,
    local_graph_tensors,
)

pytestmark = pytest.mark.skipif(
    not Path("data/processed/dti.parquet").is_file()
    or not Path("data/processed/graph_inputs.npz").is_file(),
    reason="需要 data/processed 的 DTI/PPI 产物",
)


def _features_and_drugs():
    features = build_cell_features("data/processed", split_id="p2_ldo_s0", fold=0)
    drugs = build_drug_features("data/processed")
    return features, drugs


def _degree(arrays: dict[str, np.ndarray], drug: int) -> np.ndarray:
    start, end = int(arrays["edge_ptr"][drug]), int(arrays["edge_ptr"][drug + 1])
    n_nodes = int(arrays["node_ptr"][drug + 1] - arrays["node_ptr"][drug])
    edges = arrays["edge_index"][:, start:end]
    return np.bincount(edges.reshape(-1), minlength=n_nodes)


def test_local_graph_contract(tmp_path: Path) -> None:
    features, drugs = _features_and_drugs()
    bundle = build_local_graphs(
        "data/processed",
        features,
        drugs,
        cache_dir=tmp_path,
        force=True,
    )
    arrays = bundle.arrays
    n_drugs = len(drugs.compound_uid)
    assert arrays["node_ptr"].shape == (n_drugs + 1,)
    assert arrays["edge_ptr"].shape == (n_drugs + 1,)
    assert np.all(np.diff(arrays["node_ptr"]) >= 0)
    assert np.all(np.diff(arrays["edge_ptr"]) >= 0)
    assert arrays["node_gene"].size == arrays["node_ptr"][-1]
    assert arrays["edge_index"].shape[1] == arrays["edge_ptr"][-1]
    assert arrays["edge_weight"].shape == (arrays["edge_index"].shape[1],)
    assert float(arrays["edge_weight"].min()) >= 0.7 - 1e-6
    assert float(arrays["edge_weight"].max()) <= 1.0 + 1e-6
    assert arrays["gene_is_target"].shape == (len(features.gene_ids),)
    assert int(arrays["gene_is_target"].sum()) == 301
    assert arrays["coverage"].sum() == 213
    assert bundle.spec["coverage_counts"] == {"ot": 104, "gdsc_only": 109, "none": 73}
    # 根节点必须排在每个药物节点块的前部。
    root_counts = arrays["n_roots"]
    for drug in range(n_drugs):
        start, end = int(arrays["node_ptr"][drug]), int(arrays["node_ptr"][drug + 1])
        roots = arrays["node_root"][start:end]
        assert roots[: int(root_counts[drug])].all()
        assert not roots[int(root_counts[drug]) :].any()
    # 边索引必须落在本药节点块内。
    for drug in range(n_drugs):
        start, end = int(arrays["edge_ptr"][drug]), int(arrays["edge_ptr"][drug + 1])
        n_nodes = int(arrays["node_ptr"][drug + 1] - arrays["node_ptr"][drug])
        if end > start:
            edges = arrays["edge_index"][:, start:end]
            assert edges.min() >= 0
            assert edges.max() < n_nodes


def test_local_graph_is_deterministic(tmp_path: Path) -> None:
    features, drugs = _features_and_drugs()
    config = dict(DEFAULT_LOCAL_GRAPH_CONFIG)
    first = build_local_graphs(
        "data/processed", features, drugs, config=config, cache_dir=tmp_path / "a", force=True
    )
    second = build_local_graphs(
        "data/processed", features, drugs, config=config, cache_dir=tmp_path / "b", force=True
    )
    assert first.spec["hash"] == second.spec["hash"]
    for key in (
        "node_ptr",
        "node_gene",
        "node_root",
        "edge_ptr",
        "edge_index",
        "edge_weight",
        "coverage",
        "gene_is_target",
    ):
        np.testing.assert_array_equal(first.arrays[key], second.arrays[key])


def test_degree_rewire_preserves_degrees(tmp_path: Path) -> None:
    features, drugs = _features_and_drugs()
    base_config = {"max_nodes_per_drug": 32, "max_neighbors_per_target": 4}
    true = build_local_graphs(
        "data/processed",
        features,
        drugs,
        config=base_config,
        cache_dir=tmp_path / "true",
        force=True,
    )
    rewired = build_local_graphs(
        "data/processed",
        features,
        drugs,
        config={**base_config, "rewire": "degree", "seed": 7},
        cache_dir=tmp_path / "rewired",
        force=True,
    )
    assert rewired.spec["rewire_effective"] > 0
    np.testing.assert_array_equal(true.arrays["node_ptr"], rewired.arrays["node_ptr"])
    np.testing.assert_array_equal(true.arrays["edge_ptr"], rewired.arrays["edge_ptr"])
    for drug in range(len(drugs.compound_uid)):
        np.testing.assert_array_equal(_degree(true.arrays, drug), _degree(rewired.arrays, drug))
        start, end = int(true.arrays["edge_ptr"][drug]), int(true.arrays["edge_ptr"][drug + 1])
        np.testing.assert_allclose(
            np.sort(true.arrays["edge_weight"][start:end]),
            np.sort(rewired.arrays["edge_weight"][start:end]),
        )


def test_pseudo_neighbors_keep_roots(tmp_path: Path) -> None:
    features, drugs = _features_and_drugs()
    base = build_local_graphs(
        "data/processed",
        features,
        drugs,
        config={"max_nodes_per_drug": 32, "max_neighbors_per_target": 4},
        cache_dir=tmp_path / "true",
        force=True,
    )
    pseudo = build_local_graphs(
        "data/processed",
        features,
        drugs,
        config={
            "max_nodes_per_drug": 32,
            "max_neighbors_per_target": 4,
            "neighbor_mode": "pseudo",
            "seed": 11,
        },
        cache_dir=tmp_path / "pseudo",
        force=True,
    )
    np.testing.assert_array_equal(base.arrays["node_root"], pseudo.arrays["node_root"])
    np.testing.assert_array_equal(base.arrays["n_roots"], pseudo.arrays["n_roots"])
    assert pseudo.spec["pseudo_exact"] > 0
    assert int(pseudo.arrays["n_neighbors"].sum()) == int(base.arrays["n_neighbors"].sum())


def test_root_only_config_has_no_neighbors(tmp_path: Path) -> None:
    features, drugs = _features_and_drugs()
    bundle = build_local_graphs(
        "data/processed",
        features,
        drugs,
        config={"max_neighbors_per_target": 0, "max_nodes_per_drug": 64},
        cache_dir=tmp_path,
        force=True,
    )
    assert int(bundle.arrays["n_neighbors"].sum()) == 0
    assert int(bundle.arrays["node_root"].sum()) == int(bundle.arrays["node_gene"].size)
    assert bundle.arrays["edge_weight"].shape == (bundle.arrays["edge_index"].shape[1],)


def test_cache_hash_changes_with_caps(tmp_path: Path) -> None:
    features, drugs = _features_and_drugs()
    first = build_local_graphs(
        "data/processed",
        features,
        drugs,
        config={"max_nodes_per_drug": 64},
        cache_dir=tmp_path / "a",
        force=True,
    )
    second = build_local_graphs(
        "data/processed",
        features,
        drugs,
        config={"max_nodes_per_drug": 32},
        cache_dir=tmp_path / "b",
        force=True,
    )
    assert first.spec["hash"] != second.spec["hash"]
    assert first.spec["n_nodes"] != second.spec["n_nodes"]


def test_local_graph_tensors_roundtrip(tmp_path: Path) -> None:
    features, drugs = _features_and_drugs()
    bundle = build_local_graphs("data/processed", features, drugs, cache_dir=tmp_path, force=True)
    tensors = local_graph_tensors(bundle)
    assert tensors["node_ptr"].shape == (len(drugs.compound_uid) + 1,)
    assert tensors["edge_index"].shape[0] == 2
    assert tensors["edge_weight"].dtype == torch.float32
    assert tensors["edge_weight"].shape[0] == tensors["edge_index"].shape[1]
    assert tensors["gene_is_target"].dtype == torch.bool
    assert tensors["coverage"].dtype == torch.bool

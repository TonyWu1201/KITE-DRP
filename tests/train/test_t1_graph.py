"""T1 图构建的度保持重连与结构约束测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kite_drp.train.t1_graph import (
    _rewire_bipartite,
    _rewire_undirected,
    build_t1_graph,
    graph_tensors,
)


def _degrees(edge_index: np.ndarray, n_nodes: int) -> np.ndarray:
    return np.bincount(edge_index.reshape(-1), minlength=n_nodes)


def test_rewire_undirected_preserves_degrees() -> None:
    base = np.asarray([[0, 1, 2, 3, 4, 0], [1, 2, 3, 4, 0, 2]], dtype=np.int64)
    # 实际 PPI 边是双向存储；函数会折叠后重新输出双向边。
    edge_index = np.concatenate([base, base[::-1]], axis=1)
    rewired = _rewire_undirected(edge_index, seed=0, swaps=3)
    assert rewired.shape[0] == 2
    np.testing.assert_array_equal(_degrees(edge_index, 5), _degrees(rewired, 5))
    pairs = {tuple(sorted(pair)) for pair in rewired.T.tolist()}
    assert len(pairs) == rewired.shape[1] // 2
    assert all(a != b for a, b in pairs)


def test_rewire_bipartite_preserves_degrees() -> None:
    edge_index = np.asarray([[0, 0, 1, 1, 2, 2], [0, 1, 0, 2, 1, 2]], dtype=np.int64)
    rewired = _rewire_bipartite(edge_index, seed=0, swaps=2)
    assert rewired.shape[0] == 2
    np.testing.assert_array_equal(
        np.bincount(edge_index[0], minlength=3),
        np.bincount(rewired[0], minlength=3),
    )
    np.testing.assert_array_equal(
        np.bincount(edge_index[1], minlength=3),
        np.bincount(rewired[1], minlength=3),
    )


def test_graph_tensors_roundtrip() -> None:
    arrays = {
        "gene_ids": np.asarray(["A", "B"], dtype=np.str_),
        "cell_ids": np.asarray(["c0"], dtype=np.str_),
        "drug_uids": np.asarray(["d0"], dtype=np.str_),
        "gene_full_idx": np.asarray([0, 2], dtype=np.int64),
        "gene_stats": np.ones((2, 4), dtype=np.float32),
        "cell_features": np.ones((1, 3), dtype=np.float32),
        "drug_morgan": np.ones((1, 4), dtype=np.float32),
        "drug_target_ids": np.asarray([[0]], dtype=np.int64),
        "drug_target_mask": np.asarray([[True]]),
        "drug_target_ids_full": np.asarray([[0]], dtype=np.int64),
        "drug_target_mask_full": np.asarray([[True]]),
        "ppi_edge_index": np.asarray([[0], [1]], dtype=np.int64),
        "ppi_edge_attr": np.ones((1, 1), dtype=np.float32),
        "dti_edge_index": np.asarray([[0], [0]], dtype=np.int64),
        "dti_edge_attr": np.ones((1, 1), dtype=np.float32),
        "cg_edge_index": np.asarray([[0], [0]], dtype=np.int64),
        "cg_edge_attr": np.ones((1, 4), dtype=np.float32),
    }
    tensors = graph_tensors(arrays)
    assert tensors["gene_ids"] == ["A", "B"]
    assert tensors["gene_full_idx"].tolist() == [0, 2]
    assert tensors["cg_edge_attr"].shape == (1, 4)


pytestmark = pytest.mark.skipif(
    not Path("data/processed/teacher_inputs.npz").is_file(),
    reason="需要 data/processed 自包含教师输入",
)


def test_build_t1_graph_contract(tmp_path: Path) -> None:
    cfg = {
        "gene_space": {"hvg_top": 50, "mutation_top": 20},
        "cell_gene_top_q": 0.99,
        "max_targets_per_drug": 8,
    }
    bundle = build_t1_graph(
        "data/processed",
        split_id="p2_ldo_s0",
        fold=0,
        config=cfg,
        cache_dir=tmp_path,
    )
    arrays = bundle["arrays"]
    spec = bundle["spec"]
    n_genes = spec["n_genes"]
    n_cells = spec["n_cells"]
    n_drugs = spec["n_drugs"]

    assert list(arrays["gene_ids"]) == sorted(arrays["gene_ids"].tolist())
    assert len(set(arrays["gene_full_idx"].tolist())) == n_genes
    assert arrays["gene_full_idx"].min() >= 0
    assert arrays["cell_features"].shape[0] == n_cells
    assert arrays["drug_morgan"].shape[0] == n_drugs
    assert arrays["ppi_edge_index"].max() < n_genes
    assert arrays["dti_edge_index"][0].max() < n_genes
    assert arrays["dti_edge_index"][1].max() < n_drugs
    assert arrays["cg_edge_index"][0].max() < n_genes
    assert arrays["cg_edge_index"][1].max() < n_cells
    assert np.isfinite(arrays["cg_edge_attr"]).all()
    assert arrays["drug_target_ids_full"].shape[0] == n_drugs

    cached = build_t1_graph(
        "data/processed",
        split_id="p2_ldo_s0",
        fold=0,
        config=cfg,
        cache_dir=tmp_path,
    )
    assert cached["spec"]["spec_hash"] == spec["spec_hash"]
    assert cached["arrays"]["gene_full_idx"].tolist() == arrays["gene_full_idx"].tolist()

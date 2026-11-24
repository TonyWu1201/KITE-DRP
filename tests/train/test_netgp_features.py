"""NetGP 间接靶点特征：形状、排除根、确定性、对照与缓存。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kite_drp.train.features import build_cell_features, build_drug_features
from kite_drp.train.local_graph import build_local_graphs
from kite_drp.train.netgp_features import (
    DEFAULT_NETGP_CONFIG,
    build_netgp_features,
)

pytestmark = pytest.mark.skipif(
    not Path("data/processed/dti.parquet").is_file()
    or not Path("data/processed/graph_inputs.npz").is_file()
    or not Path("data/processed/student_inputs/pathway_weights.npz").is_file(),
    reason="需要 data/processed 的 DTI/PPI/通路产物",
)


def _features_and_drugs():
    features = build_cell_features("data/processed", split_id="p2_ldo_s0", fold=0)
    drugs = build_drug_features("data/processed")
    return features, drugs


def _config(**overrides):
    config = dict(DEFAULT_NETGP_CONFIG)
    config.update({"top_k": 8, "steps": 3, "pathway_ora_top_genes": 20})
    config.update(overrides)
    return config


def _build(tmp_path: Path, **overrides):
    features, drugs = _features_and_drugs()
    return build_netgp_features(
        "data/processed",
        features,
        drugs,
        config=_config(**overrides),
        cache_dir=tmp_path,
        force=True,
    )


def test_netgp_contract(tmp_path: Path) -> None:
    bundle = _build(tmp_path)
    n_drugs = len(bundle.coverage)
    assert bundle.gene_ids.shape == (n_drugs, 8)
    assert bundle.gene_weight.shape == (n_drugs, 8)
    assert bundle.pathway_profile.shape == (n_drugs, 1000)
    assert bundle.reliability.shape == (n_drugs, 8)
    assert bundle.meta.shape == (n_drugs, 8)
    assert int(bundle.coverage.sum()) == 213
    assert np.isfinite(bundle.gene_weight).all()
    assert np.isfinite(bundle.pathway_profile).all()
    # 未覆盖药零填充。
    assert float(np.abs(bundle.gene_weight[~bundle.coverage]).max()) == 0.0
    assert float(np.abs(bundle.pathway_profile[~bundle.coverage]).max()) == 0.0
    # 覆盖药权重和为 1（允许“有 DTI 但根不在 STRING 图中”的空选择）。
    totals = bundle.gene_weight.sum(axis=1)
    nonempty = bundle.coverage & (totals > 0)
    assert int(nonempty.sum()) > 0
    assert np.allclose(totals[nonempty], 1.0, atol=1e-5)
    assert bundle.audit.shape[0] == n_drugs


def test_netgp_excludes_direct_targets(tmp_path: Path) -> None:
    features, drugs = _features_and_drugs()
    bundle = _build(tmp_path)
    local = build_local_graphs(
        "data/processed",
        features,
        drugs,
        config={
            "ppi_min_score": 0.7,
            "neighbor_mode": "ppi",
            "max_neighbors_per_target": 8,
            "max_nodes_per_drug": 64,
            "rewire": "none",
            "coverage": "any",
            "seed": 20261123,
        },
        cache_dir=tmp_path / "local_graph",
    )
    node_ptr = local.arrays["node_ptr"]
    node_gene = local.arrays["node_gene"]
    node_root = local.arrays["node_root"]
    for drug in np.flatnonzero(bundle.coverage):
        start, end = int(node_ptr[drug]), int(node_ptr[drug + 1])
        roots = set(node_gene[start:end][node_root[start:end]].tolist())
        selected = set(bundle.gene_ids[drug][bundle.gene_weight[drug] > 0].tolist())
        assert not (roots & selected), f"drug {drug} 选到了直接靶点"


def test_netgp_is_deterministic(tmp_path: Path) -> None:
    first = _build(tmp_path / "a")
    second = _build(tmp_path / "b")
    assert first.spec["hash"] == second.spec["hash"]
    np.testing.assert_array_equal(first.gene_ids, second.gene_ids)
    np.testing.assert_array_equal(first.gene_weight, second.gene_weight)
    np.testing.assert_array_equal(first.pathway_profile, second.pathway_profile)


def test_netgp_permute_keeps_coverage_but_changes_profile(tmp_path: Path) -> None:
    true = _build(tmp_path / "true")
    permuted = _build(tmp_path / "permuted", dti_control="permute_drugs", seed=7)
    np.testing.assert_array_equal(true.coverage, permuted.coverage)
    np.testing.assert_array_equal(true.tier, permuted.tier)
    assert not np.array_equal(true.gene_ids[true.coverage], permuted.gene_ids[permuted.coverage])


def test_netgp_random_matched_changes_genes(tmp_path: Path) -> None:
    true = _build(tmp_path / "true")
    random = _build(tmp_path / "random", gene_control="random_matched", seed=11)
    np.testing.assert_array_equal(true.coverage, random.coverage)
    assert not np.array_equal(true.gene_ids[true.coverage], random.gene_ids[random.coverage])
    totals = random.gene_weight.sum(axis=1)
    nonempty = random.coverage & (totals > 0)
    assert np.all(totals[nonempty] <= 1.0 + 1e-5)


def test_netgp_cache_roundtrip(tmp_path: Path) -> None:
    features, drugs = _features_and_drugs()
    config = _config()
    first = build_netgp_features(
        "data/processed", features, drugs, config=config, cache_dir=tmp_path, force=True
    )
    second = build_netgp_features(
        "data/processed", features, drugs, config=config, cache_dir=tmp_path, force=False
    )
    assert first.spec["hash"] == second.spec["hash"]
    np.testing.assert_array_equal(first.gene_ids, second.gene_ids)
    assert second.spec["cache_dir"] == first.spec["cache_dir"]

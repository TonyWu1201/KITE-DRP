"""无图 DTI/PPI 特征：形状、覆盖、确定性、随机对照与缓存。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kite_drp.train.features import build_cell_features, build_drug_features
from kite_drp.train.ppi_features import (
    DEFAULT_PPI_FEATURE_CONFIG,
    build_ppi_features,
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


def _build(tmp_path: Path, *, fold_key: str = "", **overrides):
    features, drugs = _features_and_drugs()
    config = dict(DEFAULT_PPI_FEATURE_CONFIG)
    config.update(
        {
            "max_neighbors_per_target": 4,
            "max_nodes_per_drug": 32,
            "rwr_steps": 3,
        }
    )
    config.update(overrides)
    return build_ppi_features(
        "data/processed",
        features,
        drugs,
        config=config,
        cache_dir=tmp_path,
        fold_key=fold_key,
        force=True,
    )


def test_ppi_feature_contract(tmp_path: Path) -> None:
    bundle = _build(tmp_path)
    n_drugs = len(bundle.coverage)
    n_cells = bundle.summary.shape[1]
    assert bundle.summary.shape == (n_drugs, n_cells, bundle.summary_dim)
    assert bundle.rwr.shape == (n_drugs, n_cells, bundle.rwr_dim)
    assert bundle.target_pathway.shape == (n_drugs, 1000)
    assert bundle.neighbor_pathway.shape == (n_drugs, 1000)
    assert bundle.summary_dim == 38
    assert bundle.rwr_dim == 8
    assert bundle.coverage.sum() == 213
    assert np.isfinite(bundle.summary).all()
    assert np.isfinite(bundle.rwr).all()
    assert np.isfinite(bundle.target_pathway).all()
    # 无覆盖药物必须零填充。
    assert float(np.abs(bundle.summary[~bundle.coverage]).max()) == 0.0
    assert float(np.abs(bundle.rwr[~bundle.coverage]).max()) == 0.0
    assert float(np.abs(bundle.target_pathway[~bundle.coverage]).max()) == 0.0
    # 通路 profile 要么全零，要么 L1 归一化。
    for profile in (bundle.target_pathway, bundle.neighbor_pathway):
        totals = profile.sum(axis=1)
        assert np.allclose(totals[totals > 0], 1.0, atol=1e-5)
    assert bundle.audit.shape[0] == n_drugs
    assert "rwr_top_genes" in bundle.audit.columns


def test_ppi_features_are_deterministic(tmp_path: Path) -> None:
    first = _build(tmp_path / "a")
    second = _build(tmp_path / "b")
    assert first.spec["hash"] == second.spec["hash"]
    np.testing.assert_array_equal(first.summary, second.summary)
    np.testing.assert_array_equal(first.rwr, second.rwr)
    np.testing.assert_array_equal(first.target_pathway, second.target_pathway)


def test_pseudo_neighbors_keep_roots_but_change_features(tmp_path: Path) -> None:
    true = _build(tmp_path / "true")
    pseudo = _build(tmp_path / "pseudo", neighbor_mode="pseudo", seed=11)
    np.testing.assert_array_equal(true.coverage, pseudo.coverage)
    # 根不变；邻居特征应发生变化。
    assert not np.allclose(true.summary, pseudo.summary)
    assert not np.allclose(true.neighbor_pathway, pseudo.neighbor_pathway)


def test_permuted_random_control_preserves_coverage(tmp_path: Path) -> None:
    true = _build(tmp_path / "true")
    permuted = _build(tmp_path / "permuted", random_control="permute_drugs", seed=7)
    np.testing.assert_array_equal(true.coverage, permuted.coverage)
    np.testing.assert_array_equal(true.tier, permuted.tier)
    assert not np.allclose(true.summary[true.coverage], permuted.summary[permuted.coverage])


def test_fold_key_changes_cache_hash(tmp_path: Path) -> None:
    first = _build(tmp_path / "a", fold_key="p2_ldo_s0_fold0")
    second = _build(tmp_path / "b", fold_key="p2_ldo_s0_fold1")
    assert first.spec["hash"] != second.spec["hash"]


def test_ppi_cache_roundtrip(tmp_path: Path) -> None:
    features, drugs = _features_and_drugs()
    config = dict(DEFAULT_PPI_FEATURE_CONFIG)
    config.update({"max_neighbors_per_target": 4, "max_nodes_per_drug": 32})
    first = build_ppi_features(
        "data/processed", features, drugs, config=config, cache_dir=tmp_path, force=True
    )
    second = build_ppi_features(
        "data/processed", features, drugs, config=config, cache_dir=tmp_path, force=False
    )
    assert first.spec["hash"] == second.spec["hash"]
    np.testing.assert_array_equal(first.summary, second.summary)
    assert second.spec["cache_dir"] == first.spec["cache_dir"]

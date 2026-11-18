"""教师特征装配与相似性缓存的集成测试（依赖 processed 产物）。"""

from __future__ import annotations

import numpy as np
import pytest

from kite_drp.common.paths import REPO_ROOT
from kite_drp.data import io, registry
from kite_drp.data.similarity import SimilarityConfig
from kite_drp.train.features import (
    build_cell_features,
    build_drug_features,
    build_fold_similarity,
)

PROCESSED = REPO_ROOT / "data" / "processed"
SPLIT_ID = "p1_lco_s0"


def _require() -> None:
    if not registry.resolve("teacher_inputs").is_file():
        pytest.skip("未拉取 LFS processed 产物")
    if not (REPO_ROOT / "data" / "splits" / f"{SPLIT_ID}.json").is_file():
        pytest.skip("缺少 split 文件")


@pytest.mark.integration
def test_cell_features_are_finite_and_normalized() -> None:
    _require()
    features = build_cell_features(PROCESSED, split_id=SPLIT_ID, fold=0)
    assert features.ge.shape == features.mu.shape == features.cnv.shape
    assert features.ge.shape[1] == 1000
    for values in (features.ge, features.mu, features.cnv):
        assert np.isfinite(values).all()
    assert features.mu.min() >= 0.0 and features.mu.max() <= 1.0
    assert int(features.cnv_mask.sum()) == 509


@pytest.mark.integration
def test_similarity_is_fold_safe_and_cached(tmp_path) -> None:
    _require()
    features = build_cell_features(PROCESSED, split_id=SPLIT_ID, fold=0)
    split = io.read_json(REPO_ROOT / "data" / "splits" / f"{SPLIT_ID}.json")
    train = split["folds"][0]["train"]
    neighbors, weights, is_train, meta = build_fold_similarity(
        features,
        train_cells=train,
        config=SimilarityConfig(top_n=5),
        processed_dir=PROCESSED,
        split_id=SPLIT_ID,
        fold=0,
        cache_dir=tmp_path,
    )
    assert neighbors.shape == weights.shape == (len(features.cell_ids), 5)
    assert np.allclose(weights.sum(axis=1), 1.0)
    # train 行不含自身；query 行只连 train 锚点
    for row in np.flatnonzero(is_train):
        assert row not in neighbors[row]
    for row in np.flatnonzero(~is_train):
        assert is_train[neighbors[row]].all()
    assert meta["fold_safe"] is True
    assert (tmp_path / SPLIT_ID / "fold0" / "cell_similarity.npz").is_file()


@pytest.mark.integration
def test_drug_features_have_graphs() -> None:
    _require()
    drugs = build_drug_features(PROCESSED)
    assert drugs.morgan.shape == (286, 2048)
    assert drugs.target_dim == 18396
    assert drugs.graph_has_graph is not None and drugs.graph_has_graph.all()

"""LDO/scaffold 未见药物必须可评估（label_mask/cls_mask 全 True）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kite_drp.data.dataset import TeacherPairDataset

pytestmark = pytest.mark.skipif(
    not Path("data/processed/teacher_pairs.parquet").is_file(),
    reason="需要 data/processed 自包含 pair 表",
)


@pytest.mark.parametrize("split_id", ["p2_ldo_s0", "p3_scaffold_s0"])
def test_unseen_drug_labels_are_evaluable(split_id: str) -> None:
    dataset = TeacherPairDataset(
        "data/processed",
        split_id=split_id,
        fold=0,
        pool="M-716",
        split_dir="data/splits",
    )
    for split in ("val", "test"):
        indices = dataset.indices(split)
        assert indices.size > 0
        assert dataset.label_mask[indices].all()
        assert dataset.cls_mask[indices].all()
        assert np.isfinite(dataset.y[indices]).all()
        # 未见药物使用全局 fallback，不应全部为 0
        assert np.count_nonzero(dataset.y[indices]) > 0
    assert 0 < int(dataset.known_compound_mask.sum()) < len(dataset)

"""processed 自包含产物测试：保证 clone（无 raw/canonical）也能训练。

这些测试在真实产物存在时运行；CI 未拉 LFS 时自动跳过。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kite_drp.common.paths import REPO_ROOT
from kite_drp.data import io, registry
from kite_drp.data.dataset import TeacherPairDataset
from kite_drp.data.graph_inputs import validate_graph_inputs


def _artifact(name: str) -> Path:
    path = registry.resolve(name)
    if not path.is_file() or path.stat().st_size < 1024:
        pytest.skip(f"产物不存在或未拉取 LFS：{path}")
    return path


def test_teacher_pairs_schema_and_counts() -> None:
    pairs = pd.read_parquet(_artifact("teacher_pairs"))
    expected = {
        "cell_line_uid",
        "compound_uid",
        "ln_ic50",
        "auc",
        "drug_name",
        "oncotree_lineage",
        "has_teacher_omics",
        "in_main_pool",
    }
    assert expected <= set(pairs.columns)
    assert not pairs.duplicated(["cell_line_uid", "compound_uid"]).any()
    assert len(pairs) == 235478
    assert int(pairs["has_teacher_omics"].sum()) == 176487
    assert int(pairs["in_main_pool"].sum()) == 126803


def test_drug_features_include_molecular_graphs() -> None:
    arrays = io.read_npz(_artifact("drug_features_processed"))
    for key in ("graph_atom_ptr", "graph_edge_ptr", "graph_node_feat", "graph_edge_index"):
        assert key in arrays, f"缺少分子图字段：{key}"
    assert arrays["graph_node_feat"].shape[1] == 77
    assert arrays["graph_has_graph"].sum() == len(arrays["compound_uid"])
    ptr = arrays["graph_atom_ptr"]
    assert ptr[0] == 0 and np.all(np.diff(ptr) >= 0)
    assert ptr[-1] == arrays["graph_node_feat"].shape[0]
    edge_ptr = arrays["graph_edge_ptr"]
    assert edge_ptr[-1] == arrays["graph_edge_index"].shape[1]
    assert edge_ptr[-1] % 2 == 0  # 双向边


def test_graph_inputs_are_valid() -> None:
    arrays = io.read_npz(_artifact("graph_inputs"))
    problems = validate_graph_inputs(arrays)
    assert not problems, problems
    assert arrays["ppi_edges"].shape[1] > 100_000
    assert arrays["dti_edges"].shape[1] > 3_000


def test_processed_manifest_uses_portable_paths() -> None:
    payload = io.read_json(_artifact("processed_manifest"))
    for entry in payload["artifacts"]:
        path = Path(entry["path"])
        assert not path.is_absolute(), f"{entry['name']} 使用了绝对路径：{path}"
        assert (REPO_ROOT / path).is_file(), f"{entry['name']} 指向不存在的文件：{path}"


@pytest.mark.integration
def test_teacher_pair_dataset_runs_from_processed_only() -> None:
    if not registry.resolve("teacher_pairs").is_file():
        pytest.skip("未拉取 LFS processed 产物")
    dataset = TeacherPairDataset(
        REPO_ROOT / "data" / "processed",
        split_id="p1_lco_s0",
        fold=0,
        pool="M-716",
        split_dir=REPO_ROOT / "data" / "splits",
    )
    counts = {name: int((dataset.split_label == name).sum()) for name in ("train", "val", "test")}
    assert counts["train"] > 0 and counts["val"] > 0 and counts["test"] > 0
    sample = dataset[dataset.indices("train")[0]]
    assert sample["split"] == "train"
    assert np.isfinite(sample["y"])
    assert 0.0 <= sample["y_cls"] <= 1.0

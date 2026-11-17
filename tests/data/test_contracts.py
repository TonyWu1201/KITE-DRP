"""产物契约测试：真实产物存在时校验 schema；CI 无 raw 时跳过。"""

from __future__ import annotations

import pandas as pd
import pytest

from kite_drp.data import io, registry


def _artifact(name: str):
    path = registry.resolve(name)
    if not path.is_file():
        pytest.skip(f"产物不存在（需先运行数据管线）：{path}")
    return path


def test_labels_schema_and_uniqueness() -> None:
    labels = pd.read_parquet(_artifact("labels"))
    expected = {
        "cell_line_uid",
        "compound_uid",
        "ln_ic50",
        "auc",
        "z_score_gdsc",
        "extrapolated",
        "suspect",
        "eligible_cell_line",
    }
    assert expected <= set(labels.columns)
    assert not labels.duplicated(["cell_line_uid", "compound_uid"]).any()


def test_cell_lines_schema() -> None:
    frame = pd.read_parquet(_artifact("cell_lines"))
    expected = {
        "cell_line_uid",
        "sanger_model_id",
        "source_group",
        "oncotree_lineage",
        "eligible",
        "has_ge",
        "has_mu",
        "has_cnv",
        "in_main_pool",
    }
    assert expected <= set(frame.columns)
    assert frame["cell_line_uid"].is_unique


def test_compounds_schema() -> None:
    frame = pd.read_parquet(_artifact("compounds"))
    expected = {"compound_uid", "drug_ids", "has_inchikey", "has_target", "has_smiles"}
    assert expected <= set(frame.columns)
    assert frame["compound_uid"].is_unique


def test_processed_teacher_index_matches_npz() -> None:
    path = _artifact("teacher_inputs")
    data = io.read_npz(path)
    index = io.read_json(registry.resolve("teacher_index"))
    assert data["ge"].shape == tuple(index["shapes"]["ge"])
    assert data["cnv"].shape == tuple(index["shapes"]["cnv"])
    assert data["mu_A"].shape == tuple(index["shapes"]["mu_A"])


@pytest.mark.integration
def test_main_pool_matches_label_count() -> None:
    labels = pd.read_parquet(_artifact("labels"))
    cell_lines = pd.read_parquet(_artifact("cell_lines"))
    pool = set(cell_lines.loc[cell_lines["in_main_pool"], "cell_line_uid"])
    assert len(pool) == 509
    assert int(labels["cell_line_uid"].isin(pool).sum()) == 126803

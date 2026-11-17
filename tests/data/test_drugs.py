"""药物特征 / SMILES 下载结果契约测试。"""

from __future__ import annotations

import pandas as pd
import pytest

from kite_drp.data.drugs import RDKIT_AVAILABLE, load_smiles_table, morgan_fingerprints


def test_load_smiles_table_by_compound_uid(tmp_path) -> None:
    path = tmp_path / "smiles.tsv"
    pd.DataFrame(
        {
            "compound_uid": ["inchikey:AAA", "gdscname:BBB"],
            "inchikey": ["AAA", None],
            "smiles": ["CCO", "c1ccccc1"],
        }
    ).to_csv(path, sep="\t", index=False)
    table = load_smiles_table(path)
    assert table["inchikey:AAA"] == "CCO"
    assert table["gdscname:BBB"] == "c1ccccc1"


def test_morgan_masks_missing_smiles(tmp_path) -> None:
    compounds = pd.DataFrame(
        {"compound_uid": ["a", "b"], "drug_name": ["A", "B"]}
    )
    matrix, info = morgan_fingerprints(compounds, {"a": "CCO"}, radius=2, bits=64)
    assert matrix.shape == (2, 64)
    assert info["coverage"] == 0.5
    if RDKIT_AVAILABLE:
        assert matrix[0].any()
        assert not matrix[1].any()


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="rdkit 未安装")
def test_morgan_is_deterministic() -> None:
    compounds = pd.DataFrame({"compound_uid": ["a"], "drug_name": ["ethanol"]})
    first, _ = morgan_fingerprints(compounds, {"a": "CCO"}, bits=128)
    second, _ = morgan_fingerprints(compounds, {"a": "CCO"}, bits=128)
    assert first.tolist() == second.tolist()

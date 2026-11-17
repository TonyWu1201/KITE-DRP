"""外部验证库映射测试。"""

from __future__ import annotations

import pandas as pd
import pytest

from kite_drp.data.external import (
    _inchi_key_from_smiles,
    _map_by_names,
    _model_name_index,
    build_gdsc1,
)


def test_inchi_key_from_smiles() -> None:
    if _inchi_key_from_smiles("CCO") is None:
        pytest.skip("rdkit 未安装")
    assert _inchi_key_from_smiles("CCO") == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
    assert _inchi_key_from_smiles("not-a-smiles") is None


def test_model_name_index_keeps_unique_names() -> None:
    model = pd.DataFrame(
        {
            "ModelID": ["ACH-1", "ACH-2", "ACH-3"],
            "CellLineName": ["A549", "MCF7", "A549"],
            "StrippedCellLineName": ["A549", "MCF7", "A549"],
            "CCLEName": ["A549_LUNG", "MCF7_BREAST", "A549_LUNG"],
        }
    )
    index, ambiguous = _model_name_index(model)
    assert "A549" in ambiguous
    assert index["MCF7"] == "ACH-2"
    mapped = _map_by_names(pd.Series(["MCF7", "A549", "unknown"]), index)
    assert mapped.iloc[0] == "ACH-2"
    assert pd.isna(mapped.iloc[1])
    assert pd.isna(mapped.iloc[2])


def test_build_gdsc1_compound_level_aggregation(tmp_path) -> None:
    gdsc1 = tmp_path / "gdsc1.csv"
    pd.DataFrame(
        {
            "DATASET": ["GDSC1", "GDSC1"],
            "COSMIC_ID": ["1", "1"],
            "CELL_LINE_NAME": ["A549", "A549"],
            "SANGER_MODEL_ID": ["SIDM1", "SIDM1"],
            "TCGA_DESC": ["LUNG", "LUNG"],
            "DRUG_ID": ["1", "1"],
            "DRUG_NAME": ["Erlotinib", "Erlotinib"],
            "MIN_CONC": [0.01, 0.01],
            "MAX_CONC": [1, 1],
            "LN_IC50": [1.0, 3.0],
            "AUC": [0.8, 0.9],
            "Z_SCORE": [0.0, 1.0],
            "RMSE": [0.1, 0.1],
        }
    ).to_csv(gdsc1, index=False)
    model = tmp_path / "Model.csv"
    pd.DataFrame(
        {
            "ModelID": ["ACH-1"],
            "CellLineName": ["A549"],
            "StrippedCellLineName": ["A549"],
            "CCLEName": ["A549_LUNG"],
            "SangerModelID": ["SIDM1"],
        }
    ).to_csv(model, index=False)
    meta = tmp_path / "meta.csv"
    pd.DataFrame(
        {
            "DRUG_ID": ["1"],
            "DRUG_NAME": ["Erlotinib"],
            "INCHI_KEY": ["AAKJLRGGTJKAMG-UHFFFAOYSA-N"],
        }
    ).to_csv(meta, index=False)
    compounds = pd.DataFrame(
        {
            "compound_uid": ["inchikey:AAKJLRGGTJKAMG-UHFFFAOYSA-N"],
            "inchikey": ["AAKJLRGGTJKAMG-UHFFFAOYSA-N"],
            "drug_name": ["Erlotinib"],
            "gdsc_names": [["Erlotinib"]],
        }
    )
    result = build_gdsc1(gdsc1, meta, model, compounds)
    assert len(result.frame) == 1
    assert result.frame.iloc[0]["ln_ic50"] == 2.0
    assert result.frame.iloc[0]["compound_in_gdsc2"]
    assert result.frame.iloc[0]["cell_line_uid"] == "ACH-1"

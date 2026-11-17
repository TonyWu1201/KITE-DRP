"""ID 归一与标签聚合测试（合成 fixture，不依赖 raw）。"""

from __future__ import annotations

import pandas as pd
import pytest

from kite_drp.data import ids
from kite_drp.data.labels import build_labels


def _model_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ModelID": ["ACH-000001", "ACH-000837", "ACH-002172"],
            "SangerModelID": ["SIDM00001", "SIDM00117", "SIDM00117"],
            "COSMICID": ["905933.0", None, "905967.0"],
            "CellLineName": ["NIH:OVCAR-3", "NCI-H322", "NCI-H322M"],
            "StrippedCellLineName": ["NIHOVCAR3", "NCIH322", "NCIH322M"],
            "CCLEName": ["OVCAR3_OVARY", "NCIH322_LUNG", "NCIH322M_LUNG"],
            "OncotreeLineage": ["Ovary/Fallopian Tube", "Lung", "Lung"],
            "OncotreePrimaryDisease": ["Ovarian Cancer", "NSCLC", "NSCLC"],
            "OncotreeSubtype": ["", "", ""],
            "OncotreeCode": ["OVT", "NSCLC", "NSCLC"],
            "ModelType": ["Cell Line"] * 3,
            "SourceType": ["ATCC"] * 3,
            "Sex": ["Female"] * 3,
            "Age": [60, 50, 50],
            "EngineeredModel": [None, None, None],
        }
    )


def test_gdsc_cell_line_mapping_arbitrates_one_to_many() -> None:
    labels = pd.DataFrame(
        {
            "CELL_LINE_NAME": ["NIH:OVCAR-3", "NCI-H322M"],
            "SANGER_MODEL_ID": ["SIDM00001", "SIDM00117"],
            "CANCER_TYPE": ["Ovarian", "Lung"],
        }
    )
    mapping, conflicts = ids.map_gdsc_cell_lines(labels, _model_frame())
    hit = mapping.set_index("source_id").loc["SIDM00117"]
    assert hit["cell_line_uid"] == "ACH-002172"
    assert hit["status"] == "ok"
    assert not conflicts


def test_unmapped_name_is_recorded_as_conflict() -> None:
    labels = pd.DataFrame(
        {
            "CELL_LINE_NAME": ["RH-1"],
            "SANGER_MODEL_ID": ["SIDM00427"],
            "CANCER_TYPE": ["Ewing's Sarcoma"],
        }
    )
    mapping, conflicts = ids.map_gdsc_cell_lines(labels, _model_frame())
    assert mapping.iloc[0]["status"] == "failed"
    assert conflicts and conflicts[0]["source_id"] == "SIDM00427"


def test_dual_drug_ids_merge_at_compound_layer() -> None:
    labels = pd.DataFrame(
        {
            "DRUG_ID": ["1811", "1911"],
            "DRUG_NAME": ["Dactinomycin", "Dactinomycin"],
        }
    )
    meta = pd.DataFrame(
        {
            "DRUG_ID": ["1811", "1911"],
            "DRUG_NAME": ["Dactinomycin", "Dactinomycin"],
            "INCHI_KEY": ["RJURFGZVJUGBQQ-UHFFFAOYSA-N"] * 2,
            "CHEMBL_ID": ["CHEMBL1200", "CHEMBL1200"],
            "TARGET": ["TOP2A", "TOP2A"],
            "SCREENING_SITE": ["SANGER", "SANGER"],
        }
    )
    compounds = ids.build_compounds(labels, meta)
    assert len(compounds) == 1
    assert compounds.iloc[0]["compound_uid"].startswith("inchikey:")
    assert compounds.iloc[0]["drug_ids"] == ["1811", "1911"]


def test_build_labels_averages_dual_ids_and_marks_extrapolation() -> None:
    raw = pd.DataFrame(
        {
            "CELL_LINE_NAME": ["A", "A", "A"],
            "SANGER_MODEL_ID": ["SIDM00001"] * 3,
            "CANCER_TYPE": ["Ovarian"] * 3,
            "DRUG_ID": ["1811", "1911", "2"],
            "DRUG_NAME": ["Dactinomycin", "Dactinomycin", "Other"],
            "PATHWAY_NAME": ["p"] * 3,
            "MIN_CONC": [0.01] * 3,
            "MAX_CONC": [1.0] * 3,
            "LN_IC50": [1.0, 3.0, 5.0],
            "AUC": [0.8, 0.9, 0.7],
            "RMSE": [0.1, 0.1, 0.3],
            "Z_SCORE": [0.0, 1.0, 4.0],
        }
    )
    mapping = pd.DataFrame(
        {"source_id": ["SIDM00001"], "cell_line_uid": ["ACH-000001"], "status": ["ok"]}
    )
    compounds = pd.DataFrame(
        {
            "compound_uid": ["inchikey:X", "gdscname:OTHER"],
            "drug_ids": [["1811", "1911"], ["2"]],
        }
    )
    cell_lines = pd.DataFrame({"cell_line_uid": ["ACH-000001"], "eligible": [True]})
    labels, stats = build_labels(raw, mapping, compounds, cell_lines)
    dual = labels[labels["compound_uid"] == "inchikey:X"].iloc[0]
    assert dual["ln_ic50"] == pytest.approx(2.0)
    assert dual["n_source_ids"] == 2
    assert dual["delta_dual_id"] == pytest.approx(2.0)
    assert stats["dual_id_rows_removed"] == 1
    assert labels["suspect"].sum() >= 1


def test_compound_split_groups_degrades_without_scaffold() -> None:
    from kite_drp.data.drugs import compound_split_groups

    compounds = pd.DataFrame(
        {
            "compound_uid": ["a", "b", "c"],
            "scaffold": [None, None, None],
            "target_genes": [["EGFR"], ["EGFR"], []],
        }
    )
    groups, degraded = compound_split_groups(compounds)
    assert degraded is True
    assert groups.tolist()[:2] == ["target:EGFR", "target:EGFR"]
    assert groups.tolist()[2].startswith("compound:")


def test_compound_split_groups_uses_scaffold_when_available() -> None:
    from kite_drp.data.drugs import compound_split_groups

    compounds = pd.DataFrame(
        {
            "compound_uid": ["a", "b", "c"],
            "scaffold": ["c1ccccc1", "c1ccccc1", None],
            "target_genes": [["EGFR"], ["EGFR"], ["TP53"]],
        }
    )
    groups, degraded = compound_split_groups(compounds)
    assert degraded is False
    assert groups.tolist()[0] == groups.tolist()[1] == "scaffold:c1ccccc1"
    assert groups.tolist()[2] == "target:TP53"


def test_cell_line_source_group_does_not_collapse_missing_cosmic(tmp_path) -> None:
    path = tmp_path / "Model.csv"
    _model_frame().to_csv(path, index=False)
    cell_lines = ids.build_cell_lines(path)
    assert not cell_lines["source_group"].str.contains("nan").any()
    assert cell_lines["source_group"].nunique() == len(cell_lines)

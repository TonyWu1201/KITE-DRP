"""Open Targets DTI 映射测试（不依赖真实 raw 数据）。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from kite_drp.data.genes import build_gene_master
from kite_drp.data.opentargets import build_opentargets_edges


def _master(tmp_path: Path):
    path = tmp_path / "Gene.csv"
    genes = ["TP53", "EGFR", "MYC"]
    pd.DataFrame(
        {
            "hgnc_id": [f"HGNC:{i}" for i in range(len(genes))],
            "symbol": genes,
            "name": genes,
            "locus_group": ["protein-coding gene"] * len(genes),
            "locus_type": ["gene with protein product"] * len(genes),
            "status": ["Approved"] * len(genes),
            "alias_symbol": [None] * len(genes),
            "prev_symbol": [None] * len(genes),
            "entrez_id": [str(i + 1) for i in range(len(genes))],
            "ensembl_gene_id": [f"ENSG0000000000{i + 1}" for i in range(len(genes))],
        }
    ).to_csv(path, index=False)
    return build_gene_master(path)


def _write_opentargets(tmp_path: Path) -> Path:
    root = tmp_path / "opentargets" / "26.09"
    (root / "target").mkdir(parents=True)
    (root / "drug_molecule").mkdir(parents=True)
    (root / "drug_mechanism_of_action").mkdir(parents=True)

    pd.DataFrame(
        {
            "id": ["ENSG00000000001", "ENSG00000000002"],
            "approvedSymbol": ["TP53", "EGFR"],
            "approvedName": ["Tumor protein p53", "Epidermal growth factor receptor"],
            "dbXrefs": [[], []],
        }
    ).to_parquet(root / "target" / "part.parquet", index=False)
    pd.DataFrame(
        {
            "id": ["CHEMBL1", "CHEMBL2", "CHEMBL3", "CHEMBL5", "CHEMBL6"],
            "name": ["DrugA", "DrugA salt", "DrugB", "Ambiguous", "Ambiguous"],
            "inchiKey": ["KEYA", "KEYA-SALT", "KEYB", "KEYX", "KEYY"],
            "parentId": [None, "CHEMBL1", None, None, None],
            "childChemblIds": [["CHEMBL2"], [], [], [], []],
            "synonyms": [[], [], [], [], []],
            "tradeNames": [[], [], [], [], []],
        }
    ).to_parquet(root / "drug_molecule" / "part.parquet", index=False)
    pd.DataFrame(
        {
            "chemblIds": [["CHEMBL2"], ["CHEMBL3"]],
            "targets": [["ENSG00000000001"], ["ENSG00000000002"]],
            "targetName": ["Tumor protein p53", "Epidermal growth factor receptor"],
            "actionType": ["INHIBITOR", "INHIBITOR"],
            "mechanismOfAction": ["p53 stabiliser", "EGFR inhibitor"],
            "targetType": ["single protein", "single protein"],
        }
    ).to_parquet(root / "drug_mechanism_of_action" / "part.parquet", index=False)
    return root


def _compounds() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "compound_uid": ["compound:A", "compound:B", "compound:C"],
            "drug_name": ["DrugA", "DrugB", "Ambiguous"],
            "gdsc_names": [["DrugA"], ["DrugB"], ["Ambiguous"]],
            "chembl_id": ["CHEMBL1", None, None],
            "inchikey": ["KEYA", "KEYB", None],
        }
    )


def test_opentargets_parent_and_inchikey_mapping(tmp_path: Path) -> None:
    root = _write_opentargets(tmp_path)
    master = _master(tmp_path)
    edges, info = build_opentargets_edges(
        _compounds(), master, root, allowed_genes={"TP53", "EGFR", "MYC"}
    )
    assert info["compounds_with_ot_family"] == 2
    assert info["matched_families"] == 2
    assert set(zip(edges["compound_uid"], edges["hgnc_symbol"], strict=False)) == {
        ("compound:A", "TP53"),
        ("compound:B", "EGFR"),
    }
    methods = dict(zip(edges["compound_uid"], edges["match_method"], strict=False))
    assert methods == {"compound:A": "chembl_id", "compound:B": "inchikey"}
    # 同名歧义药（CHEMBL5/6 两个族）不使用名称兜底。
    assert "compound:C" not in set(edges["compound_uid"])


def test_opentargets_gene_filter_and_missing_dir(tmp_path: Path) -> None:
    root = _write_opentargets(tmp_path)
    master = _master(tmp_path)
    edges, info = build_opentargets_edges(_compounds(), master, root, allowed_genes={"EGFR"})
    assert set(edges["hgnc_symbol"]) == {"EGFR"}
    assert info["edges_before_gene_filter"] == 2
    assert info["edges"] == 1

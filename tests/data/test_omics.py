"""组学读取/变换测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kite_drp.data.genes import build_gene_master
from kite_drp.data.omics.cnv import load_cnv
from kite_drp.data.omics.expression import load_expression, select_hvg
from kite_drp.data.omics.mutations import load_mutations


def _master(tmp_path):
    path = tmp_path / "Gene.csv"
    pd.DataFrame(
        {
            "hgnc_id": ["HGNC:1", "HGNC:2", "HGNC:3"],
            "symbol": ["TP53", "EGFR", "MYC"],
            "name": ["p53", "egfr", "myc"],
            "locus_group": ["protein-coding gene"] * 3,
            "locus_type": ["gene with protein product"] * 3,
            "status": ["Approved"] * 3,
            "alias_symbol": [None, None, None],
            "prev_symbol": [None, None, None],
            "entrez_id": ["7157", "1956", "4609"],
            "ensembl_gene_id": ["ENSG1", "ENSG2", "ENSG3"],
        }
    ).to_csv(path, index=False)
    return build_gene_master(path)


def test_expression_default_filter_and_zero_variance(tmp_path) -> None:
    master = _master(tmp_path)
    path = tmp_path / "expr.csv"
    frame = pd.DataFrame(
        {
            "SequencingID": ["s1", "s2", "s3"],
            "ModelConditionID": ["m1", "m2", "m3"],
            "ModelID": ["ACH-1", "ACH-2", "ACH-3"],
            "IsDefaultEntryForMC": ["Yes", "Yes", "Yes"],
            "IsDefaultEntryForModel": ["Yes", "No", "Yes"],
            "TP53 (7157)": [1.0, 99.0, 10.0],
            "EGFR (1956)": [0.0, 99.0, 0.0],
            "MYC (4609)": [3.0, 99.0, 4.0],
        }
    )
    frame.to_csv(path, index=False)
    data = load_expression(path, master)
    assert data.frame.shape == (2, 2)  # EGFR 全零方差被删
    assert set(data.frame.index) == {"ACH-1", "ACH-3"}
    assert data.info["zero_variance_genes"] == ["EGFR"]
    assert select_hvg(data.frame, ["ACH-1", "ACH-3"], n_top=1) == ["TP53"]


def test_cnv_log2_and_clip(tmp_path) -> None:
    master = _master(tmp_path)
    path = tmp_path / "cnv.csv"
    frame = pd.DataFrame(
        {
            "SequencingID": ["s1", "s2"],
            "ModelConditionID": ["m1", "m2"],
            "ModelID": ["ACH-1", "ACH-2"],
            "IsDefaultEntryForMC": ["Yes", "Yes"],
            "IsDefaultEntryForModel": ["Yes", "Yes"],
            "TP53 (7157)": [1.0, 2.0],
            "EGFR (1956)": [0.5, 4.0],
            "MYC (4609)": [np.nan, np.nan],
        }
    )
    frame.to_csv(path, index=False)
    data = load_cnv(path, master, clip_percentile=(0.0, 100.0))
    assert data.frame.shape == (2, 2)
    assert data.info["all_nan_columns"] == 1
    assert np.isclose(data.frame.loc["ACH-2", "TP53"], 1.0)
    assert data.frame.to_numpy().min() < 0


def test_mutation_filters_A_and_B(tmp_path) -> None:
    path = tmp_path / "mu.csv"
    frame = pd.DataFrame(
        {
            "SequencingID": ["s"] * 5,
            "ModelConditionID": ["m"] * 5,
            "ModelID": ["ACH-1"] * 5,
            "IsDefaultEntryForModel": ["Yes"] * 5,
            "VepBiotype": ["protein_coding"] * 5,
            "VariantInfo": [
                "missense_variant",
                "synonymous_variant",
                "frameshift_variant&stop_gained",
                "missense_variant",
                "intron_variant",
            ],
            "LikelyLoF": ["False", "False", "True", "False", "False"],
            "Hotspot": ["False", "False", "False", "True", "False"],
            "HugoSymbol": ["TP53", "EGFR", "MYC", "EGFR", "TP53"],
            "AF": ["0.5"] * 5,
            "DP": ["100"] * 5,
            "Chrom": ["1"] * 5,
            "Pos": ["1"] * 5,
            "Ref": ["A"] * 5,
            "Alt": ["T"] * 5,
            "VariantType": ["SNV"] * 5,
        }
    )
    frame.to_csv(path, index=False)
    data = load_mutations(path)
    assert data.info["rows_A"] == 3  # TP53 missense + MYC LoF + EGFR hotspot
    assert data.info["rows_B"] == 2
    assert data.genes_A == ["EGFR", "MYC", "TP53"]
    assert data.genes_B == ["EGFR", "MYC"]
    assert data.matrix_A.sum() == 3
    assert data.mut_burden_A["ACH-1"] == 3
    assert data.mut_burden_B["ACH-1"] == 2

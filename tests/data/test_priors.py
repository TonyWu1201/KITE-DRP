"""先验 / 通路解析测试。"""

from __future__ import annotations

import pandas as pd

from kite_drp.data.genes import build_gene_master
from kite_drp.data.priors import build_pathways


def _master(tmp_path):
    path = tmp_path / "Gene.csv"
    genes = ["TP53", "EGFR", "MYC", "BRCA1", "PTEN", "KRAS"]
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
            "ensembl_gene_id": [f"ENSG{i}" for i in range(len(genes))],
        }
    ).to_csv(path, index=False)
    return build_gene_master(path)


def test_build_pathways_infers_source_and_filters(tmp_path) -> None:
    master = _master(tmp_path)
    reactome = tmp_path / "reactome" / "r.gmt"
    go = tmp_path / "go" / "g.gmt"
    reactome.parent.mkdir()
    go.parent.mkdir()
    reactome.write_text("R1\tpath\tTP53\tEGFR\tMYC\nR2\tbig\tTP53\tEGFR\tMYC\tBRCA1\tPTEN\tKRAS\n")
    go.write_text("GO:1\tterm\tTP53\tEGFR\nGO:2\tterm\tTP53\n")
    pathways, kept = build_pathways(
        [reactome, go],
        master,
        universe=set(master.frame["hgnc_symbol"]),
        min_genes=2,
        max_genes=10,
        max_pathways=10,
    )
    assert set(pathways["source"]) == {"reactome", "go"}
    assert "go:GO:2" not in kept  # 只有 1 个基因，被 min_genes 过滤
    assert "reactome:R1" in kept
    assert "reactome:R2" in kept
    assert pathways.groupby("pathway_id")["gene"].nunique().min() >= 2


def test_build_pathways_caps_per_source(tmp_path) -> None:
    master = _master(tmp_path)
    reactome = tmp_path / "reactome" / "r.gmt"
    go = tmp_path / "go" / "g.gmt"
    reactome.parent.mkdir()
    go.parent.mkdir()
    reactome.write_text("R1\tp\tTP53\tEGFR\nR2\tp\tTP53\tMYC\nR3\tp\tEGFR\tMYC\n")
    go.write_text("G1\tp\tTP53\tEGFR\nG2\tp\tTP53\tMYC\nG3\tp\tEGFR\tMYC\n")
    pathways, kept = build_pathways(
        [reactome, go],
        master,
        universe=set(master.frame["hgnc_symbol"]),
        min_genes=2,
        max_genes=5,
        max_pathways=2,
    )
    assert len(kept) == 2
    assert set(pathways["source"]) == {"reactome", "go"}

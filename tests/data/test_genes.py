"""HGNC/基因宇宙测试。"""

from __future__ import annotations

import pandas as pd

from kite_drp.data.genes import (
    build_gene_master,
    build_universes,
    parse_gene_column,
    resolve_columns,
)


def _hgnc(tmp_path):
    path = tmp_path / "Gene.csv"
    pd.DataFrame(
        {
            "hgnc_id": ["HGNC:1", "HGNC:2", "HGNC:3", "HGNC:4"],
            "symbol": ["TP53", "EGFR", "OLD1", "ALIAS1"],
            "name": ["tumor protein p53", "epidermal growth factor receptor", "old", "alias"],
            "locus_group": ["protein-coding gene"] * 4,
            "locus_type": ["gene with protein product"] * 4,
            "status": ["Approved"] * 4,
            "alias_symbol": ["P53|TP53", None, None, None],
            "prev_symbol": [None, None, "OLD2", None],
            "entrez_id": ["7157", "1956", "100", "200"],
            "ensembl_gene_id": ["ENSG00000141510", "ENSG00000146648", "ENSG000001", "ENSG000002"],
        }
    ).to_csv(path, index=False)
    return path


def test_parse_gene_column_variants() -> None:
    assert parse_gene_column("TSPAN6 (7105)")["symbol"] == "TSPAN6"
    assert parse_gene_column("ENSG00000002586_PAR_Y")["par_y"] is True
    assert parse_gene_column("not-a-gene") is None


def test_gene_master_mapping_priority_and_alias_conflicts(tmp_path) -> None:
    master = build_gene_master(_hgnc(tmp_path))
    assert master.resolve("TP53") == ("TP53", "symbol")
    assert master.resolve("P53") == ("TP53", "alias_symbol")
    assert master.resolve("OLD2") == ("OLD1", "prev_symbol")
    assert master.resolve("ENSG00000146648") == ("EGFR", "ensembl")
    assert master.resolve("1956") == ("EGFR", "entrez")
    assert master.resolve("UNKNOWN") == (None, "unmapped")


def test_resolve_columns_and_universe_nesting(tmp_path) -> None:
    master = build_gene_master(_hgnc(tmp_path))
    mapping = resolve_columns(["TP53 (7157)", "EGFR (1956)", "BADCOL (9)"], master)
    assert mapping["hgnc_symbol"].tolist()[:2] == ["TP53", "EGFR"]
    universes = build_universes(
        master=master,
        expression_genes=["TP53", "EGFR", "OLD1"],
        cnv_genes=["TP53", "EGFR"],
        mutation_genes_A=["TP53", "ALIAS1", "EGFR"],
        mutation_genes_B=["TP53"],
        ppi_genes=["TP53", "EGFR", "ALIAS1"],
        ppi_degree={"TP53": 10, "EGFR": 5},
        pathway_genes=["TP53", "EGFR"],
        panel_genes=["TP53", "ALIAS1"],
        vae_size=1,
    )
    assert universes["teach_ge"] == ["EGFR", "TP53"]
    assert universes["teach_mu_A"] == ["EGFR", "TP53"]
    assert universes["teach_mu_B"] == ["TP53"]
    assert universes["vae"] == ["TP53"]
    assert universes["panel"] == ["TP53"]

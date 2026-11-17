"""划分协议生成与校验测试。"""

from __future__ import annotations

import pandas as pd

from kite_drp.data.splits import fold_masks, make_split, validate_split


def _fixtures():
    cells = []
    for index in range(12):
        lineage = ["Lung", "Breast", "CNS"][index % 3]
        cells.append(
            {
                "cell_line_uid": f"ACH-{index:06d}",
                "source_group": f"g{index % 6}",
                "oncotree_lineage": lineage,
                "eligible": True,
                "mapped": True,
                "in_main_pool": True,
            }
        )
    cell_lines = pd.DataFrame(cells)
    compounds = pd.DataFrame(
        {
            "compound_uid": [f"c{index}" for index in range(6)],
            "target_genes": [["EGFR"], ["EGFR"], ["TP53"], ["MYC"], [], []],
            "scaffold": [None] * 6,
        }
    )
    labels = pd.DataFrame(
        [
            {
                "cell_line_uid": cell,
                "compound_uid": compound,
                "ln_ic50": float(i + j),
                "eligible_cell_line": True,
                "mapped": True,
            }
            for i, cell in enumerate(cell_lines["cell_line_uid"])
            for j, compound in enumerate(compounds["compound_uid"])
        ]
    )
    return labels, cell_lines, compounds


def test_lco_groups_are_disjoint_and_cover_all_cells() -> None:
    labels, cell_lines, compounds = _fixtures()
    split = make_split("p1_lco", labels=labels, cell_lines=cell_lines, compounds=compounds, seed=0)
    report = validate_split(split, labels=labels, cell_lines=cell_lines, compounds=compounds)
    assert all(report["checks"].values()), report
    test_cells = []
    for fold in split["folds"]:
        test_cells.extend(fold["test"])
    assert sorted(test_cells) == sorted(cell_lines["cell_line_uid"].tolist())


def test_ldo_groups_are_disjoint() -> None:
    labels, cell_lines, compounds = _fixtures()
    split = make_split("p2_ldo", labels=labels, cell_lines=cell_lines, compounds=compounds, seed=1)
    report = validate_split(split, labels=labels, cell_lines=cell_lines, compounds=compounds)
    assert all(report["checks"].values()), report
    test_compounds = [compound for fold in split["folds"] for compound in fold["test_groups"]]
    assert sorted(test_compounds) == sorted(compounds["compound_uid"].tolist())


def test_pair_protocols_partition_pairs_deterministically() -> None:
    labels, cell_lines, compounds = _fixtures()
    for protocol in ("p0_random", "lpo"):
        split_a = make_split(protocol, labels=labels, cell_lines=cell_lines, compounds=compounds, seed=2)
        split_b = make_split(protocol, labels=labels, cell_lines=cell_lines, compounds=compounds, seed=2)
        assert split_a["folds"] == split_b["folds"]
        for fold in split_a["folds"]:
            local = dict(fold)
            local["_seed"] = 2
            masks = fold_masks(local, labels, protocol)
            assert not (masks["train"] & masks["test"]).any()
            assert not (masks["val"] & masks["test"]).any()
            assert (masks["train"] | masks["val"] | masks["test"]).all()


def test_lto_uses_only_lineages_with_enough_cells() -> None:
    labels, cell_lines, compounds = _fixtures()
    split = make_split(
        "p4_lto",
        labels=labels,
        cell_lines=cell_lines,
        compounds=compounds,
        seed=0,
        min_lineage_cells=3,
    )
    assert split["n_folds"] == 3
    for fold in split["folds"]:
        assert len(fold["test_groups"]) == 1
        assert not set(fold["train_groups"]) & set(fold["test_groups"])


def test_scaffold_protocol_degrades_without_smiles() -> None:
    labels, cell_lines, compounds = _fixtures()
    split = make_split("p3_scaffold", labels=labels, cell_lines=cell_lines, compounds=compounds, seed=0)
    assert split["degraded"] is True
    assert split["group_key"] == "target_family"
    assert split["notes"]

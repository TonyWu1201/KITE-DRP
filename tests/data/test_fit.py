"""FitContext 与扰动泄漏测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kite_drp.data.fit import FitScope, Standardizer, fit_stats_for_fold
from kite_drp.data.splits import fold_masks, make_split


def _fixtures():
    cell_lines = pd.DataFrame(
        {
            "cell_line_uid": [f"ACH-{i:06d}" for i in range(10)],
            "source_group": [f"g{i % 5}" for i in range(10)],
            "oncotree_lineage": ["Lung"] * 5 + ["Breast"] * 5,
            "eligible": True,
            "mapped": True,
            "in_main_pool": True,
        }
    )
    compounds = pd.DataFrame({"compound_uid": ["c1", "c2"], "target_genes": [["EGFR"], ["TP53"]], "scaffold": [None, None]})
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
    genes = [f"G{i}" for i in range(8)]
    expression = pd.DataFrame(
        np.arange(10 * 8, dtype=np.float32).reshape(10, 8),
        index=cell_lines["cell_line_uid"],
        columns=genes,
    )
    cnv = expression.copy()
    return labels, cell_lines, compounds, expression, cnv


def test_standardizer_only_fits_train_rows() -> None:
    frame = pd.DataFrame({"a": [1.0, 3.0], "b": [2.0, 2.0]})
    scope = FitScope("s", 0, "train", "labels")
    scaler = Standardizer().fit(frame, scope)
    transformed = scaler.transform(pd.DataFrame({"a": [2.0], "b": [99.0]}))
    assert transformed.loc[0, "a"] == 0.0
    assert transformed.loc[0, "b"] == 0.0  # 零方差列置零


def test_fit_stats_unchanged_when_test_fold_is_perturbed(tmp_path) -> None:
    labels, cell_lines, compounds, expression, cnv = _fixtures()
    split = make_split("p1_lco", labels=labels, cell_lines=cell_lines, compounds=compounds, seed=0)
    fold = dict(split["folds"][0])
    fold["_seed"] = 0
    base = fit_stats_for_fold(
        split=split,
        fold=fold,
        labels=labels,
        expression=expression,
        cnv=cnv,
        out_dir=tmp_path / "base",
        seed=0,
        min_obs_per_compound=1,
        n_hvg=4,
    )
    masks = fold_masks(fold, labels, "p1_lco")
    test_index = labels.index[masks["test"]]
    labels_perturbed = labels.copy()
    labels_perturbed.loc[test_index, "ln_ic50"] = np.roll(
        labels_perturbed.loc[test_index, "ln_ic50"].to_numpy(), 1
    )
    test_cells = sorted(labels.loc[masks["test"], "cell_line_uid"].unique())
    expression_perturbed = expression.copy()
    expression_perturbed.loc[test_cells] = np.roll(
        expression_perturbed.loc[test_cells].to_numpy(), 1, axis=0
    )
    perturbed = fit_stats_for_fold(
        split=split,
        fold=fold,
        labels=labels_perturbed,
        expression=expression_perturbed,
        cnv=cnv,
        out_dir=tmp_path / "perturbed",
        seed=0,
        min_obs_per_compound=1,
        n_hvg=4,
    )
    assert base["fit_scope"]["fingerprint"] == perturbed["fit_scope"]["fingerprint"]
    assert base["expr_fingerprint"] == perturbed["expr_fingerprint"]
    assert base["hvg_fingerprint"] == perturbed["hvg_fingerprint"]

"""LDO/scaffold 未见药物的标签 fallback（全局 mean/std/median）。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kite_drp.data.pairs import sensitivity_class, sensitivity_labels


@pytest.fixture()
def scaler() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "compound_uid": ["d1", "d2"],
            "n_train": [10, 20],
            "mean": [1.0, 2.0],
            "std": [0.5, 1.0],
            "median": [1.1, 2.1],
            "q33": [0.8, 1.8],
            "q67": [1.4, 2.4],
        }
    )


def test_known_drugs_use_per_drug_scaler(scaler: pd.DataFrame) -> None:
    pairs = pd.DataFrame(
        {
            "compound_uid": ["d1", "d1", "d2"],
            "ln_ic50": [0.5, 1.5, 1.5],
        }
    )
    y = sensitivity_labels(pairs, scaler, fallback=(3.0, 2.0))
    # y = -(ln - per_drug_mean) / per_drug_std
    np.testing.assert_allclose(y, [1.0, -1.0, 0.5], rtol=1e-6)


def test_unknown_drug_uses_global_fallback(scaler: pd.DataFrame) -> None:
    pairs = pd.DataFrame(
        {
            "compound_uid": ["d3", "d3", "d3"],
            "ln_ic50": [1.0, 3.0, 5.0],
        }
    )
    y = sensitivity_labels(pairs, scaler, fallback=(3.0, 2.0))
    # y = -(ln - 3) / 2；同药内仍是单调变换
    np.testing.assert_allclose(y, [1.0, 0.0, -1.0], rtol=1e-6)
    assert np.isfinite(y).all()


def test_unknown_drug_without_fallback_is_zero(scaler: pd.DataFrame) -> None:
    pairs = pd.DataFrame({"compound_uid": ["d3"], "ln_ic50": [1.0]})
    y = sensitivity_labels(pairs, scaler)
    np.testing.assert_allclose(y, [0.0])


def test_classification_falls_back_to_global_median(scaler: pd.DataFrame) -> None:
    pairs = pd.DataFrame(
        {
            "compound_uid": ["d1", "d1", "d3", "d3"],
            "ln_ic50": [1.0, 2.0, 1.0, 4.0],
        }
    )
    labels, mask = sensitivity_class(pairs, scaler, fallback_median=2.5)
    # d1 用逐药 median=1.1；d3 用全局 median=2.5
    np.testing.assert_allclose(labels, [1.0, 0.0, 1.0, 0.0])
    assert mask.all()


def test_classification_without_fallback_masks_unknown(scaler: pd.DataFrame) -> None:
    pairs = pd.DataFrame(
        {
            "compound_uid": ["d1", "d3"],
            "ln_ic50": [1.0, 1.0],
        }
    )
    _, mask = sensitivity_class(pairs, scaler)
    np.testing.assert_array_equal(mask, [True, False])

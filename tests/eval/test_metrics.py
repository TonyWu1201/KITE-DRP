"""指标口径测试。"""

from __future__ import annotations

import numpy as np

from kite_drp.eval.metrics import classification_metrics, regression_metrics


def test_regression_perfect() -> None:
    values = np.array([0.1, -0.2, 0.3, 0.0])
    metrics = regression_metrics(values, values)
    assert metrics["mse"] == 0.0
    assert metrics["r2"] == 1.0
    assert metrics["pcc"] > 0.999


def test_regression_constant_prediction() -> None:
    metrics = regression_metrics(np.array([0.0, 1.0, 2.0]), np.zeros(3))
    assert np.isnan(metrics["pcc"])
    assert metrics["r2"] < 0


def test_classification_metrics() -> None:
    y_true = np.array([0, 1, 0, 1, 1, 0])
    probability = np.array([0.1, 0.9, 0.2, 0.8, 0.6, 0.4])
    metrics = classification_metrics(y_true, probability)
    assert metrics["auroc"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert 0.0 <= metrics["ece"] <= 1.0

"""教师多折参数的解析测试（Hydra 逗号列表 vs 字符串）。"""

from __future__ import annotations

import pytest

pytest.importorskip("hydra")
pytest.importorskip("torch")

from kite_drp.train.teacher_train import _fold_list  # noqa: E402


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, [0]),
        ("2", [2]),
        ("0,2", [0, 2]),
        ([0, 2], [0, 2]),
        ("all", [0, 1, 2, 3, 4]),
    ],
)
def test_fold_list(value, expected) -> None:
    assert _fold_list({"fold": value}, 5) == expected


def test_fold_list_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        _fold_list({"fold": "0,5"}, 5)

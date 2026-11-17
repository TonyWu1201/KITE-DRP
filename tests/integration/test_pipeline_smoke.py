"""真实数据集成冒烟测试（默认不跑；需 raw 已就位）。

运行：``pytest -m integration tests/integration``
"""

from __future__ import annotations

import pytest

from kite_drp.common.paths import RAW_DATA_DIR
from kite_drp.data.inventory import build_inventory


@pytest.mark.integration
def test_inventory_sees_expected_raw_files() -> None:
    if not (RAW_DATA_DIR / "gdsc" / "GDSC2_fitted_dose_response_27Oct23.csv").is_file():
        pytest.skip("raw 数据未就位")
    payload = build_inventory(RAW_DATA_DIR)
    paths = {entry["relative_path"] for entry in payload["entries"]}
    assert any("GDSC2_fitted_dose_response" in path for path in paths)
    assert any("DepMap_Expression" in path for path in paths)
    assert any("9606.protein.links.v12.5" in path for path in paths)
    assert payload["total_bytes"] > 1_000_000_000


@pytest.mark.integration
def test_canonical_artifacts_exist_after_pipeline() -> None:
    from kite_drp.data.config import DataPaths

    paths = DataPaths()
    required = [
        paths.canonical / "labels.parquet",
        paths.canonical / "expression.parquet",
        paths.canonical / "cnv.parquet",
        paths.processed / "teacher_inputs.npz",
        paths.splits / "p1_lco_s0.json",
    ]
    if not all(path.is_file() for path in required):
        pytest.skip("数据管线尚未运行")
    assert all(path.stat().st_size > 0 for path in required)

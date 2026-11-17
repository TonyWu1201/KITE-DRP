"""manifest / 哈希 / 原子写测试。"""

from __future__ import annotations

import pandas as pd

from kite_drp.data import io
from kite_drp.data.manifest import Manifest


def test_artifact_hash_is_deterministic_and_sensitive(tmp_path) -> None:
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    path = io.write_parquet(frame, tmp_path / "a.parquet")
    first = io.hash_file(path)
    second = io.hash_file(io.write_parquet(frame, tmp_path / "b.parquet"))
    assert first == second
    changed = io.write_parquet(frame.assign(a=[1, 3]), tmp_path / "c.parquet")
    assert io.hash_file(changed) != first


def test_manifest_register_verify_and_tamper(tmp_path) -> None:
    frame = pd.DataFrame({"x": [1.0, 2.0]})
    path = io.write_parquet(frame, tmp_path / "artifact.parquet")
    manifest_path = tmp_path / "manifest.json"
    manifest = Manifest(manifest_path)
    manifest.register("artifact", path, layer="canonical", df=frame)
    manifest.save(code="c1", config="cfg1")
    assert Manifest(manifest_path).verify() == []

    # 篡改内容 → verify 失败
    io.write_parquet(frame.assign(x=[9.0, 9.0]), path)
    assert Manifest(manifest_path).verify()


def test_atomic_write_does_not_leave_temp_file(tmp_path) -> None:
    target = io.atomic_write_text(tmp_path / "x.txt", "hello")
    assert target.read_text() == "hello"
    assert not list(tmp_path.glob("*.tmp"))


def test_packed_matrix_roundtrip(tmp_path) -> None:
    import numpy as np

    matrix = np.array([[True, False, True], [False, True, True]])
    path = io.save_packed_matrix(tmp_path / "m.npz", matrix, ["r1", "r2"], ["c1", "c2", "c3"])
    loaded, rows, cols = io.load_packed_matrix(path)
    assert loaded.tolist() == matrix.tolist()
    assert rows == ["r1", "r2"]
    assert cols == ["c1", "c2", "c3"]

"""产物 manifest、语义哈希与可复现三件套。

约定（规划 §2.5）：

``artifact_hash = sha256(filename | columns | dtypes | content_hash)``

实现上对 parquet/表格使用 :func:`hash_dataframe`，对矩阵使用
:func:`hash_ndarray`；manifest 中同时保存输入文件哈希、配置哈希、代码版本。
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kite_drp.common.paths import REPO_ROOT
from kite_drp.data import io
from kite_drp.data.io import canonical_json, hash_dataframe, hash_file, hash_ndarray


# ---------------------------------------------------------------------------
# 代码 / 配置指纹
# ---------------------------------------------------------------------------
def code_hash(repo_root: str | Path | None = None) -> str:
    """``git HEAD + 未提交 diff`` 的哈希；非 git 环境返回内容哈希占位。"""
    root = Path(repo_root or REPO_ROOT)
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        diff = subprocess.check_output(
            ["git", "diff", "--binary"], cwd=root, stderr=subprocess.DEVNULL
        )
        untracked = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        payload = head.encode() + b"\n" + diff
        for name in sorted(untracked.splitlines()):
            candidate = root / name
            if candidate.is_file() and candidate.stat().st_size < 2_000_000:
                payload += b"\n" + name.encode() + b"\n" + candidate.read_bytes()
        return io.hash_bytes(payload)
    except (OSError, subprocess.CalledProcessError):
        return "no-git"


def config_hash(config: dict[str, Any]) -> str:
    return io.hash_bytes(canonical_json(config).encode("utf-8"))


def input_hashes(paths: dict[str, str | Path], *, algorithm: str = "sha256") -> dict[str, str]:
    """计算输入文件哈希；不存在的文件跳过并保留说明。"""
    result: dict[str, str] = {}
    for name, path in paths.items():
        candidate = Path(path)
        if candidate.is_file():
            result[name] = f"{algorithm}:{hash_file(candidate, algorithm)}"
        else:
            result[name] = "missing"
    return result


def raw_md5_from_checksums(path: str | Path = REPO_ROOT / "data" / "raw" / "CHECKSUMS.md5") -> dict[str, str]:
    """解析 raw/CHECKSUMS.md5，避免每次重算数 GB 文件的 SHA256。"""
    result: dict[str, str] = {}
    candidate = Path(path)
    if not candidate.is_file():
        return result
    for line in candidate.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            name = parts[1].lstrip("*")
            if name.startswith("./"):
                name = name[2:]
            result[name] = f"md5:{parts[0]}"
    return result


# ---------------------------------------------------------------------------
# 产物哈希
# ---------------------------------------------------------------------------
def artifact_hash(
    name: str,
    *,
    path: str | Path | None = None,
    df: pd.DataFrame | None = None,
    array: np.ndarray | None = None,
    sort_by: list[str] | None = None,
) -> str:
    digest = io.hash_bytes(name.encode("utf-8"))
    if df is not None:
        content = hash_dataframe(df, sort_by=sort_by)
        schema = canonical_json(
            {"columns": list(map(str, df.columns)), "dtypes": [str(x) for x in df.dtypes]}
        )
    elif array is not None:
        content = hash_ndarray(array)
        schema = f"{array.dtype}|{array.shape}"
    elif path is not None:
        target = Path(path)
        content = hash_file(target)
        schema = f"{target.suffix}|{target.stat().st_size if target.is_file() else -1}"
    else:
        raise ValueError("artifact_hash 需要 path/df/array 之一")
    return io.hash_bytes("|".join([name, content, schema]).encode("utf-8"))


@dataclass
class ArtifactEntry:
    name: str
    path: str
    layer: str
    sha256: str
    size: int
    schema: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "layer": self.layer,
            "sha256": self.sha256,
            "size": self.size,
            "schema": self.schema,
            "inputs": self.inputs,
            **self.meta,
        }


class Manifest:
    """``manifest.json`` 的读写与增量校验。"""

    def __init__(self, path: str | Path, *, schema_version: str = "1.0") -> None:
        self.path = Path(path)
        self.schema_version = schema_version
        self.artifacts: dict[str, dict[str, Any]] = {}
        self.meta: dict[str, Any] = {}
        if self.path.is_file():
            payload = io.read_json(self.path)
            self.schema_version = payload.get("schema_version", schema_version)
            self.meta = payload.get("meta", {})
            for entry in payload.get("artifacts", []):
                self.artifacts[entry["name"]] = entry

    def register(
        self,
        name: str,
        path: str | Path,
        *,
        layer: str,
        df: pd.DataFrame | None = None,
        array: np.ndarray | None = None,
        inputs: dict[str, str] | None = None,
        meta: dict[str, Any] | None = None,
        schema: dict[str, Any] | None = None,
    ) -> ArtifactEntry:
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(target)
        semantic = None
        if df is not None:
            semantic = artifact_hash(name, df=df)
        elif array is not None:
            semantic = artifact_hash(name, array=array)
        entry = ArtifactEntry(
            name=name,
            path=str(target),
            layer=layer,
            sha256=hash_file(target),
            size=target.stat().st_size,
            inputs=inputs or {},
            schema=schema
            or (
                {"columns": list(map(str, df.columns)), "dtypes": [str(x) for x in df.dtypes]}
                if df is not None
                else {}
            ),
            meta={**(meta or {}), "semantic_sha256": semantic} if semantic else (meta or {}),
        )
        self.artifacts[name] = entry.as_dict()
        return entry

    def should_skip(self, name: str, *, code: str, config: str) -> bool:
        entry = self.artifacts.get(name)
        if not entry:
            return False
        target = Path(entry.get("path", ""))
        return (
            target.is_file()
            and entry.get("code_hash") == code
            and entry.get("config_hash") == config
            and not entry.get("dirty", False)
        )

    def save(self, *, code: str | None = None, config: str | None = None, extra: dict[str, Any] | None = None) -> Path:
        for entry in self.artifacts.values():
            if code is not None:
                entry["code_hash"] = code
            if config is not None:
                entry["config_hash"] = config
        payload = {
            "schema_version": self.schema_version,
            "meta": {**self.meta, **(extra or {})},
            "artifacts": sorted(self.artifacts.values(), key=lambda item: item["name"]),
        }
        return io.write_json(self.path, payload)

    def verify(self) -> list[str]:
        """返回校验失败项（路径缺失或内容 hash 不匹配）。"""
        problems: list[str] = []
        for name, entry in sorted(self.artifacts.items()):
            target = Path(entry.get("path", ""))
            if not target.is_file():
                problems.append(f"{name}: missing {target}")
                continue
            actual = hash_file(target)
            if entry.get("sha256") and actual != entry["sha256"]:
                problems.append(f"{name}: hash mismatch")
        return problems


def write_stage_manifest(
    path: str | Path,
    entries: list[tuple[str, Path, str]],
    *,
    code: str,
    config: str,
    inputs: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """轻量写法：把一组文件登记进 manifest。"""
    manifest = Manifest(path)
    for name, target, layer in entries:
        manifest.register(name, target, layer=layer, inputs=inputs)
    return manifest.save(code=code, config=config, extra=extra)


def manifest_delta(before: Manifest, after: Manifest) -> dict[str, list[str]]:
    all_names = sorted(set(before.artifacts) | set(after.artifacts))
    return {
        "added": [n for n in all_names if n not in before.artifacts],
        "removed": [n for n in all_names if n not in after.artifacts],
        "changed": [
            n
            for n in all_names
            if n in before.artifacts
            and n in after.artifacts
            and before.artifacts[n].get("sha256") != after.artifacts[n].get("sha256")
        ],
    }


def copy_with_manifest(src: str | Path, dst: str | Path, manifest: Manifest, name: str) -> Path:
    """把文件复制到目标位置并登记（用于 canonical → processed 装配）。"""
    target = io.atomic_copy(src, dst)
    manifest.register(name, target, layer="processed")
    return target


__all__ = [
    "ArtifactEntry",
    "Manifest",
    "artifact_hash",
    "code_hash",
    "config_hash",
    "copy_with_manifest",
    "input_hashes",
    "manifest_delta",
    "raw_md5_from_checksums",
    "write_stage_manifest",
]

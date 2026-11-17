"""原子化 I/O、哈希与轻量 XLSX 读取。

数据管线必须满足：① 所有产物原子落盘；② 相同输入可复现；③ 不引入
openpyxl 等额外依赖（CI 无 GPU/无网络环境也能解析 TCGA 的 xlsx）。
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import posixpath
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_PKG_REL_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"

NA_TOKENS = {
    "",
    "--",
    "---",
    "Not Reported",
    "not reported",
    "Unknown",
    "unknown",
    "NA",
    "N/A",
    "nan",
    "None",
    "'--",
    "'--'",
    "--",
    "#N/A",
    "[Not Available]",
    "[Not Applicable]",
    "[Unknown]",
    "[Not Evaluated]",
}


# ---------------------------------------------------------------------------
# 原子写
# ---------------------------------------------------------------------------
def _tmp_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    return Path(name)


def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    resolved = Path(path)
    tmp = _tmp_path(resolved)
    tmp.write_bytes(data)
    os.replace(tmp, resolved)
    return resolved


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> Path:
    return atomic_write_bytes(path, text.encode(encoding))


def atomic_copy(src: str | Path, dst: str | Path) -> Path:
    resolved = Path(dst)
    tmp = _tmp_path(resolved)
    shutil.copyfile(src, tmp)
    os.replace(tmp, resolved)
    return resolved


def write_json(path: str | Path, payload: Any, *, indent: int = 2) -> Path:
    def default(value: Any) -> Any:
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"无法序列化：{type(value)!r}")

    text = json.dumps(payload, ensure_ascii=False, indent=indent, sort_keys=True, default=default)
    return atomic_write_text(path, text + "\n")


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_parquet(df: pd.DataFrame, path: str | Path, *, index: bool = False) -> Path:
    resolved = Path(path)
    tmp = _tmp_path(resolved)
    df.to_parquet(tmp, index=index, compression="zstd")
    os.replace(tmp, resolved)
    return resolved


def read_parquet(path: str | Path, *, columns: Iterable[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=list(columns) if columns else None)


def write_npz(path: str | Path, arrays: Mapping[str, Any], *, compressed: bool = True) -> Path:
    resolved = Path(path)
    tmp = _tmp_path(resolved)
    with tmp.open("wb") as handle:
        if compressed:
            np.savez_compressed(handle, **arrays)
        else:
            np.savez(handle, **arrays)
    os.replace(tmp, resolved)
    return resolved


def read_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def save_packed_matrix(
    path: str | Path,
    matrix: np.ndarray | pd.DataFrame,
    row_ids: Iterable[str],
    col_ids: Iterable[str],
    *,
    name: str = "matrix",
) -> Path:
    """把 0/1 矩阵按行 packbits 后保存（体积约为 bool 矩阵的 1/8）。"""
    values = np.asarray(matrix.values if isinstance(matrix, pd.DataFrame) else matrix, dtype=bool)
    packed = np.packbits(values, axis=1)
    return write_npz(
        path,
        {
            f"{name}_packed": packed,
            f"{name}_shape": np.asarray(values.shape, dtype=np.int64),
            "row_ids": np.asarray([str(x) for x in row_ids], dtype=np.str_),
            "col_ids": np.asarray([str(x) for x in col_ids], dtype=np.str_),
        },
    )


def load_packed_matrix(path: str | Path, *, name: str = "matrix") -> tuple[np.ndarray, list[str], list[str]]:
    data = read_npz(path)
    shape = tuple(int(x) for x in data[f"{name}_shape"])
    unpacked = np.unpackbits(data[f"{name}_packed"], axis=1, count=shape[1]).astype(bool)
    return unpacked, [str(x) for x in data["row_ids"]], [str(x) for x in data["col_ids"]]


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def set_readonly(path: str | Path, *, readonly: bool = True) -> None:
    """设置/取消只读（目录递归）。"""
    resolved = Path(path)
    paths = [resolved, *resolved.rglob("*")] if resolved.is_dir() else [resolved]
    for item in paths:
        try:
            mode = item.stat().st_mode
            if readonly:
                item.chmod(mode & ~0o222)
            else:
                item.chmod(mode | 0o200)
        except OSError:
            continue


# ---------------------------------------------------------------------------
# 哈希
# ---------------------------------------------------------------------------
def hash_bytes(data: bytes, algorithm: str = "sha256") -> str:
    return hashlib.new(algorithm, data).hexdigest()


def hash_file(path: str | Path, algorithm: str = "sha256", *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def hash_dataframe(df: pd.DataFrame, *, sort_by: Iterable[str] | None = None) -> str:
    """确定性语义哈希：列名 + dtype + 按列内容（行顺序可指定排序）。"""
    frame = df.sort_values(list(sort_by)).reset_index(drop=True) if sort_by else df.reset_index(drop=True)
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"columns": list(map(str, frame.columns)), "dtypes": [str(x) for x in frame.dtypes]},
            sort_keys=True,
        ).encode()
    )
    for column in frame.columns:
        series = frame[column]
        digest.update(str(column).encode())
        if pd.api.types.is_numeric_dtype(series):
            values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)
            digest.update(np.ascontiguousarray(values).tobytes())
        elif pd.api.types.is_bool_dtype(series):
            digest.update(np.ascontiguousarray(series.to_numpy(dtype=bool)).tobytes())
        else:
            text = series.astype("string").fillna("<NA>")
            digest.update("\x1f".join(text.tolist()).encode("utf-8"))
        digest.update(b"\x1e")
    return digest.hexdigest()


def hash_ndarray(array: np.ndarray, *, sort_rows: bool = False) -> str:
    values = np.ascontiguousarray(array)
    if sort_rows and values.ndim == 2:
        order = np.lexsort(tuple(values[:, i] for i in range(values.shape[1] - 1, -1, -1)))
        values = values[order]
    digest = hashlib.sha256()
    digest.update(f"{values.dtype}|{values.shape}".encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda value: str(value) if isinstance(value, Path) else value,
    )


def stable_seed(*parts: Any, modulo: int) -> int:
    """把任意身份信息映射为确定性整数（跨平台、跨 Python 版本一致）。"""
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % modulo


# ---------------------------------------------------------------------------
# 轻量 XLSX 读取（zipfile + ElementTree，避免 openpyxl 依赖）
# ---------------------------------------------------------------------------
def _col_index(cell_ref: str) -> int:
    letters = re.match(r"([A-Z]+)", cell_ref or "")
    if not letters:
        return 0
    index = 0
    for char in letters.group(1):
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1


def _cell_value(cell: ET.Element, shared: list[str]) -> Any:
    cell_type = cell.get("t")
    if cell_type == "inlineStr":
        node = cell.find(f"{_NS}is/{_NS}t")
        return node.text if node is not None else None
    value_node = cell.find(f"{_NS}v")
    if value_node is None or value_node.text is None:
        return None
    raw = value_node.text
    if cell_type == "s":
        try:
            return shared[int(raw)]
        except (ValueError, IndexError):
            return raw
    if cell_type == "b":
        return raw == "1"
    if cell_type in {"str", "e"}:
        return raw
    try:
        number = float(raw)
    except ValueError:
        return raw
    return int(number) if number.is_integer() else number


def read_xlsx(path: str | Path) -> dict[str, list[list[Any]]]:
    """读取 xlsx 的全部 sheet，返回 ``{sheet_name: rows}``（值已解析）。"""
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall(f"{_NS}si"):
                shared.append("".join(node.text or "" for node in item.iter(f"{_NS}t")))

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rels = {
            rel.get("Id"): rel.get("Target", "")
            for rel in rels_root.findall(f"{_PKG_REL_NS}Relationship")
        }
        sheets: dict[str, list[list[Any]]] = {}
        for sheet in workbook.findall(f".//{_NS}sheet"):
            name = sheet.get("name", "")
            rel_id = sheet.get(f"{_REL_NS}id", "")
            target = rels.get(rel_id, "")
            if not target:
                continue
            member = posixpath.normpath(posixpath.join("xl", target))
            if member not in archive.namelist():
                continue
            root = ET.fromstring(archive.read(member))
            rows: list[list[Any]] = []
            for row in root.findall(f".//{_NS}row"):
                cells = list(row.findall(f"{_NS}c"))
                if not cells:
                    rows.append([])
                    continue
                width = max(_col_index(cell.get("r", "")) for cell in cells) + 1
                values: list[Any] = [None] * width
                for cell in cells:
                    values[_col_index(cell.get("r", ""))] = _cell_value(cell, shared)
                rows.append(values)
            sheets[name] = rows
        return sheets


def xlsx_sheet_frame(path: str | Path, sheet: str, *, header_row: int = 0) -> pd.DataFrame:
    """把指定 sheet 转为 DataFrame（自动裁剪全空列/行）。"""
    rows = read_xlsx(path)[sheet]
    if not rows:
        return pd.DataFrame()
    header = rows[header_row]
    body = rows[header_row + 1 :]
    width = len(header)
    columns = [str(x) if x is not None and str(x).strip() else f"col_{i}" for i, x in enumerate(header)]
    records = []
    for row in body:
        if not row or all(value is None or str(value).strip() == "" for value in row):
            continue
        padded = list(row) + [None] * (width - len(row))
        records.append(padded[:width])
    return pd.DataFrame(records, columns=columns)


def normalize_na(series: pd.Series) -> pd.Series:
    """把 GDC/TCGA 占位符转为真正的缺失值。"""
    result = series.astype("string").str.strip()
    return result.mask(result.isin(NA_TOKENS))


def coerce_frame_for_parquet(frame: pd.DataFrame, *, numeric_threshold: float = 0.9) -> pd.DataFrame:
    """把 xlsx 混合类型列转为可写入 parquet 的稳定类型。"""
    output = frame.copy()
    for column in output.columns:
        series = output[column]
        numeric = pd.to_numeric(series, errors="coerce")
        non_null = int(series.notna().sum())
        if non_null and numeric.notna().sum() >= max(1, int(non_null * numeric_threshold)):
            output[column] = numeric
        else:
            output[column] = series.astype("string")
    return output


def strip_numeric_suffix(value: Any) -> str | None:
    """去掉 ``905967.0`` 这类由 pandas 推断产生的浮点后缀。"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    return text[:-2] if re.fullmatch(r"\d+\.0", text) else text


def open_text(path: str | Path):
    """自动识别 gzip 的文本打开方式。"""
    resolved = Path(path)
    if resolved.suffix == ".gz":
        return gzip.open(resolved, "rt", encoding="utf-8", errors="replace")
    return resolved.open(encoding="utf-8", errors="replace")


__all__ = [
    "NA_TOKENS",
    "atomic_copy",
    "atomic_write_bytes",
    "atomic_write_text",
    "ensure_dir",
    "hash_bytes",
    "hash_dataframe",
    "hash_file",
    "hash_ndarray",
    "load_packed_matrix",
    "normalize_na",
    "open_text",
    "read_json",
    "read_npz",
    "read_parquet",
    "read_xlsx",
    "save_packed_matrix",
    "set_readonly",
    "stable_seed",
    "strip_numeric_suffix",
    "write_json",
    "write_npz",
    "write_parquet",
    "xlsx_sheet_frame",
]

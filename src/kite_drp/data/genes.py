"""HGNC 基因真源、符号/Ensembl/Entrez 映射与基因宇宙定义。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from kite_drp.data import io, registry

_GENE_COL_RE = re.compile(r"^(.+?) \((\d+)\)$")
_ENSEMBL_RE = re.compile(r"^(ENSG\d+)(?:_PAR_Y)?$")


def normalize_entrez(value: object) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text[:-2] if re.fullmatch(r"\d+\.0", text) else text


def parse_gene_column(column: str) -> dict[str, object] | None:
    """解析 ``SYMBOL (ENTREZ)``；返回 symbol/entrez/par_y。"""
    if not isinstance(column, str):
        return None
    match = _GENE_COL_RE.match(column.strip())
    if match:
        return {
            "symbol": match.group(1),
            "entrez_id": match.group(2),
            "ensembl_gene_id": None,
            "par_y": False,
            "raw": column,
        }
    ensembl = _ENSEMBL_RE.match(column.strip())
    if ensembl:
        return {
            "symbol": None,
            "entrez_id": None,
            "ensembl_gene_id": ensembl.group(1),
            "par_y": column.strip().endswith("_PAR_Y"),
            "raw": column,
        }
    return None


def _split_multi(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [token.strip() for token in text.split("|") if token.strip()]


@dataclass
class GeneMaster:
    frame: pd.DataFrame
    symbol_to_row: dict[str, int]
    alias_to_symbol: dict[str, str]
    prev_to_symbol: dict[str, str]
    ensembl_to_symbol: dict[str, str]
    entrez_to_symbol: dict[str, str]
    _current_upper: set[str]

    def resolve(self, value: object) -> tuple[str | None, str]:
        """把任意基因标识解析到当前 HGNC symbol；返回 ``(symbol, method)``。"""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None, "missing"
        text = str(value).strip()
        if not text:
            return None, "missing"
        if text in self.symbol_to_row:
            return text, "symbol"
        upper = text.upper()
        if upper in self._current_upper:
            row = self.symbol_to_row.get(text)
            if row is not None:
                return text, "symbol"
        if text in self.prev_to_symbol:
            return self.prev_to_symbol[text], "prev_symbol"
        if text in self.alias_to_symbol:
            return self.alias_to_symbol[text], "alias_symbol"
        if text in self.ensembl_to_symbol:
            return self.ensembl_to_symbol[text], "ensembl"
        if text in self.entrez_to_symbol:
            return self.entrez_to_symbol[text], "entrez"
        return None, "unmapped"

    def resolve_series(self, values: Iterable[object]) -> pd.DataFrame:
        records = []
        for value in values:
            symbol, method = self.resolve(value)
            records.append({"source_id": value, "hgnc_symbol": symbol, "method": method})
        return pd.DataFrame(records)


def build_gene_master(hgnc_path: str | Path) -> GeneMaster:
    """读取 HGNC Gene.csv，构建唯一真源与唯一别名字典。"""
    raw = pd.read_csv(hgnc_path, dtype=str, low_memory=False)
    required = {"hgnc_id", "symbol", "locus_group"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"HGNC 缺列：{sorted(missing)}")

    frame = pd.DataFrame(
        {
            "hgnc_id": raw["hgnc_id"],
            "hgnc_symbol": raw["symbol"].astype(str).str.strip(),
            "name": raw.get("name", ""),
            "locus_group": raw.get("locus_group", ""),
            "locus_type": raw.get("locus_type", ""),
            "status": raw.get("status", ""),
            "entrez_id": raw.get("entrez_id", pd.Series(index=raw.index, dtype=object)).map(normalize_entrez),
            "ensembl_gene_id": raw.get("ensembl_gene_id", ""),
            "alias_json": raw.get("alias_symbol", "").map(lambda x: json.dumps(_split_multi(x))),
            "prev_json": raw.get("prev_symbol", "").map(lambda x: json.dumps(_split_multi(x))),
        }
    )
    frame = frame[frame["hgnc_symbol"].notna() & (frame["hgnc_symbol"] != "")].copy()
    frame["is_protein_coding"] = frame["locus_group"].eq("protein-coding gene")
    frame = frame.sort_values("hgnc_symbol").reset_index(drop=True)

    symbol_to_row = {symbol: int(idx) for idx, symbol in enumerate(frame["hgnc_symbol"])}
    current_upper = {symbol.upper() for symbol in symbol_to_row}

    def unique_map(pairs: Iterable[tuple[str, str]]) -> dict[str, str]:
        seen: dict[str, set[str]] = {}
        for source, target in pairs:
            seen.setdefault(source, set()).add(target)
        return {source: next(iter(targets)) for source, targets in seen.items() if len(targets) == 1}

    alias_pairs: list[tuple[str, str]] = []
    prev_pairs: list[tuple[str, str]] = []
    for symbol, alias_json, prev_json in zip(
        frame["hgnc_symbol"], frame["alias_json"], frame["prev_json"], strict=False
    ):
        for alias in json.loads(alias_json):
            if alias not in symbol_to_row:
                alias_pairs.append((alias, symbol))
        for prev in json.loads(prev_json):
            if prev not in symbol_to_row:
                prev_pairs.append((prev, symbol))

    alias_to_symbol = unique_map(alias_pairs)
    prev_to_symbol = unique_map(prev_pairs)
    # 别名/旧名不得覆盖任何当前 symbol
    alias_to_symbol = {key: value for key, value in alias_to_symbol.items() if key not in symbol_to_row}
    prev_to_symbol = {key: value for key, value in prev_to_symbol.items() if key not in symbol_to_row}

    ensembl_to_symbol = unique_map(
        (str(ensg).strip(), symbol)
        for ensg, symbol in zip(frame["ensembl_gene_id"], frame["hgnc_symbol"], strict=False)
        if ensg and str(ensg).strip() and str(ensg).lower() != "nan"
    )
    entrez_to_symbol = unique_map(
        (str(entrez), symbol)
        for entrez, symbol in zip(frame["entrez_id"], frame["hgnc_symbol"], strict=False)
        if entrez
    )
    return GeneMaster(
        frame=frame,
        symbol_to_row=symbol_to_row,
        alias_to_symbol=alias_to_symbol,
        prev_to_symbol=prev_to_symbol,
        ensembl_to_symbol=ensembl_to_symbol,
        entrez_to_symbol=entrez_to_symbol,
        _current_upper=current_upper,
    )


def build_gene_artifact(
    hgnc_path: str | Path,
    out_path: str | Path,
    *,
    universes: dict[str, list[str]] | None = None,
    inputs: dict[str, str] | None = None,
    config: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, GeneMaster]:
    master = build_gene_master(hgnc_path)
    frame = master.frame.copy()
    for name, genes in (universes or {}).items():
        members = set(genes)
        frame[f"in_{name}"] = frame["hgnc_symbol"].isin(members)
    io.write_parquet(frame, out_path)
    if universes:
        io.write_json(
            Path(out_path).parent / "gene_universes.json",
            {
                "dimensions": {name: len(genes) for name, genes in universes.items()},
                "genes": {name: sorted(genes) for name, genes in universes.items()},
                "config": config or {},
            },
        )
    return frame, master


def parse_symbol_columns(columns: Iterable[str]) -> pd.DataFrame:
    """批量解析矩阵列头，返回 raw/symbol/entrez/par_y。"""
    records = []
    for column in columns:
        parsed = parse_gene_column(str(column))
        if parsed is None:
            records.append(
                {"raw": str(column), "symbol": None, "entrez_id": None, "ensembl_gene_id": None, "par_y": False}
            )
        else:
            records.append(parsed)
    return pd.DataFrame(records)


def resolve_columns(
    columns: Iterable[str],
    master: GeneMaster,
    *,
    allow_ensembl: bool = True,
) -> pd.DataFrame:
    """把 DepMap 宽表列头映射到 HGNC symbol；无法映射者为 None。"""
    parsed = parse_symbol_columns(columns)
    records = []
    for row in parsed.itertuples(index=False):
        symbol = None
        method = "unmapped"
        if isinstance(row.symbol, str) and row.symbol:
            symbol, method = master.resolve(row.symbol)
        if symbol is None and allow_ensembl and isinstance(row.ensembl_gene_id, str):
            symbol, method = master.resolve(row.ensembl_gene_id)
        records.append(
            {
                "raw": row.raw,
                "hgnc_symbol": symbol,
                "method": method,
                "par_y": bool(row.par_y),
                "entrez_id": row.entrez_id,
            }
        )
    return pd.DataFrame(records)


def build_universes(
    *,
    master: GeneMaster,
    expression_genes: Iterable[str],
    cnv_genes: Iterable[str],
    mutation_genes_A: Iterable[str],
    mutation_genes_B: Iterable[str],
    ppi_genes: Iterable[str] = (),
    ppi_degree: dict[str, int] | None = None,
    pathway_genes: Iterable[str] = (),
    panel_genes: Iterable[str] = (),
    vae_size: int = 1600,
) -> dict[str, list[str]]:
    """按规划 §3.4 定义嵌套基因宇宙（全部去重、排序、确定性）。"""
    coding = set(master.frame.loc[master.frame["is_protein_coding"], "hgnc_symbol"])
    expr = set(expression_genes) & coding
    cnv = set(cnv_genes) & coding
    teach_ge = expr & cnv
    teach_mu_a = set(mutation_genes_A) & teach_ge
    teach_mu_b = set(mutation_genes_B) & teach_ge
    ppi = set(ppi_genes) & teach_ge
    pathways = set(pathway_genes) & teach_ge
    panel = set(panel_genes) & teach_mu_a

    degree = ppi_degree or {}
    candidates = sorted(ppi, key=lambda gene: (-degree.get(gene, 0), gene))
    vae = candidates[:vae_size]

    return {
        "hgnc_pc": sorted(coding),
        "teach_ge": sorted(teach_ge),
        "teach_mu_A": sorted(teach_mu_a),
        "teach_mu_B": sorted(teach_mu_b),
        "cnv": sorted(cnv & teach_ge),
        "vae": sorted(vae),
        "panel": sorted(panel),
        "pathway": sorted(pathways),
    }


def gene_universe_path() -> Path:
    return registry.resolve("gene_universes")


__all__ = [
    "GeneMaster",
    "build_gene_artifact",
    "build_gene_master",
    "build_universes",
    "gene_universe_path",
    "normalize_entrez",
    "parse_gene_column",
    "parse_symbol_columns",
    "resolve_columns",
]

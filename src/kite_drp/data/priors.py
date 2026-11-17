"""先验网络：STRING PPI、药物-靶点（DTI）与通路基因集。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from kite_drp.data import io
from kite_drp.data.genes import GeneMaster
from kite_drp.data.ids import normalize_drug_name


def _ensp(value: str) -> str:
    return str(value).strip().removeprefix("9606.")


@dataclass
class PpiData:
    edges: pd.DataFrame
    degree: dict[str, int]
    info: dict[str, Any] = field(default_factory=dict)


def build_ppi(
    links_path: str | Path,
    aliases_path: str | Path,
    *,
    cutoff: int = 700,
    universe: set[str] | None = None,
) -> PpiData:
    """STRING links 去重 → 阈值 → ENSP→HGNC symbol → 基因级边表。"""
    aliases = pd.read_csv(
        aliases_path,
        sep="\t",
        comment="#",
        names=["protein_id", "alias", "source"],
        dtype=str,
        low_memory=False,
    )
    aliases = aliases[aliases["source"] == "Ensembl_HGNC_symbol"].copy()
    aliases["protein_id"] = aliases["protein_id"].map(_ensp)
    symbol_by_protein = (
        aliases.dropna(subset=["alias"]).drop_duplicates("protein_id").set_index("protein_id")["alias"].to_dict()
    )

    unique_pairs: dict[tuple[str, str], int] = {}
    threshold_counts = {400: 0, 700: 0, 900: 0}
    rows = 0
    with io.open_text(links_path) as handle:
        header = handle.readline()
        if "protein1" not in header:
            raise ValueError(f"STRING links 表头异常：{header!r}")
        for line in handle:
            parts = line.rstrip("\n").split(" ")
            if len(parts) != 3:
                continue
            left, right, score_text = parts
            score = int(score_text)
            rows += 1
            for threshold in threshold_counts:
                if score >= threshold:
                    threshold_counts[threshold] += 1
            if score < cutoff:
                continue
            a, b = _ensp(left), _ensp(right)
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            previous = unique_pairs.get(key)
            if previous is None or score > previous:
                unique_pairs[key] = score

    records = []
    unmapped_proteins: set[str] = set()
    for (protein_a, protein_b), score in sorted(unique_pairs.items()):
        symbol_a = symbol_by_protein.get(protein_a)
        symbol_b = symbol_by_protein.get(protein_b)
        if symbol_a is None:
            unmapped_proteins.add(protein_a)
        if symbol_b is None:
            unmapped_proteins.add(protein_b)
        if symbol_a is None or symbol_b is None:
            continue
        if symbol_a == symbol_b:
            continue
        records.append({"gene_a": symbol_a, "gene_b": symbol_b, "combined_score": score})
    edges = pd.DataFrame(records, columns=["gene_a", "gene_b", "combined_score"])
    if not edges.empty:
        # 同一对基因在蛋白层可能多对；合并取最大分（此处只有阈值，score 统一）
        edges["pair"] = edges.apply(lambda row: tuple(sorted((row.gene_a, row.gene_b))), axis=1)
        edges = edges.drop_duplicates("pair").drop(columns="pair")
        if universe is not None:
            edges = edges[edges["gene_a"].isin(universe) & edges["gene_b"].isin(universe)]
        edges = edges.sort_values(["gene_a", "gene_b"]).reset_index(drop=True)
    degree_series = pd.concat([edges["gene_a"], edges["gene_b"]]).value_counts()
    degree = {str(gene): int(count) for gene, count in degree_series.items()}
    info = {
        "rows": rows,
        "unique_undirected_pairs_total": rows // 2,
        "unique_pairs_above_cutoff": len(unique_pairs),
        "threshold_counts_rows": threshold_counts,
        "unmapped_proteins": len(unmapped_proteins),
        "edges_gene_level": int(len(edges)),
        "genes": int(len(degree)),
        "cutoff": cutoff,
        "universe_genes": len(universe) if universe is not None else None,
    }
    return PpiData(edges=edges, degree=degree, info=info)


def build_dti(
    compounds: pd.DataFrame,
    dgidb_path: str | Path,
    master: GeneMaster,
    *,
    allowed_genes: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """合并 GDSC 靶点文本与 DGIdb（仅兜底/解释），输出 drug-target 边。"""
    allowed = set(allowed_genes) if allowed_genes is not None else None
    records: list[dict[str, Any]] = []
    for row in compounds.itertuples(index=False):
        for gene in row.target_genes:
            if allowed is None or gene in allowed:
                records.append(
                    {
                        "compound_uid": row.compound_uid,
                        "hgnc_symbol": gene,
                        "source": "gdsc_target",
                        "score": 1.0,
                    }
                )

    name_to_uid: dict[str, str] = {}
    for row in compounds.itertuples(index=False):
        for name in [row.drug_name, *getattr(row, "gdsc_names", [])]:
            key = normalize_drug_name(name)
            if key:
                name_to_uid.setdefault(key, row.compound_uid)

    interactions = pd.read_csv(dgidb_path, sep="\t", comment="#", low_memory=False)
    drug_columns = [column for column in ("drug_name", "drug_claim_name") if column in interactions.columns]
    gene_column = "gene_name" if "gene_name" in interactions.columns else "gene_claim_name"
    matched_drugs: set[str] = set()
    for row in interactions.itertuples(index=False):
        uid = None
        for column in drug_columns:
            uid = name_to_uid.get(normalize_drug_name(getattr(row, column)))
            if uid:
                break
        if not uid:
            continue
        symbol, _ = master.resolve(getattr(row, gene_column))
        if not symbol or (allowed is not None and symbol not in allowed):
            continue
        matched_drugs.add(uid)
        raw_score = getattr(row, "interaction_score", None)
        try:
            score = float(raw_score) if raw_score is not None and not pd.isna(raw_score) else 0.0
        except (TypeError, ValueError):
            score = 0.0
        records.append(
            {
                "compound_uid": uid,
                "hgnc_symbol": symbol,
                "source": "dgidb",
                "score": score,
            }
        )
    dti = pd.DataFrame(records, columns=["compound_uid", "hgnc_symbol", "source", "score"])
    if not dti.empty:
        grouped = dti.groupby(["compound_uid", "hgnc_symbol"])
        sources = grouped["source"].agg(lambda values: "+".join(sorted(set(values))))
        scores = grouped["score"].max()
        dti = (
            pd.concat([sources.rename("source"), scores.rename("score")], axis=1)
            .reset_index()
            .sort_values(["compound_uid", "hgnc_symbol"])
        )
    info = {
        "edges": int(len(dti)),
        "compounds_covered": int(dti["compound_uid"].nunique()) if not dti.empty else 0,
        "genes_covered": int(dti["hgnc_symbol"].nunique()) if not dti.empty else 0,
        "gdsc_target_compounds": int(compounds["has_target"].sum()) if "has_target" in compounds else 0,
        "dgidb_matched_compounds": len(matched_drugs),
    }
    return dti, info


def find_gmt_files(source: str | Path) -> list[Path]:
    """在目录/文件中查找 GMT。"""
    candidate = Path(source)
    if candidate.is_file() and candidate.suffix.lower() == ".gmt":
        return [candidate]
    if candidate.is_dir():
        return sorted(candidate.rglob("*.gmt"))
    return []


def build_pathways(
    gmt_files: Iterable[str | Path],
    master: GeneMaster,
    *,
    universe: set[str],
    min_genes: int = 10,
    source_name: str = "external",
) -> tuple[pd.DataFrame, list[str]]:
    """解析 GMT → (pathway_id, source, gene) 长表；只保留 >=min_genes 的通路。"""
    records: list[dict[str, Any]] = []
    pathway_genes: dict[str, set[str]] = {}
    for gmt in gmt_files:
        path = Path(gmt)
        with path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                pathway_id = parts[0].strip()
                genes = parts[2:]
                resolved = set()
                for gene in genes:
                    symbol, _ = master.resolve(gene)
                    if symbol and symbol in universe:
                        resolved.add(symbol)
                if len(resolved) >= min_genes:
                    pathway_genes[f"{source_name}:{pathway_id}"] = resolved
                    for gene in sorted(resolved):
                        records.append(
                            {
                                "pathway_id": f"{source_name}:{pathway_id}",
                                "source": source_name,
                                "gene": gene,
                            }
                        )
    pathways = pd.DataFrame(records, columns=["pathway_id", "source", "gene"])
    if not pathways.empty:
        counts = pathways.groupby("pathway_id")["gene"].transform("size")
        pathways = pathways[counts >= min_genes].sort_values(["pathway_id", "gene"]).reset_index(drop=True)
        pathways["n_genes"] = pathways.groupby("pathway_id")["gene"].transform("size")
    kept = sorted(pathway_genes)
    return pathways, kept


__all__ = [
    "PpiData",
    "build_dti",
    "build_pathways",
    "build_ppi",
    "find_gmt_files",
]

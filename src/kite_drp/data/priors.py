"""先验网络：STRING PPI、药物-靶点（DTI）与通路基因集。"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


def _array_items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    return [value]


def _load_opentargets_edges(
    opentargets_dir: str | Path,
    compounds: pd.DataFrame,
    master: GeneMaster,
    *,
    allowed: set[str] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Open Targets drug_molecule + MOA + target → GDSC DTI 边。"""
    root = Path(opentargets_dir)
    target_files = sorted((root / "target").glob("*.parquet"))
    drug_files = sorted((root / "drug_molecule").glob("*.parquet"))
    moa_files = sorted((root / "drug_mechanism_of_action").glob("*.parquet"))
    if not (target_files and drug_files and moa_files):
        return [], {"status": "missing"}
    targets = pd.concat([pd.read_parquet(path) for path in target_files], ignore_index=True)
    drugs = pd.concat([pd.read_parquet(path) for path in drug_files], ignore_index=True)
    moa = pd.concat([pd.read_parquet(path) for path in moa_files], ignore_index=True)

    ensembl_to_symbol: dict[str, str] = {}
    chembl_target_to_symbol: dict[str, str] = {}
    for row in targets.itertuples(index=False):
        symbol = str(row.approvedSymbol).strip()
        if not symbol:
            continue
        ensembl_to_symbol[str(row.id)] = symbol
        for xref in _array_items(row.dbXrefs):
            if isinstance(xref, dict) and str(xref.get("source", "")).lower() == "chembl":
                chembl_target_to_symbol[str(xref.get("id"))] = symbol

    inchikey_to_drug: dict[str, str] = {}
    name_to_drug: dict[str, str] = {}
    for row in drugs.itertuples(index=False):
        drug_id = str(row.id)
        inchikey = getattr(row, "inchiKey", None)
        if isinstance(inchikey, str) and inchikey:
            inchikey_to_drug.setdefault(inchikey, drug_id)
        for value in [getattr(row, "name", None), *_array_items(getattr(row, "synonyms", None))]:
            if isinstance(value, dict):
                value = value.get("label")
            key = normalize_drug_name(value)
            if key:
                name_to_drug.setdefault(key, drug_id)

    compound_to_drug: dict[str, str] = {}
    for row in compounds.itertuples(index=False):
        drug_id = None
        inchikey = getattr(row, "inchikey", None)
        if isinstance(inchikey, str) and inchikey:
            drug_id = inchikey_to_drug.get(inchikey)
        if drug_id is None:
            for name in [row.drug_name, *getattr(row, "gdsc_names", [])]:
                drug_id = name_to_drug.get(normalize_drug_name(name))
                if drug_id:
                    break
        if drug_id:
            compound_to_drug[row.compound_uid] = drug_id
    drug_to_compounds: dict[str, list[str]] = {}
    for compound_uid, drug_id in compound_to_drug.items():
        drug_to_compounds.setdefault(drug_id, []).append(compound_uid)

    records: list[dict[str, Any]] = []
    for row in moa.itertuples(index=False):
        target_ids = [str(value) for value in _array_items(getattr(row, "targets", None))]
        target_name = str(getattr(row, "targetName", "") or "")
        symbols = []
        for target_id in target_ids:
            symbol = ensembl_to_symbol.get(target_id)
            if symbol is None:
                symbol, _ = master.resolve(target_id)
            if symbol:
                symbols.append(symbol)
        if not symbols and target_name:
            symbol, _ = master.resolve(target_name)
            if symbol:
                symbols.append(symbol)
        if not symbols:
            continue
        for chembl_id in _array_items(getattr(row, "chemblIds", None)):
            for compound_uid in drug_to_compounds.get(str(chembl_id), []):
                for symbol in sorted(set(symbols)):
                    if allowed is not None and symbol not in allowed:
                        continue
                    records.append(
                        {
                            "compound_uid": compound_uid,
                            "hgnc_symbol": symbol,
                            "source": "opentargets",
                            "score": 1.0,
                            "action_type": str(getattr(row, "actionType", "") or ""),
                            "mechanism": str(getattr(row, "mechanismOfAction", "") or ""),
                            "target_name": target_name,
                        }
                    )
    return records, {
        "status": "ok",
        "n_targets": int(len(targets)),
        "n_drugs": int(len(drugs)),
        "n_moa": int(len(moa)),
        "matched_compounds": len(compound_to_drug),
        "edges": len(records),
    }


def _load_chembl_edges(
    chembl_dir: str | Path,
    compounds: pd.DataFrame,
    master: GeneMaster,
    *,
    allowed: set[str] | None,
    opentargets_dir: str | Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """ChEMBL mechanism → GDSC DTI 边（target ChEMBL ID 经 Open Targets dbXrefs 映射）。"""
    root = Path(chembl_dir)
    mechanism_files = sorted(root.glob("*mechanism.tsv.gz"))
    if not mechanism_files:
        return [], {"status": "missing"}
    chembl_target_to_symbol: dict[str, str] = {}
    if opentargets_dir:
        target_files = sorted(Path(opentargets_dir).glob("target/*.parquet"))
        for path in target_files:
            targets = pd.read_parquet(path, columns=["approvedSymbol", "dbXrefs"])
            for row in targets.itertuples(index=False):
                symbol = str(row.approvedSymbol).strip()
                if not symbol:
                    continue
                for xref in _array_items(row.dbXrefs):
                    if isinstance(xref, dict) and str(xref.get("source", "")).lower() == "chembl":
                        chembl_target_to_symbol[str(xref.get("id"))] = symbol

    # 与 Open Targets 共用 InChIKey/名称桥接：先建立 ChEMBL ID -> compound_uid。
    drug_files = (
        sorted(Path(opentargets_dir).glob("drug_molecule/*.parquet"))
        if opentargets_dir
        else []
    )
    inchikey_to_chembl: dict[str, str] = {}
    name_to_chembl: dict[str, str] = {}
    for path in drug_files:
        drugs = pd.read_parquet(path, columns=["id", "inchiKey", "name", "synonyms"])
        for row in drugs.itertuples(index=False):
            drug_id = str(row.id)
            if isinstance(row.inchiKey, str) and row.inchiKey:
                inchikey_to_chembl.setdefault(row.inchiKey, drug_id)
            for value in [row.name, *_array_items(row.synonyms)]:
                if isinstance(value, dict):
                    value = value.get("label")
                key = normalize_drug_name(value)
                if key:
                    name_to_chembl.setdefault(key, drug_id)
    compound_to_chembl: dict[str, str] = {}
    for row in compounds.itertuples(index=False):
        chembl_id = None
        if isinstance(getattr(row, "inchikey", None), str):
            chembl_id = inchikey_to_chembl.get(row.inchikey)
        if chembl_id is None:
            for name in [row.drug_name, *getattr(row, "gdsc_names", [])]:
                chembl_id = name_to_chembl.get(normalize_drug_name(name))
                if chembl_id:
                    break
        if chembl_id:
            compound_to_chembl[row.compound_uid] = chembl_id
    chembl_to_compounds: dict[str, list[str]] = {}
    for compound_uid, chembl_id in compound_to_chembl.items():
        chembl_to_compounds.setdefault(chembl_id, []).append(compound_uid)

    mechanism = pd.read_csv(mechanism_files[0], sep="\t", dtype=str, low_memory=False)
    records: list[dict[str, Any]] = []
    for row in mechanism.itertuples(index=False):
        symbol = chembl_target_to_symbol.get(str(getattr(row, "target_chembl_id", "")))
        if symbol is None:
            symbol, _ = master.resolve(getattr(row, "target_chembl_id", None))
        if not symbol or (allowed is not None and symbol not in allowed):
            continue
        for key in [
            getattr(row, "molecule_chembl_id", None),
            getattr(row, "parent_molecule_chembl_id", None),
        ]:
            for compound_uid in chembl_to_compounds.get(str(key), []):
                records.append(
                    {
                        "compound_uid": compound_uid,
                        "hgnc_symbol": symbol,
                        "source": "chembl",
                        "score": 1.0,
                        "action_type": str(getattr(row, "action_type", "") or ""),
                        "mechanism": str(getattr(row, "mechanism_of_action", "") or ""),
                        "target_name": str(getattr(row, "target_chembl_id", "") or ""),
                    }
                )
    return records, {
        "status": "ok",
        "n_mechanisms": int(len(mechanism)),
        "matched_compounds": len(compound_to_chembl),
        "edges": len(records),
    }


def build_dti(
    compounds: pd.DataFrame,
    dgidb_path: str | Path,
    master: GeneMaster,
    *,
    allowed_genes: Iterable[str] | None = None,
    opentargets_dir: str | Path | None = None,
    chembl_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """合并 GDSC 靶点、Open Targets、ChEMBL 与 DGIdb，输出 drug-target 边。

    Open Targets 为主源（CC0），ChEMBL 补充机制，DGIdb 仅兜底/解释。
    """
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
                        "action_type": "",
                        "mechanism": "",
                        "target_name": gene,
                    }
                )

    ot_records: list[dict[str, Any]] = []
    ot_info: dict[str, Any] = {"status": "not_configured"}
    if opentargets_dir:
        ot_records, ot_info = _load_opentargets_edges(
            opentargets_dir, compounds, master, allowed=allowed
        )
        records.extend(ot_records)

    chembl_records: list[dict[str, Any]] = []
    chembl_info: dict[str, Any] = {"status": "not_configured"}
    if chembl_dir:
        chembl_records, chembl_info = _load_chembl_edges(
            chembl_dir,
            compounds,
            master,
            allowed=allowed,
            opentargets_dir=opentargets_dir,
        )
        records.extend(chembl_records)

    name_to_uid: dict[str, str] = {}
    for row in compounds.itertuples(index=False):
        for name in [row.drug_name, *getattr(row, "gdsc_names", [])]:
            key = normalize_drug_name(name)
            if key:
                name_to_uid.setdefault(key, row.compound_uid)

    interactions = pd.read_csv(dgidb_path, sep="\t", comment="#", low_memory=False)
    drug_columns = [
        column for column in ("drug_name", "drug_claim_name") if column in interactions.columns
    ]
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
                "action_type": "",
                "mechanism": "",
                "target_name": symbol,
            }
        )

    dti = pd.DataFrame(
        records,
        columns=[
            "compound_uid",
            "hgnc_symbol",
            "source",
            "score",
            "action_type",
            "mechanism",
            "target_name",
        ],
    )
    if not dti.empty:
        grouped = dti.groupby(["compound_uid", "hgnc_symbol"])
        sources = grouped["source"].agg(lambda values: "+".join(sorted(set(values))))
        scores = grouped["score"].max()
        actions = grouped["action_type"].agg(
            lambda values: next((value for value in values if value), "")
        )
        mechanisms = grouped["mechanism"].agg(
            lambda values: next((value for value in values if value), "")
        )
        target_names = grouped["target_name"].agg(
            lambda values: next((value for value in values if value), "")
        )
        dti = (
            pd.concat(
                [
                    sources.rename("source"),
                    scores.rename("score"),
                    actions.rename("action_type"),
                    mechanisms.rename("mechanism"),
                    target_names.rename("target_name"),
                ],
                axis=1,
            )
            .reset_index()
            .sort_values(["compound_uid", "hgnc_symbol"])
        )
    by_source: dict[str, int] = {}
    for source in ("gdsc_target", "opentargets", "chembl", "dgidb"):
        by_source[source] = (
            int(dti["source"].str.contains(source, regex=False).sum()) if not dti.empty else 0
        )
    info = {
        "edges": int(len(dti)),
        "compounds_covered": int(dti["compound_uid"].nunique()) if not dti.empty else 0,
        "genes_covered": int(dti["hgnc_symbol"].nunique()) if not dti.empty else 0,
        "gdsc_target_compounds": (
            int(compounds["has_target"].sum()) if "has_target" in compounds else 0
        ),
        "dgidb_matched_compounds": len(matched_drugs),
        "by_source_edges": by_source,
        "opentargets": ot_info,
        "chembl": chembl_info,
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


def _infer_source(path: Path) -> str:
    text = str(path).lower()
    if "reactome" in text:
        return "reactome"
    if "go" in text or "goa" in text:
        return "go"
    return "external"


def build_pathways(
    gmt_files: Iterable[str | Path],
    master: GeneMaster,
    *,
    universe: set[str],
    min_genes: int = 10,
    max_genes: int | None = 500,
    max_pathways: int | None = 1000,
    source_name: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """解析 GMT → (pathway_id, source, gene) 长表。

    过滤：>=min_genes；若给 max_genes，剔除过大通路；若给 max_pathways，
    按基因数降序 + pathway_id 确定性截断（保证端侧权重矩阵可控）。
    """
    records: list[dict[str, Any]] = []
    pathway_genes: dict[str, set[str]] = {}
    for gmt in gmt_files:
        path = Path(gmt)
        source = source_name or _infer_source(path)
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
                if (
                    len(resolved) >= min_genes
                    and (max_genes is None or len(resolved) <= max_genes)
                ):
                    pathway_genes[f"{source}:{pathway_id}"] = resolved
                    for gene in sorted(resolved):
                        records.append(
                            {
                                "pathway_id": f"{source}:{pathway_id}",
                                "source": source,
                                "gene": gene,
                            }
                        )
    pathways = pd.DataFrame(records, columns=["pathway_id", "source", "gene"])
    if not pathways.empty:
        counts = pathways.groupby("pathway_id")["gene"].transform("size")
        pathways = (
            pathways[counts >= min_genes]
            .sort_values(["pathway_id", "gene"])
            .reset_index(drop=True)
        )
        pathways["n_genes"] = pathways.groupby("pathway_id")["gene"].transform("size")
        if max_pathways is not None:
            sizes = pathways[["pathway_id", "source", "n_genes"]].drop_duplicates()
            sources = sorted(sizes["source"].unique().tolist())
            per_source = max(1, max_pathways // max(len(sources), 1))
            keep_ids: set[str] = set()
            for source in sources:
                subset = sizes[sizes["source"] == source].sort_values(
                    ["n_genes", "pathway_id"], ascending=[False, True]
                )
                keep_ids.update(subset["pathway_id"].head(per_source).tolist())
            pathways = pathways[pathways["pathway_id"].isin(keep_ids)]
            pathways = pathways.sort_values(["pathway_id", "gene"]).reset_index(drop=True)
            pathways["n_genes"] = pathways.groupby("pathway_id")["gene"].transform("size")
            pathway_genes = {
                pathway_id: genes
                for pathway_id, genes in pathway_genes.items()
                if pathway_id in keep_ids
            }
    kept = sorted(pathway_genes)
    return pathways, kept


__all__ = [
    "PpiData",
    "build_dti",
    "build_pathways",
    "build_ppi",
    "find_gmt_files",
]

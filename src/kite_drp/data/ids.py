"""细胞系 / 化合物 ID 归一与审计。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from kite_drp.data import io
from kite_drp.data.genes import GeneMaster

_NON_ALNUM = re.compile(r"[^A-Za-z0-9]")


def normalize_name(value: object) -> str:
    """审计口径的名字归一：仅保留字母数字并大写。"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return _NON_ALNUM.sub("", str(value)).upper()


def normalize_drug_name(value: object) -> str:
    """药名归一：大写、去标点、去剂量后缀。"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip().upper()
    text = re.sub(r"\((?:[^)]*?(?:UM|NM|MM|MG|ML|%)[^)]*)\)", "", text)
    return _NON_ALNUM.sub("", text)


def _clean_str(value: object) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    return text


def _cosmic(value: object) -> str | None:
    text = _clean_str(value)
    if text and re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return text


def _split_targets(value: object) -> list[str]:
    text = _clean_str(value)
    if not text:
        return []
    tokens = re.split(r"[,;/&|+]+", text)
    result: list[str] = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        # 只保留像基因名的 token；机制词（Microtubule destabiliser 等）留给 unmapped
        for part in re.split(r"\s+", token):
            if part and re.fullmatch(r"[A-Za-z0-9_.-]{1,30}", part):
                result.append(part)
    return sorted(set(result))


# ---------------------------------------------------------------------------
# 细胞系
# ---------------------------------------------------------------------------
def build_cell_lines(
    model_path: str | Path,
    *,
    has_ge: set[str] | None = None,
    has_mu: set[str] | None = None,
    has_cnv: set[str] | None = None,
    has_meth: set[str] | None = None,
    mapped_cell_lines: set[str] | None = None,
    excluded_cell_lines: set[str] | None = None,
) -> pd.DataFrame:
    """构建 ``cell_lines.parquet``（Model.csv 为权威 ID 表）。"""
    raw = pd.read_csv(model_path, dtype=str, low_memory=False)
    frame = pd.DataFrame(
        {
            "cell_line_uid": raw["ModelID"].astype(str).str.strip(),
            "sanger_model_id": raw.get("SangerModelID"),
            "cosmic_id": raw.get("COSMICID").map(_cosmic),
            "ccle_name": raw.get("CCLEName"),
            "stripped_cell_line_name": raw.get("StrippedCellLineName"),
            "cell_line_name": raw.get("CellLineName"),
            "oncotree_lineage": raw.get("OncotreeLineage"),
            "oncotree_primary_disease": raw.get("OncotreePrimaryDisease"),
            "oncotree_subtype": raw.get("OncotreeSubtype"),
            "oncotree_code": raw.get("OncotreeCode"),
            "model_type": raw.get("ModelType"),
            "source_type": raw.get("SourceType"),
            "sex": raw.get("Sex"),
            "age": pd.to_numeric(raw.get("Age"), errors="coerce"),
            "engineered": raw.get("EngineeredModel").notna(),
        }
    )
    frame["source_group"] = [
        f"cosmicid:{cosmic}"
        if isinstance(cosmic, str) and cosmic.strip()
        else f"name:{normalize_name(name)}"
        for cosmic, name in zip(frame["cosmic_id"], frame["stripped_cell_line_name"], strict=False)
    ]
    # 同名归一化合并（例如 KM-H2/KMH-2 这类跨库重名/后缀变体）
    name_counts = frame.groupby("source_group")["cell_line_uid"].transform("size")
    # 若 name 组 >1，统一加前缀保留组语义；cosmic 组天然唯一
    frame["source_group"] = np.where(
        frame["source_group"].str.startswith("name:") & (name_counts > 1),
        frame["source_group"],
        frame["source_group"],
    )
    frame["has_ge"] = frame["cell_line_uid"].isin(has_ge or set())
    frame["has_mu"] = frame["cell_line_uid"].isin(has_mu or set())
    frame["has_cnv"] = frame["cell_line_uid"].isin(has_cnv or set())
    frame["has_meth"] = frame["cell_line_uid"].isin(has_meth or set())
    frame["eligible"] = ~frame["cell_line_uid"].isin(excluded_cell_lines or set())
    if mapped_cell_lines is None:
        mapped = pd.Series(True, index=frame.index)
    else:
        mapped = frame["cell_line_uid"].isin(mapped_cell_lines)
    frame["mapped"] = mapped
    frame["in_main_pool"] = (
        frame["eligible"] & frame["mapped"] & frame["has_ge"] & frame["has_mu"] & frame["has_cnv"]
    )
    return frame.sort_values("cell_line_uid").reset_index(drop=True)


def map_gdsc_cell_lines(
    labels: pd.DataFrame,
    model: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """把 GDSC 细胞系映射到 ACH。

    路径：SangerModelID 直连 → 一对多时名字仲裁 → 名字兜底。
    返回 ``(mapping, conflicts)``；mapping 每个 GDSC SID 一行。
    """
    labels = labels[["CELL_LINE_NAME", "SANGER_MODEL_ID", "CANCER_TYPE"]].drop_duplicates()
    labels["SANGER_MODEL_ID"] = labels["SANGER_MODEL_ID"].astype(str).str.strip()
    labels["norm_name"] = labels["CELL_LINE_NAME"].map(normalize_name)
    model = model.copy()
    model["ModelID"] = model["ModelID"].astype(str).str.strip()
    model["SangerModelID"] = model["SangerModelID"].map(_clean_str)
    model["norm_cell"] = model["CellLineName"].map(normalize_name)
    model["norm_stripped"] = model["StrippedCellLineName"].map(normalize_name)

    by_sid = {sid: group for sid, group in model.dropna(subset=["SangerModelID"]).groupby("SangerModelID")}
    name_index: dict[str, set[str]] = {}
    for row in model.itertuples(index=False):
        for key in {row.norm_cell, row.norm_stripped}:
            if key:
                name_index.setdefault(key, set()).add(row.ModelID)

    records: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    for row in labels.sort_values("SANGER_MODEL_ID").itertuples(index=False):
        sid = row.SANGER_MODEL_ID
        name = row.CELL_LINE_NAME
        method = "unmapped"
        status = "failed"
        cell = None
        evidence = ""
        candidates = by_sid.get(sid)
        if candidates is not None:
            if len(candidates) == 1:
                cell = candidates.iloc[0]["ModelID"]
                method = "sanger_model_id"
                status = "ok"
                evidence = "unique SangerModelID"
            else:
                hit = candidates[
                    (candidates["norm_cell"] == row.norm_name)
                    | (candidates["norm_stripped"] == row.norm_name)
                ]
                if len(hit) == 1:
                    cell = hit.iloc[0]["ModelID"]
                    method = "sanger_model_id+name"
                    status = "ok"
                    evidence = f"1->{len(candidates)} resolved by name + cosmic"
                else:
                    status = "ambiguous"
                    evidence = f"1->{len(candidates)} candidates without unique name match"
        if cell is None and status != "ok":
            candidates_ids = name_index.get(row.norm_name, set())
            if len(candidates_ids) == 1:
                cell = next(iter(candidates_ids))
                method = "normalized_name"
                status = "ok"
                evidence = "unique normalized name fallback"
            elif len(candidates_ids) > 1:
                status = "ambiguous"
                evidence = f"normalized name matched {len(candidates_ids)} models"
            else:
                status = "failed"
                evidence = "no SangerModelID and no normalized name match"
        record = {
            "source_table": "gdsc_labels",
            "source_id": sid,
            "source_name": name,
            "cancer_type": row.CANCER_TYPE,
            "cell_line_uid": cell,
            "method": method,
            "status": status,
            "evidence": evidence,
        }
        records.append(record)
        if status != "ok":
            conflicts.append(record)
    return pd.DataFrame(records), conflicts


# ---------------------------------------------------------------------------
# 化合物
# ---------------------------------------------------------------------------
def build_compounds(
    labels: pd.DataFrame,
    compound_meta: pd.DataFrame,
    *,
    master: GeneMaster | None = None,
    smiles_by_uid: dict[str, str] | None = None,
    scaffold_by_uid: dict[str, str] | None = None,
) -> pd.DataFrame:
    """按 InChIKey > 标准化药名构建化合物层表（286 行）。"""
    label_drugs = (
        labels[["DRUG_ID", "DRUG_NAME"]]
        .drop_duplicates()
        .assign(DRUG_ID=lambda d: d["DRUG_ID"].astype(str).str.strip())
    )
    meta = compound_meta.copy()
    meta["DRUG_ID"] = meta["DRUG_ID"].astype(str).str.strip()
    meta = meta.drop_duplicates("DRUG_ID").set_index("DRUG_ID")
    records: list[dict[str, Any]] = []
    for drug_id, name in label_drugs.sort_values(["DRUG_ID"]).itertuples(index=False):
        row = meta.loc[drug_id] if drug_id in meta.index else None
        inchikey = _clean_str(row.get("INCHI_KEY")) if row is not None else None
        chembl = _clean_str(row.get("CHEMBL_ID")) if row is not None else None
        target_text = _clean_str(row.get("TARGET")) if row is not None else None
        uid = f"inchikey:{inchikey}" if inchikey else f"gdscname:{normalize_drug_name(name)}"
        records.append(
            {
                "compound_uid": uid,
                "drug_id": drug_id,
                "drug_name": name,
                "inchikey": inchikey,
                "chembl_id": chembl,
                "target_text": target_text,
                "screening_site": _clean_str(row.get("SCREENING_SITE")) if row is not None else None,
            }
        )
    raw = pd.DataFrame(records)
    grouped: list[dict[str, Any]] = []
    for uid, group in raw.groupby("compound_uid", sort=True):
        target_texts = [text for text in group["target_text"].dropna().unique().tolist() if text]
        target_text = "; ".join(sorted(target_texts)) if target_texts else None
        target_tokens = _split_targets(target_text)
        target_genes: list[str] = []
        if master is not None:
            for token in target_tokens:
                symbol, _ = master.resolve(token)
                if symbol:
                    target_genes.append(symbol)
        target_genes = sorted(set(target_genes))
        smiles = (smiles_by_uid or {}).get(uid)
        scaffold = (scaffold_by_uid or {}).get(uid)
        grouped.append(
            {
                "compound_uid": uid,
                "drug_ids": sorted(group["drug_id"].unique().tolist()),
                "drug_name": sorted(group["drug_name"].unique().tolist())[0],
                "gdsc_names": sorted(group["drug_name"].unique().tolist()),
                "inchikey": group["inchikey"].dropna().iloc[0] if group["inchikey"].notna().any() else None,
                "chembl_id": group["chembl_id"].dropna().iloc[0] if group["chembl_id"].notna().any() else None,
                "targets_text": target_text,
                "target_genes": target_genes,
                "smiles": smiles,
                "scaffold": scaffold,
            }
        )
    frame = pd.DataFrame(grouped)
    frame["has_inchikey"] = frame["inchikey"].notna()
    frame["has_target"] = frame["target_genes"].map(len) > 0
    frame["has_smiles"] = frame["smiles"].notna()
    frame["has_scaffold"] = frame["scaffold"].notna()
    return frame.sort_values("compound_uid").reset_index(drop=True)


def build_id_mapping(
    cell_mapping: pd.DataFrame,
    compounds: pd.DataFrame,
    *,
    gene_mappings: list[pd.DataFrame] | None = None,
    compound_mapping: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    cell = cell_mapping.copy()
    cell["target_type"] = "cell_line"
    cell["target_id"] = cell["cell_line_uid"]
    frames.append(
        cell[
            ["source_table", "source_id", "source_name", "target_type", "target_id", "method", "status", "evidence"]
        ]
    )
    if compound_mapping is not None:
        comp = compound_mapping.copy()
        comp["target_type"] = "compound"
        frames.append(comp)
    for gene_mapping in gene_mappings or []:
        frames.append(gene_mapping)
    if not frames:
        return pd.DataFrame(
            columns=["source_table", "source_id", "source_name", "target_type", "target_id", "method", "status", "evidence"]
        )
    return pd.concat(frames, ignore_index=True).sort_values(
        ["source_table", "source_id"]
    ).reset_index(drop=True)


def build_compound_mapping(raw: pd.DataFrame, compounds: pd.DataFrame) -> pd.DataFrame:
    """DRUG_ID → compound_uid 的审计表。"""
    uid_by_id: dict[str, str] = {}
    for row in compounds.itertuples(index=False):
        for drug_id in row.drug_ids:
            uid_by_id[str(drug_id)] = row.compound_uid
    frame = raw.drop_duplicates("DRUG_ID").copy()
    frame["source_table"] = "gdsc_compounds"
    frame["source_id"] = frame["DRUG_ID"].astype(str)
    frame["source_name"] = frame["DRUG_NAME"]
    frame["target_type"] = "compound"
    frame["target_id"] = frame["source_id"].map(uid_by_id)
    frame["method"] = np.where(frame["target_id"].str.startswith("inchikey:"), "inchikey", "normalized_name")
    frame["status"] = np.where(frame["target_id"].isna(), "failed", "ok")
    frame["evidence"] = frame["target_id"].fillna("")
    return frame[["source_table", "source_id", "source_name", "target_type", "target_id", "method", "status", "evidence"]]


def write_id_conflicts(path: str | Path, conflicts: Iterable[dict[str, Any]], *, extra: dict[str, Any] | None = None) -> Path:
    payload = {"conflicts": list(conflicts), **(extra or {})}
    return io.write_json(path, payload)


__all__ = [
    "build_cell_lines",
    "build_compound_mapping",
    "build_compounds",
    "build_id_mapping",
    "map_gdsc_cell_lines",
    "normalize_drug_name",
    "normalize_name",
    "write_id_conflicts",
]

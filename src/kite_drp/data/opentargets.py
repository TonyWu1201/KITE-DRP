"""Open Targets Platform 药物–靶点（DTI）处理。

本模块只读取 Open Targets 官方 parquet（CC0 4.0），把
``drug_mechanism_of_action`` 的机制记录映射到 GDSC 化合物，并输出可追溯的
drug→gene 边。DGIdb 不再参与构建。

匹配顺序（高置信优先）：

1. GDSC 化合物自带 ``CHEMBL_ID`` 与 Open Targets ``drug_molecule.id`` 或其
   母体/盐型族完全一致；
2. GDSC 化合物 ``INCHI_KEY`` 与 Open Targets ``drug_molecule.inchiKey`` 一致；
3. GDSC 药名/别名与 Open Targets ``name`` / ``synonyms`` / ``tradeNames``
   规范化后一致；仅当该名称只对应一个 OT 药物族时采用，避免同名歧义。

Open Targets 的 ``drug_mechanism_of_action.chemblIds`` 通常是母体 ChEMBL ID，
因此匹配前用 ``parentId`` / ``childChemblIds`` 建立药物族（union-find），
把盐型、母体与 MOA 记录归到同一族。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kite_drp.data.genes import GeneMaster
from kite_drp.data.ids import normalize_drug_name


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return list(value)
    return [value]


def _clean_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def _normalize_inchikey(value: Any) -> str | None:
    text = _clean_str(value)
    if not text:
        return None
    text = text.upper()
    if text.startswith("INCHIKEY="):
        text = text.split("=", 1)[1]
    return text or None


def _label_values(value: Any) -> list[str]:
    """把 OT 的 synonyms/tradeNames（str 或 {label: ...} 字典）展开为字符串。"""
    result: list[str] = []
    for item in _as_list(value):
        if isinstance(item, dict):
            item = item.get("label")
        text = _clean_str(item)
        if text:
            result.append(text)
    return result


def _read_parquet_dir(path: Path) -> pd.DataFrame:
    files = sorted(path.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)


class _UnionFind:
    """轻量 union-find；键为 Open Targets / ChEMBL 字符串 ID。"""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, key: str) -> None:
        self.parent.setdefault(key, key)

    def find(self, key: str) -> str:
        self.add(key)
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[key] != key:
            parent = self.parent[key]
            self.parent[key] = root
            key = parent
        return root

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        # 字典序小的作根，保证跨运行稳定
        if b < a:
            a, b = b, a
        self.parent[b] = a


@dataclass
class OpenTargetsTables:
    targets: pd.DataFrame
    drugs: pd.DataFrame
    moa: pd.DataFrame
    release: str | None
    files: list[str]


@dataclass
class OpenTargetsIndex:
    tables: OpenTargetsTables
    dsu: _UnionFind
    ensembl_to_symbol: dict[str, str]
    ensembl_to_name: dict[str, str]
    inchikey_to_ids: dict[str, set[str]]
    name_to_ids: dict[str, set[str]]
    id_to_drug: dict[str, dict[str, Any]]

    def root(self, key: str) -> str:
        return self.dsu.find(key)


def load_opentargets(opentargets_dir: str | Path) -> OpenTargetsTables:
    """读取 Open Targets release 的三张必需表（target/drug/moa）。"""
    root = Path(opentargets_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Open Targets 目录不存在：{root}")
    targets = _read_parquet_dir(root / "target")
    drugs = _read_parquet_dir(root / "drug_molecule")
    moa = _read_parquet_dir(root / "drug_mechanism_of_action")
    frames = {
        "target": targets,
        "drug_molecule": drugs,
        "drug_mechanism_of_action": moa,
    }
    if targets.empty or drugs.empty or moa.empty:
        missing = [name for name, frame in frames.items() if frame.empty]
        raise FileNotFoundError(f"Open Targets 缺少 parquet 表：{missing}（{root}）")

    required = {
        "target": {"id", "approvedSymbol"},
        "drug_molecule": {"id", "name", "inchiKey"},
        "drug_mechanism_of_action": {"chemblIds", "targets"},
    }
    for name, frame in frames.items():
        missing_columns = required[name] - set(frame.columns)
        if missing_columns:
            raise ValueError(f"Open Targets {name} 缺少列：{sorted(missing_columns)}")

    release = root.name
    manifest_path = root / "download_manifest.json"
    if manifest_path.is_file():
        import json

        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        release = str(payload.get("release") or release)
    files = sorted(path.relative_to(root).as_posix() for path in root.rglob("*.parquet"))
    return OpenTargetsTables(targets=targets, drugs=drugs, moa=moa, release=release, files=files)


def build_index(tables: OpenTargetsTables) -> OpenTargetsIndex:
    """建立 Ensembl/HGNC、药名、InChIKey 与药物族索引。"""
    targets = tables.targets.copy()
    targets["id"] = targets["id"].astype(str)
    targets["approvedSymbol"] = targets["approvedSymbol"].map(_clean_str)
    targets = targets.dropna(subset=["approvedSymbol"]).drop_duplicates("id")

    ensembl_to_symbol: dict[str, str] = {}
    ensembl_to_name: dict[str, str] = {}
    for row in targets.itertuples(index=False):
        ensembl_to_symbol[str(row.id)] = str(row.approvedSymbol)
        approved_name = getattr(row, "approvedName", None)
        if _clean_str(approved_name):
            ensembl_to_name[str(row.id)] = str(approved_name)

    drugs = tables.drugs.copy()
    drugs["id"] = drugs["id"].astype(str)
    id_to_drug: dict[str, dict[str, Any]] = {}
    dsu = _UnionFind()
    for row in drugs.itertuples(index=False):
        drug_id = str(row.id)
        record = row._asdict()
        id_to_drug[drug_id] = record
        dsu.add(drug_id)
        parent = _clean_str(getattr(row, "parentId", None))
        if parent:
            dsu.union(drug_id, parent)
        for child in _as_list(getattr(row, "childChemblIds", None)):
            child_text = _clean_str(child)
            if child_text:
                dsu.union(drug_id, child_text)

    inchikey_to_ids: dict[str, set[str]] = {}
    name_to_ids: dict[str, set[str]] = {}
    for drug_id, record in id_to_drug.items():
        inchikey = _normalize_inchikey(record.get("inchiKey"))
        if inchikey:
            inchikey_to_ids.setdefault(inchikey, set()).add(drug_id)
        names = [
            record.get("name"),
            *_label_values(record.get("synonyms")),
            *_label_values(record.get("tradeNames")),
        ]
        for name in names:
            key = normalize_drug_name(name)
            if key:
                name_to_ids.setdefault(key, set()).add(drug_id)

    # MOA 的 chemblIds 可能不在 drug_molecule 表里（例如只有机制记录），也要纳入族。
    for value in tables.moa.get("chemblIds", []):
        for chembl_id in _as_list(value):
            text = _clean_str(chembl_id)
            if text:
                dsu.add(text)

    return OpenTargetsIndex(
        tables=tables,
        dsu=dsu,
        ensembl_to_symbol=ensembl_to_symbol,
        ensembl_to_name=ensembl_to_name,
        inchikey_to_ids=inchikey_to_ids,
        name_to_ids=name_to_ids,
        id_to_drug=id_to_drug,
    )


def _best_method(methods: Iterable[str]) -> str:
    priority = {"chembl_id": 0, "chembl_id_moa": 0, "inchikey": 1, "name": 2}
    return sorted(set(methods), key=lambda value: (priority.get(value, 9), value))[0]


def match_compounds(
    compounds: pd.DataFrame,
    index: OpenTargetsIndex,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """把 GDSC 化合物映射到 Open Targets 药物族。

    返回长表：``compound_uid, ot_drug_id, match_method, family_root``。
    同一化合物可能对应多个盐型/母体 ID，但每个药物族只保留最高置信匹配。
    """
    records: list[dict[str, Any]] = []
    ambiguous_names: list[str] = []
    for row in compounds.itertuples(index=False):
        uid = str(row.compound_uid)
        chembl_id = _clean_str(getattr(row, "chembl_id", None))
        inchikey = _normalize_inchikey(getattr(row, "inchikey", None))
        names = [getattr(row, "drug_name", None), *_as_list(getattr(row, "gdsc_names", None))]

        candidates: dict[str, list[str]] = {"chembl_id": [], "inchikey": [], "name": []}
        if chembl_id:
            candidates["chembl_id"].append(chembl_id)
        if inchikey:
            candidates["inchikey"].extend(sorted(index.inchikey_to_ids.get(inchikey, set())))
        name_ids: set[str] = set()
        for name in names:
            key = normalize_drug_name(name)
            if key:
                name_ids.update(index.name_to_ids.get(key, set()))
        if name_ids:
            roots = {index.root(drug_id) for drug_id in name_ids}
            if len(roots) == 1:
                candidates["name"].extend(sorted(name_ids))
            elif not candidates["chembl_id"] and not candidates["inchikey"]:
                ambiguous_names.append(uid)

        # 高置信来源存在时，不再使用低置信名称匹配，避免同名药引入假靶点。
        selected_method = None
        selected_ids: list[str] = []
        for method in ("chembl_id", "inchikey", "name"):
            if candidates[method]:
                selected_method = method
                selected_ids = candidates[method]
                break
        if not selected_ids:
            continue

        by_root: dict[str, tuple[str, str]] = {}
        for drug_id in selected_ids:
            root = index.root(drug_id)
            method = selected_method
            if method == "chembl_id" and drug_id not in index.id_to_drug:
                method = "chembl_id_moa"
            previous = by_root.get(root)
            if previous is None or _best_method([previous[1], method]) == method:
                by_root[root] = (drug_id, method)
        for root, (drug_id, method) in sorted(by_root.items()):
            records.append(
                {
                    "compound_uid": uid,
                    "ot_drug_id": drug_id,
                    "match_method": method,
                    "family_root": root,
                }
            )

    mapping = pd.DataFrame(
        records,
        columns=["compound_uid", "ot_drug_id", "match_method", "family_root"],
    )
    info = {
        "compounds_total": int(len(compounds)),
        "compounds_with_ot_family": (
            int(mapping["compound_uid"].nunique()) if not mapping.empty else 0
        ),
        "ambiguous_name_compounds": ambiguous_names,
        "match_method_compounds": (
            mapping.groupby("match_method")["compound_uid"].nunique().to_dict()
            if not mapping.empty
            else {}
        ),
    }
    return mapping, info


def _resolve_target_gene(
    ensembl_id: str | None,
    fallback_name: str | None,
    index: OpenTargetsIndex,
    master: GeneMaster,
) -> tuple[str | None, str | None]:
    """Ensembl ID → HGNC symbol；失败时用 targetName 兜底。"""
    if ensembl_id:
        symbol = index.ensembl_to_symbol.get(ensembl_id)
        if symbol:
            return symbol, ensembl_id
        symbol, _ = master.resolve(ensembl_id)
        if symbol:
            return symbol, ensembl_id
    if fallback_name:
        symbol, _ = master.resolve(fallback_name)
        if symbol:
            return symbol, None
    return None, None


def build_opentargets_edges(
    compounds: pd.DataFrame,
    master: GeneMaster,
    opentargets_dir: str | Path,
    *,
    allowed_genes: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """构建 Open Targets MoA → GDSC 化合物的 drug-target 长表。"""
    tables = load_opentargets(opentargets_dir)
    index = build_index(tables)
    mapping, mapping_info = match_compounds(compounds, index)
    allowed = set(allowed_genes) if allowed_genes is not None else None

    root_to_compounds: dict[str, list[tuple[str, str, str]]] = {}
    for row in mapping.itertuples(index=False):
        root_to_compounds.setdefault(str(row.family_root), []).append(
            (str(row.compound_uid), str(row.ot_drug_id), str(row.match_method))
        )

    records: list[dict[str, Any]] = []
    raw_edges = 0
    moa_with_targets = 0
    moa_without_targets = 0
    matched_roots: set[str] = set()
    for moa_index, row in enumerate(tables.moa.itertuples(index=False)):
        chembl_ids = [
            text for text in (_clean_str(value) for value in _as_list(row.chemblIds)) if text
        ]
        if not chembl_ids:
            moa_without_targets += 1
            continue
        roots = {index.root(chembl_id) for chembl_id in chembl_ids}
        hits: list[tuple[str, str, str]] = []
        for root in roots:
            hits.extend(root_to_compounds.get(root, []))
        if not hits:
            continue

        target_ids = [
            text for text in (_clean_str(value) for value in _as_list(row.targets)) if text
        ]
        target_name = _clean_str(getattr(row, "targetName", None))
        resolved = [
            _resolve_target_gene(ensembl_id, target_name if not target_ids else None, index, master)
            for ensembl_id in (target_ids or [None])
        ]
        resolved = [(symbol, ensembl) for symbol, ensembl in resolved if symbol]
        if not resolved:
            moa_without_targets += 1
            continue
        moa_with_targets += 1

        action_type = _clean_str(getattr(row, "actionType", None)) or ""
        mechanism = _clean_str(getattr(row, "mechanismOfAction", None)) or ""
        target_type = _clean_str(getattr(row, "targetType", None)) or ""
        for compound_uid, ot_drug_id, match_method in hits:
            matched_roots.add(index.root(ot_drug_id))
            for symbol, ensembl_id in resolved:
                raw_edges += 1
                if allowed is not None and symbol not in allowed:
                    continue
                records.append(
                    {
                        "compound_uid": compound_uid,
                        "hgnc_symbol": symbol,
                        "source": "opentargets",
                        "score": 1.0,
                        "action_type": action_type,
                        "mechanism": mechanism,
                        "target_name": target_name
                        or index.ensembl_to_name.get(ensembl_id or "", ""),
                        "target_id": ensembl_id or "",
                        "ot_drug_id": ot_drug_id,
                        "match_method": match_method,
                        "target_type": target_type,
                        "source_version": tables.release,
                        "moa_index": moa_index,
                    }
                )

    edges = pd.DataFrame(
        records,
        columns=[
            "compound_uid",
            "hgnc_symbol",
            "source",
            "score",
            "action_type",
            "mechanism",
            "target_name",
            "target_id",
            "ot_drug_id",
            "match_method",
            "target_type",
            "source_version",
            "moa_index",
        ],
    )
    info = {
        "status": "ok",
        "release": tables.release,
        "files": tables.files,
        "n_targets": int(len(tables.targets)),
        "n_drugs": int(len(tables.drugs)),
        "n_moa": int(len(tables.moa)),
        "moa_rows_with_targets": moa_with_targets,
        "moa_rows_without_targets": moa_without_targets,
        "matched_families": len(matched_roots),
        "edges_before_gene_filter": raw_edges,
        "edges": int(len(edges)),
        "compounds_with_edge": int(edges["compound_uid"].nunique()) if not edges.empty else 0,
        "genes_with_edge": int(edges["hgnc_symbol"].nunique()) if not edges.empty else 0,
        **mapping_info,
    }
    return edges, info


__all__ = [
    "OpenTargetsIndex",
    "OpenTargetsTables",
    "build_index",
    "build_opentargets_edges",
    "load_opentargets",
    "match_compounds",
]

"""药物特征装配到 processed。"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from kite_drp.data import io


def _source_count(dti: pd.DataFrame, source: str) -> tuple[int, int]:
    if dti.empty or "source" not in dti.columns:
        return 0, 0
    mask = dti["source"].astype(str).str.contains(source, regex=False)
    return int(mask.sum()), int(dti.loc[mask, "compound_uid"].nunique())


def build_dti_inputs(canonical_dir: str | Path, processed_dir: str | Path) -> dict[str, Any]:
    """把 canonical DTI（Open Targets MoA + GDSC TARGET）落成 processed 自包含表。

    processed 侧训练/图构建只读本文件；DGIdb 不参与。
    """
    canonical = Path(canonical_dir)
    processed = Path(processed_dir)
    dti = io.read_parquet(canonical / "dti.parquet")
    compounds = io.read_parquet(canonical / "compounds.parquet")
    if not dti.empty and dti["source"].astype(str).str.contains("dgidb", regex=False).any():
        raise ValueError(
            "canonical dti.parquet 仍含 DGIdb；请先运行 "
            "`kite_drp data canonical --only dti` 重建 clean DTI。"
        )
    output = processed / "dti.parquet"
    io.write_parquet(dti, output)

    source_counts = {
        source: dict(zip(("edges", "compounds"), _source_count(dti, source), strict=False))
        for source in ("gdsc_target", "opentargets", "chembl")
    }
    matched = set(dti["compound_uid"].astype(str)) if not dti.empty else set()
    has_dti = (
        compounds["has_dti"]
        if "has_dti" in compounds
        else compounds["compound_uid"].isin(matched)
    )
    unmatched = compounds.loc[
        ~has_dti.astype(bool), ["compound_uid", "drug_name", "chembl_id", "inchikey"]
    ]
    stats_path = canonical.parent / "audit" / "canonical_stats.json"
    dti_stats = {}
    if stats_path.is_file():
        dti_stats = io.read_json(stats_path).get("stats", {}).get("dti", {})
    coverage = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_policy": "Open Targets MoA (CC0 4.0) + GDSC TARGET; DGIdb excluded",
        "release": (dti_stats.get("opentargets") or {}).get("release"),
        "compounds_total": int(len(compounds)),
        "compounds_with_dti": int(has_dti.sum()),
        "coverage": float(has_dti.mean()) if len(compounds) else 0.0,
        "edges": int(len(dti)),
        "genes": int(dti["hgnc_symbol"].nunique()) if not dti.empty else 0,
        "source_versions": (
            sorted(dti["source_version"].dropna().astype(str).unique().tolist())
            if "source_version" in dti
            else []
        ),
        "by_source": source_counts,
        "match_method_edges": (
            {
                method: int(dti["match_method"].astype(str).str.contains(method, regex=False).sum())
                for method in ("gdsc_target_text", "chembl_id", "inchikey", "name")
            }
            if not dti.empty
            else {}
        ),
        "match_method_note": "counts are row-containment; one edge can contain multiple match methods",
        "dti_sources": (
            compounds["dti_sources"].value_counts().to_dict()
            if "dti_sources" in compounds
            else {}
        ),
        "unmatched_compounds": unmatched.fillna("").to_dict("records"),
        "upstream": dti_stats,
        "dti_sha256": io.hash_file(output),
    }
    io.write_json(processed / "dti_coverage.json", coverage)
    return coverage


def build_drug_inputs(canonical_dir: str | Path, processed_dir: str | Path) -> dict[str, Any]:
    canonical = Path(canonical_dir)
    processed = Path(processed_dir)
    source = canonical / "drug_features.npz"
    if not source.is_file():
        return {"status": "missing", "reason": str(source)}
    io.atomic_copy(source, processed / "drug_features.npz")
    compounds = io.read_parquet(canonical / "compounds.parquet")
    keep = [
        "compound_uid",
        "drug_ids",
        "drug_name",
        "inchikey",
        "chembl_id",
        "target_genes",
        "smiles",
        "scaffold",
        "has_inchikey",
        "has_target",
        "has_smiles",
        "has_scaffold",
        "has_dti",
        "dti_sources",
        "ot_drug_ids",
    ]
    columns = [column for column in keep if column in compounds]
    io.write_parquet(compounds[columns], processed / "compounds.parquet")
    coverage_path = canonical / "drug_feature_coverage.json"
    if coverage_path.is_file():
        io.atomic_copy(coverage_path, processed / "drug_feature_coverage.json")
    arrays = io.read_npz(source)
    return {
        "n_compounds": int(len(arrays["compound_uid"])),
        "has_smiles": int(arrays["has_smiles"].sum()),
        "has_scaffold": int((arrays["scaffold"] != "").sum()),
        "morgan_bits": int(arrays["morgan"].shape[1]),
    }


__all__ = ["build_drug_inputs", "build_dti_inputs"]

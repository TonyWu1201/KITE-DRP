"""药物特征装配到 processed。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kite_drp.data import io, registry


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
    ]
    io.write_parquet(compounds[keep], processed / "compounds.parquet")
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


__all__ = ["build_drug_inputs"]

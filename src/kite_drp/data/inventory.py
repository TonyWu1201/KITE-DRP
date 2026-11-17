"""L0 原始数据清单：文件、大小、MD5、版本推断与 TCGA 文件计数。"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from kite_drp.data import io
from kite_drp.data.manifest import raw_md5_from_checksums

KNOWN_CATEGORIES = {
    "gdsc": ["GDSC2_fitted_dose_response", "screened_compounds", "Cell_Lines_Details"],
    "depmap": ["Model.csv", "DepMap_Expression", "DepMap_SomaticMutations", "DepMap_CNGeneWGS"],
    "hgnc": ["Gene.csv"],
    "string": ["9606.protein.links", "9606.protein.aliases"],
    "dgidb": ["interactions.tsv", "drugs.tsv", "genes.tsv", "categories.tsv"],
    "ccle": ["CCLE_DNA_methylation", "Cell_lines_annotations"],
    "tcga": ["clinical", "TCGA-CDR", "ding2016", "CESC"],
}

VERSION_HINTS = {
    "GDSC2_fitted_dose_response_27Oct23": "GDSC2 release 27Oct23 / 8.5 generation",
    "screened_compounds_rel_8.4": "GDSC release-8.4",
    "screened_compounds_rel_8.5_inchi": "GDSC 8.4 + ChEMBL/InChI derived (not official 8.5)",
    "Model.csv": "DepMap Model.csv snapshot 2026-11-10 (release unverified)",
    "DepMap_Expression": "DepMap 26Q1-era snapshot 2026-07-13 (release unverified)",
    "DepMap_SomaticMutations": "DepMap 26Q1 mutation pipeline (release unverified)",
    "DepMap_CNGeneWGS": "DepMap 26Q1-era snapshot 2026-07-13 (release unverified)",
    "9606.protein.links.v12.5": "STRING v12.5",
    "9606.protein.aliases.v12.5": "STRING v12.5",
    "Gene.csv": "HGNC snapshot (latest date_modified 2025-03-31)",
}


def _category(path: Path) -> str:
    for category in KNOWN_CATEGORIES:
        if category in path.parts:
            return category
    return "other"


def _version_hint(path: Path) -> str | None:
    text = str(path)
    for key, value in VERSION_HINTS.items():
        if key in text:
            return value
    return None


def _count_lines(path: Path, *, limit_bytes: int | None = None) -> int | None:
    if path.suffix == ".gz":
        return None
    try:
        with path.open("rb") as handle:
            count = 0
            while chunk := handle.read(1 << 20):
                count += chunk.count(b"\n")
            return count
    except OSError:
        return None


def _tcga_inventory(raw: Path) -> dict[str, Any]:
    cesc = raw / "tcga" / "CESC"
    counts: dict[str, int] = {}
    if cesc.is_dir():
        for path in cesc.rglob("*"):
            if path.is_file():
                if "logs" in path.parts or path.name.endswith(".parcel"):
                    counts[".parcel"] = counts.get(".parcel", 0) + 1
                    continue
                suffix = next(
                    (
                        marker
                        for marker in (
                            "rna_seq.augmented_star_gene_counts.tsv",
                            "wxs.aliquot_ensemble_masked.maf.gz",
                            "ascat3.gene_level_copy_number.v36.tsv",
                            "methylation_array.sesame.level3betas.txt",
                            "annotations.txt",
                        )
                        if marker in path.name
                    ),
                    "other",
                )
                counts[suffix] = counts.get(suffix, 0) + 1
    manifests = raw / "tcga" / "manifests"
    manifest_rows: dict[str, int] = {}
    for path in sorted(manifests.glob("manifest_*.tsv")):
        try:
            with path.open(encoding="utf-8", errors="replace") as handle:
                manifest_rows[path.name] = max(sum(1 for _ in handle) - 1, 0)
        except OSError:
            continue
    return {"data_files": counts, "manifests": manifest_rows}


def build_inventory(raw_dir: str | Path = "data/raw") -> dict[str, Any]:
    raw = Path(raw_dir)
    md5_map = raw_md5_from_checksums(raw / "CHECKSUMS.md5")
    entries: list[dict[str, Any]] = []
    for path in sorted(raw.rglob("*")):
        if not path.is_file() or path.name in {".gitkeep", "CHECKSUMS.md5", "README.md"}:
            continue
        if "tcga" in path.parts and "CESC" in path.parts:
            continue
        relative = path.relative_to(raw)
        md5 = md5_map.get(str(relative))
        entries.append(
            {
                "path": str(path),
                "relative_path": str(relative),
                "category": _category(path),
                "size": path.stat().st_size,
                "mtime": datetime.fromtimestamp(path.stat().st_mtime, UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "md5": md5,
                "lines": _count_lines(path) if path.stat().st_size < 100_000_000 else None,
                "version_hint": _version_hint(path),
            }
        )
    return {
        "schema_version": "1.0",
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "raw_dir": str(raw),
        "n_files": len(entries),
        "total_bytes": int(sum(item["size"] for item in entries)),
        "entries": entries,
        "tcga": _tcga_inventory(raw),
        "notes": [
            "MD5 come from data/raw/CHECKSUMS.md5 when available; TCGA per-file MD5 is in GDC manifests.",
            "DepMap releases are frozen by content hash, not release number (mixed mtimes).",
        ],
    }


def write_inventory(payload: dict[str, Any], path: str | Path) -> Path:
    return io.write_json(path, payload)


__all__ = ["build_inventory", "write_inventory"]

#!/usr/bin/env python3
"""下载 Reactome GMT 与 GO GAF/OBO，并转换为管线可读的 GMT。

用法：
    .venv/bin/python scripts/download_pathways.py
    .venv/bin/python scripts/download_pathways.py --skip-go

输出：
    data/raw/reactome/ReactomePathways.gmt
    data/raw/go/goa_human.gmt
    data/raw/go/goa_human_experimental.gmt
    data/raw/pathways/download_manifest.json

说明：
- Reactome GMT 已直接按 gene symbol 提供。
- GO 官方只提供 GAF（注释表），本脚本把 human GAF 聚合为 “GO term → gene symbols” GMT；
  experimental 版本排除 IEA/ND 证据。
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import urllib.request
import zipfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REACTOME_GMT_URL = "https://reactome.org/download/current/ReactomePathways.gmt.zip"
GO_GAF_URL = "https://current.geneontology.org/annotations/goa_human.gaf.gz"
GO_OBO_URL = "https://current.geneontology.org/ontology/go-basic.obo"


def _download(url: str, target: Path, *, timeout: int = 120) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "kite-drp-data/1.0"})
    digest = hashlib.sha256()
    with (
        urllib.request.urlopen(request, timeout=timeout) as response,  # noqa: S310
        target.open("wb") as handle,
    ):
        while chunk := response.read(1 << 20):
            handle.write(chunk)
            digest.update(chunk)
    return digest.hexdigest()


def download_reactome(out_dir: Path) -> dict:
    zip_path = out_dir / "ReactomePathways.gmt.zip"
    md5 = _download(REACTOME_GMT_URL, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        members = [name for name in archive.namelist() if name.endswith(".gmt")]
        if not members:
            raise FileNotFoundError("Reactome zip 内没有 GMT")
        payload = archive.read(members[0])
    target = out_dir / "ReactomePathways.gmt"
    target.write_bytes(payload)
    n_pathways = sum(1 for line in payload.decode("utf-8", "replace").splitlines() if line.strip())
    return {
        "url": REACTOME_GMT_URL,
        "sha256": md5,
        "output": str(target),
        "n_pathways": n_pathways,
    }


def parse_obo_names(path: Path) -> dict[str, tuple[str, str]]:
    terms: dict[str, tuple[str, str]] = {}
    current: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("["):
            if current.get("id") and not current.get("is_obsolete"):
                terms[current["id"]] = (current.get("name", ""), current.get("namespace", ""))
            current = {} if line == "[Term]" else {}
            continue
        if ": " in line:
            key, value = line.split(": ", 1)
            current[key] = value
    if current.get("id") and not current.get("is_obsolete"):
        terms[current["id"]] = (current.get("name", ""), current.get("namespace", ""))
    return terms


def gaf_to_gmt(gaf_path: Path, obo_path: Path, out_path: Path, *, experimental: bool) -> dict:
    names = parse_obo_names(obo_path)
    genes_by_term: dict[str, set[str]] = defaultdict(set)
    skipped_not = skipped_taxon = skipped_evidence = 0
    excluded_evidence = {"IEA", "ND"}
    with gzip.open(gaf_path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("!"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 13:
                continue
            qualifier, go_id, evidence, gene, taxon = (
                parts[3],
                parts[4],
                parts[6],
                parts[2],
                parts[12],
            )
            if "taxon:9606" not in taxon:
                skipped_taxon += 1
                continue
            if "NOT" in qualifier.split("|"):
                skipped_not += 1
                continue
            if experimental and evidence in excluded_evidence:
                skipped_evidence += 1
                continue
            if not gene or gene == "-":
                continue
            genes_by_term[go_id].add(gene)
    lines = []
    for go_id in sorted(genes_by_term):
        genes = sorted(genes_by_term[go_id])
        if len(genes) < 5:
            continue
        name, namespace = names.get(go_id, (go_id, ""))
        label = f"{name} [{namespace}]" if namespace else name
        lines.append("\t".join([go_id, label, *genes]))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "output": str(out_path),
        "n_terms": len(lines),
        "skipped_not": skipped_not,
        "skipped_taxon": skipped_taxon,
        "skipped_evidence": skipped_evidence,
        "experimental": experimental,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 Reactome/GO 通路并转 GMT")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--skip-reactome", action="store_true")
    parser.add_argument("--skip-go", action="store_true")
    args = parser.parse_args()

    out_root: Path = args.out
    manifest: dict = {
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reactome": None,
        "go": None,
    }
    if not args.skip_reactome:
        manifest["reactome"] = download_reactome(out_root / "reactome")
    if not args.skip_go:
        go_dir = out_root / "go"
        gaf_path = go_dir / "goa_human.gaf.gz"
        obo_path = go_dir / "go-basic.obo"
        manifest["go"] = {
            "gaf_url": GO_GAF_URL,
            "gaf_sha256": _download(GO_GAF_URL, gaf_path),
            "obo_url": GO_OBO_URL,
            "obo_sha256": _download(GO_OBO_URL, obo_path),
            "all": gaf_to_gmt(gaf_path, obo_path, go_dir / "goa_human.gmt", experimental=False),
            "experimental": gaf_to_gmt(
                gaf_path, obo_path, go_dir / "goa_human_experimental.gmt", experimental=True
            ),
        }
    target = out_root / "pathways" / "download_manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

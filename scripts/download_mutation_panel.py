#!/usr/bin/env python3
"""从 cBioPortal API 下载公开 targeted panel 基因清单。

默认 MSK-IMPACT 468（与规划 D27 的 468 基因面板对齐）。可选面板：
IMPACT468、IMPACT505、IMPACT410、IMPACT341、FoundationOne。

用法：
    .venv/bin/python scripts/download_mutation_panel.py
    .venv/bin/python scripts/download_mutation_panel.py --panel IMPACT505

输出：``data/raw/panel/<panel>.txt``（每行一个 HGNC symbol）+ manifest。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "raw" / "panel"
API = "https://www.cbioportal.org/api/gene-panels/{panel}"


def fetch_panel(panel: str, *, timeout: int = 60) -> tuple[dict, bytes]:
    url = API.format(panel=panel)
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        payload = response.read()
    return json.loads(payload.decode("utf-8")), payload


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 MSK-IMPACT/FoundationOne 基因 panel")
    parser.add_argument("--panel", default="IMPACT468")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    data, raw_payload = fetch_panel(args.panel)
    genes = sorted(
        {
            str(item["hugoGeneSymbol"]).strip()
            for item in data.get("genes", [])
            if item.get("hugoGeneSymbol")
        }
    )
    if not genes:
        raise SystemExit(f"panel {args.panel} 没有返回基因")
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{args.panel}.txt"
    target.write_text("\n".join(genes) + "\n", encoding="utf-8")
    manifest = {
        "panel": args.panel,
        "description": data.get("description", ""),
        "url": API.format(panel=args.panel),
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_genes": len(genes),
        "sha256": hashlib.sha256(raw_payload).hexdigest(),
        "output": str(target),
    }
    (out_dir / f"{args.panel}_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

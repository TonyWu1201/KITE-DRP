#!/usr/bin/env python3
"""通过 PubChem PUG-REST 下载 GDSC 化合物的 SMILES（pubchempy）。

用法：
    uv sync --extra data
    .venv/bin/python scripts/download_smiles_pubchem.py
    .venv/bin/python scripts/download_smiles_pubchem.py --force --delay 0.3

输出：``data/raw/drugs/smiles.tsv``（含成功与失败记录）与同名 manifest。
已成功的 compound_uid 默认跳过，可重复运行续传。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from kite_drp.data import ids, io  # noqa: E402
from kite_drp.data.config import load_data_config  # noqa: E402

try:
    import pubchempy as pcp  # type: ignore
except ImportError:  # pragma: no cover
    print(
        "缺少 pubchempy：请先运行 `uv sync --extra data` 或 "
        "`uv pip install --python .venv/bin/python pubchempy`",
        file=sys.stderr,
    )
    raise SystemExit(2) from None


DEFAULT_OUT = ROOT / "data" / "raw" / "drugs" / "smiles.tsv"


def _candidate_names(row: pd.Series, synonyms_by_id: dict[str, list[str]]) -> list[str]:
    names = [str(row["drug_name"])]
    for drug_id in row["drug_ids"]:
        names.append(str(drug_id))
        names.extend(synonyms_by_id.get(str(drug_id), []))
    seen: set[str] = set()
    unique: list[str] = []
    for name in names:
        name = name.strip()
        if name and name.lower() not in seen:
            seen.add(name.lower())
            unique.append(name)
    return unique


def _query_one(
    *,
    compound_uid: str,
    inchikey: str | None,
    names: list[str],
    delay: float,
) -> dict[str, Any]:
    """按 InChIKey → 数字 CID → 名称 顺序查询。返回一行结果。"""
    attempts: list[tuple[str, str]] = []
    if inchikey:
        attempts.append(("inchikey", inchikey))
    for name in names:
        if name.isdigit():
            attempts.append(("cid", name))
    for name in names:
        attempts.append(("name", name))

    last_error = ""
    for namespace, query in attempts:
        try:
            found = pcp.get_compounds(query, namespace)
        except Exception as exc:  # 网络/404/名称解析失败
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(delay)
            continue
        time.sleep(delay)
        if not found:
            last_error = "not found"
            continue
        compound = found[0]
        smiles = getattr(compound, "isomeric_smiles", None) or getattr(
            compound, "canonical_smiles", None
        )
        if not smiles:
            last_error = "no smiles in PubChem record"
            continue
        return {
            "compound_uid": compound_uid,
            "inchikey": inchikey,
            "drug_name": names[0] if names else "",
            "smiles": smiles,
            "source": f"pubchem:{namespace}",
            "query": query,
            "pubchem_cid": getattr(compound, "cid", None),
            "pubchem_inchikey": getattr(compound, "inchikey", None),
            "inchikey_match": bool(
                not inchikey
                or str(getattr(compound, "inchikey", "")).startswith(inchikey.split("-")[0])
            ),
            "n_candidates": len(found),
            "status": "ok",
            "error": "",
            "fetched_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    return {
        "compound_uid": compound_uid,
        "inchikey": inchikey,
        "drug_name": names[0] if names else "",
        "smiles": "",
        "source": "",
        "query": "",
        "pubchem_cid": None,
        "pubchem_inchikey": None,
        "inchikey_match": False,
        "n_candidates": 0,
        "status": "failed",
        "error": last_error[:300],
        "fetched_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _load_compounds(config: dict[str, Any]) -> pd.DataFrame:
    canonical = ROOT / "data" / "interim" / "canonical" / "compounds.parquet"
    if canonical.is_file():
        return pd.read_parquet(canonical)
    labels_path = config.get("raw_files", {}).get(
        "dose_response", ROOT / "data/raw/gdsc/GDSC2_fitted_dose_response_27Oct23.csv"
    )
    meta_path = config.get("raw_files", {}).get(
        "compounds", ROOT / "data/raw/gdsc/screened_compounds_rel_8.5_inchi.csv"
    )
    labels = pd.read_csv(labels_path, dtype={"DRUG_ID": str, "SANGER_MODEL_ID": str})
    meta = pd.read_csv(meta_path, dtype=str)
    return ids.build_compounds(labels, meta)


def _synonyms(config: dict[str, Any]) -> dict[str, list[str]]:
    path = config.get("raw_files", {}).get(
        "compounds", ROOT / "data/raw/gdsc/screened_compounds_rel_8.5_inchi.csv"
    )
    frame = pd.read_csv(path, dtype=str)
    result: dict[str, list[str]] = {}
    for row in frame.itertuples(index=False):
        synonyms = []
        if isinstance(row.SYNONYMS, str):
            synonyms = [item.strip() for item in row.SYNONYMS.split(",") if item.strip()]
        result[str(row.DRUG_ID)] = synonyms
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="用 pubchempy 下载 GDSC 化合物 SMILES")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true", help="忽略已有成功记录，全部重查")
    parser.add_argument("--delay", type=float, default=0.25, help="每次请求后的 sleep 秒数")
    parser.add_argument("--limit", type=int, default=None, help="仅处理前 N 个（调试用）")
    parser.add_argument("--config", type=Path, default=None, help="configs/data/gdsc2.yaml")
    args = parser.parse_args()

    config = load_data_config(args.config)
    compounds = _load_compounds(config)
    synonyms = _synonyms(config)
    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cached: dict[str, dict[str, Any]] = {}
    if out_path.is_file() and not args.force:
        old = pd.read_csv(out_path, sep="\t", dtype=str).fillna("")
        for row in old.to_dict("records"):
            if row.get("status") == "ok" and row.get("smiles"):
                cached[str(row["compound_uid"])] = row

    rows: list[dict[str, Any]] = []
    total = len(compounds)
    for index, row in enumerate(compounds.itertuples(index=False), start=1):
        uid = str(row.compound_uid)
        if uid in cached:
            rows.append(cached[uid])
            continue
        names = _candidate_names(pd.Series(row._asdict()), synonyms)
        inchikey = getattr(row, "inchikey", None)
        result = _query_one(
            compound_uid=uid,
            inchikey=inchikey if isinstance(inchikey, str) else None,
            names=names,
            delay=max(args.delay, 0.0),
        )
        rows.append(result)
        marker = "OK" if result["status"] == "ok" else "FAIL"
        short_uid = uid.split(":", 1)[-1][:40]
        print(f"[{index:3d}/{total}] {marker:4s} {short_uid:40s} {result['source']}")
        if args.limit is not None and index >= args.limit:
            break
        # 每 10 条增量落盘，避免中断丢进度
        if index % 10 == 0:
            pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)

    frame = pd.DataFrame(rows).drop_duplicates("compound_uid", keep="last")
    io.write_parquet(frame, out_path.with_suffix(".parquet"))
    frame.to_csv(out_path, sep="\t", index=False)
    coverage = {
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_compounds": int(len(compounds)),
        "n_processed": int(len(frame)),
        "n_ok": int((frame["status"] == "ok").sum()),
        "n_failed": int((frame["status"] == "failed").sum()),
        "coverage": float((frame["status"] == "ok").mean()) if len(frame) else 0.0,
        "by_source": frame.loc[frame["status"] == "ok", "source"].value_counts().to_dict(),
        "failures": frame.loc[frame["status"] == "failed", ["compound_uid", "drug_name", "error"]]
        .to_dict("records"),
        "output": str(out_path),
        "note": "SMILES 来自 PubChem；再分发前核对 PubChem 条款与化合物盐型/立体化学",
    }
    io.write_json(out_path.with_name(out_path.stem + "_manifest.json"), coverage)
    print(json.dumps(coverage, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

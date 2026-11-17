#!/usr/bin/env python3
"""下载 P1 外部药敏/药物-靶点数据（见 `docs/data_gaps.md` §2）。

数据源与落盘位置：
    Open Targets 26.09   -> data/raw/opentargets/26.09/
    ChEMBL mechanism/indication -> data/raw/chembl/37/
    GDSC1 release 8.5    -> data/raw/gdsc/
    CTRPv2 v21 AUC       -> data/raw/ctrp/
    PRISM secondary      -> data/raw/prism/

用法：
    .venv/bin/python scripts/download_p1_external.py --only all
    .venv/bin/python scripts/download_p1_external.py --only opentargets,gdsc1
    .venv/bin/python scripts/download_p1_external.py --only ctrp \
        --proxy http://127.0.0.1:7897

说明：
- 支持断点续传（curl 式 `.part` 文件）；已存在的非空文件默认跳过，`--force` 重下。
- 每个来源写 `download_manifest.json`（URL、大小、sha256/md5、抓取时间）。
- 新文件的 MD5 会合并进 `data/raw/CHECKSUMS.md5`。
- ChEMBL 只抓官方 API 的 `mechanism` / `drug_indication` 两张表；5.6 GB 的
  v36 SQLite 全库不落盘。官方 API 当前为 ChEMBL 37，版本差异写入 manifest。
- Zenodo 在部分网络需代理；`--proxy` 只用于 Zenodo，其余源直连。
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
UA = "kite-drp-data/1.0 (+https://github.com/)"

OPENTARGETS_RELEASE = "26.09"
OPENTARGETS_BASE = f"https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/{OPENTARGETS_RELEASE}"
GDSC_RELEASE = "8.5"
GDSC_BASE = f"https://cog.sanger.ac.uk/cancerrxgene/GDSC_release{GDSC_RELEASE}"
CTRP_RECORD = "21721747"
CTRP_API = f"https://zenodo.org/api/records/{CTRP_RECORD}"
PRISM_FILES = {
    "20237739": "secondary-screen-dose-response-curve-parameters.csv",
    "20237757": "secondary-screen-replicate-collapsed-logfold-change.csv",
    "20237769": "secondary-screen-cell-line-info.csv",
    "20237763": "secondary-screen-replicate-collapsed-treatment-info.csv",
    "20238123": "secondary-screen-readme.txt",
}


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _opener(proxy: str | None = None) -> urllib.request.OpenerDirector:
    handlers: list[Any] = []
    if proxy:
        handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
    return urllib.request.build_opener(*handlers)


def _request(
    url: str, opener: urllib.request.OpenerDirector, *, headers: dict[str, str] | None = None
) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    return opener.open(request, timeout=180)


def _hash_file(path: Path) -> tuple[str, str]:
    sha = hashlib.sha256()
    md5 = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            sha.update(chunk)
            md5.update(chunk)
    return sha.hexdigest(), md5.hexdigest()


def _file_entry(path: Path, url: str) -> dict[str, Any]:
    sha, md5 = _hash_file(path)
    return {
        "path": str(path.relative_to(ROOT)),
        "url": url,
        "bytes": path.stat().st_size,
        "sha256": sha,
        "md5": md5,
    }


def _remote_size(url: str, opener: urllib.request.OpenerDirector) -> int | None:
    """HEAD 获取远端大小；失败/无 Content-Length 时返回 None。"""
    try:
        request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": UA})
        with opener.open(request, timeout=60) as response:
            value = response.headers.get("Content-Length")
        size = int(value) if value and value.isdigit() else 0
        return size or None
    except Exception:
        return None


def download(
    url: str,
    target: Path,
    *,
    opener: urllib.request.OpenerDirector,
    force: bool = False,
    retries: int = 5,
) -> dict[str, Any]:
    """下载单文件；存在完整目标时跳过（除非 --force）。

    对支持 HEAD 的服务器用 Content-Length 校验完整性；被截断的响应会保留
    `.part` 并在下一次尝试用 Range 续传。
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    remote_size = _remote_size(url, opener)
    part = target.with_name(target.name + ".part")

    if target.exists() and target.stat().st_size > 0 and not force:
        size = target.stat().st_size
        if remote_size is None or size == remote_size:
            print(f"[skip] {target.relative_to(ROOT)} ({size:,} B)")
            return _file_entry(target, url)
        if size < remote_size:
            if part.exists():
                target.unlink()
            else:
                target.replace(part)
            print(
                f"[resume] {target.relative_to(ROOT)}: {size:,}/{remote_size:,} B",
                file=sys.stderr,
            )
        else:
            target.unlink()

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            offset = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            expected_size = remote_size
            with _request(url, opener, headers=headers) as response:
                status = getattr(response, "status", 200)
                append = bool(offset and status == 206)
                mode = "ab" if append else "wb"
                if expected_size is None:
                    content_length = response.headers.get("Content-Length")
                    if content_length is not None:
                        expected_size = (offset if append else 0) + int(content_length)
                with part.open(mode) as handle:
                    while chunk := response.read(1 << 20):
                        handle.write(chunk)
            actual = part.stat().st_size
            if actual == 0:
                raise OSError("empty response")
            if expected_size is not None and actual != expected_size:
                raise OSError(f"incomplete response: {actual}/{expected_size} bytes")
            part.replace(target)
            entry = _file_entry(target, url)
            print(f"[done] {target.relative_to(ROOT)} ({entry['bytes']:,} B)")
            return entry
        except urllib.error.HTTPError as exc:
            complete = part.exists() and part.stat().st_size > 0
            if (
                exc.code == 416
                and complete
                and (remote_size is None or part.stat().st_size == remote_size)
            ):
                part.replace(target)
                entry = _file_entry(target, url)
                print(f"[done] {target.relative_to(ROOT)} ({entry['bytes']:,} B)")
                return entry
            last_error = exc
        except Exception as exc:  # 网络中断/超时
            last_error = exc
        wait = min(30, 2 * attempt)
        print(f"[retry {attempt}/{retries}] {url}: {last_error}; sleep {wait}s", file=sys.stderr)
        time.sleep(wait)
    raise RuntimeError(f"下载失败: {url}: {last_error}")


def _write_manifest(out_dir: Path, payload: dict[str, Any], files: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    body = {"retrieved_at": _now(), **payload, "files": files}
    (out_dir / "download_manifest.json").write_text(
        json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def update_checksums(entries: list[dict[str, Any]]) -> None:
    """把新条目合并进 data/raw/CHECKSUMS.md5（路径相对 data/raw）。"""
    if not entries:
        return
    path = RAW / "CHECKSUMS.md5"
    existing: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "  " in line:
                md5, rel = line.split("  ", 1)
                existing[rel.strip()] = md5.strip()
    for entry in entries:
        rel = Path(entry["path"]).relative_to("data/raw").as_posix()
        existing[f"./{rel}"] = entry["md5"]
    path.write_text(
        "\n".join(f"{md5}  {rel}" for rel, md5 in sorted(existing.items())) + "\n",
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# Open Targets
# --------------------------------------------------------------------------- #
def _list_parquet(url: str, opener: urllib.request.OpenerDirector) -> list[str]:
    with _request(url, opener) as response:
        html = response.read().decode("utf-8", "replace")
    names = sorted(set(re.findall(r'href="([^"]+\.parquet)"', html)))
    if not names:
        raise RuntimeError(f"目录中没有 parquet: {url}")
    return names


def fetch_opentargets(*, proxy: str | None, force: bool) -> list[dict[str, Any]]:
    out = RAW / "opentargets" / OPENTARGETS_RELEASE
    opener = _opener()
    files: list[dict[str, Any]] = []
    for table in ("target", "drug_molecule", "drug_mechanism_of_action", "clinical_indication"):
        base = f"{OPENTARGETS_BASE}/output/{table}/"
        for name in _list_parquet(base, opener):
            files.append(download(f"{base}{name}", out / table / name, opener=opener, force=force))
    files.append(
        download(
            f"{OPENTARGETS_BASE}/release_data_integrity.sha1",
            out / "release_data_integrity.sha1",
            opener=opener,
            force=force,
        )
    )
    _write_manifest(
        out,
        {
            "source": "Open Targets Platform",
            "release": OPENTARGETS_RELEASE,
            "base_url": OPENTARGETS_BASE,
            "license": "CC0 4.0",
            "tables": [
                "target",
                "drug_molecule",
                "drug_mechanism_of_action",
                "clinical_indication",
            ],
        },
        files,
    )
    return files


# --------------------------------------------------------------------------- #
# ChEMBL API
# --------------------------------------------------------------------------- #
CHEMBL_TABLES = {
    "mechanism": (
        "mechanisms",
        [
            "mec_id",
            "molecule_chembl_id",
            "parent_molecule_chembl_id",
            "target_chembl_id",
            "action_type",
            "mechanism_of_action",
            "direct_interaction",
            "disease_efficacy",
            "molecular_mechanism",
            "max_phase",
            "binding_site_comment",
            "mechanism_comment",
            "selectivity_comment",
            "variant_sequence",
            "site_id",
            "record_id",
            "mechanism_refs",
        ],
    ),
    "drug_indication": (
        "drug_indications",
        [
            "drugind_id",
            "molecule_chembl_id",
            "parent_molecule_chembl_id",
            "efo_id",
            "efo_term",
            "mesh_id",
            "mesh_heading",
            "max_phase_for_ind",
            "indication_refs",
        ],
    ),
}
CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"


def _get_json(url: str, opener: urllib.request.OpenerDirector, retries: int = 4) -> dict[str, Any]:
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with _request(url, opener, headers={"Accept": "application/json"}) as response:
                return json.load(response)
        except Exception as exc:
            last = exc
            time.sleep(min(20, 2 * attempt))
    raise RuntimeError(f"JSON 请求失败: {url}: {last}")


def fetch_chembl(*, proxy: str | None, force: bool) -> list[dict[str, Any]]:
    out = RAW / "chembl" / "37"
    out.mkdir(parents=True, exist_ok=True)
    opener = _opener()
    status = _get_json(f"{CHEMBL_API}/status.json", opener)
    version = str(status.get("chembl_db_version", "ChEMBL_37")).replace("ChEMBL_", "")
    files: list[dict[str, Any]] = []
    counts: dict[str, int] = {}

    for table, (key, columns) in CHEMBL_TABLES.items():
        jsonl_path = out / f"chembl_{version}_{table}.jsonl.gz"
        tsv_path = out / f"chembl_{version}_{table}.tsv.gz"
        if jsonl_path.exists() and tsv_path.exists() and not force:
            print(f"[skip] {jsonl_path.relative_to(ROOT)}")
            files.append(_file_entry(jsonl_path, f"{CHEMBL_API}/{table}.json"))
            files.append(_file_entry(tsv_path, f"{CHEMBL_API}/{table}.json"))
            with gzip.open(jsonl_path, "rt", encoding="utf-8") as handle:
                counts[table] = sum(1 for _ in handle)
            continue

        records: list[dict[str, Any]] = []
        offset = 0
        limit = 1000
        while True:
            url = f"{CHEMBL_API}/{table}.json?limit={limit}&offset={offset}"
            payload = _get_json(url, opener)
            page = payload.get(key, [])
            records.extend(page)
            total = int(payload.get("page_meta", {}).get("total_count", 0))
            offset += len(page)
            print(f"[chembl] {table}: {offset}/{total}")
            if not page or offset >= total:
                break

        tmp_jsonl = jsonl_path.with_name(jsonl_path.name + ".part")
        tmp_tsv = tsv_path.with_name(tsv_path.name + ".part")
        with (
            gzip.open(tmp_jsonl, "wt", encoding="utf-8") as raw_handle,
            gzip.open(tmp_tsv, "wt", encoding="utf-8", newline="") as tsv_handle,
        ):
            writer = csv.DictWriter(tsv_handle, fieldnames=columns, delimiter="\t")
            writer.writeheader()
            for record in records:
                raw_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                row = {column: record.get(column) for column in columns}
                for ref_key in ("mechanism_refs", "indication_refs"):
                    if isinstance(row.get(ref_key), (list, dict)):
                        row[ref_key] = json.dumps(row[ref_key], ensure_ascii=False)
                writer.writerow(row)
        tmp_jsonl.replace(jsonl_path)
        tmp_tsv.replace(tsv_path)
        files.append(_file_entry(jsonl_path, f"{CHEMBL_API}/{table}.json"))
        files.append(_file_entry(tsv_path, f"{CHEMBL_API}/{table}.json"))
        counts[table] = len(records)
        print(f"[done] {tsv_path.relative_to(ROOT)} ({len(records):,} rows)")

    (out / "chembl_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    files.append(_file_entry(out / "chembl_status.json", f"{CHEMBL_API}/status.json"))
    _write_manifest(
        out,
        {
            "source": "ChEMBL",
            "planned_version": "36",
            "api_version": status.get("chembl_db_version"),
            "release_date": status.get("chembl_release_date"),
            "api_base": CHEMBL_API,
            "license": "CC BY-SA 3.0 Unported",
            "counts": counts,
            "note": "官方 API 仅提供当前版本；本次为 ChEMBL 37，未下载 5.6 GB v36 SQLite 全库。",
        },
        files,
    )
    return files


# --------------------------------------------------------------------------- #
# GDSC1
# --------------------------------------------------------------------------- #
def fetch_gdsc1(*, proxy: str | None, force: bool) -> list[dict[str, Any]]:
    out = RAW / "gdsc"
    opener = _opener()
    files: list[dict[str, Any]] = []
    for name in (
        "GDSC1_fitted_dose_response_27Oct23.csv",
        "screened_compounds_rel_8.5.csv",
        "GDSC_Fitted_Data_Description.pdf",
    ):
        files.append(
            download(
                f"{GDSC_BASE}/{name}",
                out / name,
                opener=opener,
                force=force,
                retries=20 if name.endswith(".csv") else 5,
            )
        )
    _write_manifest(
        out,
        {
            "source": "GDSC1",
            "release": f"release-{GDSC_RELEASE} / 27Oct23",
            "base_url": GDSC_BASE,
            "license": "GDSC academic use; redistribution pending review",
            "note": "GDSC1 仅用于 P5 外部验证，不参与训练。",
        },
        files,
    )
    return files


# --------------------------------------------------------------------------- #
# CTRPv2
# --------------------------------------------------------------------------- #
def fetch_ctrp(*, proxy: str | None, force: bool) -> list[dict[str, Any]]:
    out = RAW / "ctrp"
    if not proxy:
        proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    opener = _opener(proxy)
    record = _get_json(CTRP_API, opener)
    (out / f"ctrp_zenodo_record_{CTRP_RECORD}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    files: list[dict[str, Any]] = []
    for name in (
        "v21_auc_sensitivities.tsv",
        "treatment.tsv",
        "sample.tsv",
        "gene_annotation.tsv",
        "molecular_feature_annotation.tsv",
    ):
        files.append(
            download(f"{CTRP_API}/files/{name}/content", out / name, opener=opener, force=force)
        )
    files.append(_file_entry(out / f"ctrp_zenodo_record_{CTRP_RECORD}.json", CTRP_API))
    _write_manifest(
        out,
        {
            "source": "CTRPv2",
            "release": "CTRPv2.1 / v21 AUC (Zenodo processed release)",
            "base_url": CTRP_API,
            "license": "CC BY 4.0",
            "note": "原 NCI/Broad 直链已下线；使用 Haibe-Kains 实验室 v21 AUC 处理版。",
        },
        files,
    )
    return files


# --------------------------------------------------------------------------- #
# PRISM
# --------------------------------------------------------------------------- #
def fetch_prism(*, proxy: str | None, force: bool) -> list[dict[str, Any]]:
    out = RAW / "prism"
    opener = _opener()
    files: list[dict[str, Any]] = []
    for file_id, name in PRISM_FILES.items():
        url = f"https://ndownloader.figshare.com/files/{file_id}"
        files.append(download(url, out / name, opener=opener, force=force))
    _write_manifest(
        out,
        {
            "source": "PRISM Repurposing Secondary Screen",
            "release": "2019-11-22 processed",
            "base_url": "https://depmap.org/repurposing/",
            "license": "CC BY 4.0 (DepMap/CCLE; verify latest terms)",
            "figshare_file_ids": PRISM_FILES,
        },
        files,
    )
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 P1 外部数据")
    parser.add_argument(
        "--only",
        default="all",
        help="逗号分隔: all,opentargets,chembl,gdsc1,ctrp,prism（默认 all）",
    )
    parser.add_argument("--force", action="store_true", help="已存在也重下")
    parser.add_argument(
        "--proxy",
        default=None,
        help="仅用于 Zenodo 的 HTTP 代理，例如 http://127.0.0.1:7897",
    )
    args = parser.parse_args()

    wanted = {item.strip().lower() for item in args.only.split(",") if item.strip()}
    if "all" in wanted:
        wanted = {"opentargets", "chembl", "gdsc1", "ctrp", "prism"}
    tasks = {
        "opentargets": fetch_opentargets,
        "chembl": fetch_chembl,
        "gdsc1": fetch_gdsc1,
        "ctrp": fetch_ctrp,
        "prism": fetch_prism,
    }
    unknown = wanted - tasks.keys()
    if unknown:
        parser.error(f"未知数据源: {sorted(unknown)}")

    all_files: list[dict[str, Any]] = []
    for name in ("opentargets", "chembl", "gdsc1", "ctrp", "prism"):
        if name not in wanted:
            continue
        print(f"\n===== {name} =====")
        all_files.extend(tasks[name](proxy=args.proxy, force=args.force))

    update_checksums(all_files)
    print(f"\n[checksums] {len(all_files)} 个文件已合并进 data/raw/CHECKSUMS.md5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

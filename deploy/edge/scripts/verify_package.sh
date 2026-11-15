#!/usr/bin/env bash
# =============================================================================
# KITE-DRP 部署包校验：manifest.json + model.onnx SHA256（总计划书 §9.3 #10）
#
# 用法：
#   ./verify_package.sh <部署包目录>
#
# 校验内容：
#   1) <包>/manifest.json 存在、可解析、含 model_version 与 outputs.response
#   2) <包>/model.onnx（或 manifest.model_file）存在
#   3) 实际 SHA256 与 manifest.model_sha256 完全一致（缺失/不匹配 → 失败）
#   4) 打印 基因列表长度 / 药表大小 / 文件大小，便于人工复核部署包内容
#
# 退出码：0=通过；1=校验失败；2=用法或文件缺失
# 依赖：python3 标准库（树莓派/医生站/训练机均可用），不需要 onnxruntime。
# =============================================================================
set -euo pipefail

usage() {
    echo "用法: $0 <部署包目录>" >&2
}

if [[ $# -ne 1 ]]; then
    usage
    exit 2
fi

PKG_DIR="$1"
if [[ ! -d "$PKG_DIR" ]]; then
    echo "[FAIL] 部署包目录不存在：$PKG_DIR" >&2
    exit 2
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "[FAIL] 未找到 python3（校验脚本仅依赖 Python 标准库）" >&2
    exit 2
fi

exec python3 - "$PKG_DIR" <<'PY'
"""校验 KITE-DRP 部署包（manifest + model SHA256）。"""
import hashlib
import json
import sys
from pathlib import Path

pkg = Path(sys.argv[1])
manifest_path = pkg / "manifest.json"
if not manifest_path.is_file():
    print(f"[FAIL] 缺少 manifest.json：{manifest_path}", file=sys.stderr)
    sys.exit(1)

try:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    print(f"[FAIL] manifest.json 不可解析：{exc}", file=sys.stderr)
    sys.exit(1)

if not isinstance(manifest, dict):
    print("[FAIL] manifest.json 顶层必须是 JSON 对象", file=sys.stderr)
    sys.exit(1)

missing = [key for key in ("model_version", "model_sha256") if not manifest.get(key)]
outputs = manifest.get("outputs") or {}
if "response" not in outputs:
    missing.append("outputs.response")
if missing:
    print(f"[FAIL] manifest.json 缺少必填字段：{', '.join(missing)}", file=sys.stderr)
    sys.exit(1)

model_file = str(manifest.get("model_file") or "model.onnx")
model_path = pkg / model_file
if not model_path.is_file():
    print(f"[FAIL] 模型文件不存在：{model_path}", file=sys.stderr)
    sys.exit(1)

digest = hashlib.sha256()
with model_path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
actual = digest.hexdigest()
expected = str(manifest["model_sha256"]).strip().lower()
if actual != expected:
    print("[FAIL] model SHA256 不匹配（拒绝部署）", file=sys.stderr)
    print(f"       manifest = {expected}", file=sys.stderr)
    print(f"       actual   = {actual}", file=sys.stderr)
    sys.exit(1)

size_mb = model_path.stat().st_size / (1024 * 1024)
print("[OK] 部署包校验通过")
print(f"     model_version  = {manifest['model_version']}")
print(f"     model_file     = {model_file} ({size_mb:.2f} MiB)")
print(f"     model_sha256   = {actual}")
print(f"     mutation_genes = {len(manifest.get('mutation_genes') or [])}")
print(f"     expression_genes = {len(manifest.get('expression_genes') or [])}")
print(f"     drug_index     = {len(manifest.get('drug_index') or {})}")
PY

#!/usr/bin/env bash
# 一键重建 G0 数据管线（规划 §8.6）。
#
# 用法：
#   bash scripts/preprocess_all.sh [--profile core|full] [--force] [--skip-online]
#
# 说明：
#   1. 默认使用仓库内 .venv；也可用 PYTHON=/path/to/python 覆盖。
#   2. core  = 离线可跑部分（inventory/canonical/splits/fit-stats/build/validate/report）
#      full  = 额外尝试 GDC file index 与联网资源（当前仅实现 file index 拉取）
#   3. --skip-online 传给 canonical，禁止任何联网步骤。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
PROFILE="core"
FORCE=""
SKIP_ONLINE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --force) FORCE="--force"; shift ;;
    --skip-online) SKIP_ONLINE="--skip-online"; shift ;;
    *) echo "未知参数：$1" >&2; exit 2 ;;
  esac
done

if [[ ! -x "$PYTHON" ]]; then
  echo "未找到 Python：$PYTHON（请先 uv sync 或设置 PYTHON）" >&2
  exit 2
fi

echo "[1/7] inventory"
"$PYTHON" -m kite_drp data inventory

echo "[2/7] canonical"
if [[ "$PROFILE" == "full" && -z "$SKIP_ONLINE" ]]; then
  "$PYTHON" -m kite_drp data canonical --fetch-file-index $FORCE
else
  "$PYTHON" -m kite_drp data canonical $FORCE
fi

echo "[3/7] splits"
"$PYTHON" -m kite_drp data splits --all --seeds 0,1,2

echo "[4/7] fit-stats"
"$PYTHON" -m kite_drp data fit-stats $FORCE

echo "[5/7] build"
"$PYTHON" -m kite_drp data build --stage all

echo "[6/7] validate"
"$PYTHON" -m kite_drp data validate --all

echo "[7/7] report"
"$PYTHON" -m kite_drp data report

echo "完成。质量报告：data/interim/audit/quality_report.md"

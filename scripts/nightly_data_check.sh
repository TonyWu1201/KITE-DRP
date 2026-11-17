#!/usr/bin/env bash
# 夜间全量数据检查（规划 §8.5）。建议 cron 每周运行一次，不依赖网络。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"

"$PYTHON" -m kite_drp data validate --all --strict
"$PYTHON" -m pytest -q
"$PYTHON" -m kite_drp data report

echo "夜间数据检查通过。"

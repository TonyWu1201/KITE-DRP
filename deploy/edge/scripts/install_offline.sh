#!/usr/bin/env bash
# =============================================================================
# 树莓派/离线医生站：KITE-DRP 本机服务离线安装脚手架（一期文档化脚本）
#
# 定位：把「离线 bundle」安装成可启动的端侧服务。**尚未在真实 Pi 5 上验证**；
# 离线环境的依赖安装方式（wheelhouse / uv 缓存 / 系统包）需按现场情况调整，
# 详见 deploy/edge/README.md 与 deploy/device/README.md。
#
# 离线 bundle 约定（由训练机打包，U 盘拷贝进院）：
#   <bundle>/
#   ├── server/                 # deploy/server 源码目录（含 pyproject.toml/uv.lock/app/）
#   ├── packages/current/       # 部署包：manifest.json + model.onnx（SHA256 校验）
#   └── wheelhouse/             # 离线依赖 wheel（fastapi/pydantic/uvicorn/onnxruntime/...）
#
# 用法：
#   ./install_offline.sh --bundle /media/usb/kite-drp-offline [--target DIR]
#
# 退出码：0=安装成功；1=安装/校验失败；2=用法错误
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE=""
TARGET="${KITE_INSTALL_DIR:-${HOME}/.local/share/kite-drp/server}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
    echo "用法: $0 --bundle <离线bundle目录> [--target <安装目录>]" >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bundle) BUNDLE="${2:-}"; shift 2 ;;
        --target) TARGET="${2:-}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "未知参数：$1" >&2; usage; exit 2 ;;
    esac
done

if [[ -z "$BUNDLE" ]]; then
    usage
    exit 2
fi
if [[ ! -d "$BUNDLE/server" ]]; then
    echo "[FAIL] 离线 bundle 缺少 server/ 目录：$BUNDLE/server" >&2
    exit 1
fi
if [[ ! -d "$BUNDLE/packages/current" ]]; then
    echo "[FAIL] 离线 bundle 缺少 packages/current/ 部署包" >&2
    exit 1
fi

# 1) Python >= 3.12（服务 requires-python；Pi OS Bookworm 自带 3.11 需升级/改用 Trixie）
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "[FAIL] 未找到 $PYTHON_BIN（需要 Python >= 3.12）" >&2
    exit 1
fi
PY_VERSION="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
PY_MAJOR="${PY_VERSION%%.*}"
PY_MINOR="${PY_VERSION#*.}"
if [[ "$PY_MAJOR" -lt 3 ]] || [[ "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 12 ]]; then
    echo "[FAIL] Python $PY_VERSION < 3.12（Pi OS Bookworm 需升级到 Trixie 或使用 uv 管理的 3.12）" >&2
    exit 1
fi
echo "[OK] Python $PY_VERSION"

# 2) 校验部署包（manifest + model SHA256，计划书 §9.3 #10）
"${SCRIPT_DIR}/verify_package.sh" "$BUNDLE/packages/current"

# 3) 复制服务源码到安装目录（保留旧版本以便回滚；设备侧正式流程见 device/README.md）
mkdir -p "$TARGET"
if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete --exclude .venv --exclude tests "$BUNDLE/server/" "$TARGET/"
else
    cp -a "$BUNDLE/server/." "$TARGET/"
fi
echo "[OK] 已复制服务源码到 $TARGET"

# 4) 离线安装依赖（优先 uv + wheelhouse；否则 python -m venv + pip）
export KITE_MODEL_DIR="${KITE_MODEL_DIR:-$BUNDLE/packages/current}"
if command -v uv >/dev/null 2>&1 && [[ -d "$BUNDLE/wheelhouse" ]]; then
    echo "[INFO] 使用 uv 从 wheelhouse 离线同步（$BUNDLE/wheelhouse）"
    (
        cd "$TARGET"
        UV_LINK_MODE=copy uv sync --frozen --offline --no-index \
            --find-links "$BUNDLE/wheelhouse" --extra onnx
    )
elif [[ -d "$BUNDLE/wheelhouse" ]]; then
    echo "[INFO] 使用 python -m venv + pip 从 wheelhouse 离线安装"
    "$PYTHON_BIN" -m venv "$TARGET/.venv"
    "$TARGET/.venv/bin/python" -m pip install --no-index \
        --find-links "$BUNDLE/wheelhouse" fastapi pydantic uvicorn onnxruntime
else
    echo "[FAIL] 缺少 wheelhouse/：端侧离线安装必须随包携带依赖 wheel" >&2
    echo "        参考软件版本：fastapi>=0.115 / pydantic>=2.9 / uvicorn>=0.32 / onnxruntime>=1.20" >&2
    exit 1
fi

# 5) 汇总与下一步
cat <<EOF
[OK] 离线安装完成
     服务目录     : $TARGET
     部署包       : $KITE_MODEL_DIR
     启动命令     : cd "$TARGET" && KITE_MODEL_DIR="$KITE_MODEL_DIR" \\
                    .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
     服务化/自启  : 见 deploy/device/README.md（systemd unit + 双目录原子替换 + 回滚）
     健康检查     : curl -s http://127.0.0.1:8000/health
EOF

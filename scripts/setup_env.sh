#!/usr/bin/env bash
# =============================================================================
# KITE-DRP 环境安装脚本（Linux / WSL）
#
# 用法：
#   ./scripts/setup_env.sh                       # 默认 minimal：核心依赖 + dev 组（不装 CUDA torch）
#   ./scripts/setup_env.sh --profile train       # 训练机：train + graph + explain（需 NVIDIA GPU）
#   ./scripts/setup_env.sh --profile deploy      # 导出/量化 + 临床验证（CPU 可跑）
#   ./scripts/setup_env.sh --profile full        # 全部 extras（约 3–4 GB 下载，需驱动 R560+）
#   ./scripts/setup_env.sh --extra graph --extra deploy    # 在 minimal 上叠加任意 extras
#   ./scripts/setup_env.sh --profile train --extra clinical  # profile 与 --extra 可组合
#   ./scripts/setup_env.sh --no-check            # 安装后跳过 scripts/check_env.py 自检
#   ./scripts/setup_env.sh --help                # 查看帮助
#
# profile 与 extras 的对应关系（详见 docs/environment.md）：
#   minimal : （无 extra）核心依赖 + dev 组；无 GPU 的纯数据/算法开发
#   train   : train graph explain          教师图模型 + 学生训练 + 可解释性（需 GPU）
#   deploy  : deploy clinical              ONNX 导出/量化 + 临床生存分析（CPU 可用）
#   full    : train graph explain deploy clinical（以上全部）
#
# 约定：
#   - 不 sudo、不改系统 Python、不改全局配置；只操作本仓库的 .venv 与 uv 缓存；
#   - uv 未安装时只打印官方安装命令（安全起见不自动 curl | sh）；
#   - 依赖严格按已提交的 uv.lock 安装（--locked）；改了 pyproject.toml 请先 `uv lock`；
#   - CUDA 12.6：torch 来自 download.pytorch.org/whl/cu126，NVIDIA 驱动要求 R560+；
#   - 结束后运行 scripts/check_env.py 自检，并打印下一步命令。
# =============================================================================
set -euo pipefail

PYTHON_VERSION="3.12"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if [ -t 1 ] && [ "${TERM:-dumb}" != "dumb" ]; then
    C_BOLD=$'\033[1m'
    C_BLUE=$'\033[34m'
    C_YELLOW=$'\033[33m'
    C_RED=$'\033[31m'
    C_RESET=$'\033[0m'
else
    C_BOLD=""
    C_BLUE=""
    C_YELLOW=""
    C_RED=""
    C_RESET=""
fi

log() { printf '\n%s==> %s%s\n' "${C_BOLD}${C_BLUE}" "$*" "${C_RESET}"; }
warn() { printf '%s警告：%s%s\n' "${C_YELLOW}" "$*" "${C_RESET}" >&2; }
die() { printf '%s错误：%s%s\n' "${C_RED}" "$*" "${C_RESET}" >&2; exit 1; }

usage() {
    cat <<'USAGE'
用法：scripts/setup_env.sh [选项]

  --profile <name>   minimal（默认）| train | deploy | full
                     minimal: 核心依赖 + dev 组，不装 CUDA torch（无 GPU 开发）
                     train  : train + graph + explain（教师图模型/训练/可解释性，需 GPU）
                     deploy : deploy + clinical（ONNX 导出/量化 + 临床生存分析，CPU 可跑）
                     full   : train + graph + explain + deploy + clinical（约 3–4 GB 下载）
  --extra <name>     在 profile 基础上叠加 extras，可重复：
                     train | graph | explain | deploy | clinical | all
  --no-check         安装后跳过 scripts/check_env.py 自检
  -h, --help         显示本帮助

示例：
  ./scripts/setup_env.sh
  ./scripts/setup_env.sh --profile train
  ./scripts/setup_env.sh --extra graph --extra deploy
  ./scripts/setup_env.sh --profile full --no-check

说明：不 sudo、不改系统 Python；依赖严格按 uv.lock 安装（--locked）；
      CUDA 12.6 需要 NVIDIA 驱动 R560+；详细文档见 docs/environment.md。
USAGE
}

# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------
PROFILE="minimal"
RUN_CHECK=1
EXTRAS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --profile)
            [ $# -ge 2 ] || die "--profile 需要一个参数（minimal|train|deploy|full）"
            PROFILE="$2"
            shift 2
            ;;
        --profile=*)
            PROFILE="${1#*=}"
            shift
            ;;
        --extra)
            [ $# -ge 2 ] || die "--extra 需要一个参数（train|graph|explain|deploy|clinical|all）"
            EXTRAS+=("$2")
            shift 2
            ;;
        --extra=*)
            EXTRAS+=("${1#*=}")
            shift
            ;;
        --no-check)
            RUN_CHECK=0
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            die "未知参数：$1（--help 查看用法）"
            ;;
    esac
done

case "${PROFILE}" in
    minimal) ;;
    train) EXTRAS+=("train" "graph" "explain") ;;
    deploy) EXTRAS+=("deploy" "clinical") ;;
    full) EXTRAS+=("train" "graph" "explain" "deploy" "clinical") ;;
    *) die "未知 profile：${PROFILE}（可选：minimal|train|deploy|full）" ;;
esac

# 去重并校验 extra 名称
VALID_EXTRAS="train graph explain deploy clinical all"
UNIQUE_EXTRAS=()
for extra in "${EXTRAS[@]}"; do
    case " ${VALID_EXTRAS} " in
        *" ${extra} "*) ;;
        *) die "未知 extra：${extra}（可选：${VALID_EXTRAS}）" ;;
    esac
    # shellcheck disable=SC2076  # 这里需要按字面量匹配，不是正则
    if [[ ! " ${UNIQUE_EXTRAS[*]:-} " =~ " ${extra} " ]]; then
        UNIQUE_EXTRAS+=("${extra}")
    fi
done

# ---------------------------------------------------------------------------
# 平台与仓库检查
# ---------------------------------------------------------------------------
case "$(uname -s)" in
    Linux*)
        case "$(uname -m)" in
            x86_64 | amd64) ;;
            aarch64 | arm64)
                warn "当前为 aarch64：端侧/推理可用（onnxruntime），但 cu126 训练未在本项目验证；建议训练机用 x86_64。"
                ;;
            *)
                warn "未识别的 CPU 架构：$(uname -m)，预编译 wheel 覆盖可能不全。"
                ;;
        esac
        ;;
    Darwin*)
        die "本项目不支持 macOS（PyTorch cu126 无 macOS wheel，见 docs/environment.md 与 ADR 0002）。"
        ;;
    MINGW* | MSYS* | CYGWIN*)
        warn "检测到 Windows 上的 bash（Git Bash/MSYS2）：依赖安装请优先使用 scripts/setup_env.ps1。"
        ;;
    *)
        warn "未识别的操作系统：$(uname -s)，脚本按 Linux 流程继续。"
        ;;
esac

cd "${REPO_ROOT}"
[ -f "${REPO_ROOT}/pyproject.toml" ] || die "未找到 pyproject.toml，请确认在 KITE-DRP 仓库内运行本脚本。"
[ -d "${REPO_ROOT}/scripts" ] || die "仓库结构异常：缺少 scripts/ 目录。"

log "KITE-DRP 环境安装：profile=${PROFILE}，extras=[${UNIQUE_EXTRAS[*]:-无}]"
printf '仓库根目录：%s\n' "${REPO_ROOT}"
printf 'Python 版本：%s（与 .python-version 一致）\n' "${PYTHON_VERSION}"

# ---------------------------------------------------------------------------
# 1) uv
# ---------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    printf '\n%s未找到 uv。%s请先安装（无需 sudo，任选其一）：\n\n' "${C_YELLOW}" "${C_RESET}" >&2
    printf '  # 官方安装脚本（Linux）\n' >&2
    printf '  curl -LsSf https://astral.sh/uv/install.sh | sh\n\n' >&2
    printf '  # 或通过包管理器（若系统已支持）\n' >&2
    printf '  pipx install uv          # 需要已安装 pipx\n\n' >&2
    printf '安装后重新打开终端，或执行：source "$HOME/.local/bin/env"\n' >&2
    printf '官方文档：https://docs.astral.sh/uv/getting-started/installation/\n' >&2
    exit 1
fi
log "uv 版本：$(uv --version)"

if [ -n "${UV_CACHE_DIR:-}" ]; then
    printf 'uv 缓存目录（UV_CACHE_DIR）：%s\n' "${UV_CACHE_DIR}"
fi
if [ -n "${UV_DEFAULT_INDEX:-}" ]; then
    printf '自定义默认索引（UV_DEFAULT_INDEX）：%s\n' "${UV_DEFAULT_INDEX}"
fi

# ---------------------------------------------------------------------------
# 2) Python 3.12（uv 管理的独立解释器，不动系统 Python）
# ---------------------------------------------------------------------------
log "准备 Python ${PYTHON_VERSION}"
uv python install "${PYTHON_VERSION}"

# ---------------------------------------------------------------------------
# 3) 依赖安装（严格按 uv.lock）
# ---------------------------------------------------------------------------
if [ ! -f "${REPO_ROOT}/uv.lock" ]; then
    warn "未找到 uv.lock，先运行 uv lock 生成"
    uv lock
fi

log "uv lock --check（确认锁文件与 pyproject.toml 一致）"
uv lock --check

if [ "${#UNIQUE_EXTRAS[@]}" -gt 0 ]; then
    log "uv sync --locked（安装 extras：${UNIQUE_EXTRAS[*]}）"
    SYNC_ARGS=(sync --locked)
    for extra in "${UNIQUE_EXTRAS[@]}"; do
        SYNC_ARGS+=(--extra "${extra}")
    done
    uv "${SYNC_ARGS[@]}"
else
    log "uv sync --locked（核心依赖 + dev 组，不安装 CUDA torch）"
    uv sync --locked
fi

# 训练 extras 体积较大，提前提示磁盘
if command -v df >/dev/null 2>&1; then
    free_kb="$(df -Pk "${REPO_ROOT}" | awk 'NR==2 {print $4}')"
    if [ -n "${free_kb}" ] && [ "${free_kb}" -lt 8388608 ]; then
        warn "仓库所在磁盘剩余不足 8 GB：训练 extras 的 cu126 torch 约需 3–4 GB 下载、6–8 GB 磁盘。"
    fi
fi

# ---------------------------------------------------------------------------
# 4) 环境自检
# ---------------------------------------------------------------------------
if [ "${RUN_CHECK}" -eq 1 ]; then
    log "运行环境自检：uv run python scripts/check_env.py"
    if ! uv run python scripts/check_env.py; then
        warn "环境自检未通过（见上方 [FAIL] 项）。修复后重跑：uv run python scripts/check_env.py"
        exit 1
    fi
else
    printf '\n已跳过自检（--no-check）；可手动运行：uv run python scripts/check_env.py\n'
fi

# ---------------------------------------------------------------------------
# 5) 下一步与 CUDA 说明
# ---------------------------------------------------------------------------
log "安装完成，下一步"
cat <<'EOF'
  uv run python scripts/check_env.py     # 环境自检（训练机加 --require-train 验收 GPU）
  uv run kite-drp paths                  # 确认仓库/数据目录解析
  uv run kite-drp --version              # 检查 CLI 入口
  uv run pytest -q                       # 单元测试
  uv run pre-commit install              # 启用提交前检查（可选）

CUDA 12.6 提示（仅 train/graph/explain 等 GPU 依赖需要）：
  - 需要 NVIDIA 驱动 R560+（CUDA 12.6 GA 驱动；Linux 高于 525.60.13 可经 CUDA
    minor version compatibility 运行，仍建议 R560+），见 docs/environment.md；
  - 验证命令：
      nvidia-smi
      uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
    期望输出包含 2.14.1+cu126 / 12.6 / True；
  - 无 GPU 机器请使用默认 minimal 或 --profile deploy（本脚本不会替你装 cu126 torch）。
EOF

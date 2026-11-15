#Requires -Version 5.1
<#
.SYNOPSIS
    KITE-DRP Windows 环境安装脚本（Windows PowerShell 5.1+ / 无需管理员权限）。

.DESCRIPTION
    在仓库根目录创建 .venv、按已提交的 uv.lock 安装依赖，并运行环境自检。

    profile 与 extras 的对应关系（详见 docs/environment.md）：
      minimal : （无 extra）核心依赖 + dev 组；无 GPU 的纯数据/算法开发（默认）
      train   : train graph explain          教师图模型 + 学生训练 + 可解释性（需 NVIDIA GPU）
      deploy  : deploy clinical              ONNX 导出/量化 + 临床生存分析（CPU 可用）
      full    : train graph explain deploy clinical（以上全部）

    约定：
      - 不修改系统 Python、不需要管理员权限；只操作本仓库 .venv 与 uv 缓存；
      - uv 未安装时只打印官方安装命令（安全起见不自动执行远程脚本）；
      - 依赖严格按 uv.lock 安装（--locked）；改了 pyproject.toml 请先 uv lock；
      - CUDA 12.6：torch 来自 download.pytorch.org/whl/cu126，NVIDIA 驱动要求 R560+。

.PARAMETER Profile
    minimal（默认）| train | deploy | full

.PARAMETER Extra
    在 profile 基础上叠加 extras，可多个（逗号分隔）：
    train | graph | explain | deploy | clinical | all
    例：-Extra graph,deploy

.PARAMETER NoCheck
    安装后不运行 scripts/check_env.py 自检。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1 -Profile train

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1 -Extra graph,deploy

.NOTES
    本文件必须保存为 UTF-8 with BOM：Windows PowerShell 5.1 在无 BOM 时按系统 ANSI
    代码页解析脚本，中文会乱码/解析异常；请勿用会去掉 BOM 的编辑器重存。
    若提示"禁止运行脚本"，可临时使用：
      powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1
    或仅对当前用户放开：
      Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
#>
[CmdletBinding()]
param(
    [Alias("Profile")]
    [ValidateSet("minimal", "train", "deploy", "full")]
    [string]$EnvProfile = "minimal",

    # 不加 ValidateSet：需要同时兼容 PS 控制台的 `-Extra graph,deploy`（数组）
    # 与 `powershell -File ... -Extra graph,deploy`（单个字符串）两种传参方式。
    [string[]]$Extra = @(),

    [switch]$NoCheck
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
}
catch {
    # 某些受限控制台不允许修改输出编码，忽略即可
}

# ---------------------------------------------------------------------------
# 帮助函数
# ---------------------------------------------------------------------------
function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Write-WarnLine {
    param([string]$Message)
    Write-Host "警告：$Message" -ForegroundColor Yellow
}

function Fail {
    param([string]$Message)
    throw $Message
}

function Invoke-Uv {
    param([string[]]$UvArgs)
    Write-Host ("uv " + ($UvArgs -join " ")) -ForegroundColor DarkGray
    & uv @UvArgs
    if ($LASTEXITCODE -ne 0) {
        throw "uv $($UvArgs -join ' ') 失败（退出码 $LASTEXITCODE）"
    }
}

# ---------------------------------------------------------------------------
# 平台与仓库检查
# ---------------------------------------------------------------------------
try {
if (-not [Environment]::Is64BitOperatingSystem) {
    Fail "需要 64-bit Windows 10/11 x64；当前系统不是 64-bit。"
}
if (-not [Environment]::Is64BitProcess) {
    Fail "当前是 32-bit PowerShell。请使用 64-bit PowerShell 运行本脚本（开始菜单默认即是）。"
}

$scriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$repoRoot = Split-Path -Parent $scriptDir
Set-Location -Path $repoRoot

if (-not (Test-Path -LiteralPath (Join-Path $repoRoot "pyproject.toml"))) {
    Fail "未找到 pyproject.toml，请确认在 KITE-DRP 仓库内运行本脚本。"
}

# ---------------------------------------------------------------------------
# profile / extras 组合
# ---------------------------------------------------------------------------
$selected = New-Object System.Collections.Generic.List[string]
switch ($EnvProfile) {
    "minimal" { }
    "train" {
        $selected.Add("train")
        $selected.Add("graph")
        $selected.Add("explain")
    }
    "deploy" {
        $selected.Add("deploy")
        $selected.Add("clinical")
    }
    "full" {
        $selected.Add("train")
        $selected.Add("graph")
        $selected.Add("explain")
        $selected.Add("deploy")
        $selected.Add("clinical")
    }
}
$validExtras = @("train", "graph", "explain", "deploy", "clinical", "all")
$extraItems = New-Object System.Collections.Generic.List[string]
foreach ($item in $Extra) {
    foreach ($part in ($item -split ",")) {
        $trimmed = $part.Trim()
        if ($trimmed) {
            $extraItems.Add($trimmed)
        }
    }
}
foreach ($item in $extraItems) {
    if ($validExtras -notcontains $item) {
        Fail "未知 extra：$item（可选：$($validExtras -join ' | ')）"
    }
    if (-not $selected.Contains($item)) {
        $selected.Add($item)
    }
}
$extraText = if ($selected.Count -gt 0) { $selected -join " " } else { "无" }

Write-Step "KITE-DRP 环境安装：profile=$EnvProfile，extras=[$extraText]"
Write-Host "仓库根目录：$repoRoot"
Write-Host "Python 版本：3.12（与 .python-version 一致）"

# ---------------------------------------------------------------------------
# 1) uv
# ---------------------------------------------------------------------------
$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvCommand) {
    Write-WarnLine "未找到 uv。请先安装（任选其一，无需管理员权限）："
    Write-Host "  winget install --id=astral-sh.uv -e"
    Write-Host '  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"'
    Write-Host "安装后重新打开 PowerShell 再运行本脚本。"
    Write-Host "官方文档：https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
}
Write-Step "uv 版本：$(& uv --version)"
Write-Host "uv 可执行文件：$($uvCommand.Path)"

if ($env:UV_CACHE_DIR) {
    Write-Host "uv 缓存目录（UV_CACHE_DIR）：$env:UV_CACHE_DIR"
}
if ($env:UV_DEFAULT_INDEX) {
    Write-Host "自定义默认索引（UV_DEFAULT_INDEX）：$env:UV_DEFAULT_INDEX"
}

# ---------------------------------------------------------------------------
# 2) Python 3.12（uv 管理的独立解释器，不动系统 Python）
# ---------------------------------------------------------------------------
Write-Step "准备 Python 3.12"
Invoke-Uv -UvArgs @("python", "install", "3.12")

# ---------------------------------------------------------------------------
# 3) 依赖安装（严格按 uv.lock）
# ---------------------------------------------------------------------------
if (-not (Test-Path -LiteralPath (Join-Path $repoRoot "uv.lock"))) {
    Write-WarnLine "未找到 uv.lock，先运行 uv lock 生成"
    Invoke-Uv -UvArgs @("lock")
}

Write-Step "uv lock --check（确认锁文件与 pyproject.toml 一致）"
Invoke-Uv -UvArgs @("lock", "--check")

$syncArgs = @("sync", "--locked")
foreach ($item in $selected) {
    $syncArgs += @("--extra", $item)
}
if ($selected.Count -gt 0) {
    Write-Step "uv sync --locked（安装 extras：$extraText）"
}
else {
    Write-Step "uv sync --locked（核心依赖 + dev 组，不安装 CUDA torch）"
}
Invoke-Uv -UvArgs $syncArgs

# 训练 extras 体积较大，提前提示磁盘
$drive = (Get-Item -LiteralPath $repoRoot).PSDrive
if ($null -ne $drive) {
    $freeGb = [math]::Round($drive.Free / 1GB, 1)
    if ($freeGb -lt 8) {
        Write-WarnLine "仓库所在磁盘剩余 $freeGb GB：训练 extras 的 cu126 torch 约需 3–4 GB 下载、6–8 GB 磁盘。"
    }
    else {
        Write-Host "磁盘剩余：$freeGb GB"
    }
}

# ---------------------------------------------------------------------------
# 4) 环境自检
# ---------------------------------------------------------------------------
if (-not $NoCheck) {
    Write-Step "运行环境自检：uv run python scripts/check_env.py"
    & uv run python scripts/check_env.py
    if ($LASTEXITCODE -ne 0) {
        Write-WarnLine "环境自检未通过（见上方 [FAIL] 项）。修复后重跑：uv run python scripts/check_env.py"
        exit 1
    }
}
else {
    Write-Host ""
    Write-Host "已跳过自检（-NoCheck）；可手动运行：uv run python scripts/check_env.py"
}

# ---------------------------------------------------------------------------
# 5) 下一步与 CUDA 说明
# ---------------------------------------------------------------------------
Write-Step "安装完成，下一步"
Write-Host @"
  uv run python scripts/check_env.py      # 环境自检（训练机加 --require-train 验收 GPU）
  uv run kite-drp paths                   # 确认仓库/数据目录解析
  uv run kite-drp --version               # 检查 CLI 入口
  uv run pytest -q                        # 单元测试
  uv run pre-commit install               # 启用提交前检查（可选）
"@

Write-Host ""
Write-Host "CUDA 12.6 提示（仅 train/graph/explain 等 GPU 依赖需要）：" -ForegroundColor Cyan
Write-Host "  - 需要 NVIDIA 驱动 R560+（CUDA 12.6 GA 驱动；高于 527.41 可经 CUDA"
Write-Host "    minor version compatibility 运行，仍建议 R560+），见 docs/environment.md；"
Write-Host "  - 验证命令："
Write-Host '      nvidia-smi'
Write-Host '      uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"'
Write-Host "    期望输出包含 2.14.1+cu126 / 12.6 / True；"
Write-Host "  - 无 GPU 机器请使用默认 minimal 或 -Profile deploy（本脚本不会替你装 cu126 torch）。"
}
catch {
    Write-Host ""
    Write-Host "错误：$($_.Exception.Message)" -ForegroundColor Red
    Write-Host "安装未完成，请修复后重试（详细说明见 docs/environment.md）。" -ForegroundColor Red
    exit 1
}

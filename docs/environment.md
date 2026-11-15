# 环境配置手册（Windows + Linux / Python 3.12 / CUDA 12.6）

> 本文回答三件事：**要装什么、怎么装、出问题怎么排查**。
> 配套文件：
> - 依赖声明与锁定：[`pyproject.toml`](../pyproject.toml)、[`uv.lock`](../uv.lock)
> - 一键安装：[`scripts/setup_env.sh`](../scripts/setup_env.sh)（Linux/WSL）、[`scripts/setup_env.ps1`](../scripts/setup_env.ps1)（Windows PowerShell 5.1+）
> - 环境自检：[`scripts/check_env.py`](../scripts/check_env.py)（`uv run kite-drp doctor` 等价）
> - 决策记录：[`docs/adr/0002-python-environment-cuda126.md`](adr/0002-python-environment-cuda126.md)
>
> 数据目录约定见计划书 §3.8；端侧部署见 `deploy/README.md`（树莓派只装 onnxruntime，见本文 §8）。

## 1. 平台矩阵

| 维度 | 支持情况 | 说明 / 官方依据 |
|---|---|---|
| Windows | **Windows 10 / 11 x64** | PyTorch cu126 提供 `win_amd64` cp312 wheel；Windows ARM64 无 wheel（只能用 WSL2 或 x64 机器） |
| Linux | **Ubuntu 22.04 / 24.04 / 26.04 x64**（或同代发行版） | cu126 wheel 为 `manylinux_2_28`，要求 **glibc ≥ 2.28**（Ubuntu 22.04 起满足；CentOS 7 等旧系统不行）→ [manylinux 规范](https://github.com/pypa/manylinux) |
| CPU 架构 | 训练/导出：x86_64；端侧推理：aarch64（树莓派） | 见 §8 |
| Python | **3.12.x**（uv 管理，`requires-python = ">=3.12"`） | 与 [`.python-version`](../.python-version) 一致；CUDA 12.6 的 torch 2.14.1 提供 cp312 wheel |
| GPU | NVIDIA RTX 3090 / 4060 Ti（项目现有） | 需 NVIDIA 驱动 **R560+**（见下） |
| CUDA | **12.6（cu126）** | torch `2.14.1+cu126`，通过 `download.pytorch.org/whl/cu126` 索引安装，见 [ADR 0002](adr/0002-python-environment-cuda126.md) |
| NVIDIA 驱动 | **Linux ≥ 560.28.03，Windows ≥ 560.76**（CUDA 12.6 GA）；最低 ≥ 525.60.13 / ≥ 527.41（CUDA 12.x 兼容层） | [CUDA Toolkit Release Notes（驱动版本表）](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)、[CUDA Minor Version Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html) |
| macOS | **不支持** | PyTorch 不为 cu126 提供 macOS wheel；`[tool.uv] environments` 已限定 linux/win32，`uv lock` 在 macOS 上会直接报错，避免产生不可用锁文件 |
| 磁盘 | 最小环境 ≈ 1 GB；训练环境下载 ≈ 3.4 GB、安装后约 7–9 GB | 见 §5、§10 |
| uv | **0.12.20**（CI 固定此版本；本机升级 uv 后需重跑 `uv lock`） | [uv 安装文档](https://docs.astral.sh/uv/getting-started/installation/) |

> 为什么 CUDA 固定 12.6、torch 固定 2.14.1、只支持 Win/Linux：见 [ADR 0002](adr/0002-python-environment-cuda126.md)。
> PyTorch 版本与 wheel 清单可直接查看索引：<https://download.pytorch.org/whl/cu126>。

## 2. 快速开始（推荐：一键脚本）

### 2.1 Linux / WSL

```bash
# 前置（仅训练需要）：NVIDIA 驱动 R560+，并确认 nvidia-smi 可用
cd KITE-DRP

./scripts/setup_env.sh                       # minimal：核心 + dev（不装 CUDA torch）
./scripts/setup_env.sh --profile deploy      # 导出/量化 + 临床验证（CPU 可跑）
./scripts/setup_env.sh --profile train       # 训练机：train + graph + explain（需 GPU）
./scripts/setup_env.sh --profile full        # 全部 extras（下载约 3.4 GB）
./scripts/setup_env.sh --extra graph --extra deploy   # 在 minimal 上叠加任意 extras
```

脚本行为：检查 uv（缺失时打印官方安装命令，不自动执行）→ `uv python install 3.12` →
`uv lock --check` → `uv sync --locked [--extra ...]` → `uv run python scripts/check_env.py` → 打印下一步。
不会 `sudo`、不改系统 Python、依赖严格按 `uv.lock` 安装。

### 2.2 Windows（PowerShell 5.1+，无需管理员权限）

```powershell
cd KITE-DRP

powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1 -Profile deploy
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1 -Profile train
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1 -Profile full
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1 -Extra graph,deploy
```

> `setup_env.ps1` 以 **UTF-8 with BOM** 保存（Windows PowerShell 5.1 才能正确解析中文），
> 请勿用会去掉 BOM 的编辑器重存；脚本帮助见文件头注释（`Get-Help .\scripts\setup_env.ps1 -Full`）。

### 2.3 手动安装（不用脚本时）

```bash
# 1) 安装 uv（Linux/macOS 官方脚本；Windows 用 winget 或官方 PowerShell 脚本）
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows PowerShell（二选一）：
#   winget install --id=astral-sh.uv -e
#   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2) 准备 Python 3.12（uv 管理的独立解释器）
uv python install 3.12

# 3) 校验锁文件并用它安装依赖
uv lock --check
uv sync --locked                                  # 核心 + dev
uv sync --locked --extra deploy --extra clinical  # 导出 + 临床（CPU）
uv sync --locked --extra train --extra graph --extra explain   # GPU 训练机
uv sync --locked --extra all                      # 全部 extras

# 4) 自检
uv run python scripts/check_env.py
```

几点约定：

- **所有命令用 `uv run ...` 执行**（自动使用 `.venv`），不要 `pip install` 或 `conda install` 混装；
- 每次 `uv sync` 未显式带 `--extra` 时，uv 会**移除**之前装的 extras——切换组合时把 extras 一起写上；
- 修改 `pyproject.toml` 后先 `uv lock`（同步提交 `pyproject.toml` + `uv.lock`），CI 会跑 `uv lock --check`。

## 3. uv 项目结构说明

| 文件/配置 | 作用 |
|---|---|
| `pyproject.toml` → `[project].dependencies` | 核心依赖（Windows/Linux 都有 wheel，CI 与无 GPU 开发默认装） |
| `pyproject.toml` → `[project.optional-dependencies]` | `train` / `graph` / `explain` / `deploy` / `clinical` / `all` extras |
| `pyproject.toml` → `[dependency-groups].dev` | pytest / pytest-cov / ruff / mypy / pre-commit / ipykernel，`uv sync` 默认安装 |
| `pyproject.toml` → `[[tool.uv.index]]` + `[tool.uv.sources]` | torch 钉到 cu126 索引；默认索引显式钉 PyPI 官方源 |
| `pyproject.toml` → `[tool.uv] environments` | 只解析 `linux` 与 `win32` 两个平台（排除 macOS） |
| `uv.lock` | 精确锁定 146 个包的版本/哈希/索引，`uv lock --check` 校验 |

CUDA 12.6 的关键配置（**不要删**，否则 torch 会装成 PyPI 默认的 CPU/cu13x 构建）：

```toml
[[tool.uv.index]]
name = "pypi"
url = "https://pypi.org/simple"
default = true

[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu126" }]
```

> uv 0.12 的项目接口没有 `--torch-backend`，显式 index + sources 是官方推荐做法：
> <https://docs.astral.sh/uv/guides/integration/pytorch/>。`explicit = true` 保证该索引只服务 torch，
> 不会"截胡"其他包；`default = true` 的 PyPI 索引保证本地与 CI 解析一致（UV 全局镜像配置见 §9）。

## 4. 依赖分层与安装组合

| 组合 | 命令 | 内容 | 适用机器 | 额外下载（Linux） |
|---|---|---|---|---|
| minimal（默认） | `uv sync --locked` | 核心 14 包 + dev 组 | 无 GPU 的数据/算法开发、CI | ≈ 456 MB（.venv 约 1 GB） |
| deploy | `uv sync --locked --extra deploy --extra clinical` | ONNX/onnxruntime、lifelines、scikit-survival | 导出/量化、临床生存分析（CPU） | ≈ 42 MB |
| train | `uv sync --locked --extra train --extra graph --extra explain` | torch cu126、PyG、RDKit、Captum、SHAP、gseapy | **NVIDIA GPU 训练机** | train+graph+explain 合计 ≈ 3.4 GB |
| full/all | `uv sync --locked --extra all` | 以上全部 | GPU 训练机（含导出/临床） | ≈ 3.5 GB |

- Windows 的 torch cu126 wheel 自带 CUDA 运行库（单 wheel ≈ 2.4 GB），无需另装 CUDA Toolkit；
- Linux 的 torch wheel 通过 PyPI 的 `nvidia-*` 包补 CUDA 运行库（`cuda-toolkit[...]` 等），这也是 Linux 下载量更大的原因；
- 安装/运行训练环境不需要系统级 CUDA Toolkit（不需要 `nvcc`），只需 **NVIDIA 驱动**；
- 端侧（树莓派）**不要**同步本环境，见 §8。

## 5. 环境自检（`check_env.py`）

```bash
uv run python scripts/check_env.py                 # 日常自检（无 GPU 机器应返回 0）
uv run python scripts/check_env.py --strict        # CI/验收：WARN 也算失败
uv run python scripts/check_env.py --require-train # GPU 训练机：要求 cu126 torch + CUDA 可用 + 驱动 R560+
uv run kite-drp doctor                             # 等价入口
```

检查项：Python/位数/平台/glibc/虚拟环境、uv 项目与锁文件（torch 是否来自 cu126）、核心依赖版本与可导入性、
extras 组、torch 版本与 `torch.version.cuda == "12.6"`、`torch.cuda.is_available()`、`nvidia-smi` 驱动版本、
data/ 各目录与 `.gitkeep`、uv/git/git-lfs/pre-commit。

输出标记含义：`[ OK ]` 通过；`[WARN]` 不阻塞（可选组件/硬件/数据目录缺失，默认退出码仍为 0）；
`[FAIL]` 必须修复（退出码 1）；`[SKIP]` 未安装但当前场景可接受。

## 6. GPU / CUDA 验证

```bash
# 1) 驱动与驱动支持的最高 CUDA 版本（输出里要有 Driver Version / CUDA Version 12.6+）
nvidia-smi

# 2) PyTorch 构建与 CUDA 可用性（训练机期望输出 2.14.1+cu126 / 12.6 / True）
uv run python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

# 3) 完整训练环境验收（缺 torch/GPU/驱动会 [FAIL] 并返回 1）
uv run python scripts/check_env.py --require-train
```

期望结果示例（RTX 3090 + 驱动 R560+）：

```text
2.14.1+cu126
12.6
True
NVIDIA GeForce RTX 3090
```

多卡/指定卡：`CUDA_VISIBLE_DEVICES=0` / `CUDA_VISIBLE_DEVICES=0,1`（Windows PowerShell：`$env:CUDA_VISIBLE_DEVICES="0"`）。

## 7. 无 GPU / 纯数据开发（最小安装）

```bash
uv sync --locked          # 只装核心 + dev，不装 torch/CUDA
uv run pytest -q
uv run kite-drp paths
```

可用：数据下载/预处理（pandas/pyarrow）、经典 ML 基线（scikit-learn/xgboost）、调参（optuna）、
统计与绘图（scipy/matplotlib/seaborn）、配置（hydra）、ONNX 推理（若加 `--extra deploy`）。
**不可用**：教师/学生模型训练、VAE、可解释性（Captum/SHAP 的深度模型部分）——需要 §2 的 train 组合 + GPU。

## 8. 树莓派端侧（只装 onnxruntime，不进主环境）

树莓派上**不要**执行 `uv sync`（会拉 torch/CUDA 训练依赖，Pi 上无意义且装不下）。
端侧只需 onnxruntime + numpy，用独立 venv（部署包与脚本见 `deploy/edge/README.md`）：

```bash
# Raspberry Pi OS 64-bit (aarch64)；Python 3.12 可用 uv 安装
uv python install 3.12
uv venv --python 3.12 .venv-edge
uv pip install --python .venv-edge "onnxruntime>=1.32" "numpy>=1.26"

# 验证（输出 Providers 应包含 CPUExecutionProvider）
.venv-edge/bin/python -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"
```

- 装置侧**不装 PyTorch**（省约 1 GB 依赖），训练/导出/量化在 x86 训练机完成（计划书 §9.2）；
- onnxruntime 提供 aarch64 官方 wheel：<https://pypi.org/project/onnxruntime/>；
- Pi 4 默认 FP32、Pi 5 可动态 INT8（以实测为准，计划书 §9.2 工程点 4）。

## 9. 国内网络与镜像（可选）

**本项目 `uv.lock` 以 PyPI 官方源（`https://pypi.org/simple`）生成**，并显式写进 `pyproject.toml` 的
`default = true` 索引，因此本地、CI、其他机器解析结果一致。`~/.config/uv/uv.toml` 里配置的全局镜像
（例如 USTC）**不会**影响本项目解析（项目级配置优先）。

如果确实需要镜像加速（如校园网限制）：

```bash
# 仅对本次命令生效（不写全局配置）；可用 TUNA/阿里云等 PyPI 镜像
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple     # Linux
$env:UV_DEFAULT_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"  # Windows
uv sync --locked
```

注意：

1. 切换索引后 `uv lock --check` 会认为锁文件与索引不一致（uv 把索引视为锁状态的一部分），
   需要本地 `uv lock`——**不要把镜像生成的 `uv.lock` 提交**，否则 CI 与官方源解析结果不一致；
2. PyTorch cu126 的 wheel **只由** `download.pytorch.org` 提供，PyPI 镜像不镜像它；
   该站是 CDN，国内通常可访问，慢时可用 `UV_HTTP_TIMEOUT=300` 并保留缓存重试；
3. 长期建议团队统一网络方案（代理或镜像 + 约定不提交镜像锁文件），不要各自产生不同 lock。

## 10. 常见问题（FAQ）

### Q1 驱动过旧 / `torch.cuda.is_available() == False`

先看 `nvidia-smi` 的 Driver Version：

- **< 525.60.13（Linux）/ < 527.41（Windows）**：无法运行 CUDA 12.x，必须升级驱动；
- **525.60.13 – 560.x**：可经 CUDA minor version compatibility 运行 cu126，但建议升级到 **R560+**
  （CUDA 12.6 GA 驱动：Linux 560.28.03 / Windows 560.76）；
- 升级驱动后重启，再跑 §6 验证。Windows 可从 <https://www.nvidia.com/Download/index.aspx> 下载；
  Ubuntu 可用「软件和更新 → 附加驱动」，或参考 <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/>。

### Q2 torch 装错了（CPU 版 / cu13x / PyPI 默认版）

症状：`torch.version.cuda is None`（CPU 构建）或为 `13.x`；`uv run python scripts/check_env.py` 会 `[FAIL]`。

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda)"
# 期望：2.14.1+cu126 12.6

# 修复：确认 pyproject 的 cu126 index/sources 未被删改，然后重装
uv lock --check
uv sync --locked --extra train --reinstall
```

**不要**用 `pip install torch` / `conda install pytorch` 绕过 uv.lock——那会装成 PyPI 默认构建。

### Q3 `uv lock --check` 失败："The lockfile at uv.lock is outdated"

`pyproject.toml` 改了依赖但 `uv.lock` 没更新。执行 `uv lock`，检查 diff 后把 **两个文件一起** 提交；
CI 会跑 `uv lock --check` 拦截。升级/降级 uv 大版本后也可能需要重跑 `uv lock`（锁格式版本变化）。

### Q4 显式索引与锁文件不一致 / 出现镜像 429

如果某台机器在 `~/.config/uv/uv.toml` 配了全局默认镜像，而项目又显式钉了 PyPI，通常不会有问题
（项目级优先）。若手工用 `UV_DEFAULT_INDEX` 切了镜像，`uv lock --check` 会要求重解析——见 §9，不要提交镜像锁。
镜像返回 `429 Too Many Requests` 时换官方源或稍后重试（本仓库用的 PyPI 官方源在 CI 上稳定）。

### Q5 Windows 长路径（260 字符限制）

uv 缓存/`.venv` 路径过深时可能报 "path too long"：

```powershell
# 管理员 PowerShell 打开长路径（注册表）
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force
# git 同步放开长路径
git config --system core.longpaths true
```

参考：<https://learn.microsoft.com/windows/win32/fileio/maximum-file-path-limitation>。
建议仓库放在 `C:\dev\KITE-DRP` 这类短路径下。

### Q6 Windows 缺 VC++ 运行库（`DLL load failed while importing ...`）

torch / onnxruntime / pyarrow 需要 **Microsoft Visual C++ Redistributable 2015–2022 x64**：

```powershell
winget install Microsoft.VCRedist.2015+.x64
# 或手动下载：https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist
```

装完重启终端，再 `uv sync --locked --reinstall`。

### Q7 uv 缓存 / 离线 / 无家目录（沙箱、CI）

```bash
export UV_CACHE_DIR=/path/to/writable/uv-cache          # 缓存不可写时
export UV_PYTHON_INSTALL_DIR=/path/to/writable/uv-python
export UV_LINK_MODE=copy                                 # 跨文件系统（缓存与 .venv 不同盘）时避免硬链接失败
uv cache dir    # 查看当前缓存目录；uv cache clean 可清理
```

uv 默认缓存位置见 <https://docs.astral.sh/uv/concepts/cache/> 与
<https://docs.astral.sh/uv/reference/environment/>。

### Q8 Git LFS 未安装

`uv run python scripts/check_env.py` 会 `[WARN] git-lfs`。`data/processed/`、`data/splits/`、
`checkpoints/`、`deploy/packages/` 走 Git LFS（见 `.gitattributes`），提交/拉取前装好：

```bash
# Linux: https://git-lfs.com 提供 apt/yum 包；Windows: winget install GitHub.GitLFS
git lfs install
```

### Q9 pre-commit 钩子 / PowerShell 执行策略

```bash
uv run pre-commit install          # 安装 git hook（每个 clone 一次）
uv run pre-commit run --all-files  # 全量跑
uv run pre-commit autoupdate       # 升级 hook 版本（同时同步 ruff 版本）
```

Windows 若提示"禁止运行脚本"：`powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1`，
或对当前用户放开：`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`。
`end-of-file-fixer` 已排除 `**/.gitkeep`（空占位文件），无需处理。

### Q10 `uv sync` 报 "does not include the current environment"

说明当前 OS 不在 `[tool.uv] environments`（linux/win32）中，例如 macOS 或 WSL 之外的路径。
本项目不支持 macOS：请用 Linux/Windows 训练/导出；只在 macOS 上做文档工作可 `uv run --no-project` 或直接用 Python 工具链。

### Q11 aarch64 / Windows ARM64

- Linux aarch64：cu126 索引有 aarch64 wheel，但本项目训练未验证；端侧推理用 onnxruntime（§8）；
- Windows ARM64：cu126 无 `win_arm64` wheel，请用 x64 机器/虚拟机或 WSL2。

### Q12 磁盘不足 / 训练依赖下载慢

`--extra train`（Linux 约 3.4 GB 下载、安装后 7–9 GB）需要预留磁盘；下载慢可设
`UV_HTTP_TIMEOUT=300`、保留 uv 缓存重试（已下载的包不会重复下载）。CI **不装** torch，见 §12。

### Q13 `kite-drp: command not found`

用 `uv run kite-drp ...`（`uv sync` 会把项目本身以可编辑方式装入 `.venv`）；
若仍未安装，先 `uv sync --locked`。CLI 入口定义在 `pyproject.toml` 的 `[project.scripts]`。

### Q14 `check_env.py` 的 WARN 要不要管？

与核心环境无关的 WARN（无 GPU、git-lfs 未装、data/licenses 为空）不影响开发；
训练机验收请用 `--require-train`，完整验收用 `--strict`。CI 不跑 `check_env.py`（避免 runner 硬件差异）。

## 11. 验收命令清单

| # | 命令 | 期望结果 |
|---|---|---|
| 1 | `uv lock --check` | 无输出 / 退出码 0（锁文件与 pyproject 一致） |
| 2 | `uv sync --locked` | 成功；`.venv` 就绪 |
| 3 | `uv run python scripts/check_env.py` | 退出码 0；无 `[FAIL]` |
| 4 | `uv run ruff check .` | `All checks passed!` |
| 5 | `uv run ruff format --check .` | 无文件需要重排 |
| 6 | `uv run pytest -q` | 全部通过 |
| 7 | `uv run pre-commit run --all-files` | 全部钩子通过 |
| 8 | `uv run kite-drp --version` | `kite-drp 0.1.0` |
| 9 | （训练机）`uv run python scripts/check_env.py --require-train` | 退出码 0；torch `2.14.1+cu126`、`torch.version.cuda=12.6`、`CUDA 可用` |
| 10 | （训练机）`nvidia-smi` | Driver Version ≥ 560（或 ≥ 525 兼容层） |
| 11 | （训练机）`uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"` | `2.14.1+cu126 12.6 True` |
| 12 | （CI）GitHub Actions | ubuntu-latest / windows-latest 两个 job 全绿（不装 CUDA） |

## 12. CI 说明（`.github/workflows/ci.yml`）

- 矩阵：`ubuntu-latest` + `windows-latest`，Python 3.12，uv 0.12.20；
- 流程：`uv lock --check` → `uv sync --locked`（核心 + dev，**不装 CUDA extras**）→
  `uv run ruff check .` → `uv run ruff format --check .` → `uv run pytest -q`；
- CPU-only runner 不做 GPU/CUDA 验证；训练环境的验收在 GPU 训练机上执行第 9–11 条命令；
- 不拉取 Git LFS 大文件（`lfs: false`），CI 不依赖数据/权重。

## 13. 维护与升级流程

| 场景 | 步骤 |
|---|---|
| 新增/升级普通依赖 | 改 `pyproject.toml` 下界 → `uv lock` → `uv sync --locked` → `uv run pytest -q` → 提交 `pyproject.toml` + `uv.lock` |
| 升级 torch / CUDA | 先核对目标索引是否有 cp312 的 win_amd64 + manylinux wheel（<https://download.pytorch.org/whl/cu126>），改 `pyproject.toml` 的 extras → `uv lock` → 训练机验收 → 更新 ADR 0002 |
| 升级 uv | 更新 CI 的 `version` 与文档 → `uv lock`（锁格式可能升级）→ 提交 |
| 新增 extra | `[project.optional-dependencies]` + `all` 并集 + §4 表格 + `setup_env.*` 的 profile 映射（如需） |
| 训练机环境漂移 | 训练前跑 `uv run python scripts/check_env.py --require-train`（建议写入训练脚本前置检查） |

## 14. 官方资料索引

- uv：安装 <https://docs.astral.sh/uv/getting-started/installation/>，项目 <https://docs.astral.sh/uv/concepts/projects/>，
  索引 <https://docs.astral.sh/uv/concepts/indexes/>，PyTorch 集成 <https://docs.astral.sh/uv/guides/integration/pytorch/>，
  环境变量 <https://docs.astral.sh/uv/reference/environment/>
- PyTorch CUDA 12.6 wheel 索引：<https://download.pytorch.org/whl/cu126>；历史版本 <https://pytorch.org/get-started/previous-versions/>
- NVIDIA CUDA 12.6：Release Notes <https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html>，
  驱动兼容 <https://docs.nvidia.com/deploy/cuda-compatibility/>
- manylinux（glibc ≥ 2.28）：<https://github.com/pypa/manylinux>
- pre-commit：<https://pre-commit.com>；ruff：<https://docs.astral.sh/ruff/>
- GitHub Actions + uv：<https://github.com/astral-sh/setup-uv>
- Git LFS：<https://git-lfs.com>
- Windows 长路径：<https://learn.microsoft.com/windows/win32/fileio/maximum-file-path-limitation>
- VC++ 运行库：<https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist>

# ADR 0002：Python 依赖环境 —— uv + pyproject / CUDA 12.6（cu126）

- **状态**：已采纳（Accepted）
- **日期**：2026-11-15
- **决策者**：环境/平台负责人（env-engineer）；由项目组评审
- **相关文档**：计划书 §9（端侧部署）、§13.3（软件工具链）、附录 B.2（工程规范）；
  [`pyproject.toml`](../../pyproject.toml)、[`uv.lock`](../../uv.lock)、
  [`docs/environment.md`](../environment.md)、
  [`scripts/setup_env.sh`](../../scripts/setup_env.sh)、[`scripts/setup_env.ps1`](../../scripts/setup_env.ps1)、
  [`scripts/check_env.py`](../../scripts/check_env.py)、[`.github/workflows/ci.yml`](../../.github/workflows/ci.yml)
- **替代/关联 ADR**：无（0001 为仓库/目录结构，见 `docs/adr/`）

## 1. 背景（Context）

KITE-DRP 是教师-学生知识蒸馏项目：训练侧需要 **PyTorch + PyG + RDKit + Captum/SHAP** 等 GPU 依赖，
端侧部署侧需要 **ONNX Runtime**（不装 PyTorch），另有 CPU 侧的数据处理/经典 ML/临床统计依赖。
团队环境为 **Windows 10/11 x64 与 Ubuntu x64 双平台**，GPU 为 RTX 3090 / 4060 Ti（计划书 §13.4），
要求"一键可复现"（附录 B.2：`git commit + config + seed` 三件套）与 CI 可校验。

需要在本 ADR 中确定四件事：

1. 用什么依赖管理工具与工程格式（可复现、跨平台、GPU/CPU 依赖分层）；
2. CUDA 版本与 PyTorch 构建来源（避免成员各自装成 CPU 版/不同 CUDA 版本）；
3. torch 精确版本的锁定策略；
4. 支持哪些操作系统，以及如何在锁文件层面表达这种限制。

约束/事实（2026-11 核实）：

- 目标 CUDA 线为 **12.6（cu126）**：本地驱动/显卡（RTX 3090、4060 Ti）与现有 CUDA 12.x 生态一致，
  且 12.6 是 PyTorch 官方仍在发布 wheel 的 CUDA 线之一；
- PyPI 上 `torch` 最新版为 **2.15.0**；`download.pytorch.org/whl/cu126` 上最新可用的 torch 为
  **2.14.1**，提供 cp312 的 `win_amd64` 与 `manylinux_2_28_x86_64`/`aarch64` wheel；
  **2.15.0 没有 cu126 发布**（2.15.0 只在 cu132/PyPI 默认索引出现）；
- cu126 索引上的 torch wheel 版本形如 `2.14.1+cu126`（PEP 440 本地版本标签）；
- Linux cu126 torch 依赖 PyPI 上的 `cuda-toolkit`/`nvidia-*-cu12` 等运行库包；
  Windows cu126 torch 单 wheel 自带 CUDA 运行库（约 2.4 GB）；
- PyTorch **不为 cu126 提供 macOS wheel**。

## 2. 决策驱动因素（Decision Drivers）

- **可复现**：任何机器（Windows/Linux、CI、服务器）用同一条命令得到同一组精确版本；
- **正确性**：torch 必须是 cu126 构建，不能因默认索引回退成 CPU/cu13x；`torch.version.cuda == "12.6"` 可校验；
- **分层**：无 GPU 的数据/算法开发、CPU 导出、GPU 训练、端侧推理各自安装最小集合；
- **CI 成本**：CI runner 无 GPU，不应下载 GB 级 CUDA wheel；
- **工程规范**：符合附录 B.2（配置优先、可复现、ADR、CI）；
- **团队现实**：成员使用 Windows 与 Ubuntu 两种开发机，需统一的安装/排障文档。

## 3. 备选方案（Options）

### 3.1 依赖管理工具

| 方案 | 优点 | 缺点 | 结论 |
|---|---|---|---|
| A. `conda`/`environment.yml` + pip | 科学计算生态成熟；CUDA 可走 conda 渠道 | 双求解器/双锁机制复杂，跨平台复现弱；conda 与 pip 混装易冲突；CUDA 渠道与 PyPI wheel 版本漂移；CI 慢 | 不选 |
| B. `requirements.txt` + pip-tools | 简单直接；团队熟悉 | 无 GPU/CPU 依赖分层表达力（extra 组），跨平台条件依赖需手写 marker；无项目元数据/CLI 入口管理 | 不选 |
| C. Poetry | 有 lock 与 extras | PyTorch 官方 cu126 索引需 hack（source 优先级/依赖解析器较慢）；对 `[[tool.uv.index]]` 式显式索引支持弱 | 不选 |
| D. **uv + pyproject.toml + uv.lock** | 单一工具链（Python 管理/项目/锁/运行）；**原生多索引与 `[tool.uv.sources]`**，可把 torch 精确钉到 cu126；`environments` 可按平台裁剪解析；速度最快；CI 集成成熟（`setup-uv`） | 生态较新；锁格式版本绑定 uv 版本；成员需接受 `uv run` 工作流 | **选 D** |

选择 uv 的具体依据：
<https://docs.astral.sh/uv/>、<https://docs.astral.sh/uv/guides/integration/pytorch/>、
<https://docs.astral.sh/uv/concepts/indexes/>。

### 3.2 CUDA 版本与 PyTorch 来源

| 方案 | 优点 | 缺点 | 结论 |
|---|---|---|---|
| A. 让成员各自 `pip install torch`（默认 PyPI） | 随意 | 装成 CPU 版或不同 CUDA 版本，训练结果不可比；计划书要求可复现 | 不选 |
| B. `pip install torch --index-url .../whl/cu126` 手工命令 | 简单 | 无法被 lock/CI 固化，易漂移 | 不选 |
| C. uv `--torch-backend`（自动按 CUDA 选构建） | 一行命令 | **uv 0.12 的项目接口不支持该 flag**（仅 `uv pip install` 侧有）；不能在 pyproject/CI 中固化 | 不选 |
| D. **`[[tool.uv.index]]`（explicit）+ `[tool.uv.sources].torch` 钉死 cu126** | 在 pyproject 中声明式固化，lock/CI/所有成员一致；可校验 | 索引 URL 变更需改配置；需注意不要被全局镜像覆盖 | **选 D** |

决策：CUDA 固定 **12.6（cu126）**，torch 来自 <https://download.pytorch.org/whl/cu126>。
理由：项目显卡/驱动与 CUDA 12.x 匹配；cu126 有 cp312 的 Windows 与 Linux wheel；PyTorch 官方文档
推荐用"显式索引 + sources"管理带本地版本号的 CUDA wheel。

### 3.3 torch 精确版本

| 候选 | 结论 |
|---|---|
| PyPI 默认 `torch`（当前 2.15.0，CPU/cu13x） | 不选：不是 cu126 构建 |
| `torch==2.15.0` + cu126 | 不可行：cu126 索引无 2.15.0 发布（已核实 2026-11-15） |
| **`torch==2.14.1`（+cu126）** | **选**：cu126 索引上最新，且同时有 win_amd64 与 manylinux cp312 wheel |
| 将来 2.15+ 的 cu126 | 每当升级前重新核对索引；见 §7 触发条件 |

`==2.14.1` 按 PEP 440 可匹配 `2.14.1+cu126`（本地版本标签），因此 extras 中同时写
`torch==2.14.1` 与 sources 即可锁定到 cu126 构建。

### 3.4 支持平台

| 方案 | 优点 | 缺点 | 结论 |
|---|---|---|---|
| 全平台（含 macOS） | 开发者自由 | **macOS 无 cu126 torch wheel**，lock 会解析失败；项目也不需要 macOS 训练/导出机 | 不选 |
| **Linux + Windows（`[tool.uv] environments`）** | 与真实训练/部署硬件一致；lock 不含 macOS | macOS 上无法 `uv sync`（文档明确说明） | **选** |

同时显式钉默认 PyPI 索引（`default = true`）：避免成员机器的全局镜像配置（如 USTC）让 uv 认为锁文件
索引不一致而重新解析，保证"本地 = CI = 服务器"。

## 4. 决策（Decision）

1. **使用 uv 0.12.x 作为唯一的 Python 环境管理器**，工程元数据写在 `pyproject.toml`（hatchling、src 布局、
   CLI `kite-drp = "kite_drp.cli:main"`），`uv.lock` 提交入库；CI 固定 uv 0.12.20。
2. **CUDA 12.6 / torch 2.14.1+cu126**：
   ```toml
   [[tool.uv.index]]
   name = "pytorch-cu126"
   url = "https://download.pytorch.org/whl/cu126"
   explicit = true

   [tool.uv.sources]
   torch = [{ index = "pytorch-cu126" }]
   ```
   extras：`train`（torch）、`graph`（+PyG/RDKit）、`explain`（+Captum/SHAP/gseapy）、
   `deploy`（ONNX/onnxruntime，无 torch）、`clinical`（lifelines/scikit-survival）、`all`（并集）。
3. **只支持 Linux + Windows**：`[tool.uv] environments = ["sys_platform == 'linux'", "sys_platform == 'win32'"]`；
   glibc ≥ 2.28（manylinux_2_28）。
4. **默认索引钉 PyPI 官方源**（`default = true`），保证锁文件可复现；国内网络方案见 `docs/environment.md` §9。
5. **CI 不安装 CUDA extras**：`uv sync --locked`（核心 + dev）→ ruff → pytest → `uv lock --check`；
   GPU 验收由训练机用 `uv run python scripts/check_env.py --require-train` 完成。
6. **安装与验收脚本化**：`scripts/setup_env.sh` / `scripts/setup_env.ps1` 一键安装（profile：minimal/train/deploy/full），
   `scripts/check_env.py` 自检（Python/依赖/torch cu126/CUDA/驱动/git-lfs/data 目录），退出码 0/1。

## 5. 后果（Consequences）

### 5.1 正面

- 一份 lock 覆盖 Windows/Linux，`uv lock --check` 在 CI 与本地均可校验；
- GPU/CPU 依赖显式分层，无 GPU 成员与 CI 不下载 CUDA wheel（CI 安装量约 0.5 GB）；
- `uv sync --locked` 严格按锁安装，杜绝"我机器上能跑"；
- 安装/验收/排障全部有脚本与文档（`docs/environment.md`、FAQ 14 条、验收清单 12 条）；
- 后续迁移成本低：换 CUDA 线（如 cu128/cu13x）或换构建来源只需改 pyproject 的 index/sources 与 ADR。

### 5.2 代价与风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| cu126 不是"未来 CUDA 线" | PyTorch 新版本可能不再发 cu126（2.15.0 已确认无 cu126） | torch 固定在 2.14.1；升级需评估 CUDA 线迁移（§7） |
| 训练依赖体积大（Linux 下载 ≈3.4 GB，安装后 7–9 GB） | 新成员首次安装慢/占盘 | profile 分层；文档给出体积预算；CI 不装 |
| torch 精确 pin 导致错过上游修复/安全更新 | 版本滞后 | 定期检查 cu126 索引；升级走 §7 流程 |
| 不支持 macOS | macOS 开发者无法 `uv sync` | 文档明确；macOS 仅适合改文档/纯 Python 工具（`--no-project`） |
| 全局镜像配置与 lock 索引不一致 | uv 重新解析/429/误以为锁过期 | 项目显式钉 PyPI 默认索引；镜像用法与限制写入文档 §9 |
| uv 锁格式随版本变化 | 升级 uv 后 CI `uv lock --check` 可能失败 | CI 固定 uv 0.12.20；升级 uv 时同步 `uv lock` 并提交 |
| Windows 无 win_arm64 wheel | ARM64 Windows 不可用 | 文档说明用 x64/WSL2 |

### 5.3 验证证据（2026-11-15 实测）

```text
# cu126 索引提供 cp312 win_amd64 / manylinux 轮子（curl 索引页）
torch-2.14.1+cu126-cp312-cp312-manylinux_2_28_x86_64.whl
torch-2.14.1+cu126-cp312-cp312-win_amd64.whl
# 2.15.0 在 cu126 索引中不存在（最新为 2.14.1；PyPI 默认索引最新为 2.15.0）

$ uv lock                              # 146 packages
$ uv lock --check                      # 退出码 0
$ uv sync --locked                     # 退出码 0
$ uv sync --no-install-project --extra deploy --extra clinical   # 退出码 0
$ uv run pytest -q                     # 39 passed
$ uv run python scripts/check_env.py   # 退出码 0（无 GPU 机器：torch/nvidia-smi 为 SKIP）
```

说明：本机无 NVIDIA GPU，`--extra train` 未实际下载（GB 级 + 无 GPU 验证价值低），
以 `uv lock` 解析成功 + cu126 索引 wheel 存在性 + `--extra deploy/clinical` 实际安装作为验证下限；
GPU 训练环境验收命令已写入 `docs/environment.md` §11（第 9–11 条）。

## 6. 被否决方案的影响（不采纳的代价）

若未来改用 conda 或 Poetry，需要重写依赖声明、重新建立 GPU/CPU 分层与 CI，且要重新解决
"torch cu126 + 本地版本标签"在多索引下的解析问题；收益不足以抵消迁移成本，故维持 uv。

## 7. 重新评估的触发条件

出现以下任一情况时，重新评估本 ADR：

1. 项目需要 CUDA 12.6 之外的能力（新显卡/新特性/上游不再维护 12.6），需迁移到 cu128/cu13x；
2. PyTorch 在 cu126 线上发布 2.15+（届时可升级 torch 版本并更新本 ADR 与 lock）；
3. 需要 macOS 原生开发（如引入 Apple Silicon 端侧推理），需拆分为独立环境或引入平台条件依赖；
4. uv 项目接口开始支持 `--torch-backend` 等价配置，可简化 index/sources；
5. 团队网络环境要求强制使用内网镜像（需重新评估锁文件索引策略与 CI）。

## 8. 参考链接

- uv 文档：<https://docs.astral.sh/uv/>；PyTorch 集成：<https://docs.astral.sh/uv/guides/integration/pytorch/>；
  索引：<https://docs.astral.sh/uv/concepts/indexes/>；项目配置：<https://docs.astral.sh/uv/concepts/projects/config/>
- PyTorch cu126 索引：<https://download.pytorch.org/whl/cu126>；历史版本：<https://pytorch.org/get-started/previous-versions/>
- CUDA 12.6 Release Notes（驱动版本表）：<https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html>
- CUDA 兼容性：<https://docs.nvidia.com/deploy/cuda-compatibility/>
- PEP 440（本地版本标签 `+cu126`）：<https://peps.python.org/pep-0440/>；PEP 621（pyproject）：<https://peps.python.org/pep-0621/>
- manylinux / glibc：<https://github.com/pypa/manylinux>
- 计划书：`plans&notes/端侧肿瘤药敏师生模型_详细计划书.md` §9、§13.3、附录 B

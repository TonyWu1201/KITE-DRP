# KITE-DRP

**K**nowledge-**I**nformed **T**eacher–student for **E**dge **D**rug **R**esponse **P**rediction

端侧肿瘤药物敏感性预测的师生迁移模型（工作代号 **TRACE-DRP**，中文简称「端侧药敏·师生迁移模型」）。

> **研究目标**：用「全组学 + 药物-基因（蛋白）互作网络」的**图教师**，通过四层知识蒸馏 + 组学补全（VAE），
> 把知识迁移给一个**只输入临床可及的面板突变与表达**、不含 GNN、可在树莓派 5 上离线推理的**轻量学生**；
> 并用 TCGA 分层验证（L1 补全保真 / L2 弱标签 / L2′ 生存 / L3 域适应 / L4 案例）补齐「贴近临床」的证据链。
>
> 完整计划书（12 个月路线、实验/临床/端侧规范、决策日志 D1–D19）：
> [`plans&notes/端侧肿瘤药敏师生模型_详细计划书.md`](<plans&notes/端侧肿瘤药敏师生模型_详细计划书.md>)

## 项目状态（2026-11-15）

| 部分 | 状态 |
|---|---|
| 仓库结构 / Git + Git LFS 策略 / 文档体系 | ✅ 已就绪 |
| Python 环境：Windows + Linux、CUDA 12.6（cu126）、uv 锁文件、CI、pre-commit | ✅ 已就绪并通过本地验证 |
| 端侧部署脚手架：本机 Web 服务（前端页面 + HTTP API + 后端排队，离线合规） | ✅ 可运行（demo 模式，待接真实 ONNX 模型） |
| 数据管线（下载/预处理/split）与算法代码（教师/学生/蒸馏/评估） | 🚧 待实现（G0 门禁前完成）；`src/kite_drp/**` 目前为带说明的骨架 |

本轮结构与工程配置的实施记录见
[`plans&notes/2026-11-15_仓库结构与工程配置说明.md`](<plans&notes/2026-11-15_仓库结构与工程配置说明.md>)；
三个工程决策见 [`docs/adr/`](docs/adr/)。

## 快速开始

### 环境要求

| 项 | 要求 |
|---|---|
| 操作系统 | Windows 10/11 x64；Ubuntu 22.04/24.04/26.04 x64（或 WSL2）；**暂不支持 macOS** |
| Python | 3.12（由 uv 管理，无需系统安装） |
| GPU / CUDA | NVIDIA 驱动 **R560+**；PyTorch 固定 **CUDA 12.6**（`torch==2.14.1+cu126`） |
| 工具 | [uv](https://docs.astral.sh/uv/) 0.12+、git；使用 Git LFS 数据前需安装 git-lfs |
| 无 GPU 机器 | 可用：核心依赖 + 开发工具 + 部署/临床 extras，仅训练需要 CUDA |

详细的平台矩阵、GPU 验证、Windows 常见问题、国内镜像与排障见
[`docs/environment.md`](docs/environment.md)。

### 一键安装

```bash
# Linux / WSL —— 默认 minimal：核心依赖 + dev 组（不装 CUDA torch）
./scripts/setup_env.sh

# 训练机（CUDA 12.6）：教师图模型 + 学生训练 + 可解释性
./scripts/setup_env.sh --profile train

# ONNX 导出/量化 + 临床生存分析（CPU 可跑）
./scripts/setup_env.sh --profile deploy

# 全部 extras（约 3–4 GB 下载）
./scripts/setup_env.sh --profile full
```

```powershell
# Windows PowerShell 5.1+（无需管理员权限）
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1 -Profile train
```

### 手动安装（等价命令）

```bash
uv python install 3.12
uv sync --locked                                  # 核心 + dev（无 GPU 也可用）
uv sync --locked --extra train --extra graph --extra explain   # GPU 训练机
uv sync --locked --extra deploy --extra clinical               # 导出/量化 + 临床验证
uv sync --locked --extra all                                   # 全部
```

### 验证

```bash
uv run kite-drp doctor            # 环境自检（等价 scripts/check_env.py）
uv run pytest -q                  # 39 项单元/结构测试
uv run python scripts/check_env.py --require-train   # 训练机：要求 torch 2.14.1+cu126 / CUDA 12.6 / is_available=True
```

期望（GPU 训练机）：`torch.__version__ = 2.14.1+cu126`、`torch.version.cuda = 12.6`、`torch.cuda.is_available() = True`。

## 目录结构

```
KITE-DRP/
├── README.md                  # 本文件
├── pyproject.toml / uv.lock   # uv 项目：依赖、extras、CUDA 12.6 索引、工具链配置
├── .python-version            # 3.12
├── .gitignore / .gitattributes / .editorconfig / .pre-commit-config.yaml
├── .github/workflows/ci.yml   # Ubuntu + Windows 双平台 CI（不装 CUDA）
├── configs/                   # Hydra 组合配置：paths/data/split/teacher/student/kd/edge
├── data/                      # 数据分层（见下文「数据与 Git LFS」）
│   ├── raw/                   #   原始数据（git 忽略，仅保留 README/CHECKSUMS/.gitkeep）
│   ├── interim/               #   中间产物（git 忽略）
│   ├── processed/             #   模型输入缓存（Git LFS）
│   ├── splits/                #   数据划分（Git LFS）
│   └── licenses/              #   数据源许可与引用要求
├── src/kite_drp/              # 主包（可安装；对应计划书附录 B.1）
│   ├── common/                #   仓库路径解析等公共工具
│   ├── data/  models/{teacher,student}/  distill/  train/  eval/  explain/
│   └── deploy/                #   训练机侧 ONNX 导出/量化/端侧基准代码
├── deploy/                    # 部署资产与独立服务（见下文「端侧部署」）
│   ├── server/                #   本机 Web：前端 + HTTP API + 后台排队（独立 uv 项目/环境）
│   ├── edge/                  #   树莓派侧脚手架（安装/校验脚本）
│   ├── device/                #   systemd/离线签名/回滚规划
│   └── packages/              #   部署包产物（Git LFS）
├── scripts/                   # setup_env.sh/.ps1、check_env.py；后续一键复现脚本
├── experiments/               # 实验输出（git 忽略，保留 README/.gitkeep）
├── checkpoints/               # 正式模型权重（Git LFS）
├── tests/                     # pytest：路径/CLI/仓库结构守卫
├── docs/                      # 长期文档：ADR / 数据卡 / 模型卡 / 泄漏审查 / 环境
└── plans&notes/               # Agent 生成的计划与过程文档（含主计划书）
```

## 数据与 Git LFS

- `data/raw/` 约 8.3 GB（GDSC2、DepMap/CCLE、STRING、DGIdb、HGNC、TCGA-CESC 等），**不提交 git**；
  数据索引与校验和（`data/raw/README.md`、`CHECKSUMS.md5`）随仓库跟踪。
- **Git LFS 跟踪**：`data/processed/**`、`data/splits/**`、`deploy/packages/**`，以及
  `*.pt/*.pth/*.ckpt/*.safetensors/*.onnx/*.npz/*.parquet` 等大文件扩展名。
- 被忽略目录内均保留 `.gitkeep`，保证 clone 后目录结构完整。

```bash
git lfs install      # 每台机器一次（需先安装 git-lfs）
git lfs pull         # clone 后拉取 LFS 文件
git lfs ls-files     # 查看 LFS 跟踪文件
```

数据卡与许可见 [`docs/data_cards/`](docs/data_cards/)、[`data/licenses/`](data/licenses/)；
数据缺口与补数据顺序见 [`docs/data_gaps.md`](docs/data_gaps.md)；
**公开 release 不打包受限数据**（TCGA 受控层、DrugBank 等），只提供下载脚本与派生特征。

### 数据管线（G0）

原始数据就位后，用统一 CLI 重建 canonical → splits → fit_stats → processed：

```bash
# 药物结构（scaffold/P3 需要）：PubChem → data/raw/drugs/smiles.tsv
uv sync --extra data --extra graph
.venv/bin/python scripts/download_smiles_pubchem.py

# 先验资源（学生路线 A / 学生 mutation panel）
.venv/bin/python scripts/download_pathways.py
.venv/bin/python scripts/download_mutation_panel.py

# TCGA file→case（联网一次；之后冻结）
.venv/bin/python -m kite_drp data canonical --only tcga --fetch-file-index

# 一键（离线 core profile；缺失资源自动降级并写报告）
.venv/bin/python -m kite_drp data pipeline --profile core

# 分步
.venv/bin/python -m kite_drp data inventory
.venv/bin/python -m kite_drp data canonical
.venv/bin/python -m kite_drp data splits --all --seeds 0,1,2
.venv/bin/python -m kite_drp data fit-stats
.venv/bin/python -m kite_drp data build --stage all
.venv/bin/python -m kite_drp data validate --all
.venv/bin/python -m kite_drp data report
.venv/bin/python -m kite_drp eval naive --split p1_lco_s0
```

产物层次：`data/interim/canonical/`（不提交，本地重建）、`data/processed/`（LFS，模型输入）、
`data/splits/`（LFS，公开可复现）。质量报告写入 `data/interim/audit/quality_report.md`。
**禁止先 z-score 再切分**：所有 fold 内统计量由 `src/kite_drp/data/fit.py` 的 `FitScope` 生成。

## 端侧部署（`deploy/`）

`deploy/server` 是一个**独立 uv 项目**（自带 `pyproject.toml` 与 `.venv`，不污染训练环境）：

```bash
cd deploy/server
uv sync
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
# 浏览器打开 http://127.0.0.1:8000
```

- 前端：纯本地中文单页（无 CDN/外部字体/外网请求），展示健康状态、提交样本、轮询结果、最近任务；
- HTTP API：`POST /api/v1/jobs`（入队，202）、`GET /api/v1/jobs/{id}`、`GET /api/v1/jobs`、`GET /health`、`GET /api/v1/schema`；
- 后端：`asyncio.Queue` + 可配 workers 排队执行，任务状态机 `queued/running/succeeded/failed`；
- 预测器：优先加载 `deploy/packages/current` 的 ONNX 模型（校验 manifest + SHA256），缺失时进入 **demo 模式**并在 UI/响应显式标记；
- 合规：默认仅绑定 `127.0.0.1`、运行期零外网请求、日志不含输入数据；
- 接口契约、Windows/Linux 启动、部署包 manifest 见 [`deploy/server/README.md`](deploy/server/README.md)；
  树莓派与设备运维见 [`deploy/README.md`](deploy/README.md)。

> 当前状态：**科研原型，不用于临床决策**。真实 ONNX 模型加载路径待训练侧导出包联调。

## 配置与实验

- 所有实验由配置驱动（Hydra/OmegaConf），组合方式：`data × split × teacher/student × kd × edge`；
- 运行示例（算法代码实现后）：

  ```bash
  uv run python -m kite_drp.train.teacher_train --config-name config teacher=t0 split=lco
  uv run python -m kite_drp.train.student_train --config-name config student=a kd=default
  ```

- 实验输出写入 `experiments/`（git 忽略），正式权重放 `checkpoints/`（Git LFS）；
- 可复现三件套：`git commit + 配置快照 + seed`，主结果 ≥3 seeds；数据泄漏审查用
  [`docs/leakage_checklist.md`](docs/leakage_checklist.md)。

## 工程约定

- **代码质量**：`uv run ruff check .`、`uv run ruff format --check .`、`uv run pytest -q`；pre-commit 钩子
  （`uv run pre-commit install`）在提交前自动检查；CI 在 Ubuntu + Windows 双平台跑同样的检查。
- **路径**：统一通过 `kite_drp.common.paths` 解析，禁止硬编码绝对路径；可用环境变量 `KITE_DRP_ROOT` 覆盖仓库根。
- **文档分工**：`plans&notes/` 放 Agent 生成的计划与过程文档；`docs/` 放人读的长期文档（ADR/数据卡/模型卡/泄漏审查/环境）。
- **Git 策略说明**：见本文「数据与 Git LFS」及
  [`plans&notes/2026-11-15_仓库结构与工程配置说明.md`](<plans&notes/2026-11-15_仓库结构与工程配置说明.md>)。

## 参考与合规

- 第三方数据许可与引用要求：`data/licenses/`；学术使用前请核对原始条款；
- **科研原型声明**：本仓库代码与模型仅用于科学研究，不构成医疗器械，**不用于临床诊断或治疗决策**；
- 计划书 §9.4：离线运行、数据不出院、日志不含 PHI、患者 ID 伪名化。

## 许可证

代码许可证**尚未最终确定**（计划书建议在 MIT 与 Apache-2.0 之间选择）。公开 release 前必须：
1. 在 `pyproject.toml` 中更新 `license` 字段；
2. 添加对应 `LICENSE` 文件；
3. 确认第三方数据/代码兼容性（见 `data/licenses/`）。

在此之前请将其视为「保留所有权利」的内部开发仓库。

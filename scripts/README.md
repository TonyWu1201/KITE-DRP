# `scripts/` —— 环境与一键复现脚本

| 脚本 | 平台 | 作用 |
|---|---|---|
| [`setup_env.sh`](setup_env.sh) | Linux（bash） | 安装 uv / Python 3.12，按 profile 创建环境 |
| [`setup_env.ps1`](setup_env.ps1) | Windows（PowerShell 5.1+） | 同上（Windows 版） |
| [`check_env.py`](check_env.py) | 跨平台 | 环境自检：Python/依赖版本/CUDA 12.6/驱动/git-lfs/目录 |
| [`preprocess_all.sh`](preprocess_all.sh) | Linux（bash） | G0 数据管线一键重建（inventory→canonical→splits→fit-stats→build→validate→report） |
| [`download_smiles_pubchem.py`](download_smiles_pubchem.py) | 跨平台 | 用 pubchempy 从 PubChem 下载 GDSC 化合物 SMILES（可续传/限速） |
| [`download_pathways.py`](download_pathways.py) | 跨平台 | 下载 Reactome GMT + GO GAF/OBO，转换为 GMT |
| [`download_mutation_panel.py`](download_mutation_panel.py) | 跨平台 | 从 cBioPortal API 下载 MSK-IMPACT panel 基因清单 |
| [`download_p1_external.py`](download_p1_external.py) | Linux/macOS | 下载 P1：Open Targets / ChEMBL / GDSC1 / CTRPv2 / PRISM（断点续传 + manifest） |
| [`download_p1_external.py`](download_p1_external.py) | 跨平台 | 下载 P1 外部数据：Open Targets / ChEMBL / GDSC1 / CTRPv2 / PRISM（可续传/校验/写 manifest） |
| [`nightly_data_check.sh`](nightly_data_check.sh) | Linux（bash） | 夜间全量校验 + pytest + 质量报告 |

## 环境自检

```bash
uv run python scripts/check_env.py     # 完整环境（已 uv sync）
uv run kite-drp doctor                 # 等价入口（CLI）
```

## 数据管线

```bash
uv sync --extra data                                # pubchempy（SMILES 下载）
uv sync --extra graph                               # rdkit/torch-geometric（FP/scaffold/图）
.venv/bin/python scripts/download_smiles_pubchem.py # → data/raw/drugs/smiles.tsv
.venv/bin/python scripts/download_pathways.py       # → Reactome/GO GMT
.venv/bin/python scripts/download_mutation_panel.py # → MSK-IMPACT 468
.venv/bin/python scripts/download_p1_external.py    # → Open Targets/ChEMBL/GDSC1/CTRPv2/PRISM
bash scripts/preprocess_all.sh --profile core       # 离线全量重建
bash scripts/preprocess_all.sh --profile full       # 额外尝试 GDC file index
bash scripts/nightly_data_check.sh                  # validate --strict + pytest + report
```

## 计划中的其他一键脚本（后续里程碑）

| 脚本 | 作用 | 计划书 |
|---|---|---|
| `download_data.sh` / `download_data.ps1` | 下载 GDSC2/DepMap 等原始数据并校验 SHA256（raw 不入库） | §3.2、§3.8 |
| `train_all.sh` | 教师 → VAE → 学生 → KD → QAT 全流程 | §6.3 |
| `export_deploy.sh` | ONNX 导出 + 量化 + 部署包（调用 `kite_drp.deploy`） | §9.2 |
| `bench_pi.sh` | 树莓派端侧延迟/内存基准与验收报告 | §9.3 |

> 约定：脚本全部用 `uv run ...` 执行（自动使用项目环境），禁止 `sudo`；
> 路径通过 `kite_drp.common.paths` 或环境变量 `KITE_DRP_ROOT` 解析，不硬编码绝对路径。
> 完整环境说明见 [`docs/environment.md`](../docs/environment.md)。

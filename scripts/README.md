# `scripts/` —— 环境与一键复现脚本

| 脚本 | 平台 | 作用 |
|---|---|---|
| [`setup_env.sh`](setup_env.sh) | Linux（bash） | 安装 uv / Python 3.12，按 profile 创建环境 |
| [`setup_env.ps1`](setup_env.ps1) | Windows（PowerShell 5.1+） | 同上（Windows 版） |
| [`check_env.py`](check_env.py) | 跨平台 | 环境自检：Python/依赖版本/CUDA 12.6/驱动/git-lfs/目录 |

## 环境自检

```bash
uv run python scripts/check_env.py     # 完整环境（已 uv sync）
uv run kite-drp doctor                 # 等价入口（CLI）
```

## 计划中的一键复现脚本（G0 门禁前补齐）

| 脚本 | 作用 | 计划书 |
|---|---|---|
| `download_data.sh` / `download_data.ps1` | 下载 GDSC2/DepMap 等原始数据并校验 SHA256（raw 不入库） | §3.2、§3.8 |
| `preprocess.sh` | raw → interim → processed + 生成 splits（LFS） | §3.3、§7.2 |
| `train_all.sh` | 教师 → VAE → 学生 → KD → QAT 全流程 | §6.3 |
| `export_deploy.sh` | ONNX 导出 + 量化 + 部署包（调用 `kite_drp.deploy`） | §9.2 |
| `bench_pi.sh` | 树莓派端侧延迟/内存基准与验收报告 | §9.3 |

> 约定：脚本全部用 `uv run ...` 执行（自动使用项目环境），禁止 `sudo`；
> 路径通过 `kite_drp.common.paths` 或环境变量 `KITE_DRP_ROOT` 解析，不硬编码绝对路径。
> 完整环境说明见 [`docs/environment.md`](../docs/environment.md)。

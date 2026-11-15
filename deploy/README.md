# KITE-DRP 端侧部署（`deploy/`）

本目录是 KITE-DRP 的**实际部署资产**：把训练好的端侧学生模型做成可离线分发的
「本机 Web 服务 + 前端单页 + HTTP API + 后端排队」，并提供树莓派边缘端与设备侧
运维脚手架。计划书依据：§9.2–§9.4（推理栈 / 性能预算 / 工程与合规约束）、附录 B。

> ⚠️ **科研原型，不用于临床决策。** 本目录的全部服务与页面都不宣称诊断用途；
> 进入临床决策需按 NMPA/CE-MDR 路径注册与前瞻验证（计划书 §9.4）。

---

## 0. 与 `src/kite_drp/deploy/` 的分工（不要混淆）

| 位置 | 角色 | 内容 | 运行环境 |
|---|---|---|---|
| `src/kite_drp/deploy/` | **训练机侧**导出/量化代码 | `export_onnx.py`（opset 17/batch=1）、`quantize.py`（动态 INT8 / 静态 PTQ / QAT）、`package.py`（ONNX + 配置 + gene list + SHA256 打包）、`benchmark.py` | 仓库根 uv 项目 |
| `deploy/server/` | **部署侧**本机服务 | FastAPI + 纯本地前端单页 + asyncio 队列 + ONNX 预测器（可选）+ 独立 uv 项目/.venv | **独立** uv 项目 |
| `deploy/edge/` | 树莓派侧脚手架 | 离线安装、onnxruntime(aarch64)、systemd 服务化、签名包校验 | 装置本机 |
| `deploy/device/` | 设备侧运维规划 | 开机自启、双目录原子替换、回滚、签名校验（一期文档化） | 装置本机 |
| `deploy/packages/` | 部署产物 | ONNX + `manifest.json` + gene list + SHA256（**Git LFS**，见根 `.gitattributes`） | 只读数据 |

**边界原则**：`deploy/server` 是独立可分发单元，**不 import** 训练侧 `kite_drp` 包；
两边只通过「部署包目录 + manifest 契约」和「HTTP API 契约」交互。这也是未来把
`deploy/server` 拆成 git submodule 的迁移前提（见 [ADR 0003](../docs/adr/0003-deploy-server-and-submodule.md)）。

---

## 1. 目录地图

```text
deploy/
├── README.md                      # 本文件：部署总览/数据流/合规红线/启动入口
├── server/                        # 独立 uv 项目：本机 Web 服务（详见 server/README.md）
│   ├── pyproject.toml             #   name=kite-drp-server，[tool.uv] package=false
│   ├── uv.lock                    #   独立锁文件（与根项目互不影响）
│   ├── .python-version            #   固定 Python 3.12
│   ├── .env.example               #   配置模板（复制为 .env，不入库）
│   ├── README.md                  #   启动/测试/API 契约/安全声明（最详细的部署文档）
│   ├── app/                       #   FastAPI 应用、队列、预测器、schema
│   │   └── static/                #   index.html / style.css / app.js（纯本地，无外链）
│   └── tests/test_api.py          #   接口/队列/静态页/隐私/配置测试
├── edge/                          # 树莓派边缘端脚手架（README + 最小脚本）
│   ├── README.md
│   └── scripts/
│       ├── install_offline.sh     #   离线安装脚手架（wheelhouse + 部署包校验）
│       └── verify_package.sh      #   部署包 manifest/SHA256 校验（可直接复用）
├── device/                        # 设备侧运维规划（systemd/原子替换/回滚/签名）
│   └── README.md
└── packages/                      # 部署产物（Git LFS）；当前仅 .gitkeep 占位
```

---

## 2. 部署形态（计划书 §9.4「工作流集成」）

```text
科研/验收 ──▶ CLI（训练机/装置终端）                     一期
医生站   ──▶ 本机 Web UI（FastAPI，浏览器仅访问 127.0.0.1）  一期 ← deploy/server
平板/触屏 ──▶ 本机 Web UI 的浏览器入口（同一 127.0.0.1 服务） 一期（界面自适应）
HIS/LIS  ──▶ 院内系统对接（接口网关 + 认证 + 审计）          二期
```

- 服务**只监听 `127.0.0.1`**，浏览器与服务在同一台机器（医生站主机或树莓派），
  天然不暴露到医院网络；不实现内网穿透、上报、云调用。
- 树莓派既可以只跑「装置侧推理」，也可以跑**同一个** `deploy/server`（界面自适应手机/平板），
  两者共用同一部署包契约（见 §5）。

---

## 3. 数据流（端到端）

```text
训练机（x86）                                       端侧装置（x86 医生站 / 树莓派）
─────────────────────────────────────────────      ─────────────────────────────────────────────
src/kite_drp/deploy/export_onnx.py  ─┐              deploy/server 启动
src/kite_drp/deploy/quantize.py     ─┼─▶ 部署包 ───▶  ├─ 读取 deploy/packages/current/manifest.json
src/kite_drp/deploy/package.py      ─┘   ├ ONNX         ├─ 校验 model.onnx SHA256（不过 → 拒绝加载）
                                          ├ manifest.json├─ onnxruntime CPU EP 加载（可选依赖）
                                          ├ gene list    └─ 缺失/失败 → demo 模式（显式标记）
                                          └ SHA256        │
                                                          ▼
样本 JSON（伪名化）─HTTP POST /api/v1/jobs─▶ 校验(422) ─▶ asyncio 队列(满→503)
                                                          │ worker（默认 1，asyncio.to_thread）
                                                          ▼
                                          预测（ONNX 或 demo）→ 任务状态机(queued/running/succeeded/failed)
                                                          │ GET /api/v1/jobs/{job_id} 轮询
                                                          ▼
前端单页：response + ci + model_version + model_sha256 + demo 标记 + 「科研原型」声明
```

**数据不出院**：输入特征只在本机进程内存中处理；任务表在内存中（重启清空），
日志只记录 job_id/状态/耗时，**不落盘输入数据、不记录 PHI 与原始基因值**。

---

## 4. 离线与隐私红线（验收必查，计划书 §9.3 #10、§9.4）

| # | 红线 | 本仓库落实方式 |
|---|---|---|
| 1 | 运行期 **0 次外网请求** | 前端只发同源相对请求，页面含 CSP `default-src 'self'`；无 CDN/外部字体；Swagger/ReDoc 默认 UI（从 CDN 拉资源）已关闭，仅保留本地 `/api/v1/openapi.json` |
| 2 | 仅本机访问 | 默认 `KITE_SERVER_HOST=127.0.0.1`；非回环地址启动会打印高亮警告（正式部署禁止） |
| 3 | 日志不含 PHI/原始基因值 | 队列日志只含 job_id/状态/耗时；测试 `test_logs_do_not_leak_sample_id_or_gene_values` 守护 |
| 4 | 输入伪名化 | `sample_id` 只允许字母/数字/`._-`；UI 明示禁止姓名/住院号等 PHI |
| 5 | 模型 SHA256 校验通过才加载 | 预测器校验失败一律拒绝加载真实模型并回退 demo（计划书 §9.3 #10） |
| 6 | 离线可复现 | demo 模式确定性伪分数；真实模型 ONNX + 固定基因列表 + mask 对齐，无需联网 |
| 7 | 不宣称临床用途 | `/health`、`POST` 响应、每次结果 `warnings`、前端顶部横幅均含「科研原型，不用于临床决策」 |
| 8 | 不输出硬分类/临床阈值 | 只输出 0–1 科研未校准分数 + 区间（附录 C：临床呈现以分数+CI 为主） |

---

## 5. 部署包契约（训练机导出 ↔ 端侧加载）

部署包 = 一个目录（默认 `deploy/packages/current`，由 `KITE_MODEL_DIR` 覆盖）：

```text
<deploy-package>/
├── manifest.json    # 版本、SHA256、基因列表、药表、ONNX 输入输出名（必选）
└── model.onnx       # 固定 batch=1、opset 17 的 ONNX（文件名可由 manifest 指定）
```

`manifest.json` 最小字段（完整说明见 [server/README.md](server/README.md#模型包契约)）：

```json
{
  "schema_version": "1.0",
  "model_version": "0.1.0",
  "model_file": "model.onnx",
  "model_sha256": "<model.onnx 的 64 位十六进制 SHA256>",
  "mutation_genes": ["EGFR", "TP53", "..."],
  "expression_genes": ["EGFR", "MYC", "..."],
  "drug_index": {"Osimertinib": 0},
  "inputs": {
    "mutation": "mutation", "mutation_mask": "mutation_mask",
    "expression": "expression", "expression_mask": "expression_mask",
    "drug": "drug_id"
  },
  "outputs": {"response": "response", "ci_low": "ci_low", "ci_high": "ci_high"},
  "ci_level": 0.9
}
```

关键约定（来自计划书 §9.2）：

- **batch=1、固定 shape**；基因顺序由部署包 gene list 对齐；
- **缺失基因用显式 mask**（`mutation_mask/expression_mask`），**不能用 0 冒充缺失**
  （0 只表示「未突变」）；
- VAE 采样层在导出时确定性化（取 μ）；解码器「解释模式」单独子图，一期服务暂不加载；
- `response` 必须是 **0–1 的 `kite_response_score_v1`**（导出侧负责定标/量化），
  越高越可能敏感；服务会做越界钳制并给出 warning；
- 量化保护层（解码器/回归头 FP32）与 INT8/FP32 选择（Pi5 INT8 / Pi4 默认 FP32）
  由训练机侧导出流程负责，端侧只消费结果。

---

## 6. 启动步骤（两套：Linux/macOS 与 Windows PowerShell）

前置：安装 [uv](https://docs.astral.sh/uv/)（任意平台，无需管理员权限）；Python 3.12
由 uv 自动获取（`deploy/server/.python-version` 固定 3.12）。

### 6.1 Linux / macOS（bash）

```bash
# 1) 进入独立服务项目并创建隔离环境（不污染仓库根环境）
cd deploy/server
uv sync                       # 只有 fastapi/uvicorn/pydantic + dev 组；不含 torch

# 2) 启动（默认 127.0.0.1:8000；无模型时自动 demo 模式）
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000

# 3) 浏览器打开 http://127.0.0.1:8000  → 本机单页界面
# 4) 健康检查（另开终端）
curl -s http://127.0.0.1:8000/health
```

可选：加载真实 ONNX 模型（模型包放到默认目录或被 `KITE_MODEL_DIR` 指定）。

```bash
# 方式 A：默认目录 deploy/packages/current
# 方式 B：环境变量覆盖（相对路径以 deploy/server 为基准）
export KITE_MODEL_DIR=/opt/kite-drp/packages/0.1.0
uv sync --extra onnx          # 安装 onnxruntime（离线装置用 wheelhouse，见 deploy/edge/README.md）
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### 6.2 Windows PowerShell（医生站/工作站）

```powershell
# 前置：安装 uv（官方脚本，用户级安装，无需管理员权限）
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# 若执行策略受限，也可先：Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# 1) 进入独立服务项目并创建隔离环境
cd deploy\server
uv sync

# 2)（可选）配置环境变量/加载真实模型
$env:KITE_MODEL_DIR = "D:\kite-drp\packages\current"
$env:KITE_WORKERS   = "1"
$env:KITE_SERVER_PORT = "8000"
uv sync --extra onnx

# 3) 启动（默认只绑定 127.0.0.1，无需在防火墙开放端口）
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000

# 4) 浏览器打开 http://127.0.0.1:8000
# 5) 健康检查（另开 PowerShell）
Invoke-RestMethod http://127.0.0.1:8000/health | ConvertTo-Json -Depth 4
```

> Windows 提示：`.env` 请保存为 **UTF-8（无 BOM）**；日志/中文输出正常。
> 需要开机自启时优先用「任务计划程序」或 NSSM 包装 `uv.exe run uvicorn ...`（见
> `deploy/device/README.md` 的 Windows 备选小节）。

### 6.3 从仓库根目录的一条命令

```bash
cd deploy/server && uv sync && uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### 6.4 测试

```bash
cd deploy/server
uv sync
uv run pytest -q            # 32 个测试：健康/契约/任务终态/422/404/503/静态页无外链/demo/隐私
```

---

## 7. 环境隔离（为什么单独配 Python 环境）

用户要求「为部署服务单独配置 Python 环境」，本仓库的落实方式：

1. `deploy/server/pyproject.toml` 是**独立项目**（`name=kite-drp-server`），
   `deploy/server/.venv` 由 `uv sync` 创建、已被根 `.gitignore` 忽略；
2. `[tool.uv] package = false`：按源码目录分发，不构建 wheel；
3. **不加入仓库根项目的 uv workspace**：根 `pyproject.toml` 与 `deploy/server/pyproject.toml`
   互不引用，根环境不安装 fastapi/uvicorn（训练机不需要 Web 依赖，端侧不需要 torch）；
4. 依赖面刻意最小：基础依赖仅 `fastapi/pydantic/uvicorn`，ONNX Runtime 是**可选 extra**，
   Pi/医生站按需安装；测试依赖 `pytest/httpx` 只在 dev 组；
5. 验证隔离：

   ```bash
   cd deploy/server && uv run python -c "import fastapi, uvicorn; print('server env OK')"
   cd <repo-root> && uv run python -c "import fastapi"   # 期望 ImportError：根环境不含 fastapi
   ```

---

## 8. 部署产物与 Git LFS

- `deploy/packages/**` 由根 `.gitattributes` 规则走 **Git LFS**（ONNX/权重/部署包），
  目录当前只有 `.gitkeep` 占位；
- 正式发布包建议目录名带版本：`deploy/packages/0.1.0/`（ONNX + manifest + SHA256SUMS），
  端侧用 `current` 软链/目录指向当前版本（见 `deploy/device/README.md` 双目录原子替换）；
- 受控数据（TCGA 受控层、DrugBank 等）**禁止**随部署包提交（计划书 §3.8、附录 B.2-8）；
- `deploy/server/.venv/`、`deploy/server/var/`、`deploy/server/uploads/` 已被根 `.gitignore` 忽略。

---

## 9. 一期范围 vs 二期

**一期（本次交付，可运行）**

- 本机 Web 服务：`/health`、`/api/v1/schema`、`/api/v1/jobs`（创建/查询/列表）；
- asyncio 队列 + 任务状态机 + 单 worker 串行（可配多 worker）；
- demo 模式（无模型显式标记）+ ONNX 预测器（可选依赖，SHA256 校验通过才加载）；
- 纯本地前端单页（输入表单、轮询、结果、最近任务、健康条、合规声明）；
- 树莓派/设备侧**文档化**脚手架（离线安装、systemd、回滚、签名规划）。

**二期（需要另立任务/ADR）**

- HIS/LIS 对接（网关、认证、审计、脱敏，涉及院内网络边界与合规评审）；
- 批量 CSV 推理与结果导出；多模型/多癌种路由；模型热切换；
- 前瞻性小样本人工复核流程（§9.3 #12）与分布漂移监控；
- `deploy/server` 拆为独立 git submodule 仓库（触发条件见 ADR 0003）。

---

## 10. 子模块化决策摘要

当前选择「**`deploy/` 子目录 + 独立 uv 项目**」而不是立即拆 git submodule：一期代码量
（~3k 行）与团队规模不足以抵消 submodule 的协作/CI 成本；但接口（HTTP API + manifest +
目录边界）已按可迁移方式设计。**触发拆分子模块的条件、迁移成本与步骤**见
[`docs/adr/0003-deploy-server-and-submodule.md`](../docs/adr/0003-deploy-server-and-submodule.md)。

---

## 11. 文档索引

- 服务启动 / 环境隔离 / **API 契约完整定义（请求响应示例 + curl）** / 模型包契约 / 排障：
  [`deploy/server/README.md`](server/README.md)
- 树莓派边缘端：离线 wheelhouse、onnxruntime(aarch64)、systemd：
  [`deploy/edge/README.md`](edge/README.md)
- 设备侧：systemd/开机自启/双目录原子替换/回滚/签名校验：
  [`deploy/device/README.md`](device/README.md)
- 架构决策：`docs/adr/0003-deploy-server-and-submodule.md`
- 训练机侧导出/量化代码说明：`src/kite_drp/deploy/__init__.py`
- 总计划书：`plans&notes/端侧肿瘤药敏师生模型_详细计划书.md`（§9、附录 B）

---

## 12. 免责声明

本目录所有代码与文档均为**科研原型**：不用于临床决策、不构成医疗器械软件、
不宣称任何诊断/治疗用途。临床使用须完成注册审批、前瞻验证与伦理合规（计划书 §9.4）。

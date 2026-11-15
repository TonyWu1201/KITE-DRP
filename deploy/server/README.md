# kite-drp-server — KITE-DRP 本机 Web 推理服务（科研原型）

`deploy/server` 是 **独立 uv 项目**：FastAPI 后端 + 纯本地前端单页 + asyncio 任务队列 +
可选的 ONNX Runtime 预测器。它把一个端侧药敏学生模型包装成「医生站本机浏览器可用」的
离线服务，默认**只绑定 `127.0.0.1`**、运行期**零外网请求**。

> ⚠️ **科研原型，不用于临床决策。** 服务不宣称诊断用途；`response` 是 0–1 的科研未校准
> 分数，不是临床概率，也不输出「敏感/耐药」硬分类。进入临床决策需按 NMPA/CE-MDR
> 完成注册与前瞻验证（总计划书 §9.4、附录 B.2-9）。

## 目录

- [1. 环境隔离（为什么单独配 Python 环境）](#1-环境隔离为什么单独配-python-环境)
- [2. 快速开始](#2-快速开始)
- [3. 配置项](#3-配置项)
- [4. API 契约](#4-api-契约)
- [5. 任务队列语义](#5-任务队列语义)
- [6. 模型包契约（ONNX）](#6-模型包契约onnx)
- [7. demo 模式](#7-demo-模式)
- [8. 前端页面](#8-前端页面)
- [9. 离线与隐私红线](#9-离线与隐私红线)
- [10. 测试](#10-测试)
- [11. 排障 FAQ](#11-排障-faq)
- [12. 一期范围与二期](#12-一期范围与二期)

---

## 1. 环境隔离（为什么单独配 Python 环境）

本目录是**独立 uv 项目**，与仓库根项目互不共享环境：

| 项目 | 根项目 | `deploy/server` |
|---|---|---|
| `pyproject.toml` | 训练/数据/导出依赖（torch 等） | 仅 `fastapi/pydantic/uvicorn`（+ dev `pytest/httpx`） |
| Python | 3.12（根 `.python-version`） | 3.12（本目录 `.python-version`） |
| 虚拟环境 | 根 `.venv` | **本目录 `.venv`**（根 `.gitignore` 已忽略） |
| 打包方式 | hatchling wheel | `[tool.uv] package = false`（源码目录直接运行） |
| uv workspace | 不属于任何 workspace | **不加入根项目 workspace** |

好处：医生站/树莓派只装 Web 服务所需的小依赖（不装 torch，省 ~1 GB+）；训练机不需要
fastapi。验证隔离：

```bash
cd deploy/server
uv run python -c "import fastapi, uvicorn; print('server env OK')"

cd <repo-root>
uv run python -c "import fastapi"     # 期望 ModuleNotFoundError：根环境不含 fastapi
```

> 本目录的 `uv.lock` 是独立锁文件，与根 `uv.lock` 无关；两者都在各自目录执行
> `uv sync` 即可，互不覆盖。

---

## 2. 快速开始

前置：安装 [uv](https://docs.astral.sh/uv/)（用户级安装即可，无需管理员权限）。

### 2.1 Linux / macOS

```bash
cd deploy/server
uv sync                                              # 建独立 .venv（首次会取 Python 3.12）
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
# 浏览器打开 http://127.0.0.1:8000
```

另开终端验证：

```bash
curl -s http://127.0.0.1:8000/health
```

### 2.2 Windows PowerShell

```powershell
# 安装 uv（若尚未安装；用户级，无需管理员）
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

cd deploy\server
uv sync
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
# 浏览器打开 http://127.0.0.1:8000

# 另开 PowerShell 验证
Invoke-RestMethod http://127.0.0.1:8000/health | ConvertTo-Json -Depth 4
```

> `.env` 请保存为 UTF-8（无 BOM）。安装失败时检查 PowerShell 执行策略：
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`（可选，用户级）。

### 2.3 从仓库根目录一条命令

```bash
cd deploy/server && uv sync && uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### 2.4 加载真实 ONNX 模型（可选）

```bash
# 部署包目录内含 manifest.json + model.onnx（契约见 §6）
export KITE_MODEL_DIR=/opt/kite-drp/packages/0.1.0      # Windows: $env:KITE_MODEL_DIR = "..."
uv sync --extra onnx                                    # 安装 onnxruntime（CPU EP）
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
curl -s http://127.0.0.1:8000/health                     # status=ok, demo=false
```

不配置模型、目录不存在、校验失败或未装 onnxruntime 时，服务**自动进入 demo 模式**并在
`/health` 与每个结果中显式标记 `demo=true`（见 §7）。默认模型目录：
`deploy/packages/current`。

### 2.5 无网络/沙箱环境的 uv 缓存

若环境不允许写 `~/.cache`（CI 沙箱等），把缓存重定向到可写目录：

```bash
export UV_CACHE_DIR="$PWD/../../.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/../../.cache/uv-python"
export UV_LINK_MODE=copy
cd deploy/server && uv sync && uv run pytest -q
```

---

## 3. 配置项

配置来源优先级：**进程环境变量 > `deploy/server/.env`（仅 `KITE_` 前缀）> 内置默认值**。
启动时非法配置直接报错（fail fast）。模板见 `.env.example`。

| 变量 | 默认值 | 说明 |
|---|---|---|
| `KITE_SERVER_HOST` | `127.0.0.1` | 监听地址。**正式部署必须保持回环地址**；非回环会打警告（仅限受控测试） |
| `KITE_SERVER_PORT` | `8000` | 监听端口（1–65535）。仅 `python -m app.main`/`run()` 使用；uvicorn CLI 以命令行为准 |
| `KITE_MODEL_DIR` | `deploy/packages/current` | 部署包目录（含 `manifest.json`）。相对路径以 `deploy/server` 为基准；**空值 = 显式 demo** |
| `KITE_QUEUE_MAX` | `64` | 队列容量（1–10000）。满时 `POST /api/v1/jobs` 返回 503 |
| `KITE_WORKERS` | `1` | 推理 worker 数（0–8）。默认 1 = 串行；**0 仅供测试/维护**（只入队不执行） |
| `KITE_JOB_RETENTION` | `200` | 内存中保留任务数上限（1–10000）；超出淘汰最旧终态任务 |
| `KITE_LOG_LEVEL` | `INFO` | `DEBUG/INFO/WARNING/ERROR/CRITICAL` |
| `KITE_ENV_FILE` | `deploy/server/.env` | 可选：指定其他 `.env` 路径 |

> 日志只包含 job_id/状态/耗时/模型版本，**不包含输入特征、样本 ID、基因名、PHI**。

---

## 4. API 契约

### 4.1 基础约定

- Base URL：`http://127.0.0.1:8000`，所有路径以 `/` 开头；仅同源调用。
- 请求/响应均为 `application/json; charset=utf-8`；时间均为 **UTC ISO 8601**（`...Z`）。
- 机器可读契约：**`GET /api/v1/openapi.json`**（OpenAPI 3 JSON）。
  **`/docs`、`/redoc` 已刻意关闭**：其默认 UI 会从 CDN 拉取前端资源，违反「运行期零外网请求」
  （总计划书 §9.3 #10）。需要浏览契约时请保存该 JSON 到本机离线查看。
- 错误响应统一为：

  ```json
  {"detail": {"code": "job_not_found", "message": "中文说明（不含输入数据）"}}
  ```

  其中 422 校验失败遵循 pydantic 标准格式（`detail` 为数组）。
- 接口版本策略：`/api/v1` 内**向后兼容**（只允许新增可选字段/接口）；破坏性变更必须
  新增 `/api/v2` 并写 ADR；数据 schema 有独立 `schema_version`（当前 `1.0`）。

### 4.2 任务状态机

```text
POST /api/v1/jobs ──▶ queued ──▶ running ──▶ succeeded   （result 有值）
                                    │
                                    └────▶ failed       （error 有值，result=null）
```

任务只存内存，**服务重启后清空**（再查旧 job_id 返回 404，属预期）。

### 4.3 `GET /health`

健康检查 + 模型/队列状态（前端状态条、运维探活、demo 标记）。

```bash
curl -s http://127.0.0.1:8000/health
```

```json
{
  "status": "degraded",
  "service": "kite-drp-server",
  "version": "0.1.0",
  "schema_version": "1.0",
  "model_version": "demo-0.0.0",
  "model_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
  "demo": true,
  "engine": "demo",
  "demo_reason": "模型目录不存在：/home/.../deploy/packages/current",
  "model_dir": "/home/.../deploy/packages/current",
  "host": "127.0.0.1",
  "port": 8000,
  "queue": {
    "size": 0, "capacity": 64, "active_workers": 1,
    "running": 0, "succeeded": 0, "failed": 0
  },
  "uptime_s": 0.342,
  "time": "2026-11-15T06:31:14.024873Z",
  "notice": "科研原型，不用于临床决策",
  "offline": true
}
```

- `status`：`ok`（已加载真实 ONNX）/ `degraded`（demo 或模型校验失败）；
- `demo_reason`：demo/降级原因（排障入口）；
- `model_sha256`：demo 模式固定为 64 个 `0`，真实模型为实际 SHA256。

### 4.4 `GET /api/v1/schema`

返回输入/输出 JSON Schema、说明与**可直接 POST 的示例**（脚本/前端据此构造请求）。

```bash
curl -s http://127.0.0.1:8000/api/v1/schema
```

```json
{
  "schema_version": "1.0",
  "description": "单样本药敏响应预测输入/输出 schema（对齐计划书 §9.4）",
  "input_schema": {"type": "object", "properties": {"schema_version": {}, "sample_id": {}, "...": {}}},
  "output_schema": {"type": "object", "properties": {"response": {}, "ci": {}, "model_version": {}, "model_sha256": {}, "...": {}}},
  "example": {
    "schema_version": "1.0",
    "sample_id": "DEMO-0001",
    "assay": "DNA panel + RNA-seq",
    "mutation": {"EGFR": 1, "TP53": 1, "KRAS": 0},
    "expression": {"EGFR": 5.2, "TP53": 4.1, "MYC": 6.0},
    "drug": "Osimertinib",
    "meta": {"cancer_type": "LUAD", "stage": "IV"}
  },
  "notes": ["schema_version 当前固定为 1.0；...", "sample_id 必须是伪名化 ID；..."]
}
```

输入字段语义（对齐总计划书 §9.4）：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `schema_version` | string | 否（默认 `1.0`） | 只接受 `"1.0"`，其他值 422 |
| `sample_id` | string | 是 | 伪名化 ID，`^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$`；禁止 PHI |
| `assay` | string/null | 否 | 检测平台/方法，≤128 字符 |
| `mutation` | object[str, 0/1] | 二选一* | 基因→0/1；缺失基因省略，**不要用 0 冒充缺失** |
| `expression` | object[str, number] | 二选一* | 基因→表达值（log2(TPM+1) 或通路分数，需与训练一致） |
| `drug` | string | 是 | 药物名，≤128 字符（真实模型需在部署包药表内） |
| `meta` | object[str, scalar] | 否 | 低敏临床变量（癌种/分期等），键名同 `sample_id` 字符集 |

\* `mutation` 与 `expression` **至少提供一个**；仅突变 panel 的降级输入是合法形态（§9.5）。
未知字段一律 422（`extra="forbid"`，契约严格，避免拼写错误静默丢弃）。

### 4.5 `POST /api/v1/jobs`（202 Accepted）

提交单样本任务并入队，立即返回 job_id（不等待推理完成）。

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/jobs \
  -H 'Content-Type: application/json' \
  -d '{
    "schema_version": "1.0",
    "sample_id": "DEMO-CURL-1",
    "assay": "DNA panel + RNA-seq",
    "mutation": {"EGFR": 1, "TP53": 1},
    "expression": {"EGFR": 5.2, "MYC": 6.0},
    "drug": "Osimertinib"
  }'
```

响应 `202`：

```json
{
  "job_id": "23a576c644d14d06980f8ccee9c7fa3f",
  "status": "queued",
  "created_at": "2026-11-15T06:31:14.032945Z",
  "poll_url": "/api/v1/jobs/23a576c644d14d06980f8ccee9c7fa3f",
  "demo": true,
  "notice": "科研原型，不用于临床决策"
}
```

错误：

- `422`：字段缺失/类型错误/`mutation[A]` 非 0/1/同时缺 mutation 与 expression/未知字段等；
- `503`：队列已满（`detail.code = "queue_full"`），稍后重试。

### 4.6 `GET /api/v1/jobs/{job_id}`

轮询任务状态；成功时带 `result`，失败时带 `error`。

```bash
JOB_ID=23a576c644d14d06980f8ccee9c7fa3f
curl -s "http://127.0.0.1:8000/api/v1/jobs/$JOB_ID"
```

```json
{
  "schema_version": "1.0",
  "job_id": "23a576c644d14d06980f8ccee9c7fa3f",
  "status": "succeeded",
  "created_at": "2026-11-15T06:31:14.032945Z",
  "updated_at": "2026-11-15T06:31:14.034100Z",
  "started_at": "2026-11-15T06:31:14.033165Z",
  "finished_at": "2026-11-15T06:31:14.034100Z",
  "input": {
    "schema_version": "1.0",
    "sample_id": "DEMO-CURL-1",
    "assay": null,
    "mutation": {"EGFR": 1, "TP53": 1},
    "expression": {"EGFR": 5.2, "MYC": 6.0},
    "drug": "Osimertinib",
    "meta": {}
  },
  "result": {
    "schema_version": "1.0",
    "sample_id": "DEMO-CURL-1",
    "drug": "Osimertinib",
    "response": 0.4036,
    "response_scale": "kite_response_score_v1",
    "ci": {"level": 0.9, "low": 0.3215, "high": 0.4857},
    "model_version": "demo-0.0.0",
    "model_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
    "demo": true,
    "engine": "demo",
    "warnings": [
      "科研原型，不用于临床决策",
      "demo 模式：未加载真实 ONNX 模型，response/ci 为确定性伪分数，禁止用于科研结论或任何临床判断。"
    ],
    "explanation": {
      "method": "demo-hash（伪归因，仅用于界面联通性验证，不代表真实生物学原因）",
      "top_features": [{"gene": "MYC", "contribution": 0.988}]
    },
    "elapsed_ms": 0.77
  },
  "error": null
}
```

失败示例（`status="failed"`，`result=null`）：`"error": "RuntimeError: ..."`（错误摘要，**不含输入数据**）。

- 未知 `job_id` → `404`，`detail.code = "job_not_found"`（服务重启后旧 ID 必然 404，属预期）。

### 4.7 `GET /api/v1/jobs?limit=N`

最近任务列表（创建时间倒序，仅内存）。

```bash
curl -s "http://127.0.0.1:8000/api/v1/jobs?limit=5"
```

```json
{"count": 1, "total": 1, "items": [{"job_id": "...", "status": "succeeded", "...": "..."}]}
```

`limit` 范围 1–100（默认 10）；越界 → `422`。

### 4.8 错误码速查

| HTTP | 场景 | `detail.code` |
|---|---|---|
| 200 | 查询/健康/schema 正常 | — |
| 202 | 任务已接受入队 | — |
| 404 | job_id 不存在（含服务重启清空） | `job_not_found` |
| 422 | 请求体校验失败 / limit 越界 | pydantic 标准数组 |
| 503 | 队列已满 | `queue_full` |
| 503 | 服务尚未完成 lifespan 初始化（极短窗口） | `not_ready` |

### 4.9 端到端 curl 演示（可直接复制执行）

```bash
BASE=http://127.0.0.1:8000

# 1) 健康 & 契约
curl -s "$BASE/health"
curl -s "$BASE/api/v1/schema" | head -c 400

# 2) 提交任务（从 /api/v1/schema 的 example 复制，或自定义）
JOB=$(curl -s -X POST "$BASE/api/v1/jobs" -H 'Content-Type: application/json' \
  -d '{"sample_id":"DEMO-0001","drug":"Osimertinib","mutation":{"EGFR":1},"expression":{"EGFR":5.2}}')
echo "$JOB"
JOB_ID=$(printf '%s' "$JOB" | python3 -c 'import json,sys; print(json.load(sys.stdin)["job_id"])')

# 3) 轮询到终态（demo 模式毫秒级；真实 ONNX 也在毫秒–十毫秒量级）
for i in 1 2 3 4 5; do
  STATE=$(curl -s "$BASE/api/v1/jobs/$JOB_ID")
  STATUS=$(printf '%s' "$STATE" | python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])')
  echo "poll $i: $STATUS"
  [ "$STATUS" = "succeeded" ] && { printf '%s\n' "$STATE"; break; }
  [ "$STATUS" = "failed" ] && { printf '%s\n' "$STATE"; break; }
  sleep 0.2
done

# 4) 最近任务 / 错误码
curl -s "$BASE/api/v1/jobs?limit=5"
curl -s -o /dev/null -w 'unknown job -> %{http_code}\n' "$BASE/api/v1/jobs/nope"
curl -s -o /dev/null -w 'invalid input -> %{http_code}\n' -X POST "$BASE/api/v1/jobs" \
  -H 'Content-Type: application/json' -d '{"sample_id":"X","drug":"D","mutation":{"A":2}}'
```

Windows PowerShell 等价：

```powershell
$base = "http://127.0.0.1:8000"
Invoke-RestMethod "$base/health" | ConvertTo-Json -Depth 4

$body = '{"sample_id":"DEMO-0001","drug":"Osimertinib","mutation":{"EGFR":1}}'
$ack  = Invoke-RestMethod -Method Post -Uri "$base/api/v1/jobs" `
          -ContentType 'application/json; charset=utf-8' -Body $body
$ack.job_id

Invoke-RestMethod "$base/api/v1/jobs/$($ack.job_id)" | ConvertTo-Json -Depth 6
Invoke-RestMethod "$base/api/v1/jobs?limit=5" | ConvertTo-Json -Depth 4
```

---

## 5. 任务队列语义

- `asyncio.Queue` + 固定 worker 池（`KITE_WORKERS`，默认 1 = 单 worker 串行，便于端侧限流/复现）；
- 容量 `KITE_QUEUE_MAX`；`size` 统计排队中任务（不含执行中的）；满 → 503；
- 预测经 `asyncio.to_thread` 执行，阻塞推理不会卡住 `/health`；
- 每次状态变更刷新 `updated_at`；终态后不可再变更；
- 任务表保留最近 `KITE_JOB_RETENTION` 条（淘汰最旧终态任务，不淘汰排队/执行中）；
- **不落盘**：重启清空；不写日志输入数据；不提供任务删除接口（内存数据，重启即清）。

---

## 6. 模型包契约（ONNX）

训练机侧用 `src/kite_drp/deploy/package.py` 生成部署包目录（计划书 §9.2）：

```text
<deploy-package>/
├── manifest.json
└── model.onnx          # 固定 batch=1、opset 17、CPU EP 可运行；文件名可由 manifest 指定
```

### 6.1 `manifest.json` 字段

| 字段 | 必填 | 说明 |
|---|---|---|
| `schema_version` | 否 | 部署包 schema 版本，默认 `1.0` |
| `model_version` | **是** | 语义化版本，出现在 `/health` 与结果里 |
| `model_file` | 否 | ONNX 文件名，默认 `model.onnx` |
| `model_sha256` | **是** | `model.onnx` 的 64 位小写十六进制 SHA256；**缺失/不匹配拒绝加载** |
| `mutation_genes` | 二选一* | 突变基因顺序（与 ONNX 输入第 1 维对齐） |
| `expression_genes` | 二选一* | 表达基因顺序（与 ONNX 输入第 1 维对齐） |
| `drug_index` | 否 | 药物名→整数索引；提供时未知药物 → 任务 failed（明确报错） |
| `inputs` | 否 | 逻辑名→ONNX 输入名映射（默认见 §6.2） |
| `outputs` | **是** | 必须含 `response`；可选 `ci_low`/`ci_high` |
| `ci_level` | 否 | 置信水平（默认 0.9，仅用于展示） |
| `ci_half_width` | 否 | 模型无 CI 输出时的半宽回退（默认 0.05） |

\* `mutation_genes` 与 `expression_genes` **至少一个**。

### 6.2 张量契约（batch=1）

| 逻辑输入（可在 `inputs` 改名） | dtype | shape | 必填 | 语义 |
|---|---|---|---|---|
| `mutation` | int64 | `[1, G_m]` | 提供突变基因时 | 0=未突变，1=突变（缺失基因填 0） |
| `mutation_mask` | int64 | `[1, G_m]` | 同上 | 1=该样本显式提供该基因，0=缺失（**显式 mask，不要用 0 冒充缺失**） |
| `expression` | float32 | `[1, G_e]` | 提供表达基因时 | 表达值（log2(TPM+1)/通路分数，与训练一致；缺失填 0） |
| `expression_mask` | int64 | `[1, G_e]` | 同上 | 1=提供，0=缺失 |
| `drug_id` | int64 | `[1]` | 提供 `drug_index` 时 | 药物索引 |

| 逻辑输出（可在 `outputs` 改名） | 必填 | 说明 |
|---|---|---|
| `response` | **是** | 标量或 `[1,1]`；`kite_response_score_v1`：0≈更可能耐药、1≈更可能敏感；服务对越界值钳制并给出 warning |
| `ci_low` / `ci_high` | 否 | 区间下/上界；缺失时用 `response ± ci_half_width` |

### 6.3 示例 `manifest.json`

```json
{
  "schema_version": "1.0",
  "model_version": "0.1.0",
  "model_file": "model.onnx",
  "model_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "mutation_genes": ["EGFR", "TP53", "KRAS"],
  "expression_genes": ["EGFR", "MYC"],
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

### 6.4 加载与校验顺序（失败即回退 demo，绝不静默降级）

1. `manifest.json` 存在且可解析、含 `model_version`/`outputs.response`；
2. `model.onnx` 存在；
3. `sha256(model.onnx) == manifest.model_sha256`（**合规硬要求**，§9.3 #10）；
4. `import onnxruntime`（可选依赖 `uv sync --extra onnx`）；
5. `InferenceSession(..., providers=["CPUExecutionProvider"])` 创建成功。

任一步失败：**拒绝加载真实模型 → demo 模式**，原因写入 `/health.demo_reason`，
前端顶部红色横幅展示。**不要**在 demo 模式下对外声称是模型结果。

> 部署包内**不要放**受控数据（重测序原始数据、受控 TCGA、受限注释库）；
> 只放 ONNX + manifest + gene list + SHA256（§3.8、附录 B.2-8）。部署包由
> 根 `.gitattributes` 的 `deploy/packages/**` 规则走 Git LFS。

---

## 7. demo 模式

触发条件（任一）：

- 未配置 `KITE_MODEL_DIR`（默认目录不存在也算）；
- 目录缺 `manifest.json` / `model.onnx`；
- `model_sha256` 缺失或不匹配；
- 未安装 `onnxruntime`；
- ONNX Session 创建失败（算子/版本不兼容等）。

行为与标记：

- `/health`：`status="degraded"`、`demo=true`、`engine="demo"`、`model_version="demo-0.0.0"`、
  `model_sha256` 为 64 个 `0`、`demo_reason` 给出具体原因；
- 每个结果：`demo=true`，且 `warnings` 同时包含「科研原型，不用于临床决策」与 demo 警示；
- 前端：顶部红色 DEMO 横幅 + 结果区红色「DEMO 输出·禁止用于任何判断」标签；
- demo 分数是**确定性哈希伪分数**（同一输入恒定输出，用于联调/演示/培训），
  **不是模型推理结果**，explanation 为伪归因占位。

---

## 8. 前端页面

`http://127.0.0.1:8000/` 单页中文界面（`app/static/index.html` + `style.css` + `app.js`）：

- 顶部：「科研原型，不用于临床决策」固定横幅 + 服务状态条（健康/demo/队列/模型版本）；
- 输入表单：`sample_id`、`drug`、`assay`、`meta`（可选），突变/表达支持
  **JSON 对象**或**粘贴文本**（`EGFR:1,KRAS:0`、`EGFR,TP53`、`EGFR:5.2,MYC:6.0`）；
- 提交后轮询任务（800 ms 间隔，60 s 超时），展示 `response`、`ci`、`model_version`、
  `model_sha256`、`engine`、耗时、warnings、解释占位；
- 「最近任务」表格（点击回填结果）、「填入示例」按钮；
- 全部资源本地：**无 CDN、无外部字体、无外链**；页面带 CSP
  `default-src 'self'`，浏览器层面兜底禁止外联；所有用户输入用 `textContent` 渲染（防 XSS）。

---

## 9. 离线与隐私红线

- 运行期 **0 次外网请求**：前端仅同源相对 fetch；后端无任何出网 SDK/调用；
  `/docs`、`/redoc` 关闭（默认 UI 依赖 CDN）；仅保留本地 `/api/v1/openapi.json`；
- 只监听 `127.0.0.1`；无认证/内网穿透/上报功能（不要给公网或医院网卡开放）；
- 日志不含 PHI、样本 ID、基因名、表达值（有测试守护）；
- 输入伪名化；页面明示禁止填入姓名/住院号等 PHI；
- 任务/结果只在内存，重启清空；不提供导出接口（一期）；
- 模型 SHA256 校验通过才加载；部署包与模型卡随版本管理（§9.3 #10/#11）；
- 不输出「敏感/耐药」硬分类与临床阈值（附录 C），只给分数 + 区间 + 免责声明。

---

## 10. 测试

```bash
cd deploy/server
uv sync
uv run pytest -q
```

当前真实结果（Python 3.12.14 / uv 0.12.20）：**32 passed**。

覆盖范围：

| 编号 | 测试点 |
|---|---|
| 1 | `/health` 状态、模型版本/SHA256、demo 标记、队列统计、loopback/offline 声明 |
| 2 | `/api/v1/schema` 契约字段、示例可被 pydantic 校验、notes 合规条目 |
| 3 | 创建任务 → 轮询到 `succeeded` → result 字段（response/ci/model_version/model_sha256） |
| 4 | 仅突变输入（§9.5 降级形态）可推理 |
| 5 | 最近任务列表、`limit` 越界 422、最新在前 |
| 6 | demo 分数确定性（输入字典顺序无关） |
| 7 | 9 类非法输入 → 422 |
| 8 | 未知 job → 404（`job_not_found`） |
| 9 | 队列满 → 503（`queue_full`） |
| 10 | 预测器抛异常 → `failed` 状态机 + error 摘要 |
| 11 | 日志不含样本 ID/原始表达值/基因名 |
| 12 | 首页/样式/JS 200、`static/` 仅 3 个本地文件、**无 http(s):// / 无 @import / 无协议相对外链**、CSP |
| 13 | `/docs`、`/redoc`、`/openapi.json` 404；本地 `/api/v1/openapi.json` 200 |
| 14 | 预测器回退（未配置/目录不存在/SHA256 不匹配 → demo + 原因） |
| 15 | 配置默认回环、环境变量覆盖、.env 优先级/相对路径基准、非法配置 fail fast |

冒烟（可选，真实起服务）：

```bash
cd deploy/server
uv run uvicorn app.main:app --host 127.0.0.1 --port 8877 &
sleep 2
curl -s -o /dev/null -w 'GET / -> %{http_code}\n' http://127.0.0.1:8877/
curl -s http://127.0.0.1:8877/health
kill %1
```

---

## 11. 排障 FAQ

| 现象 | 排查 |
|---|---|
| `Address already in use` | 换 `--port 8001`，或 `ss -ltnp | grep 8000` / `Get-NetTCPConnection -LocalPort 8000` 找占用进程 |
| `/health` 一直 `degraded` | 看 `demo_reason`：目录不存在→检查 `KITE_MODEL_DIR`；SHA256 失败→重新打包；onnxruntime 缺失→`uv sync --extra onnx` |
| 提交任务 503 | 队列满：调大 `KITE_QUEUE_MAX` 或等 worker 消费（`/health.queue`） |
| 任务一直 `queued` | `KITE_WORKERS` 是否为 0（仅测试模式） |
| 旧 job_id 404 | 服务重启会清空内存队列，属预期；请重新提交 |
| 根环境 `import fastapi` 成功 | 说明误把 fastapi 装进根环境；根项目不应有该依赖，请只在本目录 `uv sync` |
| Windows 中文乱码 | `.env` 保存为 UTF-8（无 BOM）；控制台可用 `chcp 65001` |
| uv 缓存不可写（CI/沙箱） | 见 §2.5 的 `UV_CACHE_DIR`/`UV_PYTHON_INSTALL_DIR`/`UV_LINK_MODE` |
| 想在浏览器看 OpenAPI | 下载 `/api/v1/openapi.json` 到本机用离线工具查看（不要重新打开 Swagger CDN UI） |

日志：默认输出到 stdout/stderr（uvicorn/journald 收集）；级别用 `KITE_LOG_LEVEL`。
排查完成后建议恢复 `INFO`，避免 DEBUG 输出过多运行细节。

---

## 12. 一期范围与二期

**一期（已实现）**：本机 Web 服务 + 单页前端 + 异步队列/状态机 + demo/ONNX 预测器 +
schema 契约 + 测试 + 文档。

**二期（另立任务）**：HIS/LIS 网关对接（认证/审计/脱敏）、批量 CSV 推理与导出、
模型热切换/多模型路由、内存外任务持久化（SQLite）、`deploy/server` 独立仓库（见 ADR 0003）。

---

## 安全声明

本服务是**科研原型**：不用于临床决策，不构成医疗器械软件，不宣称诊断/治疗用途。
`response` 与 `ci` 是科研分数，未经临床校准；demo 模式输出为伪数据。临床使用须完成
NMPA/CE-MDR 注册路径、前瞻验证、伦理与《个人信息保护法》《人类遗传资源管理条例》合规
（总计划书 §9.4）。

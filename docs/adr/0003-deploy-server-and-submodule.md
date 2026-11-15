# ADR 0003：本机 Web 推理服务的部署形态与子模块化决策

- **日期**：2026-11-15
- **状态**：Accepted（已采纳）
- **决策者**：KITE-DRP 项目组（部署/工程方向）
- **相关**：总计划书 §9.2–§9.4、§9.3 #10/#11、附录 B；ADR 0002（Python 环境）；`deploy/README.md`、`deploy/server/README.md`

---

## 1. 背景（Context）

用户要求为 KITE-DRP 提供「实际端侧部署」配套：**一个可分发的本机 Web 服务器**
（前端展示页面 + 接收数据/查询的 HTTP API + 后端排队），并**单独配置 Python 环境**；
同时明确要求考虑「工程量增长时迁移为 git submodule」的决策。

端侧部署需求（总计划书 §9）：

- 形态：CLI（科研/验收）→ **本机 Web UI/FastAPI（医生站浏览器，仅 127.0.0.1，一期）** →
  平板/触屏终端；院内 HIS/LIS 对接为二期；
- 推理栈：训练机 PyTorch → ONNX（opset 17、batch=1）→ **装置侧仅 onnxruntime CPU EP**，
  装置不装 PyTorch；部署包 ≤50 MB；
- 合规红线：**运行期 0 次外网请求**、装置全离线、数据不出院、日志不含 PHI/原始基因值、
  **模型 SHA256 校验通过才加载**；一期定位「科研原型，不用于临床决策」；
- 环境约束：部署侧依赖（fastapi/uvicorn/pydantic/onnxruntime）与训练侧依赖
  （torch/rdkit/hydra/CUDA 等）**体积与平台都不同**，不应共享同一 Python 环境。

当前仓库布局：训练机侧代码在 `src/kite_drp/`（根 uv 项目，hatchling 打包）；部署资产放在
仓库根 `deploy/`。本 ADR 决定 `deploy/server` 的组织边界，以及何时/如何拆成 git submodule。

已观察到的约束与数据点（本 ADR 落笔时）：

- `deploy/server` 当前约 3k 行（含测试/前端），依赖面小（4 个运行时包 + 2 个 dev 包）；
- 服务端与训练侧只通过「部署包目录 + manifest 契约」和「HTTP API 契约」交互，
  **`deploy/server` 不 import `kite_drp`**；
- 根项目 CI（lint/test/lock check）与端侧服务 CI（独立环境 pytest）目标不同；
- 一期团队规模小（学生 + 导师），发版频率预计 ≤1 次/月；二期可能引入独立部署负责人
  与院内交付/运维需求。

---

## 2. 决策驱动因素（Decision Drivers）

1. **可复现与隔离**：部署服务必须能在他人的 x86 医生站/树莓派上独立安装、不污染训练环境；
2. **离线合规**：依赖与产物需可整体审计/打包（wheelhouse、部署包 SHA256/签名）；
3. **协作成本**：一期小团队、单仓库协作效率优先；避免 submodule 带来的指针/权限/CI 复杂度；
4. **演进空间**：接口边界必须稳定，未来拆仓时端侧代码/部署流程可平滑迁移；
5. **权限与发版解耦**（未来）：部署团队可能独立发版、独立 CI 与镜像，仅需读训练侧产物。

---

## 3. 备选方案（Options）

### 方案 A：并入主包（`src/kite_drp/server/`，与训练代码同一 uv 项目）

- 优点：单项目单环境、零额外配置、导入/类型/CI 统一；小团队心智负担最低。
- 缺点：**违反「单独配置 Python 环境」要求**；端侧依赖（fastapi/uvicorn/onnxruntime）
  被塞进训练环境（或反向要求训练机装 Web 依赖）；部署机安装训练依赖（torch ~GB 级）
  不可接受；根 `uv.lock` 被 Web/ONNX 依赖污染；端侧离线分发无法做到最小依赖面；
  安全/合规审计边界模糊（训练代码与临床端服务混在一起）。

### 方案 B：`deploy/` 子目录 + **独立 uv 项目**（当前选择）

```text
deploy/
├── server/            # 独立 pyproject.toml + uv.lock + .venv（[tool.uv] package=false）
├── edge/  device/     # 端侧脚手架与运维文档
└── packages/          # 部署产物（Git LFS）
```

- 优点：
  - **独立 Python 环境**：`cd deploy/server && uv sync` 只装 fastapi/pydantic/uvicorn
    （+ 可选 onnxruntime），端侧无需 torch；根环境不装 Web 依赖；
  - 独立 `uv.lock` 与 `.python-version`，依赖可离线 wheelhouse 化，满足「运行期 0 外网」；
  - 源码目录直接分发（`package=false`），树莓派/医生站拷贝即可运行，无需构建/发 PyPI；
  - 与主包**零 import 耦合**：只依赖部署包目录（manifest+SHA256）与 HTTP API 契约；
  - 单仓库协作简单：一次 PR 同时改训练侧导出与部署侧消费，评审/回溯方便；
  - 未来可整体平移为 git submodule / 独立仓库（见 §4），迁移成本已被边界设计压低。
- 缺点：
  - 需要两套 `uv sync`（根 + `deploy/server`），CI 需分别检查（已通过文档与独立测试覆盖）；
  - 根 `.gitignore`/`.gitattributes` 需显式为 `deploy/` 留规则（已落实：`.venv`/`var` 忽略、
    `deploy/packages/**` 走 LFS）；
  - 仓库体积随时间增长（部署包/ wheelhouse 走 LFS，前台代码量仍可控）。

### 方案 C：拆为 git submodule（独立仓库 `kite-drp-server`）

- 优点：部署团队独立发版/独立 CI/独立权限；主仓库不背部署包与 wheelhouse 体积；
  端侧镜像可只 checkout 服务仓库；安全审计边界清晰（临床端代码单独评审）。
- 缺点：**一期成本过高**——submodule 指针更新、权限/分支策略、双仓库 PR 协调、
  新人配置（`git submodule update --init`）、CI 跨仓取产物/版本对齐、离线镜像构建；
  在代码量 ~3k 行、发版 ≤1 次/月的情况下属于过度工程（YAGNI）；且当前团队共用同一套
  仓库权限，拆仓没有实际权限收益。

---

## 4. 决策（Decision）

**采纳方案 B**：在仓库根 `deploy/` 下以**独立 uv 项目**的形式承载本机 Web 推理服务
（`deploy/server`：`pyproject.toml` + `uv.lock` + `.python-version` + 独立 `.venv`，
`[tool.uv] package = false`，**不加入根项目 uv workspace**），并以目录 + 契约隔离
训练侧与部署侧：

1. **源码边界**：`deploy/server` 不 import `kite_drp`；训练机侧导出/量化仍在
   `src/kite_drp/deploy/`；
2. **产物边界**：只通过 `deploy/packages/<version>/`（`manifest.json` + `model.onnx` +
   SHA256）交互，部署包由根 `.gitattributes` 走 Git LFS；
3. **接口边界（稳定契约，拆仓后保持不变）**：
   - HTTP API：`GET /health`、`GET /api/v1/schema`、`POST /api/v1/jobs`（202 + job_id）、
     `GET /api/v1/jobs/{job_id}`、`GET /api/v1/jobs?limit=N`；错误码 422/404/503；
     `/api/v1` 内向后兼容，破坏性变更走 `/api/v2` + 新 ADR；
   - 机器可读契约：`/api/v1/openapi.json`（本地 JSON；Swagger/ReDoc UI 因 CDN 依赖被刻意关闭）；
   - 数据契约：输入 `schema_version/sample_id/assay/mutation/expression/drug/meta`，
     输出 `response/ci/model_version/model_sha256`（+ demo/engine/warnings 可审计字段）；
   - 部署包契约：`manifest.json` 字段与 ONNX 张量（batch=1、基因 mask、输出 `[0,1]` 分数）
     见 `deploy/server/README.md` §6；
4. **触发拆分子模块的条件**（任一满足即重新评估，建议按季度检查）：
   - **代码量**：`deploy/server` 超过约 **2 万行**（或团队并行提交频率导致主仓库 CI 显著变慢）；
   - **独立发版**：部署侧需要**独立于训练代码的发版节奏**（如 ≥1 次/周，或需要独立语义化版本
     + 发布说明），且主仓库 release 与之解耦；
   - **独立团队/权限**：出现专职部署/运维负责人，或需要**独立读写权限**（部署代码仓库对
     外部实施/合作方开放、训练代码不开放）；
   - **镜像/分发体积**：需要为端侧单独构建最小镜像/离线包，且部署产物（wheelhouse/部署包）
     已不适合长期放在训练主仓库（即使有 LFS）；
   - **CI 资源**：端侧测试（多 Python/多平台/装置仿真）在根仓库 CI 中成为瓶颈或需要自托管
     runner；
   - **依赖冲突**：部署侧依赖与训练侧出现不可调和的版本/平台冲突（如 onnxruntime 与
     torch/onnx 的版本矩阵冲突）。
5. **迁移路径与成本（预估）**：
   - 建独立仓库 `kite-drp-server`（保留 `deploy/server` 目录结构、提交历史可用
     `git subtree split -P deploy/server` 保留）；
   - 本仓库改为 `git submodule add <server-repo> deploy/server`（或 `deploy/` 整体子模块）；
   - CI：根仓库只保留契约测试（对部署产物做 smoke），部署侧仓库独立 lint/test/lock/发布；
   - 版本对齐：主仓库记录 submodule commit + 部署包 `model_version`/`manifest.sha256`；
   - 预估工程量：**2–4 人日**（仓库拆分 + CI + 文档 + 一次完整离线部署演练），
     远低于一期引入 submodule 的持续协作成本；
   - 触发后应新写 ADR（000x）记录迁移决策与验收。

---

## 5. 后果（Consequences）

**正面**

- 满足「单独配置 Python 环境」：部署环境最小化、可离线 wheelhouse 化、训练环境不被污染；
- 端侧合规可控：运行期零外网、模型 SHA256 校验、日志无 PHI、仅 127.0.0.1；
- 单仓库协作、一次 PR 打通「导出 → 部署 → 测试」；迁移接口已稳定，拆仓风险低；
- 部署包与产物通过 LFS/托管目录管理，不进入代码评审主路径。

**负面/风险与缓解**

- 双环境/双锁文件：文档明确两条 `uv sync` 路径；CI 分别检查（根项目 CI 排除 `deploy/`，
  部署侧自带 `Pytest` 与依赖锁）；
- 根仓库体积增长：部署包/wheelhouse 走 Git LFS，且只保留必要版本；
- 根 `.gitignore`/`.gitattributes` 对 `deploy/` 的规则需要维护：已加注释与相应规则
  （`.venv`/`var`/`uploads` 忽略，`deploy/packages/**` LFS）；
- 未来拆仓时的版本对齐风险：以 `/api/v1` + `manifest` 契约为唯一接口，并在拆仓 ADR 中
  固化验收清单。

**未决策/遗留**

- 批量 CSV 推理、HIS/LIS 对接、任务持久化（SQLite）为二期内容，届时可能进一步影响
  部署形态（是否需要独立网关服务），届时按同样流程评估是否触发拆仓条件。

---

## 6. 相关文档

- `deploy/README.md`：部署总览、数据流、离线红线、Linux/Windows 启动；
- `deploy/server/README.md`：独立环境、API 契约（示例 + curl）、模型包契约、demo 模式、测试；
- `deploy/edge/README.md`、`deploy/device/README.md`：树莓派与设备侧运维；
- `docs/adr/0002-python-environment-cuda126.md`：根项目 Python/CUDA 环境决策；
- `src/kite_drp/deploy/__init__.py`：训练机侧导出/量化/打包模块职责；
- 总计划书 §9（端侧部署）、附录 B（仓库结构与工程规范）。

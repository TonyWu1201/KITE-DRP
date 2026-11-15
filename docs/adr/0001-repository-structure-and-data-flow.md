# ADR-0001：仓库结构与数据分层重构

- 状态：已接受
- 日期：2026-11-15
- 相关：计划书 §3.8、附录 B.1/B.2；`plans&notes/2026-11-15_仓库结构与工程配置说明.md`

## 背景

仓库初版只有空目录骨架（`src/data-process`、`src/models` 等扁平目录），存在以下问题：

1. 与计划书附录 B.1 的建议结构（`src/data`、`src/models/teacher|student`、`src/train`…）
   不一致，且 `src/data-process` 这种带连字符的目录无法作为 Python 包导入；
2. 原始数据（`data/raw`，约 8.3 GB）与未来实验输出没有进入忽略规则，容易被误提交；
3. 预处理数据、split、模型权重等可复现产物没有大文件策略；
4. Agent 生成的计划/文档与长期维护的项目文档混放，缺少约定；
5. 端侧部署需要独立的前后端服务与独立 Python 环境，目录边界未定义。

## 选项

1. **保持扁平 `src/*` 目录，不引入 Python 包**
   - 优点：改动小。
   - 缺点：不可安装、不可导入、无法打包 CLI；跨目录 import 依赖 `sys.path` hack。
2. **单一可安装包 `src/kite_drp/`（采用）**
   - 优点：`uv sync` 可编辑安装；`kite-drp` CLI 可用；测试/CI 直接 import；子包按计划书
     章节切分（data/models/distill/train/eval/explain/deploy），与计划书 B.1 一一对应。
   - 缺点：路径比计划书示意多一层 `kite_drp`；需要在文档里说明映射。
3. **多包 monorepo（pip 包 + 部署服务 package）**
   - 优点：边界最清晰。
   - 缺点：当前工程量不需要；部署服务已用独立 `deploy/server/pyproject.toml` 解决环境隔离。

## 决策

- 采用 **src-layout 单一可安装包** `src/kite_drp/`，hatchling 打包，CLI 入口 `kite-drp`；
- 目录映射计划书附录 B.1：`src/kite_drp/data` ≈ `src/data`，`src/kite_drp/models/{teacher,student}`
  ≈ `src/models/...`，其余同名对应；
- **数据分层**（计划书 §3.8）：
  - `data/raw/` 原始数据 → `.gitignore` 忽略（只跟踪 `README.md`、`CHECKSUMS.md5` 与 `.gitkeep`）；
  - `data/interim/` 中间产物 → 忽略；
  - `data/processed/`、`data/splits/` → **Git LFS** 跟踪；
  - `data/licenses/` → 跟踪（许可说明为文本）；
- **实验输出** `experiments/` → 忽略（保留 `.gitkeep` 与 `README.md`）；
- **正式权重** `checkpoints/` → 大文件走 Git LFS（扩展名兜底规则）；
- **文档分工**：`plans&notes/` = Agent 生成的计划与过程文档；`docs/` = 人读的长期文档
  （ADR/数据卡/模型卡/泄漏审查/环境文档）；
- **部署分工**：
  - `src/kite_drp/deploy/`：训练机侧导出/量化/基准代码（Python 库）；
  - `deploy/`：部署资产与独立服务（`server/` 独立 uv 项目、`edge/` 树莓派脚手架、
    `device/` 设备运维规划、`packages/` 打包产物走 LFS）；
- 所有被忽略目录内保留 `.gitkeep`，使 `git clone` 后目录结构完整可建。

## 后果

**正面**

- 代码可 `uv sync` 安装进任意 venv；测试与 CI 无需路径 hack；
- 原始数据/实验输出不会被误提交；大文件走 LFS，仓库历史保持轻量；
- 目录职责与计划书章节对应，新人/新 Agent 可按图索骥；
- 部署服务与训练主环境隔离，避免把 FastAPI/服务依赖塞进训练环境。

**负面 / 权衡**

- `git clone` 后需执行 `git lfs install`（每用户一次）并 `git lfs pull` 才能拿到
  processed/split/权重；未安装 LFS 时文件会以指针形式（仅 130 字节）存在；
- 主包路径多一层 `kite_drp`，与计划书示意略有差异，已在 `docs/` 与 `plans&notes/` 说明；
- 约定需要人工遵守（pre-commit / CI 只能部分自动检查）。

**后续动作**

- 每个被忽略目录的 `.gitkeep` 与 `README.md` 保持更新；
- 新增的数据源先写 `data/licenses/` 与 `docs/data_cards/`，再写下载脚本；
- 代码组织若需要拆分多包，另开 ADR 记录迁移。

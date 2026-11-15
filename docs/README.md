# `docs/` —— 长期维护的项目文档

本目录存放**人读的、长期维护**的项目文档；由 Agent 生成的计划、调研笔记与过程记录
统一放 `plans&notes/`（见该目录 README）。两者不要混用。

## 文档地图

| 文件/目录 | 内容 |
|---|---|
| [`adr/`](adr/) | 架构决策记录（ADR）：每个重要工程决策的背景/选项/决策/后果 |
| [`data_cards/`](data_cards/) | 数据集数据卡（来源/版本/许可/预处理/已知缺陷），配合 `data/licenses/` |
| [`model_cards/`](model_cards/) | 模型卡（训练数据版本/超参/量化配置/验证指标/适用癌种/使用限制） |
| [`leakage_checklist.md`](leakage_checklist.md) | 数据泄漏审查表（提交论文/发布前逐条勾选） |
| [`search_log.md`](search_log.md) | 文献检索式与检索记录（计划书 §1.2 引用） |
| [`environment.md`](environment.md) | Python/uv/CUDA 12.6 环境配置（Windows + Linux） |

## 相关目录

- `plans&notes/`：Agent 生成的计划与过程文档（含主计划书）
- `data/raw/README.md`：原始数据索引与下载清单（小型元数据，随仓库跟踪）
- `data/licenses/`：各数据源许可说明与引用要求
- `README.md`（仓库根）：快速开始与总览

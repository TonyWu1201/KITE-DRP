# 数据卡：外部突变 panel（MSK-IMPACT 468）

- 状态：已生成（2026-11-17）
- 更新日期：2026-11-17
- 负责人：数据管线 / A
- 相关：计划书 §5.1、D27；`scripts/download_mutation_panel.py`

## 1. 基本信息

| 项 | 值 |
|---|---|
| 数据源 | cBioPortal gene-panel API：MSK-IMPACT 468 |
| URL | `https://www.cbioportal.org/api/gene-panels/IMPACT468` |
| 下载时点 | 2026-11-17 |
| 文件 | `data/raw/panel/IMPACT468.txt`（每行一个 HGNC symbol） |
| 规模 | 469 个 symbol（含融合/别名条目；与 U_teach_mu 交集 462） |
| manifest | `data/raw/panel/IMPACT468_manifest.json` |

## 2. 许可与引用

- 基因列表是事实信息；cBioPortal 数据条款以官网为准。
- 需引用 MSK-IMPACT 原始文献与 cBioPortal。
- 详见 `data/licenses/cbioportal.md`。

## 3. 用途

- 学生 MU 输入维度：`U_panel = panel ∩ U_teach_mu(A)`，避免 18k 全基因矩阵。
- 与临床/TCGA 靶向测序 panel 的域对齐。
- 端侧模型输入固定，便于部署。

## 4. 预处理

- 脚本下载 JSON → 提取 `hugoGeneSymbol` → 排序去重 → 文本清单。
- 通过 HGNC 当前 symbol/alias 唯一映射；无法映射者丢弃并报告。
- 产物：`data/processed/student_inputs/panel_mutations_A.npz`（716 × 462，packbits）。

## 5. 已知缺陷与风险

- MSK-IMPACT 468 是历史 panel；MSK-IMPACT 505 / FoundationOne CDx 324 可作为敏感性分析。
- panel 版本更新会改变输入维度；必须冻结文件 + sha256。
- 基因列表不含拷贝数/融合的具体探针定义，仅用于突变输入。

## 6. 使用计划

- 学生 MU 主输入；教师 MU 仍用全基因 B 口径。
- 外部验证（TCGA/临床 panel）时保持同一 gene list。

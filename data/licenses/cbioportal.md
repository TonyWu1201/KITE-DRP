# 数据源许可：cBioPortal gene panels

- 核实状态：待核实（基因列表为事实信息；数据门户条款以官网为准）
- 核实日期：2026-11-17
- 官方条款链接：https://www.cbioportal.org/faq
- 数据卡：`docs/data_cards/mutation_panel_msk_impact.md`

## 1. 基本信息

| 项 | 值 |
|---|---|
| 数据源 | cBioPortal gene-panel API（MSK-IMPACT 468） |
| 版本/时点 | 2026-11-17 抓取 |
| 本地位置 | `data/raw/panel/IMPACT468.txt` |
| 规模 | 469 个 HGNC symbol |

## 2. 许可与再分发

- 基因列表是事实信息，通常不构成版权作品；仍建议核对 cBioPortal/MSK 条款。
- 是否允许再分发原始列表：待核实；本项目只保留派生输入矩阵与统计。
- 署名要求：引用 MSK-IMPACT 原始文献与 cBioPortal。

## 3. 引用文献

- Cheng DT, et al. Memorial Sloan Kettering-Integrated Mutation Profiling of Actionable Cancer Targets (MSK-IMPACT). *J Mol Diagn.* 2015.
- Cerami E, et al. The cBio Cancer Genomics Portal. *Cancer Discov.* 2012.

## 4. 使用限制与本项目约定

- 允许：学生突变 panel 输入、域对齐、派生统计。
- 公开 release：优先发布派生输入与统计；原始 panel 清单需在发布前复核。

# 数据源许可：HGNC

- 核实状态：待核实（公开基因命名资源；再分发需署名）
- 核实日期：2026-11-17
- 官方条款链接：https://www.genenames.org/about/license/
- 数据卡：`docs/data_cards/priors_string_hgnc_dgidb.md`

## 1. 基本信息

| 项 | 值 |
|---|---|
| 数据源 | HGNC Gene.csv |
| 版本 | 本地快照，最新 `date_modified` 2025-03-31 |
| 本地位置 | `data/raw/hgnc/Gene.csv` |

## 2. 许可与再分发

- 许可类型：CC0 / 公开领域（以官网为准；本文件待最终核实）。
- 是否允许再分发：允许（建议署名）。
- 是否允许商业用途：允许。
- 署名要求：引用 HGNC 与 Seal et al. 2023。

## 3. 引用文献

- Seal RL, et al. genenames.org: the HGNC resources in 2023. *Nucleic Acids Res.* 2023.

## 4. 使用限制与本项目约定

- 允许：基因 ID 映射、基因宇宙定义、派生注释表。
- 注意：alias/prev_symbol 多义，必须唯一才采用。
- 公开 release：可保留派生 gene_master 与映射统计。

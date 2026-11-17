# 数据源许可：Gene Ontology (GO)

- 核实状态：已核实（CC BY 4.0）
- 核实日期：2026-11-17
- 官方条款链接：https://geneontology.org/docs/go-citation-policy/
- 数据卡：`docs/data_cards/pathways_reactome_go.md`

## 1. 基本信息

| 项 | 值 |
|---|---|
| 数据源 | GO Consortium：goa_human.gaf + go-basic.obo |
| 版本 | 2026-05/08 构建（下载时点 2026-11-17） |
| 本地位置 | `data/raw/go/` |

## 2. 许可与再分发

- 许可类型：**CC BY 4.0**。
- 是否允许再分发：允许，需署名。
- 是否允许商业用途：允许。
- 署名要求：引用 GO Consortium 与 GOC 最新论文。

## 3. 引用文献

- Gene Ontology Consortium. The Gene Ontology knowledgebase in 2023. *Genetics.* 2023.

## 4. 使用限制与本项目约定

- 允许：通路/功能特征、解释、派生权重矩阵。
- 注意：GAF 含 IEA 电子推断；主表用全量，experimental 版（排除 IEA/ND）仅消融。
- 公开 release：可保留派生权重/统计，需署名与记录 release。

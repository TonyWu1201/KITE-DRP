# 数据源许可：STRING

- 核实状态：已核实（v12.5 为 CC BY 4.0）
- 核实日期：2026-11-17
- 官方条款链接：https://string-db.org/cgi/access
- 数据卡：`docs/data_cards/priors_string_hgnc_dgidb.md`

## 1. 基本信息

| 项 | 值 |
|---|---|
| 数据源 | STRING protein-protein interaction network |
| 版本 | v12.5（主）；v12.0（留档） |
| 本地位置 | `data/raw/string/` |
| 原始文件 | `9606.protein.links.v12.5.txt.gz`、`9606.protein.aliases.v12.5.txt.gz` |

## 2. 许可与再分发

- 许可类型：**CC BY 4.0**（含商业使用）。
- 是否允许再分发原始文件/派生网络：允许，需署名。
- 是否允许商业用途：允许。
- 署名要求：引用 STRING v12.5 与 STRING Consortium 论文。

## 3. 引用文献

- Szklarczyk D, et al. The STRING database in 2023/2025. *Nucleic Acids Res.*
- von Mering C, et al. STRING: known and predicted protein-protein associations. *Nucleic Acids Res.* 2005.

## 4. 使用限制与本项目约定

- 允许：训练、评估、派生 PPI 图与子图。
- 注意：links 文件每对互作双向两行，必须去重后使用。
- 公开 release：可保留去重后的派生 PPI 子图，需在数据卡/论文署名。

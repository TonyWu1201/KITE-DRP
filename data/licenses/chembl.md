# 数据源许可：ChEMBL

- 核实状态：已核实（CC BY-SA 3.0；再分发需同许可）
- 核实日期：2026-11-17
- 官方条款链接：https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/
- 数据卡：`docs/data_cards/opentargets_chembl.md`

## 1. 基本信息

| 项 | 值 |
|---|---|
| 数据源 | ChEMBL 37 mechanism / drug indication |
| 本地位置 | `data/raw/chembl/37/` |

## 2. 许可与再分发

- 许可类型：**CC BY-SA 3.0**（copyleft）。
- 是否允许再分发：允许，但派生数据需同许可并署名。
- 是否允许商业用途：允许。
- 署名要求：引用 ChEMBL 与原始来源。

## 3. 引用文献

- Zdrazil B, et al. The ChEMBL Database in 2023. *Nucleic Acids Res.* 2024.

## 4. 使用限制与本项目约定

- **默认不参与 DTI**（`priors.dti.include_chembl=false`）；Open Targets 已覆盖其 MoA 主路径。
- 允许：显式开启后的 DTI/机制补充、药名桥接、派生统计。
- 注意：若公开 release 包含 ChEMBL 派生边表，需标注 CC BY-SA 并评估 copyleft 影响。
- KEGG/DrugBank 不在本许可内，仍禁止打包。

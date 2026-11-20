# 数据源许可：Open Targets Platform

- 核实状态：已核实（CC0 4.0）
- 核实日期：2026-11-17
- 官方条款链接：https://platform.opentargets.org/downloads
- 数据卡：`docs/data_cards/opentargets_chembl.md`

## 1. 基本信息

| 项 | 值 |
|---|---|
| 数据源 | Open Targets Platform 26.09 |
| 本地位置 | `data/raw/opentargets/26.09/` |
| 表 | target / drug_molecule / drug_mechanism_of_action / clinical_indication |

## 2. 许可与再分发

- 许可类型：**CC0 4.0**（公共领域贡献）。
- 是否允许再分发：允许。
- 是否允许商业用途：允许。
- 署名要求：建议引用 Open Targets，但 CC0 不强制。

## 3. 引用文献

- Ochoa D, et al. The Open Targets Platform: supporting systematic drug-target identification and prioritisation. *Nucleic Acids Res.* 2021/2023.

## 4. 使用限制与本项目约定

- 允许：DTI、靶点/机制注释、派生图与统计。
- 公开 release：可保留派生 DTI 边表，建议署名。
- **本项目主 DTI 源**：`build_dti` 只使用 Open Targets `drug_mechanism_of_action`
  + GDSC `TARGET` 文本；DGIdb 已停用，ChEMBL 默认不参与。
- processed 产物：`data/processed/dti.parquet`、`data/processed/dti_coverage.json`。

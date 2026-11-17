# 数据源许可：PubChem（SMILES）

- 核实状态：待核实（公共数据；再分发前核对 NCBI 条款）
- 核实日期：2026-11-17
- 官方条款链接：https://pubchem.ncbi.nlm.nih.gov/docs/terms
- 数据卡：`docs/data_cards/drug_smiles_pubchem.md`

## 1. 基本信息

| 项 | 值 |
|---|---|
| 数据源 | PubChem PUG-REST / `pubchempy` |
| 版本/时点 | 2026-11-17 抓取 |
| 本地位置 | `data/raw/drugs/smiles.tsv` |
| 规模 | 286/286 GDSC 化合物 |

## 2. 许可与再分发

- 许可类型：PubChem 公共数据库；以 NCBI 条款为准。
- 是否允许再分发原始 SMILES：待核实；SMILES 本身是事实性结构描述，但批量抓取/派生数据仍建议审阅条款。
- 是否允许商业用途：待核实。
- 署名要求：引用 PubChem 与 `pubchempy`。

## 3. 引用文献

- Kim S, et al. PubChem 2023 update. *Nucleic Acids Res.* 2023.
- `pubchempy` Python package.

## 4. 使用限制与本项目约定

- 允许：内部研究、Morgan FP/分子图/scaffold 计算。
- 公开 release：优先发布下载脚本与覆盖率统计；是否打包 SMILES/FP 需在发布前复核。
- 派生特征（FP/scaffold）与 PubChem 记录绑定，需记录抓取日期与 query。

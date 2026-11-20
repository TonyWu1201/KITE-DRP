# 数据源许可：DGIdb（已停用）

- 核实状态：**已停用**（混合来源；再分发条款不清）
- 核实日期：2026-11-20
- 官方条款链接：https://dgidb.org/、https://github.com/dgidb
- 数据卡：`docs/data_cards/priors_string_hgnc_dgidb.md`

> **2026-11-20 决策**：DTI 主源改为 Open Targets Platform 26.09（CC0 4.0）。
> `build_dti` 不再读取 DGIdb；`data/raw/dgidb/` 仅保留作历史审计，不进入
> canonical / processed，也不应再打包发布。

## 1. 基本信息

| 项 | 值 |
|---|---|
| 数据源 | DGIdb v5.0.13 |
| 版本 | data 2026-10；v5.0.13 |
| 本地位置 | `data/raw/dgidb/{interactions,drugs,genes,categories}.tsv` |

## 2. 许可与再分发

- 许可类型：混合来源（ChEMBL、DTC、GuideToPharmacology、Drug Repurposing Hub、TTD 等），逐源条款不同。
- 是否允许再分发原始文件：**不建议**；按受限数据处理。
- 是否允许商业用途：待核实。
- 署名要求：引用 DGIdb 与各来源数据库。

## 3. 引用文献

- Freshour SL, et al. Integration of the Drug–Gene Interaction Database (DGIdb 4.0) with open crowdsource efforts. *Nucleic Acids Res.* 2021.

## 4. 使用限制与本项目约定

- 允许：内部研究、解释校验、DTI 兜底。
- 禁止：公开 release 打包 DGIdb 原始文件或少见来源的派生边表。
- 公开 release：只保留覆盖率统计与不泄露来源条款的聚合结果。

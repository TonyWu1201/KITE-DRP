# 数据源许可：TCGA-CDR（PanCan-Clinical-2018）

- 核实状态：已核实（补充表，需引用）
- 核实日期：2026-11-17
- 官方条款链接：https://doi.org/10.1016/j.cels.2018.03.002
- 数据卡：`docs/data_cards/tcga_cesc.md`

## 1. 基本信息

| 项 | 值 |
|---|---|
| 数据源 | TCGA-CDR Supplemental Table S1 |
| 版本 | PanCan-Clinical-2018 |
| 本地位置 | `data/raw/tcga/cdr/TCGA-CDR-SupplementalTableS1.xlsx` |
| 规模 | 11,160 例 / 33 癌种；CESC 307 |

## 2. 许可与再分发

- 许可类型：学术补充材料；引用 Liu et al. 2018。
- 是否允许再分发：派生统计可；原始 xlsx 不进入公开 release（保守处理）。
- 是否允许商业用途：待核实。
- 署名要求：引用 Liu et al. 2018。

## 3. 引用文献

- Liu J, et al. An integrated TCGA pan-cancer clinical data resource to drive high-quality survival outcome analytics. *Cell.* 2018;173(2):400–416.

## 4. 使用限制与本项目约定

- 允许：L2′ 生存终点（PFI 主、OS 敏感性）。
- 注意：DFI 仅 176/307；OS 与 PFI 在 243/307 完全一致，不当独立终点。
- 公开 release：只发布派生生存分析结果，不打包原始表。

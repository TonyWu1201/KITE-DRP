# 数据源许可：DepMap / CCLE

- 核实状态：待核实（公开数据，但再分发/商用条款需复核）
- 核实日期：2026-11-17
- 官方条款链接：https://depmap.org/portal/、https://www.broadinstitute.org/ccle
- 数据卡：`docs/data_cards/depmap_omics.md`、`docs/data_cards/ccle_methylation.md`

## 1. 基本信息

| 项 | 值 |
|---|---|
| 数据源 | DepMap Model/Expression/Mutations/CNV；CCLE 旧资源 |
| 版本 | 组学 mtime 2026-07-13；Model.csv 2026-11-10；CCLE METH 2018 |
| 本地位置 | `data/raw/depmap/`、`data/raw/ccle/` |

## 2. 许可与再分发

- 许可类型：DepMap/CCLE 公开数据；具体条款待官方页面复核。
- 是否允许再分发原始文件：**待核实**；本项目默认不公开原始/逐细胞系组学矩阵。
- 是否允许商业用途：待核实。
- 署名要求：引用 DepMap 2026 release 与 CCLE 相关论文。

## 3. 引用文献

- DepMap 2026 release（Broad Institute）。
- Ghandi M, et al. Next-generation characterization of the Cancer Cell Line Encyclopedia. *Nature.* 2019.

## 4. 使用限制与本项目约定

- 允许：内部研究、训练/评估、派生特征计算。
- 禁止：未经确认将原始组学或逐细胞系派生矩阵放入公开 release。
- `data/processed/` 仅私有 LFS；公开 release 只含代码/脚本/split/模型卡/派生统计。
- CCLE RRBS 甲基化仅作可选消融，不用于跨平台补全。

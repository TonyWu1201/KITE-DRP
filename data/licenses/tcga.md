# 数据源许可：TCGA-CESC（GDC 开放层）

- 核实状态：已核实（开放层，需引用）
- 核实日期：2026-11-17
- 官方条款链接：https://www.cancer.gov/ccg/research/genome-sequencing/tcga/using-tcga-data
- 数据卡：`docs/data_cards/tcga_cesc.md`

## 1. 基本信息

| 项 | 值 |
|---|---|
| 数据源 | TCGA Cervical Squamous Cell Carcinoma (CESC) |
| 版本 | GDC Data Release 46.0；下载 2026-11-14 |
| 本地位置 | `data/raw/tcga/CESC/`、`data/raw/tcga/clinical/` |
| 规模 | 1,204 数据文件 / 6.421 GB；clinical 307 例 |

## 2. 许可与再分发

- 许可类型：GDC 开放层（NIH open access）。
- 是否允许再分发：开放层可再分发/共享，需遵守 GDC 数据使用与署名要求。
- 是否允许商业用途：按 GDC/NIH 条款。
- 受控层数据（如有）**不得**进入仓库。

## 3. 引用文献

- The Cancer Genome Atlas Research Network. Integrated genomic and molecular characterization of cervical cancers. *Nature.* 2017.
- Grossman RL, et al. Toward a shared vision for cancer genomic data. *NEJM.* 2016.

## 4. 使用限制与本项目约定

- 允许：评估、域适应研究、派生统计。
- 禁止：把 TCGA 用于主模型训练/选模/早停（L3 半监督单列）。
- 仓库不提交 TCGA 原始数据；`data/processed/tcga_inputs` 默认不生成，需 D29 映射表后按需装配。
- 公开 release 只含脚本、聚合统计与不泄露样本级组学的派生结果。

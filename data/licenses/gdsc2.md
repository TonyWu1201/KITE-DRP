# 数据源许可：GDSC2

- 核实状态：待核实（本地使用；公开 release 前需复核）
- 核实日期：2026-11-17
- 官方条款链接：https://www.cancerrxgene.org/（使用条款/下载页）
- 数据卡：`docs/data_cards/gdsc2.md`

## 1. 基本信息

| 项 | 值 |
|---|---|
| 数据源 | Genomics of Drug Sensitivity in Cancer (GDSC2) |
| 版本 | 本地文件名 27Oct23（8.5 代，release 号待核） |
| 本地位置 | `data/raw/gdsc/` |
| 原始文件 | `GDSC2_fitted_dose_response_27Oct23.csv`、`screened_compounds_rel_8.4.csv`、`screened_compounds_rel_8.5_inchi.csv`、`Cell_Lines_Details.xlsx` |

## 2. 许可与再分发

- 许可类型：学术使用需引用；具体条款待官方页面复核。
- 是否允许再分发原始文件：**待核实**；本项目默认不提交 git、不公开原始/逐细胞系派生矩阵。
- 是否允许商业用途：待核实。
- 署名要求：引用 GDSC 原始论文与数据版本。

## 3. 引用文献

- Yang W, et al. Genomics of Drug Sensitivity in Cancer (GDSC): a resource for therapeutic biomarker discovery in cancer cells. *Nucleic Acids Res.* 2013.
- Iorio F, et al. A landscape of pharmacogenomic interactions in cancer. *Cell.* 2016.

## 4. 使用限制与本项目约定

- 允许：内部研究、训练/评估、派生特征计算。
- 禁止：未经确认将原始标签或逐细胞系派生矩阵放入公开仓库/发布包。
- 公开 release：只发布代码、下载脚本、split 文件、模型卡与不含逐细胞系矩阵的派生统计。
- `data/processed/` 仅在私有 LFS 内共享（D30）。

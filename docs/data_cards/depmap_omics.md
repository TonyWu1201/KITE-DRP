# 数据卡：DepMap 细胞系与组学

- 状态：已核实（本地实测，2026-11-17）
- 更新日期：2026-11-17
- 负责人：数据审计 A/B
- 相关：计划书 §3.2–§3.3；`configs/data/gdsc2.yaml`；`src/kite_drp/data/omics/`

## 1. 基本信息

| 项 | 值 |
|---|---|
| 数据集全名 | DepMap/CCLE Model、Expression、SomaticMutations、CNV |
| 提供方 / 项目 | Broad Institute DepMap |
| 版本 / 发布时点 | 组学 mtime 2026-07-13（疑 26Q1）；Model.csv 2026-11-10（疑更晚季度）；**以内容 MD5 冻结** |
| 下载入口 | https://depmap.org/portal/ |
| 文件 | `Model.csv`、`DepMap_ExpressionTPMLogp1HumanProteinCodingGenes.csv`、`DepMap_SomaticMutations.csv`、`DepMap_CNGeneWGS.csv` |
| 文件大小 | 0.7 MB / 291 MB / 554 MB / 390 MB |
| 存储位置 | `data/raw/depmap/` |

## 2. 许可与引用

- 许可类型：DepMap/CCLE 公开数据，再分发与商用条款待核实（`data/licenses/depmap.md`）。
- 再分发：原始文件不提交 git；processed 组学仅在私有 LFS 内共享（D30）。
- 引用：DepMap 2026 release；CCLE 相关论文（Ghandi et al. 2019 等）。

## 3. 内容与口径

| 模态 | 原始 | default 过滤后 | 备注 |
|---|---|---|---|
| Model | 2,154 模型 × 49 列 | — | ACH 主键；SangerModelID 1,220 非空 |
| GE | 1,775 × 19,215 基因 | **1,719** 模型 | log2(TPM+1)；删 5 个零方差基因 |
| MU | 1,172,688 行 × 68 列 | **736,094 行 / 1,968 模型** | A 口径 730,652 行 / 18,620 基因 |
| CNV | 1,132 × 19,955 列 | **1,118** 模型 | 线性相对 CN；811 列全 NaN |

- 主训练池：GDSC2 可映射 968 → ∩GE 717 / ∩MU 959 / ∩CNV 513 → **三模态 509**；教师矩阵用 M-716（GE+MU，CNV 缺失以 mask 标记）。
- 组学版本不一致，但 ID 集合已验证为 Model.csv 子集（0 孤儿 ID）。

## 4. 预处理

- 脚本：`src/kite_drp/data/omics/{expression,mutations,cnv}.py`。
- GE：`index_col=0` 跳过无名首列；`IsDefaultEntryForModel=='Yes'`；不再取 log；不插补零表达。
- MU：A 口径 = protein_coding + `VariantInfo` 非沉默；B 口径 = LikelyLoF/Hotspot；`mut_burden` 为官方预筛后的突变基因数（**不是 TMB**）。教师默认 B、学生 panel 默认 A（D31）。
- CNV：`log2(x)`；裁剪边界在 fit_stats 内按训练 fold 估计；删 811 个全 NaN 列；PAR_Y/裸 ENSG 标记或丢弃。
- 产物：`data/interim/canonical/{expression,cnv,mutations_A,mutations_B,mutation_long}.parquet|npz`。

## 5. 已知缺陷与风险

- 三文件与 Model.csv 版本窗口不同；release 号待官方确认。
- CNV 只有 1,118 模型，是主池瓶颈（509/968）。
- 突变文件已官方预筛（AF≥0.15、DP≥5、非沉默 + rescued），不能当原始突变全集。
- 表达 log 底数由列名推断为 log2；不影响 z-score/排序。
- CNV 的 811 个全 NaN 列中含 753 个可解析基因列，导致实际 `U_teach_ge=18,396`，低于规划估计的 19,125；详见 `docs/adr/0004-data-pipeline-g0-decisions.md` 与 `data/interim/canonical/manifest.json`。

## 6. 使用计划

- 教师：GE+MU+CNV 三模态；METH 可选（D21）。
- 学生：MU panel + TMB；GE 通路分数（路线 A）或 HVG（路线 B）。
- VAE：目标 CNV 空间按 PPI 度规则取 1,600 基因（D26）。

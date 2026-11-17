# 数据卡：TCGA-CESC 四模态（评估域）

- 状态：已核实（离线部分，2026-11-17）；file→case 映射表待生成
- 更新日期：2026-11-17
- 负责人：数据审计 TCGA / A
- 相关：计划书 §3.7、§8；`configs/data/tcga.yaml`；`src/kite_drp/data/tcga.py`

## 1. 基本信息

| 项 | 值 |
|---|---|
| 数据集 | TCGA Cervical Squamous Cell Carcinoma (CESC) |
| 提供方 | GDC / NCI |
| 版本 | GDC Data Release 46.0；下载 2026-11-14 |
| 文件 | `data/raw/tcga/CESC/` 1,204 个数据文件（GE 309 / MU 289 / CNV 294 / METH 312） |
| 大小 | 6.421 GB |
| 临床/CDR/Ding | `data/raw/tcga/{clinical,cdr,ding2016}/` |

## 2. 许可与引用

- 许可：GDC 开放层（NIH open access），需引用 TCGA 与 GDC。
- 受控层不进入仓库；公开 release 只含脚本/派生统计。
- 详见 `data/licenses/tcga.md`。

## 3. 内容与口径

| 模态 | 文件数 | 关键事实 |
|---|---:|---|
| GE | 309 | STAR counts；两行表头；GENCODE v36 versioned ENSG；4 个 `N_*` 行需剔除 |
| MU | 289 | 287 例患者；MAF 140 列；`IMPACT` 列名不是 VepImpact；2 例 06A 转移灶 |
| CNV | 294 | ASCAT3 整数总 CN（0–111，1.33% 空）；20/294 文件 CN==2 占比 >90% |
| METH | 312 | HM450 sesame level3 betas；486,427 探针；缺失中位 14.77% |
| clinical | — | 202 列 × 1,454 行 / 307 例；一行=一条治疗记录；43 个 treatments.* 字段 |
| CDR | — | 11,160 例 / 33 癌种；CESC 307；PFI 307（事件 71）、OS 307 |
| Ding2016 | — | CESC 97 条 / 69 例；Cisplatin 62 条 / 59 例（CR 47 / PD 8 / PR 5 / SD 2） |

- 离线可确证：MU 287 例、MU∩clinical=287、clinical∩CDR=307。
- 四模态交集 274/281 为前次审计值，**需 file→case 映射表复核**。

## 4. 预处理

- file→case：`fetch_file_index()` 调 GDC API，冻结 `data/interim/tcga/file_index.parquet`（D29）。
- MU：MAF → patient barcode 前 12 位 → A/B 口径 0/1 + TMB；样本类型 01A/01B 主评估，06A 标记。
- GE：`skiprows=1`、剔 `N_*`、`tpm_unstranded` → log2(TPM+1)、ENSG 去版本→HGNC→symbol、对齐 U_teach_ge。
- CNV：`log2(max(cn,0.5)/2)`；QC 剔除 CN==2 占比 >90% 的文件。
- clinical：`'--'/Not Reported/Unknown → NA`；按 patient 聚合 treatments。
- CDR：主终点 PFI，敏感性 OS；DFI 仅 176 例子集。
- Ding：CESC 用全称标签；仅 Cisplatin 达 n≥30，只评估不训练。

## 5. 已知缺陷与风险

- **P0 阻断：文件名/ manifest/.parcel/annotations 均无 barcode**，GE/CNV/METH 患者级矩阵依赖 API 映射表。
- METH 缺失高且与 CCLE RRBS 跨平台，本期只做统计。
- 20 个正常样 CNV 文件需 QC 排除。
- Ding 标签严重不平衡（顺铂严格响应 52/62），且与 clinical 顺铂 128 例不能混用。
- treatment 字段稀疏；`best_overall_response` 全占位不可用。

## 6. 使用计划

- L1：`(MU, GE) → CNV` 补全评估（四模态交集，待复核）。
- L2：Cisplatin-CESC 弱标签，仅评估、不调参。
- L2′：PFI 生存一致性（C-index/HR）。
- L3：无标签域适应，独立实验、不进主表（代码级 `purpose=eval` 红线）。

# 数据缺口清单（2026-11-17 复核）

> 目的：对照 `plans&notes/2026-11-17_数据预处理详细规划.md` 与当前仓库实测，列出
> **已就绪 / 仍缺 / 可选** 的数据资源。`data/raw/` 与 `data/interim/` 不提交 git；
> 可复现入口以脚本和 processed manifest 为准。

## 1. 已就绪（G0 主流程可用）

| 数据 | 本地位置 | 状态 |
|---|---|---|
| GDSC2 标签 + 药物/细胞系元数据 | `data/raw/gdsc/` | ✅ 969 株 / 295 DRUG_ID / 286 化合物 |
| DepMap Model/GE/MU/CNV | `data/raw/depmap/` | ✅ 1,719 GE / 1,968 MU / 1,118 CNV；主池 509 |
| STRING v12.5 / HGNC / DGIdb | `data/raw/{string,hgnc,dgidb}/` | ✅ PPI 282,864 对；DTI 4,378 边 / 237 化合物 |
| CCLE RRBS 甲基化（可选） | `data/raw/ccle/` | ✅ 默认关闭（D21） |
| TCGA-CESC GE/MU/CNV/METH 原始文件 | `data/raw/tcga/CESC/` | ✅ 1,204 文件 |
| TCGA clinical / CDR / Ding2016 | `data/raw/tcga/{clinical,cdr,ding2016}/` | ✅ 307 例 / 307 例 / 59 例顺铂 |
| GDC file→case 映射 | `data/interim/tcga/file_index.parquet` | ✅ 1,204/1,204 |
| GDSC SMILES（PubChem） | `data/raw/drugs/smiles.tsv` | ✅ 286/286，InChIKey 全匹配 |
| Reactome + GO 通路 | `data/raw/{reactome,go}/` | ✅ 2,874 + 7,542；主表 1,000 条 |
| MSK-IMPACT 468 panel | `data/raw/panel/IMPACT468.txt` | ✅ 469 symbol / 462 交集 |

当前 processed：M-716 教师输入、M-509 主评估子集、1,000 通路路线 A、462 基因学生 panel、
260 例 TCGA GE+CNV+MU 评估集、18 splits × 3 seeds、90 fold fit-stats。

## 2. 补数清单（按优先级）

> 2026-11-17 更新：P1 五项已全部落盘（见下方“P1 补数结果”）；P2 仍为可选/后续里程碑。

### P1 — 已补齐并预处理（✅ 完成）

| 缺口 | 用途 | 下载方式 | 影响 |
|---|---|---|---|
| **Open Targets Platform**（targets/drugs parquet） | A_DG 药物-靶点主源；补 GDSC/DGIdb 未覆盖的 49 个化合物 | https://platform.opentargets.org/downloads （CC0 4.0） | ✅ 已下载 26.09 并预处理：466 条 MOA 边；合并 DTI 覆盖 **237/286** 化合物（49 个内部化合物仍无 DTI） |
| **ChEMBL v36**（mechanisms / drug indications） | 药名桥接、DTI 补充、SMILES 兜底 | https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/ （CC BY-SA **3.0**） | ✅ 已抓取 **ChEMBL 37** mechanism（7,561 行）与 drug_indication；新增 242 条机制边；v36 未落盘，版本差异见 manifest |
| **GDSC1** 标签 | P5 外部验证：GDSC2 训练 → GDSC1 零样本 | https://www.cancerrxgene.org/downloads/bulk_download | ✅ 已下载 release 8.5 / 27Oct23（CancerRxGene 已 410，改用 Sanger COG 镜像）；外部验证主项；不混训 |
| **CTRPv2** | P5 外部验证（AUC） | https://portals.broadinstitute.org/ctrp/ | ✅ 已下载 v21 AUC（Zenodo 处理版；原 NCI/Broad 直链已下线） |
| **PRISM** | P5 外部验证（LFC/AUC，非 IC50） | https://depmap.org/repurposing/ | ✅ 已下载并预处理 AUC（701,004 行）+ LFC long（6,360,912 行）；主评估用 AUC/排序 |

#### P1 补数结果（2026-11-17）

| 来源 | 本地位置 | 本次文件 | 备注 |
|---|---|---|---|
| Open Targets 26.09 | `data/raw/opentargets/26.09/` | `target/*.parquet`（78,733 靶点）、`drug_molecule`（19,170 分子）、`drug_mechanism_of_action`（6,092 行）、`clinical_indication`（86,451 行） | CC0 4.0；官方 FTP 直连 |
| ChEMBL 37 | `data/raw/chembl/37/` | `chembl_37_mechanism.{jsonl,tsv}.gz`、`chembl_37_drug_indication.{jsonl,tsv}.gz`、`chembl_status.json` | 规划 v36；官方 API 仅当前版本，故为 37；CC BY-SA 3.0 |
| GDSC1 8.5 | `data/raw/gdsc/` | `GDSC1_fitted_dose_response_27Oct23.csv`（54.1 MB；333,161 行/970 株/402 drugs/19 列）、`screened_compounds_rel_8.5.csv`、`GDSC_Fitted_Data_Description.pdf` | CancerRxGene 返回 410；Sanger COG 镜像下载；学术使用条款 |
| CTRPv2.1 | `data/raw/ctrp/` | `v21_auc_sensitivities.tsv`（10.4 MB；326,148 AUC 行/481 drugs/835 cell lines）、`treatment.tsv`、`sample.tsv`、`gene_annotation.tsv`、`molecular_feature_annotation.tsv` | 原 NCI `CTRPv2.0_2015_ctd2_ExpandedDataset.zip` 直链已 301 到退役页面；改用 Haibe-Kains 实验室 Zenodo v21 AUC 处理版；CC BY 4.0 |
| PRISM | `data/raw/prism/` | `secondary-screen-dose-response-curve-parameters.csv`（252 MB；701,004 行 × 20 列，含 AUC/IC50/EC50）、`secondary-screen-replicate-collapsed-logfold-change.csv`（87 MB，LFC 矩阵）、cell-line-info、collapsed treatment-info、readme | figshare `ndownloader` 直连；CC BY 4.0（DepMap/CCLE，发布前复核） |

- 复现脚本：`scripts/download_p1_external.py`（支持 `--only`、`--force`、Zenodo `--proxy`）。
- 每个来源落盘 `download_manifest.json`，新文件 MD5 已并入 `data/raw/CHECKSUMS.md5`。
- 许可说明：`data/licenses/{opentargets,chembl,gdsc1,ctrp,prism}.md`。
- **已做预处理（2026-11-17）**：
  - DTI 合并 GDSC + Open Targets + ChEMBL + DGIdb：4,395 边 / 237 化合物 / 1,663 靶点基因；Open Targets 466 边、ChEMBL 242 边。
  - GDSC1：314,970 条化合物层标签；969 细胞系（509 在主池）；377 化合物，其中 127 个与 GDSC2 共有；交集 109,636 行。
  - CTRPv2：326,148 行 AUC；832 细胞系（420 在主池）；84 个与 GDSC2 共有化合物；交集 59,245 行。
  - PRISM：701,004 行 AUC；480 细胞系（280 在主池）；122 个与 GDSC2 共有化合物；交集 75,488 行。
  - 产物：`data/interim/external/{gdsc1_labels,ctrpv2_auc,prism_auc,prism_lfc}.parquet`；清单 `data/splits/external_validation_manifest.json`。
  - PRISM LFC：6,360,912 行 / 489 细胞系 / 1,502 treatments；122 个 GDSC2 共有化合物；672,864 行交集。
- 统一方向：CTRP/PRISM AUC 越大越耐药（与 GDSC2 LN_IC50 正相关 r≈0.68/0.57），输出 `sensitivity_score=-AUC`。

### P2 — 可选/后续里程碑

| 缺口 | 用途 | 说明 |
|---|---|---|
| TCGA RPPA（172 文件） | 蛋白模态扩展 | 当前只有 manifest，数据未下载；本期不做 |
| TCGA 非 ascat3 CNV | 复核前次 281 例交集 | 当前 ascat3 + QC 后 260 例 |
| TCGA-BRCA / LUAD | L3 域适应扩展 | 当前仅 CESC |
| CCLE legacy 450K | 替代 RRBS 的 METH 主模态 | 若 E2-METH 显示收益再补；D21 默认关闭 |
| TCGA HM450 患者级 METH 矩阵 | L1 METH 补全 | 当前只生成 `tcga_meth_stats.parquet`，未进入主评估 |
| MSigDB Hallmark | 通路集对照 | 学术注册后本地使用，不得再分发原始 GMT |
| IMPACT505 / FoundationOne CDx | panel 敏感性分析 | 可用 `scripts/download_mutation_panel.py --panel ...` |
| Edge 预印本 1,603 基因清单 | VAE 目标空间对照 | D26 已改为规则化 PPI∩CNV，不阻塞 |
| 官方 GDSC 8.5 screened compounds | 元数据版本核对 | 本地 8.5_inchi 是 8.4 派生；标签/靶点已够用 |
| KEGG / DrugBank | 通路/DTI 补充 | 许可限制，禁止打包分发；不建议纳入 release |

## 3. 未核实但不算“缺数据”

| 项 | 现状 | 处理 |
|---|---|---|
| DepMap release 号 | 组学 2026-07-13 / Model.csv 2026-11-10，混合版本 | 已按内容 MD5 冻结；数据卡标注待官方确认 |
| GDSC 本地 release | 文件名 27Oct23，数据卡称 8.5 | 论文引用前核对官方 |
| TCGA 四模态交集 | 当前 QC 后 **260**；前次审计 274/281 | 260 是剔除 16 个正常样 CNV + 3 个非肿瘤 GE 后的口径；见 `tcga_inputs/index.json` |
| CNV 裸 ENSG 映射 | 531 列无法映射到当前 HGNC | `U_teach_ge=18,396`，见 ADR-0004 |
| 20 个正常样 CNV 文件 | 当前 QC 剔除 16 个（`CN==2>90%`） | 其余按 case 映射；报告中注明 |

## 4. 推荐补数据顺序

1. ~~**Open Targets**（CC0）→ 提升 DTI/药物图覆盖~~ ✅ 2026-11-17 已下载 26.09。
2. ~~**GDSC1 + CTRPv2** → P5 外部验证~~ ✅ 2026-11-17 已下载 GDSC1 8.5 / CTRPv2.1 v21 AUC。
3. ~~**PRISM** → 与 CTRPv2 并列的 external validation~~ ✅ 2026-11-17 已下载 secondary dose-response 参数。
4. 下一步：把 P1 表接入 A_DG / P5 预处理，复核 DTI 覆盖与跨库药名映射。
5. 可选：MSigDB Hallmark、IMPACT505、TCGA RPPA/BRCA/LUAD。

## 5. 下载脚本/入口

```bash
# 已实现且已运行
.venv/bin/python scripts/download_smiles_pubchem.py
.venv/bin/python scripts/download_pathways.py
.venv/bin/python scripts/download_mutation_panel.py --panel IMPACT468
.venv/bin/python -m kite_drp data canonical --only tcga --fetch-file-index

# P1 外部库（Open Targets / ChEMBL / GDSC1 / CTRPv2 / PRISM）
.venv/bin/python scripts/download_p1_external.py --only all

# 本环境 Zenodo 需代理；脚本只对 Zenodo 走代理，其余源直连
.venv/bin/python scripts/download_p1_external.py --only ctrp --proxy http://127.0.0.1:7897

# 下载前先在 data/licenses/<source>.md 写许可，再落 data/raw/<source>/ 并更新 CHECKSUMS。
# 每个来源会写 download_manifest.json（URL/大小/sha256/md5/抓取时间）。
```

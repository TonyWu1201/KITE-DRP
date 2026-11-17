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

## 2. 仍缺（按优先级）

### P1 — 建议在 M1/M2 前补

| 缺口 | 用途 | 下载方式 | 影响 |
|---|---|---|---|
| **Open Targets Platform**（targets/drugs parquet） | A_DG 药物-靶点主源；补 GDSC/DGIdb 未覆盖的 49 个化合物 | https://platform.opentargets.org/downloads （CC0 4.0） | DTI 覆盖 237/286 → 预计 260+；药物图/解释更完整 |
| **ChEMBL v36**（mechanisms / drug indications） | 药名桥接、DTI 补充、SMILES 兜底 | https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/ （CC BY-SA 4.0） | 改善跨库药名与靶点；注意 copyleft |
| **GDSC1** 标签 | P5 外部验证：GDSC2 训练 → GDSC1 零样本 | https://www.cancerrxgene.org/downloads/bulk_download | 外部验证主项；不混训 |
| **CTRPv2** | P5 外部验证（AUC） | https://portals.broadinstitute.org/ctrp/ | 外部验证首选之一 |
| **PRISM** | P5 外部验证（LFC/AUC，非 IC50） | https://depmap.org/repurposing/ | 需转换为排序/二分类 |

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

1. **Open Targets**（CC0）→ 提升 DTI/药物图覆盖，许可最友好。
2. **GDSC1 + CTRPv2** → P5 外部验证，G0 之后最先需要。
3. **PRISM** → 与 CTRPv2 并列的 external validation。
4. 可选：MSigDB Hallmark、IMPACT505、TCGA RPPA/BRCA/LUAD。

## 5. 下载脚本/入口

```bash
# 已实现且已运行
.venv/bin/python scripts/download_smiles_pubchem.py
.venv/bin/python scripts/download_pathways.py
.venv/bin/python scripts/download_mutation_panel.py --panel IMPACT468
.venv/bin/python -m kite_drp data canonical --only tcga --fetch-file-index

# 待补外部库：建议新增 scripts/download_external.sh
#   Open Targets / ChEMBL / GDSC1 / CTRPv2 / PRISM
# 下载前先在 data/licenses/<source>.md 写许可，再落 data/raw/<source>/ 并更新 CHECKSUMS。
```

# `data/raw/` 原始数据目录（2026-11-14 整理）

> 目的：把此前散落在 `data/` 顶层、`dgi/` 与工作区根目录的原始数据统一归档到 `data/raw/`，按**数据来源**分类，对齐计划书 §3.8 与 `work/notes/data_readiness_audit_2026-11-14.md` §8 的建议结构。
> 原则：`raw/` 只放**下载得到的原始文件**（+ 下载清单/元数据文件），不做预处理；预处理产物应放 `data/interim/`、`data/processed/`（尚未创建）。
> 完整性：非 TCGA 数据文件的 MD5 见 [`CHECKSUMS.md5`](CHECKSUMS.md5)（30 个数据文件）；TCGA-CESC 1,204 个文件的 MD5 见 `tcga/manifests/manifest_cesc_*.tsv`（gdc-client 下载时已校验，2026-11-14 复核 1204/1204 文件、6.421 GB 与 manifest 完全一致）。

---

## 1. 目录结构

```
data/raw/
├── README.md                       ← 本文件（数据索引/数据卡摘要）
├── CHECKSUMS.md5                   ← 非 TCGA 数据文件 MD5
├── gdsc/                           ← GDSC 药敏标签 + 药物/细胞系元数据
│   ├── GDSC2_fitted_dose_response_27Oct23.csv
│   ├── screened_compounds_rel_8.4.csv
│   ├── screened_compounds_rel_8.5_inchi.csv
│   ├── Cell_Lines_Details.xlsx
│   └── README.md                   ← 该来源的数据卡（由 2026-11-10 笔记更新路径）
├── depmap/                         ← CCLE/DepMap 组学（教师 GE/MU/CNV + 模型交叉表）
│   ├── Model.csv
│   ├── DepMap_ExpressionTPMLogp1HumanProteinCodingGenes.csv
│   ├── DepMap_SomaticMutations.csv
│   ├── DepMap_CNGeneWGS.csv
│   └── 26Q1_Mutation_Pipeline_Documentation.pdf
├── ccle/                           ← 旧 CCLE 资源（甲基化、旧细胞系注释，均非权威主表）
│   ├── CCLE_DNA_methylation_TSS1kb.txt
│   └── Cell_lines_annotations_20181226.txt
├── hgnc/                           ← HGNC 基因主表（symbol/Ensembl/Entrez 映射）
│   └── Gene.csv
├── string/                         ← STRING PPI（v12.5 为主，v12.0 保留）
│   ├── 9606.protein.links.v12.5.txt.gz
│   ├── 9606.protein.aliases.v12.5.txt.gz
│   ├── 9606.protein.links.v12.0.txt
│   └── 9606.protein.aliases.v12.0.txt
├── dgidb/                          ← DGIdb v5.0.13（2026-10）药物-基因交互
│   ├── interactions.tsv
│   ├── drugs.tsv
│   ├── genes.tsv
│   └── categories.tsv
├── drugs/                          ← PubChem 下载的 GDSC 化合物 SMILES（2026-11-17）
│   ├── smiles.tsv
│   ├── smiles.parquet
│   └── smiles_manifest.json
├── reactome/                       ← Reactome current GMT（2026-11-17）
│   └── ReactomePathways.gmt
├── go/                             ← GO human GAF/OBO 与转换 GMT（2026-11-17）
│   ├── goa_human.gaf.gz
│   ├── go-basic.obo
│   ├── goa_human.gmt
│   └── goa_human_experimental.gmt
├── panel/                          ← cBioPortal MSK-IMPACT 468（2026-11-17）
│   ├── IMPACT468.txt
│   └── IMPACT468_manifest.json
└── tcga/                           ← TCGA（当前只有 CESC；BRCA/LUAD 未下载）
    ├── CESC/                       ← GDC processed 四模态原始下载（1,204 个 parcel 目录）
    ├── clinical/                   ← GDC Portal 临床导出（CESC，tar.gz）
    ├── cdr/                        ← TCGA-CDR 泛癌生存终点表（xlsx）
    ├── ding2016/                   ← Ding 2016 响应标签补充材料（zip）
    └── manifests/                  ← GDC 下载清单（按模态筛选 + 源清单 + 下载命令）
```

---

## 2. 文件清单（来源 / 版本 / 大小 / 用途 / 许可）

| 分类 | 文件 | 大小 | 版本/时点 | 计划书用途 | 许可/引用 |
|---|---|---|---|---|---|
| gdsc | `GDSC2_fitted_dose_response_27Oct23.csv` | 35 MB | 文件名 27Oct23（8.5 代）；242,036 行/969 株/295 DRUG_ID | 主训练标签（§3.2-#1，D11） | GDSC 学术使用需引用；再分发前确认 |
| gdsc | `screened_compounds_rel_8.4.csv` | 46 KB | release-8.4（2022-07-24） | 药物元数据兜底（§3.2-#20） | GDSC |
| gdsc | `screened_compounds_rel_8.5_inchi.csv` | 61 KB | 8.5 代 + ChEMBL/InChI 派生（2026-11-10） | 药物 SMILES/靶点/scaffold 前置（§3.3-6、D3） | GDSC；派生字段来自 ChEMBL（CC BY-SA） |
| gdsc | `Cell_Lines_Details.xlsx` | 117 KB | release-8.4，1,002 行 × 13 列 | 旧细胞系元数据/TCGA 标签（已由 `depmap/Model.csv` 取代，作 fallback） | GDSC |
| depmap | `Model.csv` | 0.7 MB | 2,154 行（2026-11-10 下载；疑 26Q3） | **权威 ID 交叉表**（SangerModelID↔ACH↔Oncotree/组织） | DepMap Public |
| depmap | `DepMap_ExpressionTPMLogp1HumanProteinCodingGenes.csv` | 291 MB | 1,775 行/1,719 模型 × 19,215 基因（2026-07-13；疑 26Q1 代） | 教师 GE + 学生表达输入（§3.2-#3） | DepMap/CCLE |
| depmap | `DepMap_SomaticMutations.csv` | 554 MB | 1,172,688 行/1,968 模型（默认条目 736,094） | 教师 MU + panel 突变（§3.2-#3） | DepMap/CCLE |
| depmap | `DepMap_CNGeneWGS.csv` | 390 MB | 1,132 行/1,118 模型 × 19,955 基因（WGS 线性相对 CN） | 教师 CNV + VAE 补全目标（§3.3、§5.3） | DepMap/CCLE |
| depmap | `26Q1_Mutation_Pipeline_Documentation.pdf` | 0.2 MB | DepMap 26Q1 突变流程说明 | 突变过滤口径依据 | DepMap |
| ccle | `CCLE_DNA_methylation_TSS1kb.txt` | 142 MB | 21,338 TSS × 843 列（**RRBS，非 450K**，2018 年版） | 教师 METH（可选）/VAE METH 头（D2/D5） | CCLE（旧资源） |
| ccle | `Cell_lines_annotations_20181226.txt` | 0.3 MB | 1,461 行，非权威 | 旧名称映射 fallback（正式管线用 Model.csv） | CCLE（旧资源） |
| hgnc | `Gene.csv` | 17 MB | 44,083 行（19,291 蛋白编码；`date_modified` 最晚 2025-03-31） | 基因 ID 三层映射（§3.3-1） | HGNC（引用） |
| string | `9606.protein.links.v12.5.txt.gz` | 62.8 MB | v12.5；10,826,054 行 / 5,413,027 唯一无向边 / 19,649 蛋白 | **PPI 主版本**（A_GG，§4.4） | **CC BY 4.0**（仅需署名） |
| string | `9606.protein.aliases.v12.5.txt.gz` | 19.8 MB | v12.5；3,889,207 行 | ENSP↔symbol 映射 | CC BY 4.0 |
| string | `9606.protein.links.v12.0.txt` | 630.9 MB | v12.0（2023） | 旧版本留档/对照 | CC BY 4.0 |
| string | `9606.protein.aliases.v12.0.txt` | 199.7 MB | v12.0 | 旧版本留档/对照 | CC BY 4.0 |
| dgidb | `interactions.tsv` | 19.3 MB | DGIdb v5.0.13，data 2026-10；122,905 条 claim | E11 解释命中率校验 + A_DG 兜底（§4.6） | DGIdb（混合来源，再分发受限，内部研究用） |
| dgidb | `drugs.tsv` / `genes.tsv` / `categories.tsv` | 14.7 MB | 同上（80,483 / 85,738 / 33,890 行） | 药/基因/类别映射 | 同上 |
| drugs | `smiles.tsv` / `smiles.parquet` | ~80 KB | PubChem 2026-11-17；286/286 化合物 | 药物图/Morgan FP/scaffold（§4.7） | PubChem 公共数据；再分发前核对 NCBI 条款 |
| tcga | `CESC/`（1,204 文件） | 6.421 GB | GDC Data Release 46.0；ascat3-only CNV | L1/L2/L3（§3.7、§8） | GDC 开放层（NIH open access） |
| tcga | `clinical/clinical.project-tcga-cesc.2026-11-11.tar.gz` | 363 KB | GDC Portal 2026-11-11 导出；202 列、1,454 行/307 例 | L2 治疗记录/L2′ 协变量（§8.4） | GDC 开放层 |
| tcga | `cdr/TCGA-CDR-SupplementalTableS1.xlsx` | 2.9 MB | PanCan-Clinical-2018；11,160 例/33 癌种 | L2′ 生存终点（OS/PFI/DFI/DSS） | 引用 Liu et al. 2018 |
| tcga | `ding2016/bioinformatics_32_19_2891_s1.zip` | 1.8 MB | Ding, Zu & Gu 2016 补充材料；Table S2=2,572 条响应记录 | **L2 唯一响应标签源**（§8.4a） | 引用原文；标签为 TCGA 自由文本再整理 |
| tcga | `manifests/`（7 个 tsv + 源 manifest + README + 下载命令） | 3.4 MB | GDC 2026-11-11 导出/筛选 | 下载可复现/完整性校验 | — |

> **许可统一说明**：以上许可为当前已知信息；完整许可说明与引用要求待建 `data/licenses/`（计划书 §3.4/§3.8、审计缺口 P0-结构件）。开源仓库只发布派生特征与下载脚本，不打包 STRING/KEGG/DrugBank/TCGA 受控层等受限数据。

---

## 3. 迁移映射（2026-11-14 之前 → 之后）

| 旧路径 | 新路径 |
|---|---|
| `data/drug_sensitivity/GDSC2_fitted_dose_response_27Oct23.csv` | `data/raw/gdsc/GDSC2_fitted_dose_response_27Oct23.csv` |
| `data/cell_line_omics/Model.csv` | `data/raw/depmap/Model.csv` |
| `data/cell_line_omics/DepMap_{Expression…,SomaticMutations,CNGeneWGS}.csv` | `data/raw/depmap/` 同名 |
| `data/cell_line_omics/26Q1_Mutation_Pipeline_Documentation.pdf` | `data/raw/depmap/` |
| `data/cell_line_omics/CCLE_DNA_methylation_TSS1kb.txt` | `data/raw/ccle/` |
| `data/cell_line_omics/Cell_lines_annotations_20181226.txt` | `data/raw/ccle/` |
| `data/cell_line_omics/Gene.csv` | `data/raw/hgnc/Gene.csv` |
| `data/protein_protein_interaction/9606.protein.*` | `data/raw/string/` 同名 |
| `dgi/{interactions,drugs,genes,categories}.tsv` | `data/raw/dgidb/` 同名 |
| 根目录 `clinical.project-tcga-cesc.2026-11-11.tar.gz` | `data/raw/tcga/clinical/` |
| 根目录 `TCGA-CDR-SupplementalTableS1.xlsx` | `data/raw/tcga/cdr/` |
| 根目录 `bioinformatics_32_19_2891_s1.zip` | `data/raw/tcga/ding2016/` |
| 根目录 `gdc_manifest.2026-11-11.112720.txt` | `data/raw/tcga/manifests/` |
| `work/tcga_manifests/*.tsv` + `README.md` | `data/raw/tcga/manifests/` |
| 根目录 `1.txt`（下载命令） | `data/raw/tcga/manifests/download_command.txt` |

未移动（非数据）：`gdc-client`（下载工具，仍在根目录）、`plan/`、`work/`（笔记与文献）、论文 PDF 目录、`压缩文件.zip`。

---

## 4. 已知数据缺陷/使用注意（详见 `work/notes/data_readiness_audit_2026-11-14.md`）

1. **GDSC2 药身份**：按 (细胞系, DRUG_ID) 无重复，但 **9 个化合物各有 2 个 DRUG_ID**（6,288 个名称级重复对），去重必须在化合物层（InChIKey/药名）而非曲线层；295 ID vs 286 药名口径需冻结。
2. **DepMap 版本混用风险**：组学文件无 release 标签（mtime 2026-07-13，疑 26Q1 代），`Model.csv` 为 2026-11-10（疑 26Q3）→ 需定版并记录 MD5（P1）。`SIDM00117` 有 1→2 映射碰撞，用 COSMICID=905967 → ACH-002172 仲裁。
3. **METH 平台**：CCLE 为 **RRBS TSS1kb（2018）**，非计划书的 450K；且 843 列中 10 列整列全空。与 TCGA HM450 跨平台，L1 建议默认只补 CNV。
4. **CNV 格式**：DepMap 为 WGS **线性相对拷贝数**（非 log2，约 4% 缺失）；TCGA ascat3 为**整数总拷贝数**（0–111，1.3% 缺失）——两者需 log2/离散化对齐。
5. **TCGA-CESC 下载结构**：每个文件一个 `<file_uuid>/` 目录（内含数据文件 + `logs/` + `.parcel`；部分含 `annotations.txt`）。`logs/`、`.parcel`、`annotations.txt` 不是数据；**文件名不含 TCGA barcode**，L1/L2/L3 前必须用 GDC API 生成 file_id→case 映射派生表（见审计 P0-5）。CNV/MU 的 `associated_entities` 指向 10A 正常样本，**不能**用于判 sample_type。
6. **TCGA 四模态**：CESC 下载集（ascat3-only）四模态交集 **274 例**（补非 ascat3 CNV 可到 281）；METH 单样本 ~14.9% NA；GE 为 GENCODE v36 versioned Ensembl ID。
7. **Ding 2016 标签 provenance**：TCGA 自由文本治疗结局的再整理（非独立 RECIST 复核）；CESC 顺铂可用 **59 例（62 条）**，不是治疗记录数 128；使用前需标签 QC。
8. **STRING 许可**：v12.5 起为 **CC BY 4.0**（含商用，仅需署名），计划书 §3.4 旧表述已过时。

---

## 5. 待补（本轮未做，见审计缺口清单）

- `data/licenses/`（许可说明/引用要求）、`data/interim/`、`data/processed/`、`data/splits/`（计划书 §3.8；本轮按要求只整理 `raw/`）；
- 缺失数据源：GDSC1、CTRPv2（P1）；TCGA-BRCA/LUAD（P1/P2）；Reactome/GO/Hallmark（P0）；panel 清单（P0）；Open Targets/ChEMBL（P0）；RPPA（可选）等（详见审计笔记 §4）。

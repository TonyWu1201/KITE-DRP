# TCGA 可得性审计（CESC 四模态 / clinical / CDR / Ding2016）

- 审计日期：2026-11-17 ｜ 审计员：audit-tcga（task-4）｜ 环境：`.venv/bin/python`（pandas 3.1.1，**无 openpyxl/xlrd**，xlsx 用 stdlib zipfile+ElementTree 解析）
- 范围：`data/raw/tcga/`（CESC、clinical、cdr、ding2016、manifests）+ `data/raw/README.md` §4 上轮（2026-11-14）结论复核
- 结论先行：**1) README 的 1,204 文件 / 6.421 GB 精确成立；2) MU 模态可离线拿到 287 例患者真值（MAF 数据行含 TCGA barcode）；3) GE/CNV/METH 的患者级映射仍依赖 GDC API file_id→case 派生表（P0 缺口，本机无网络未执行），因此四模态交集只能引用上轮值 274/281 并标注"待复核"；4) Ding2016 的 CESC 顺铂 59 例/62 条实测量化成立，但响应极不平衡（CR+PR=52/62）。**

---

## 1) 盘点与复现命令

```bash
cd /home/tonywu/workspace/KITE-DRP
# 目录数 / 全部文件数 / 体积
ls data/raw/tcga/CESC | wc -l                                   # 1204 个 UUID 目录
find data/raw/tcga/CESC -type f | wc -l                         # 2725（含 .parcel 与 annotations.txt）
du -sh data/raw/tcga/CESC                                       # 6.0G（GiB）≈ 6.421 GB（十进制）
# 分类计数（区分数据 / .parcel / annotations）
for s in 'methylation_array.sesame.level3betas.txt' 'rna_seq.augmented_star_gene_counts.tsv' \
         'wxs.aliquot_ensemble_masked.maf.gz' 'ascat3.gene_level_copy_number.v36.tsv' 'annotations.txt'; do
  printf '%-58s %s\n' "$s" "$(find data/raw/tcga/CESC -name "*$s" | wc -l)"; done
find data/raw/tcga/CESC -name '*.parcel' | wc -l                 # 1204
# manifest 联表 + 体积：id 与磁盘目录名 100% 一致
.venv/bin/python - <<'PY'
import csv,os
base='data/raw/tcga/CESC'
man={r['id']:r for r in csv.DictReader(open('data/raw/tcga/manifests/manifest_cesc_4modalities.tsv'),delimiter='\t')}
dirs=set(os.listdir(base)); print(len(man), len(dirs), len(dirs-set(man)), len(set(man)-dirs))
print('sum size =', sum(int(r['size']) for r in man.values()))  # 6421376002 B = 6.421 GB
PY
# 解压（/tmp 在本会话内跨 bash 调用不保留，必须与解析写在同一条命令里）
mkdir -p /tmp/ta && tar -xzf data/raw/tcga/clinical/clinical.project-tcga-cesc.2026-11-11.tar.gz -C /tmp/ta
mkdir -p /tmp/ding && unzip -o -q data/raw/tcga/ding2016/bioinformatics_32_19_2891_s1.zip -d /tmp/ding
```

实测盘点（**关键口径**）：

| 类别 | 数量 | 说明 |
|---|---|---|
| UUID 目录 | 1,204 | 每个目录 = 1 个 GDC file；目录名 = manifest `id`（file UUID），**不等于 barcode** |
| 数据文件 | **1,204** | GE 309 + MU 289 + CNV 294 + METH 312；README 的 1,204 指此数 |
| `.parcel` | 1,204 | 全部在 `<uuid>/logs/` 下；**二进制 pickle**（intervaltree + 分块 md5），非文本、无 case 信息 |
| `annotations.txt` | 317 | GDC annotation 通知（entity_type=case/aliquot/analyte/masked_somatic_mutation），**不含 TCGA barcode** |
| `find -type f` 合计 | 2,725 | = 1,204 + 1,204 + 317，与 README 不矛盾；直接用 `find` 计数会高估 2.26 倍 |
| manifest 体积和 | 6,421,376,002 B | 6.421 GB 十进制 / 5.98 GiB，与 README"6.421 GB"逐字节一致 |

各模态文件数与大小（manifest 实测）：

| 模态 | 后缀 | 文件数 | 大小 | RPPA |
|---|---|---|---|---|
| GE（RNA-seq） | `*.rna_seq.augmented_star_gene_counts.tsv` | 309 | 1.309 GB | — |
| MU（体细胞突变） | `*.wxs.aliquot_ensemble_masked.maf.gz` | 289 | 0.023 GB | — |
| CNV（ascat3 gene-level） | `*.ascat3.gene_level_copy_number.v36.tsv` | 294 | 1.013 GB | — |
| METH（HM450 β） | `*.methylation_array.sesame.level3betas.txt` | 312 | 4.077 GB | — |
| 合计 | | **1,204** | **6.421 GB** | `manifest_cesc_RPPA_optional.tsv` 有 172 文件清单，**未下载**（盘上 0 个） |

---

## 2) 各模态患者覆盖与交集矩阵（最关键输出）

### 2.1 可直接离线推导的部分

- **MU（唯一可实测患者级）**：289 个 MAF 文件 → **287 例唯一 participant**（TCGA-XX-XXXX）。
  - 每个 MAF 恰含 1 个 tumor barcode + 1 个 normal barcode；2 例患者各 2 个 MAF 文件（`TCGA-UC-A7PG`、`TCGA-HM-A6W2`）。
  - tumor 样本类型：`01A`×281、`01B`×6、`06A`×2（=转移灶，L3 需注意）；matched normal：`10A`×267、`10B`×17、`11A`×3、`11C`×2。
  - MAF 数据行含 barcode 与 `case_id`（UUID）；`case_id` 唯一数 = 287，与 barcode 去重一致。
- **clinical**：`clinical.tsv` 307 例（`cases.submitter_id` 唯一；`cases.case_id` UUID 唯一 307）。
- **交集实测**：`MU ∩ clinical = 287`（MU 全部在 clinical 中）；clinical 有 **20 例无 MU**（C5-A1BQ/A2LZ/A3HD/A3HE/A3HL、DS-A1O9/A1OC、EA-A44S/A4BA/A50E/A556、EK-A2IR/A3GJ、FU-A57G、HM-A4S6、IR-A3LC、MU-A51Y、MY-A5BD/A5BE/A5BF）。
- **CDR**：CESC 307 例，与 clinical **307/307** 完全对齐；`CDR ∩ MU = 287`。

### 2.2 交集矩阵（现状：部分单元不可离线计算）

| | MU(287) | GE(?≤309) | CNV(?≤294) | METH(?≤312) | clinical(307) | CDR(307) |
|---|---|---|---|---|---|---|
| **MU** | — | 待映射表 | 待映射表 | 待映射表 | **287** | **287** |
| **GE** | | — | 待映射表 | 待映射表 | 待映射表 | 待映射表 |
| **CNV** | | | — | 待映射表 | 待映射表 | 待映射表 |
| **METH** | | | | — | 待映射表 | 待映射表 |
| **clinical** | | | | | — | **307** |

- 上轮审计值（**前次审计值，待 file_id→case 映射表重建后复核**）：四模态患者交集 **274 例**；补非 ascat3 CNV 后 **281 例**。
- 真值上界：GE 309、CNV 294、METH 312 个文件，患者数 ≤ 上述文件数、且 ≤ 307（CESC 总病例数，clinical/CDR 一致）；MU 实测 287，落在 274–287 的合理区间，与"274 交集"不矛盾。

### 2.3 为什么本地算不出 GE/CNV/METH 的患者数（P0 缺口）

- 文件名只有 file UUID；manifest 仅 5 列（id/filename/md5/size/state）；`.parcel` 是下载分块 pickle；`annotations.txt` 不含 barcode。**四者都不能给 barcode**。
- 唯一可用的 GDC 派生映射是 `file_id → case_id`，需调用 GDC API（`/files?filters=...&expand=associated_entities` 或 `/files/{id}`），**本机无网络，未执行**。
- **部分映射证据（方法学已验证，但样本太少）**：317 个 `annotations.txt` 中 29 个含 case 级 annotation（entity_type=case，entity_id=case UUID）。用 clinical 的 `cases.case_id→cases.submitter_id` 联表 **29/29 命中**；其中 7 个 MAF 目录的 annotation case UUID 与 MAF 数据行 `case_id` **7/7 一致**，证明"annotation case_id = 该目录文件的 case"。这 29 个目录仅覆盖 **8 个 case UUID**（GE 7、CNV 7、METH 8、MU 7 个文件），来自有 QC 标注的病例，**不足以重建 274 矩阵**。
- 复现（需联网，本机跳过）：
```bash
# 1) 拉 file→case 映射（UUID→barcode）并存派生表，冻结 md5
curl -s 'https://api.gdc.cancer.gov/files?filters={"op":"in","content":{"field":"file_id","value":[...]}}&fields=file_id,file_name,cases.case_id,cases.submitter_id,associated_entities.entity_submitter_id,associated_entities.entity_type,data_type,platform&size=1204&format=JSON'
# 2) 用 MAF 的 287 例 barcode 校验映射表；3) 再算四模态交集
```

---

## 3) 各模态内容 schema（实测抽 1–2 个文件）

### 3.1 GE（RNA-seq STAR Counts，309 文件）
- **两行表头**：第 1 行 `# gene-model: GENCODE v36`（必须 `skiprows=1`）；第 2 行为列名。
- 9 列：`gene_id, gene_name, gene_type, unstranded, stranded_first, stranded_second, tpm_unstranded, fpkm_unstranded, fpkm_uq_unstranded`。
- 60,666 行 = 60,664 数据行（4 行 `N_*` 汇总行：`N_unmapped/N_multimapping/N_noFeature/N_ambiguous` + 60,660 基因）。**309 个文件行数完全一致**。
- `gene_id` 为 **GENCODE v36 带版本 Ensembl**（如 `ENSG00000000003.15`，60,660/60,660 匹配 `ENSG\d+\.\d+`）；`gene_name` 为 HGNC symbol（4 个 N_ 行为空）。
- 数值口径：`unstranded/stranded_*` 为**整数原始 read count**（未标准化，实测全整数）；`tpm_unstranded`、`fpkm_unstranded`、`fpkm_uq_unstranded` 为浮点（TPM 均值 ~16.5，最大 18,982；4 个 N_ 行 TPM 为空）。`gene_type`：protein_coding 19,962 / lncRNA 16,901 / processed_pseudogene 10,167…
- **陷阱**：基因矩阵含 4 个非基因 `N_*` 行；TPM 的 `tpm_unstranded` 是 unstranded 口径（与 stranded 库不一致时需说明）。

### 3.2 MU（MAF，289 文件）
- 140 列；头部注释 7 行：`#version gdc-1.0.0`、`#annotation.spec gdc-2.0.0-aliquot-merged-masked`、`#contigs`、`#sort.order BarcodesAndCoordinate`、`#filedate`、`#normal.aliquot <UUID>`、`#tumor.aliquot <UUID>`；第 8 行为表头。
- 关键列：`Hugo_Symbol, Entrez_Gene_Id, Variant_Classification, Variant_Type, Chromosome, Start_Position, dbSNP_RS, Tumor_Sample_Barcode, Matched_Norm_Sample_Barcode, Tumor_Sample_UUID, Matched_Norm_Sample_UUID, HGVSc/HGVSp/HGVSp_Short, t_depth/t_ref_count/t_alt_count, n_depth/..., One_Consequence, Consequence, SYMBOL, HGNC_ID, BIOTYPE, **IMPACT（=VEP impact，不叫 VepImpact）**, SIFT, PolyPhen, gnomAD_AF, CLIN_SIG, SOMATIC, **case_id（UUID）**, GDC_FILTER, COSMIC, hotspot, RNA_Support/RNA_depth/RNA_ref_count/RNA_alt_count, callers`。
- 全量 289 文件合计 **68,893 条突变**；`Hugo_Symbol` 非空率 **100%**；`IMPACT`：MODERATE 43,337 / LOW 17,617 / HIGH 6,350 / MODIFIER 1,589；`Variant_Classification` top：Missense 43,155、Silent 17,183、Nonsense 3,862、Frame_Shift_Del 1,192、Splice_Site 770、Intron 659。
- 每文件突变数：min 2、median 103、max 13,729（存在超突变样本，**L2/L3 特征需 log 变换或按 TMB 分层**）。
- **`Tumor_Sample_Barcode` 是完整 TCGA aliquot barcode**（`TCGA-C5-A7CM-01A-11D-A33O-09`），由此可判 sample type；`Matched_Norm_Sample_Barcode` 为 10A/10B/11A/11C。**这是唯一可离线做患者映射的模态。**

### 3.3 CNV（ASCAT3 gene-level，294 文件）
- 8 列：`gene_id, gene_name, chromosome, start, end, copy_number, min_copy_number, max_copy_number`；**无 barcode**。
- 60,623 基因行（**294 文件行数完全一致**）；`gene_id` 为 versioned Ensembl，`gene_name` 为 symbol，chromosome 为 `chr1..chr22,chrX,chrY`（24 种）。
- `copy_number` 为**整数总拷贝数**（ASCAT3，非 log2、非相对值）；全库实测值域 **0–111**，缺失为空字段，**缺失率 1.33%**（逐文件 1.32–1.43%，几乎固定，为无编码基因无 CN 值所致）。
- 每文件 `copy_number==2` 基因占比中位 64.2%（range 宽）；**20/294 文件的 `==2` 占比 >90%（正常样/极低肿瘤纯度）→ 用前需按纯度/CN 分布 QC**；非编码基因 `copy_number` 为空（不可当 0）。
- README 旧称"0–104"与实测 **0–111** 略有差异（见 §7）。

### 3.4 METH（Illumina HM450，sesame level3 betas，312 文件）
- **平台实测：Illumina Human Methylation 450（HM450）**，文件名 `methylation_array.sesame.level3betas.txt`；**无表头、无 barcode、单样本单文件**。
- 486,427 行 = 486,427 个探针（**312 文件行数完全一致**）：`cg` 482,421 / `ch` 3,091 / `ct` 850 / `rs` 65；两列 `probe<TAB>beta`（无列名）。
- beta 值域 [0,1]（实测 min≈0.005、max≈0.994）。
- **缺失实测（全 312 文件扫描，非抽样）：缺失率 min 13.62% / 中位 14.77% / 均值 15.44% / max 21.80%；全库 2343 万 / 1.52 亿 单元格缺失**。上轮"~14.9%"基本成立（中位口径）。
- **与细胞系 CCLE 的跨平台问题**：CCLE 甲基化是 RRBS TSS1kb（2018，843 列，且 10 列全空），与 TCGA HM450 探针/dimension 不可直接对齐，**L1 默认不建议把 METH 作为补全目标**（沿用上轮结论）。

### 3.5 RPPA
- 仅有 manifest（172 文件 / 0.004 GB），**数据未下载**；本轮不评估。

---

## 4) 与细胞系域（DepMap）基因空间对接

- **GE**：DepMap `DepMap_ExpressionTPMLogp1HumanProteinCodingGenes.csv` 为 1,775 行 × **19,221 列**（6 元数据列 + **19,215 基因列**，列名格式 `SYMBOL (ENTREZ)`，如 `TSPAN6 (7105)`；值为 TPM logp1）。
  - GDC RNA 的 protein_coding symbol 唯一数 19,938，与 DepMap 基因 symbol 交集 **18,818（94.4%）**；DepMap 有 397 个 symbol 不在 GDC RNA（版本/命名差异）→ **映射路线：`gene_id` 去版本 → HGNC `ensembl_gene_id→symbol`（HGNC 44,083 行可覆盖 41,357/60,664 全基因，其中 symbol 与 GDC gene_name 一致率 92.5%）→ 再对 DepMap `SYMBOL (ENTREZ)` 按 symbol+Entrez 双键匹配**。建议以 HGNC 与 DepMap 共同的 protein-coding 基因集（≈19.2k）为冻结基因空间，GE 用 TPM（或 log1p），与 DepMap log TPM 做分位数/秩变换对齐。
- **CNV**：DepMap `DepMap_CNGeneWGS.csv` 为 19,955 基因列（19,366 个 `SYMBOL (ENTREZ)` + 589 个 `ENSG..._PAR_Y` 列）；GDC CNV 60,623 行（全 biotype），其中 **19,013 个 gene_name 命中 DepMap CNV symbol**（≈全部 protein-coding 基因）；GDC 为整数总 CN（0–111），DepMap 为线性相对 CN（≈4% 缺失）→ **对齐：GDC `log2(max(copy_number,0.5)/2)`，DepMap 可直接 log2**；两者都需限制到共同基因集并处理 PAR_Y。
- **MU**：DepMap 突变列含 `HugoSymbol/EnsemblGeneID/VepImpact/VariantType/ProteinChange`，与 MAF 的 `Hugo_Symbol/IMPACT/Variant_Classification/HGVSp_Short` 可映射；建议统一到 HugoSymbol + VepImpact 二值（HIGH/MODERATE）矩阵。
- **陷阱**：GDC `gene_id` 版本号（GENCODE v36 = Ensembl 98）与 DepMap 当前注释不同，**必须去版本后经 HGNC/Entrez 双键映射**，不要直接按字符串 `gene_id` 合并。

---

## 5) clinical / TCGA-CDR / Ding2016 关键字段结论

### 5.1 clinical（GDC Portal 导出，2026-11-11）
- tar.gz 内含 5 表：`clinical.tsv`（**202 列 × 1,454 行**；README 称 207 列，实测差 5，见 §7）、`follow_up.tsv`、`pathology_detail.tsv`、`exposure.tsv`、`family_history.tsv`；**无独立 treatment.tsv**，治疗字段以 `treatments.*` 扁平列并入 `clinical.tsv`（共 **43 个**）。
- 结构：307 cases，**1,454 行 = 1,398 个唯一 `treatments.treatment_id` + 56 行无治疗记录**；一行 = 一条治疗记录（非一例一行），按 `cases.submitter_id` 聚合需去重。
- 关键字段真实（非 `'--'`/Not Reported/Unknown 占位）覆盖：
  - `treatments.therapeutic_agents`：**288 行 / 144 例**（65% 缺失，**字段稀疏，不能当"全治疗史"**）；精确药名 top：Cisplatin 128 例、Carboplatin 18、Paclitaxel 16、Fluorouracil 8、Gemcitabine 3+3、Bevacizumab 3…
  - `treatments.treatment_dose` 414 行/171 例；`number_of_cycles` 100/82；`prescribed_dose` 54/39；`days_to_treatment_start/end` 470/177、425/167 例。
  - `treatments.treatment_outcome`：**532 行 / 127 例**（CR 382、PD 85、PR 42、SD 21、Ongoing 2）→ 可作 L2 响应标签候选，但需按治疗记录→药物去卷积。
  - `treatments.treatment_effect`、`drug_category`、`best_overall_response` **全为 `'--'` 占位符（不可用）**；`diagnoses.progression_or_recurrence/last_known_disease_status/days_to_recurrence` 同样全占位。
  - 人口学/分期可用：`demographic.sex_at_birth` 全部 female、`age_at_index`/`age_at_diagnosis`、`ajcc_pathologic_stage`、`figo_stage`、`primary_diagnosis`、`tumor_grade`、`prior_malignancy/prior_treatment`、`vital_status`。
- **交集**：307 例，与 MU 交 **287**；与 CDR 交 307。

### 5.2 TCGA-CDR（PanCan-Clinical-2018，SupplementalTableS1.xlsx）
- 主 sheet `TCGA-CDR`：**11,160 例 / 33 癌种**，34 列（含 `bcr_patient_barcode,type,age_at_initial_pathologic_diagnosis,gender,race,ajcc_pathologic_tumor_stage,vital_status,tumor_status,last_contact_days_to,death_days_to,cause_of_death,new_tumor_event_*,treatment_outcome_first_course,margin_status,residual_tumor, OS, OS.time, DSS, DSS.time, DFI, DFI.time, PFI, PFI.time, Redaction`）。
- **CESC：307 行**，barcode 与 clinical **307/307 完全一致**，`∩MU = 287`。
- CESC 可用性（非空/事件数）：`OS` 307（事件 71）、`PFI` 307（事件 71）、`DSS` 303（事件 54）、**`DFI` 仅 176/307（事件 26；43% 不可用）**；`tumor_status` 281 非空（TUMOR FREE 200 / WITH TUMOR 80）；`treatment_outcome_first_course` 241 非空。
- 实测 CESC 中 OS 与 PFI 完全一致 **243/307**（两列高度相关，L2′ 不要当独立终点重复使用）；另有 `ExtraEndpoints` sheet 提供 PFI.1/PFS/DSS_cr 等补充终点（11,160 行）；`TCGA-CDR_Notes` 明确推荐：**PFI 优先、OS 次之；DFI 由 treatment_outcome_first_course+residual_tumor+margin_status 派生**。
- 建议 L2′ 口径：主终点 `PFI/PFI.time`（307 全覆盖），敏感性分析 `OS`；`DFI` 只做子集分析（n=176）。

### 5.3 Ding2016（`bioinformatics_32_19_2891_s1.zip`）
- zip 内容 2 文件：`bioinfo16_supplementary_tables.xlsx`（233 KB，9 sheets）+ `bioinfo16_supplementary_materials.docx`（1.6 MB）。
- 表结构：`Summary / Table S1（363 行药名标准化映射：Recorded Name→DrugBank ID→Standard drug name）/ S2（响应记录）/ S3（同药不一致响应）/ S4（按 drug-cancer pair 的分子数据集 NR/R 计数）/ S5–S8（建模与验证）`。
- **Table S2**：标题+表头+CDE_ID 行之后有 **2,572 条数据行**（README 称 2,573，差 1）；有表头 10 列：`Cancer(全称标签), bcr_patient_barcode, drug_name, DrugBank ID, measure_of_response, days_to_drug_therapy_start, days_to_drug_therapy_end, days_to_initial_pathologic_diagnosis, method_of_sample_procurement, days_to_sample_procurement`，另有 4 个无表头补充列（第 11–14 列，NA/Yes/No 类）。**无剂量/周期列**。
- 28 个癌种，**CESC 用全称标签** `Cervical squamous cell carcinoma and endocervical adenocarcinoma (CESC)`（按缩写 `CESC` 过滤会得 0 行，这是易踩坑）。
- **CESC 实测**：97 条记录 / **69 例患者** / 12 种药；**Cisplatin = 62 条记录 / 59 例患者**（与上轮"59 例 62 条"完全一致 ✓）；响应分布 **CR 47、Clinical PD 8、PR 5、SD 2**（严格响应者 CR+PR=52/62，**严重不平衡**）。CESC 其他药均 <10 例（Paclitaxel 9、Carboplatin 9、Fluorouracil 4、Bleomycin 3…）。69 个 barcode **全部**能在 clinical 找到（69/69）。
- **Table S4 交叉验证 L2 样本量**：cisplatin-CESC 有分子数据的 NR/R 计数示例如 CNA 9/49、methylation 3/32、miRNA 9/50、mRNA 9/…，即**可建模的 CIS-CESC 响应者约 49–50 例**。
- **L2 可用性判断**：CESC 只有 **Cisplatin** 达到计划书 "n≥30" 门槛（59 例；严格 responder 52 条记录），其余癌种-药物均 <30 → **只做机制描述**。标签为 TCGA 自由文本治疗结局再整理（非独立 RECIST 复核），无剂量、无周期，使用前需与 clinical `treatment_outcome` 交叉 QC（clinical 中 Cisplatin 128 例，与 Ding 59 例相差 69 例，**不能混用**）。

---

## 6) 关键缺陷与陷阱（供 L1/L2/L2′/L3 预处理引用）

1. **P0（阻断）**：**file_id→case 映射缺失**。CESC 数据仅按 file UUID 落盘；文件名/5 列 manifest/`.parcel`(pickle)/`annotations.txt` 均不含 barcode（`annotations.txt` 只有 UUID），四模态交集与逐样本表型必须调用 GDC API 生成派生表（本机无网络未执行）。上轮 274/281 只能作"前次审计值"引用。
2. **模态计数口径**：`find -type f` = 2,725 ≠ 1,204；数据文件 = 1,204（GE 309/MU 289/CNV 294/METH 312）+ `.parcel` 1,204 + annotations 317。任何"文件数"结论必须说明口径。
3. **RA 缺失/N_ 行**：GE 前 4 行 `N_*` 是 QC 汇总，必须先剔除再做基因×样本矩阵；TPM 为 unstranded 口径。
4. **METH 缺失率高**：单样本中位 14.8%/均值 15.4% 探针为 NA（最大 21.8%），且**无表头/无 barcode**；与 CCLE RRBS 跨平台不可直接补全。
5. **CNV 非 log2**：ASCAT3 为整数总拷贝数（0–111，1.33% 空），DepMap 为线性相对 CN；且 **20/294 文件 `CN==2` 占比 >90%**（低纯度/正常样），用前必须 QC。
6. **MU 样本类型混杂**：287 例中 6 例 01B、**2 例 06A（转移灶）**；判"原发肿瘤"需过滤 `-01A/-01B`。
7. **clinical 稀疏与占位符**：`treatments.*` 43 列中仅 6 列有一定覆盖；`'--'` 是**占位符而非 NA**（pandas 默认会读成字符串，需显式 map 为缺失）；`best_overall_response/treatment_effect/drug_category` 全占位，不可用；一行=一条治疗记录而非一例。
8. **Ding 标签**：CESC 类标签用全称而非缩写；响应 CR 占 47/62（84%），正负样本极不平衡；无剂量/周期；`days_to_drug_therapy_start` 是相对"initial_pathologic_diagnosis"的天数（基线=0），可与 clinical 时间轴对齐但需检查负值。
9. **CDR**：CESC `DFI` 仅 176/307；OS 与 PFI 在 243/307 例完全一致，不能当两个独立终点；`Redaction` 列需检查。
10. **旧 README 数字偏差**：clinical 列数 202（README 207）；Ding S2 2,572 行（README 2,573）；CNV 值域 0–111（README 0–104）；MU/CNV `associated_entities` 指向 10A 正常样本的说法**本轮离线无法复核**（待 API）。
11. **RPPA 未下载**（manifest 172 文件），如后续要蛋白模态需另下。
12. **/tmp 不跨 bash 调用保留**（本会话沙箱特性）：解压与解析必须写在同一命令内；复现脚本需注意。

---

## 7) 对 TCGA 预处理的影响与建议口径

| 用途 | 建议口径（本轮审计支撑） |
|---|---|
| **L1 组学补全（GE↔CNV）** | 先建 file→case 派生表（P0），取四模态交集（前次 274，**待复核**）作为可评估集；GE 用 `tpm_unstranded`（剔除 `N_*`，Ensembl 去版本经 HGNC→symbol 对齐 DepMap 19.2k 基因空间）；CNV 用 `copy_number`→`log2(max(cn,0.5)/2)`，先剔除"正常样"文件（`==2`>90% 的 20 个）；**不建议默认补 METH**（跨平台 RRBS/HM450 + 15% 缺失）。 |
| **L2 弱标签** | 唯一可用药-癌种对为 **Cisplatin-CESC（59 例/62 条，严格响应者 52 条）**，用 Ding Table S2 + Table S4 的 NR/R 口径；其余药 <30 例 → 只做机制/描述；标签用前与 clinical `treatment_outcome`（127 例）交叉 QC，**不要混用** clinical 的 128 例 Cisplatin（含无响应标签者）。 |
| **L2′ 生存** | CDR 主表直接可用：`PFI/PFI.time`（307）为主、`OS`（307）为敏感性；`DSS` 303、`DFI` 仅 176（子集）；CDR 与 clinical 307/307 对齐，可按 barcode 直接 join（无需 file 映射）。 |
| **L3 域适应** | TCGA 侧可评估肿瘤样本：MU 287 例（含 2 例 06A 转移灶，需单独标记）；GE/CNV/METH 的实际患者数待映射表；normal 对照（10A/10B/11A/11C）仅来自 MAF matched-normal 字段，样本级正常组学文件是否存在依赖映射表（CNV/METH 文件数 294/312 与 307 例的差额需 API 判定）。 |
| **派生表（P0 建议冻结）** | `file_uuid, modality, data_type, platform, case_id(uuid), case_submitter_id(barcode), sample_type, aliquot_uuid`；用 MAF 的 **287 例 barcode 真值**校验；建表命令与 md5 写入 `manifests/` 旁 README。 |

**需联网处理的步骤（本机未执行，单列）**：
1. GDC API 拉 1,204 个 file 的 `associated_entities`（case_id/case_submitter_id/sample_type/aliquot）→ 生成 `file_id→case` 映射表（可分批 100 个/请求）；
2. 复核 MU/CNV 的 `associated_entities` 是否指向 10A 正常样本（决定 sample_type 判定口径）；
3. 拉取 CESC 全部已发布 CNV（非 ascat3 套）以复现"281"口径；
4. 复核上轮"274/281"交集数字；
5. 如需 RPPA（172 文件）再补下载。

---

## 8) 待核实项

1. 四模态交集 **274 / 281**（前次审计值）——需 file_id→case 映射表重建后逐例复核；本审计只验证了 MU=287 与 MU∩clinical=287 的相容性。
2. GE 309 / CNV 294 / METH 312 文件对应的**唯一患者数**与样本类型分布（需 API 映射表）。
3. METH 312 文件 > 307 例：多出的 5 个样本是重复 aliquot、正常样本还是转移灶？（需 API/映射表）。
4. CNV 294 文件 vs 307 例：缺 13 例还是含正常样本？（同上）。
5. README 数字偏差：clinical 202 vs 207 列、Ding S2 2,572 vs 2,573 行、CNV 0–111 vs 0–104 — 建议修 README。
6. 20 个"正常样 CNV 文件"（`copy_number==2`>90%）的样本性质与是否应排除出 L1/L3。
7. MAF 超突变样本（max 13,729 突变/文件）对应的 barcode 与 TMB 分层策略。
8. clinical `treatments.treatment_outcome` 与 Ding `measure_of_response` 的逐例一致性（127 例 vs 69 例 CESC），决定 L2 标签源唯一性。
9. 是否需要 RPPA（manifest 172 文件已列、未下载）。

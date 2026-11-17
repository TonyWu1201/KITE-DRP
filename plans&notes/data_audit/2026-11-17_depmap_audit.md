# DepMap 组学（Model/表达/突变/CNV）与 GDSC 细胞系匹配审计

- 审计人：audit-depmap（task-2）｜日期：2026-11-17｜环境：`.venv/bin/python` + pandas 3.1.1 / numpy 2.5.4
- 范围：`data/raw/depmap/` 四文件只读实测 + `data/raw/gdsc/GDSC2_fitted_dose_response_27Oct23.csv` 标签侧；未修改任何既有文件
- 结论先行（最关键数字）：**有 GDSC2 药敏标签、且 GE/MU/CNV 三模态齐全（default entry 过滤后）的细胞系 = 509 株**（GDSC 可映射 968 株的 52.6%）；三模态组学样本池 = 1,105 个 ModelID。计划书 §3.2 的第三方转述数字未采用，全部为本次实测。

## 1. 文件与方法

### 1.1 文件清单（MD5 经 `CHECKSUMS.md5` 逐文件校验，全部 OK）

| 文件（`data/raw/depmap/`） | 大小/字节 | mtime | MD5 | 行（含表头） |
|---|---|---|---|---|
| `Model.csv` | 697,455 | 2026-11-10 | a15d75dffcc5219111ca39598948df9a | 2,155（2,154 行数据 × 49 列） |
| `DepMap_ExpressionTPMLogp1HumanProteinCodingGenes.csv` | 305,007,605 | 2026-07-13 | 559245dfb8dc496a8e30bba4c97d9b25 | 1,776（1,775 行 × 19,221 列） |
| `DepMap_SomaticMutations.csv` | 580,755,338 | 2026-07-13 | d5b0b87ba8852c50da54ba1442b7518f | 1,172,689（1,172,688 行 × 68 列） |
| `DepMap_CNGeneWGS.csv` | 409,005,582 | 2026-07-13 | 96a13bf6531b7140ce4b8188d7d73bc6 | 1,133（1,132 行 × 19,961 列） |
| `26Q1_Mutation_Pipeline_Documentation.pdf` | 218,876 | 2026-07-16 | 08a18e8154876f621572e4eb371d9983 | —（突变口径依据，已 `pdftotext` 抽取） |

### 1.2 方法要点

- 宽表（表达/CNV）只读元数据列 + 抽样/全量基因列；突变长表用 `usecols` 只取所需列；三者均未整表加载非必要列。
- 列数用 `head -1 | tr ',' '\n' | wc -l` 先探；再按 0-based 位置切基因列。**注意 CSV 首列无列名**（见陷阱 T2）。
- 匹配口径：DepMap `Model.csv.SangerModelID` ↔ GDSC `SANGER_MODEL_ID` 为主键；名字兜底用 `norm=re.sub('[^A-Za-z0-9]','',s).upper()` 对 `CellLineName`/`StrippedCellLineName`。
- 全流程只读；本笔记为唯一写出物。

## 2. 实测数字表

### 2.1 Model.csv（2,154 行 × 49 列，非"约 50 列"）

| 项目 | 实测值 | 备注 |
|---|---|---|
| ModelID（ACH-######） | 2,154 唯一，0 重复 | 全部符合 `^ACH-\d{6}$` |
| ModelType | Cell Line 2,130 / Organoid 24 | |
| SourceType | 缺失 36；ATCC 550 / DSMZ 327 / Wellcome Sanger 243 / RIKEN 113 / HSRRB 113 / KCLB 82 / Sigma-Aldrich 81 / JCRB 68 / CCLF 68 / ECACC 63 / Academic lab 50 / DFCI 45 | 来源批次可作协变量/QC |
| SangerModelID | **934 缺失（43.4%）**；非空 1,220 → 1,219 唯一，**1 个一对多** | 仅 `SIDM00117` 一对多（见 2.3） |
| COSMICID | 1,177 缺失（54.6%）；非空 977 唯一，0 重复 | 可作仲裁但覆盖不足 |
| CCLEName | 124 缺失；与 `StrippedCellLineName` **2,154/2,154 全部不同**；归一化后仅 14 行相等 | CCLEName 是旧 CCLE `NAME_TISSUE` 格式，不可当纯名字用 |
| OncotreeLineage | 34 类，13 空 + 13 `Normal` | top：Lymphoid 266/Lung 261/Skin 150/CNS 127/…；`Fibroblast` 45 等非肿瘤系需剔除 |
| Sex / Age / AgeCategory | Male 1,143 / Female 869 / Unknown 119 / 空 23；Age 空 434（其余均数值）；Adult 1,480 / Pediatric 291 / Unknown 371 / Fetus 12 | |
| ModelAvailableInDbgap | 空 420；Approved-CCLE 1,094 / HMB-MDS 548 / In Progress 83 / GRU 7 / DS-CA-MDS 2 | 与公开可用性无关，勿当过滤条件 |
| EngineeredModel | 空 2,133；drug adapted 8 / engineered clone 7 / KO 5 / KD 1 | 21 个工程化模型建议标记 |

### 2.2 GDSC2 标签侧（`GDSC2_fitted_dose_response_27Oct23.csv`）

242,036 行 / **969 个 SANGER_MODEL_ID（且 CELL_LINE_NAME 亦 969 唯一）** / 295 个 DRUG_ID（286 个药名）；`LN_IC50`、`AUC` 均 0 缺失（969 株全部至少 1 条标签）。

### 2.3 GDSC2 ↔ Model.csv 匹配率（实测）

| 口径 | 命中 | 覆盖率 | 说明 |
|---|---|---|---|
| 仅 SangerModelID | 967/969 个 SID（968 个 ModelID 行） | 99.79% | `SIDM00117` 1→2 行，故行数 968 > SID 数 967 |
| 仅名字（归一化 Stripped） | 941/969 | 97.11% | 归一化后 GDSC 969 名塌缩为 967 唯一（`KM-H2`/`KMH-2`→KMH2、`TT`/`T-T`→TT 两组碰撞） |
| ID∪名字兜底 | **968/969** | **99.90%** | 唯一失败：`RH-1`（SIDM00427），无同名 Model 行 |
| 名字兜底救回（无 SID 行） | 1 株：`NCI-H748`（SIDM01117）→ ACH-003071 | — | |

- **全 Model.csv 只有 1 个一对多 SangerModelID**：`SIDM00117` → ACH-000837（NCI-H322，COSMICID 空）/ ACH-002172（NCI-H322M，COSMICID=905967）。GDSC 侧名字 = `NCI-H322M` → **仲裁到 ACH-002172**（名字 + COSMICID 双重一致）。**两个候选都在三模态池中，故主数字 509 不受仲裁影响**。
- ID 命中但名字不直接相等的 27 株为别名（如 `OVCAR-3`→`NIH:OVCAR-3`、`U-266`→`U266B1`）；**ID↔名字冲突 = 0**（不存在 ID 指一个模型、名字指另一个模型的情况），故 SangerModelID 可作主键。
- 名字兜底歧义：5 株 GDSC 线名归一化后匹配到 >1 个 Model 行（如 `MS-1`→ACH-001131/MS1 与 ACH-001958/MSDASH1；`KM-H2`→ACH-000815/KMH2 与 ACH-002397/KMHDASH2），名字兜底必须人工核对。

### 2.4 组学三文件：行数 / default 过滤 / 覆盖

| 模态 | 原始行 | 唯一 ModelID | `IsDefaultEntryForModel=Yes` 后 | 覆盖 Model.csv | 基因列 | 备注 |
|---|---|---|---|---|---|---|
| GE 表达 | 1,775 | 1,719（另 56 行非默认；54 个模型多行，最多 3 行） | **1,719 行/模型，唯一** | 1,719/2,154 = 79.8% | 19,215 | `IsDefaultEntryForMC` Yes 1,720（比模型级多 1，见 T4） |
| MU 突变 | 1,172,688 | 1,968（2,581 个 MC） | **736,094 行（62.77%）→ 1,968 模型，每模型恰 1 个默认 MC，唯一** | 1,968/2,154 = 91.4% | —（长表） | 0 个模型无 default；`IsDefaultEntryForModel` 可安全作唯一过滤键 |
| CNV | 1,132 | 1,118（14 行重复 ModelID，同 MC 双 SequencingID） | **1,118 行/模型，唯一** | 1,118/2,154 = 51.9% | 19,955 | 14 个重复模型各有一行 `No` 一行 `Yes`，default 过滤即解决 |
| 三模态（default 后） | — | — | **1,105** | 1,105/2,154 = 51.3% | — | 非默认行不影响交集结果（原始集合交集同为 1,105） |

- 三文件 ModelID 全部 ⊆ Model.csv（0 个孤儿 ID）→ 版本窗口内 ID 一致；但 Model.csv 有 186 个模型在 MU 中不存在、435 个不在 GE、1,036 个不在 CNV。
- 交集矩阵（|A∩B|，default 后）：GE∩MU = **1,643**；GE∩CNV = **1,105**；MU∩CNV = **1,118**（= CNV 全集）；**GE∩MU∩CNV = 1,105**。CNV 是唯一瓶颈模态。

### 2.5 最关键：GDSC2 × 三模态（default 后）

| 集合 | 细胞系数 | 占 GDSC 可映射 968 |
|---|---|---|
| GDSC ∩ GE | 717 | 74.1% |
| GDSC ∩ MU | 959 | 99.1% |
| GDSC ∩ CNV | 513 | 53.0% |
| GDSC ∩ GE ∩ MU（不含 CNV） | **716** | 74.0% |
| **GDSC ∩ GE ∩ MU ∩ CNV（主训练可用）** | **509** | **52.6%** |
| 三模态之外 | 仅 GE & MU 207 株；仅 MU & CNV 4 株；仅 GE 1 株；仅 MU 239 株；无任一模态 8 株 | —（不适用） |
| 缺模态原因（互有重叠） | 缺 GE 251 / 缺 MU 9 / 缺 CNV 455 | —（不适用） |

509 株的 OncotreeLineage top：Lung 99、Lymphoid 50、Breast 41、Esophagus/Stomach 34、Skin 28、CNS/Brain 28、Bowel 27、Ovary/Fallopian Tube 24、Pancreas 23、Myeloid 19、Bone 18、Kidney 17（其余 15 个类 ≤15）。

### 2.6 基因列解析与基因空间

列结构（两文件同构）：`<无名索引列>,SequencingID,ModelConditionID,ModelID,IsDefaultEntryForMC,IsDefaultEntryForModel,<基因列>…`；即首字段无列名，6 个 token 中 5 个是命名元数据列，其后才是 19,215（GE）/ 19,955（CNV）个基因列。

| 项目 | GE（19,215 列） | CNV（19,955 列） |
|---|---|---|
| 命名格式 | 100% 可解析 `^(.+) \((\d+)\)$` | 19,366 列可解析；**589 列非此格式**（`ENSG00000002586_PAR_Y` 等 `_PAR_Y` 与裸 ENSG） |
| 重复 symbol / 重复 Entrez | **0 / 0** | **0 / 0**（按可解析列） |
| 符号怪字符（逗号/引号/分号） | 0 | 0 |
| 全 NaN 基因列 | 0（矩阵全稠密） | **811 列（4.06%）整列全 NaN** → 有效基因仅 19,144 |
| 两文件 symbol 交集 | **19,130**（跨两列：GE-only 85；CNV 可解析列中 CNV-only 236） | 同名对应上一行说明 |

### 2.7 表达值域（default 后 1,719 × 19,215，实测全量）

- 数值 = `log2(TPM+1)` 形态（列名 TPMLogp1；文件内无元数据标注底数，推断为 log2）：**NaN 0、Inf 0、负值 0**；min 0 / max 17.36。
- 分位数：q01=0、q25=0.0673、**q50=2.4152**、q75=4.6181、q99=9.243、q99.9=12.126。
- 零膨胀：**精确 0 占 17.60%**，≤0.01 占 18.77%；**5 个全零方差基因**（`DEFB106A`、`KRTAP23-1`、`KRTAP20-3`、`DEFB113`、`SCYGR7`）须删。

### 2.8 突变长表值域与候选过滤口径（default 736,094 行）

列清单（68 列，仅列一遍）：SequencingID, ModelID, ModelConditionID, IsDefaultEntryForModel, IsDefaultEntryForMC, Chrom, Pos, Ref, Alt, AF, DP, RefCount, AltCount, GT, PS, VariantType, VariantInfo, DNAChange, ProteinChange, HugoSymbol, Exon, Intron, EnsemblGeneID, EnsemblFeatureID, HgncName, HgncFamily, UniprotID, DbsnpRsID, GcContent, NMD, MolecularConsequence, VepImpact, VepBiotype, VepHgncID, VepExistingVariation, VepManeSelect, VepENSP, VepSwissprot, Sift, Polyphen, GnomadeAF, GnomadgAF, VepClinSig, VepSomatic, VepPliGeneValue, VepLofTool, OncogeneHighImpact, TumorSuppressorHighImpact, TranscriptLikelyLof, Brca1FuncScore, CivicID, CivicDescription, CivicScore, LikelyLoF, HessDriver, HessSignature, RevelScore, PharmgkbId, GwasDisease, GwasPmID, GtexGene, ProveanPrediction, AMClass, AMPathogenicity, Rescue, RescueReason, Hotspot, EntrezGeneID。
关键 QC：`HugoSymbol` 0 缺失（20,147 个 symbol）、`EntrezGeneID` 缺失 346 行（0.03%，20,085 个；symbol↔Entrez 在文件内**双射，0 冲突**）；重复变异行（ModelID+Chrom+Pos+Ref+Alt）= 0。
`VepBiotype`（全表 1,172,688 行）：protein_coding 1,162,092 行（99.1%）/ protein_coding_LoF 3,328 / 其余 lncRNA、IG/TR 基因等合计 <0.5%。
**官方文件本身已是预筛选后的"高质量非沉默 + rescued"变异集**（PDF §Variant Selection：先过滤 GATK bad + `AF<0.15` + `DP<5`），故 `VepImpact` 几乎只有 MODERATE/HIGH（全表 MODERATE 975,164、HIGH 196,993、MODIFIER 474、LOW 57）；`VariantType`：SNV 1,026,811 / deletion 79,086 / substitution 38,385 / insertion 28,406。
`MolecularConsequence` **default 行 95.75% 缺失**（仅 31,261 行有值）；`VepSomatic` 缺失 65.86% 且为 `0&1&1` 型多值索引串；`AF` 0 缺失，min 0.15、中位 0.449、`AF>0.6` 17.31%、`AF≥0.9` 6.80%；`GT`：0/1 650,982、`1|1` 49,937（6.8%）、0|1 32,450、1|0 2,725。

候选矩阵口径（均为 default 行 → `(ModelID,HugoSymbol)` 二值去重，样本数均 1,968）：

| 口径 | 变异行 | 基因数 | ModelID-基因对 | 每样本中位/均值突变数 | 基因∩GE |
|---|---|---|---|---|---|
| A0 全行 | 736,094 | 20,069 | 663,606 | 174 / 337.2 | 18,500 |
| A1 protein_coding(+_LoF) | 731,024 | 18,620 | 658,750 | 172 / 334.7 | 18,495 |
| **A 推荐宽松**：A1 + `VariantInfo` 非沉默 token | 730,652 | 18,620 | 658,404（密度 1.797%） | 172 / 334.6（min 21, max 7,828） | 18,495 |
| **B 推荐严格**：A1 + (`LikelyLoF`=True 或 `Hotspot`=True) | 124,610 | 17,186 | 118,933（密度 0.352%） | 25 / 60.4（min 1, max 1,727） | 17,074 |
| B1 仅 `LikelyLoF`=True（= 官方 MatrixDamaging 口径） | 122,513 | 17,172 | 116,916 | 24 / — | — |
| B2 仅 `Hotspot`=True（= 官方 MatrixHotspot，已排除 HLA-A/HLA-B） | 3,860 | 506 | 3,731 | 2 / — | — |

- `VepImpact in (HIGH,MODERATE)` 与 `VariantInfo` 非沉默近似等价（因文件已预筛，A1 与 A 差 372 行）。
- 突变矩阵与 GDSC 的交集样本数：MU default 1,968 中 **959** 株在 GDSC（宽松/严格口径相同；全部样本至少 1 个突变，因 AF≥0.15 已保证）。

### 2.9 CNV 值域（default 1,118 × 19,955）

- **线性相对拷贝数**（非 log2）：min ≈1.4e-6、最大 254.49；q0.1%=0.0063、q1%=0.463、q25=0.9098、**q50=1.0202**、q75=1.1482、q95=1.5333、q99=2.011、q99.9=3.5255；0 值 0、负值 0、无 bool 型异常列（19,955 列全 float64）。
- 缺失：**4.06% 全部集中在 811 个整列全 NaN 的基因列**，其余 19,144 列完全观测。
- 转 log2 后：中位 0.0288、q1%=-1.111、q99%=1.008、min -19.93、max 7.99；分布：< -1 占 1.54%、< -0.585（约半拷贝缺失）6.41%、**[-2,2] 内占 99.79%**、区间外仅 0.211%。**先取 log2，再谈 -2..2 阈值**。

## 3. 关键缺陷与陷阱（含证据）

- **T1 版本混用且无 release 标签**：三个组学文件 mtime 2026-07-13，Model.csv mtime 2026-11-10（差约 4 个月）；组学文件名不含 release 号，仅突变 PDF 标明 26Q1 流程。证据：`ls -l --time-style=full-iso`；MD5 见 1.1。ID 集合上三组学 ⊆ Model.csv（0 孤儿），说明 Model.csv 是更晚/超集快照，但**不能据此认定同一版本**。
- **T2 宽表 CSV 首列无列名**：表达/CNV 文件头以逗号开头（`od -c` 首字节 `,`），因此"列数" 19,221/19,961 = 无名索引列 + 5 个元数据列 + 19,215/19,955 基因列；pandas 默认命名 `Unnamed: 0`。**天真把前 6 列当元数据会整体错位一位**。
- **T3 CNV 缺失是"整列"缺失 + 行重复**：811/19,955 基因列全 NaN（不是随机缺失 4%）；1,132 行中 14 个 ModelID 各出现 2 次（同 ModelConditionID、双 SequencingID，一行 default=Yes 一行 No）。不做 default 过滤会得到 1,132 行/重复样本。
- **T4 表达 default 过滤的隐藏不一致**：1,775 行 → 1,719 模型；56 行 `IsDefaultEntryForModel=No`；其中 **ACH-003012 一行 `IsDefaultEntryForMC=Yes` 但模型级为 No**（`CDS-OcWb8D / MC-003012-BOcy`）。模型级口径必须用 `IsDefaultEntryForModel`；另 54 个模型有 2–3 行（多条件）。
- **T5 `MolecularConsequence` 不可用于非沉默过滤**：default 行 **95.75% 为 NaN**（PDF 说明该列是 "Comma separated list of molecular consequence from ClinVar/SO"，仅部分变异有）；用它做 `synonymous` 剔除等于丢弃 70 万行或全部放行。**应改用 `VariantInfo`（0 缺失，`&` 连接需 tokenize）**。
- **T6 `VepSomatic` 不是体细胞过滤列**：65.86% 缺失，且取值为 `0&1`、`0&1&1`、`0&0&1` 等**索引串**（PDF：它是对 `VepExistingVariation` 多值的 somatic 状态索引）；不能 `==1` 过滤。
- **T7 突变文件已官方预筛选，行数不等于原始突变全集**：PDF §Variant Selection：仅保留"high quality non-synonymous somatic + rescued 癌症相关"变异（AF≥0.15、DP≥5、去 clusted/segdup/repeat、gnomAD≤1e-5 等）。证据：synonymous_variant 仅 32 行、VepImpact 99.96% 为 HIGH/MODERATE、AF 最小值恰 0.15。**不要把 1.17M 行当"全突变"或用于突变负荷绝对值**。
- **T8 GT=`1|1`（6.8%）与高 AF 不是 germline 证据**：PDF：深度≥8 且 AF>0.9 时用 bcftools 把 0/1 改写为 1|1；且 M2 额外参数 `--genotype-germline-sites true --genotype-pon-sites true` 会带入 germline 位点。AF 中位 0.449、>0.6 占 17.3%，**低频 germline 残留比例未量化**（见 §5）。
- **T9 ID 匹配陷阱**：Model.csv 43.4% 无 SangerModelID（934/2,154），且恰好 1 个 SID 一对多（SIDM00117）。名字兜底会因归一化碰撞（KM-H2/KMH-2、TT/T-T）和多模型同名候选（5 株）产生歧义；`CCLEName` 为旧 CCLE `NAME_TISSUE` 格式（124 缺失、与 Stripped 全不同）不能用于匹配。RH-1 无任何映射。
- **T10 CNV 是"线性相对 CN"而非 log2**：中位 1.0202；直接套计划书 -2/-1/0/1/2 离散阈值在原始尺度无意义（几乎全 >0 且 <2），必须先 log2（log2 后 99.79% 落在 [-2,2]）。极端值 max 254 提示需截断/分位数裁剪。
- **T11 CNV 覆盖率是主训练集瓶颈**：1,118/2,154 模型（51.9%）；GDSC 侧缺 CNV 455/968 株；三模态∩GDSC 仅 509（52.6%）。若放弃 CNV 可回到 716 株（GE∩MU）。
- **T12 基因空间不完全对齐**：突变 18,620 编码基因中 18,495 在表达空间（125 个不在）；表达- CNV symbol 交集 19,130（GE-only 85）；CNV 另有 589 个非 `SYMBOL (ENTREZ)` 列（PAR_Y/裸 ENSG）。

## 4. 对数据预处理的影响（建议口径）

1. **定版**：以本笔记 §1.1 的 MD5 清单冻结四个文件（`data/raw/CHECKSUMS.md5` 已覆盖，实测全部 OK）；数据卡中记录 release 待核实（§5-1）。以 `Model.csv` 为唯一 ID 权威表（组学 ID 全为其子集）。
2. **default 条目过滤（强制）**：GE/CNV 用 `IsDefaultEntryForModel=='Yes'`（1,719 / 1,118）；MU 同（736,094 行 → 1,968 模型 × 1 个默认 MC）；**不要用 `IsDefaultEntryForMC`**（GE 有 1 行 MC=Yes/model=No）。过滤后三模态样本池 = 1,105。
3. **基因列解析**：读表时跳过首列（`index_col=0` 或 `usecols=range(1, ...)`）；基因名统一正则 `^(.+) \((\d+)\)$` 抽 symbol/Entrez（GE 100% 成功，0 重复）；CNV 589 个非标准列：`_PAR_Y` 单独标记/保留（与 X/Y 同源基因，勿与常染色体同名基因合并），裸 ENSG 用 HGNC 映射后保留或显式丢弃，并记录数量。
4. **表达**：已是 log2(TPM+1)（推断）；无需再 log；0 值占 17.6%（生物学零表达，勿当缺失）；**删除 5 个全零方差基因**；z-score/HVG 等 fit 类操作只在训练 fold 内做；1,719 模型中可用 1,719（无缺失）。
5. **突变矩阵**：主口径建议 **B（严格 driver）**：`IsDefaultEntryForModel=Yes` → `VepBiotype ∈ {protein_coding, protein_coding_LoF}` → (`LikelyLoF==True` 或 `Hotspot==True`) → `(ModelID,HugoSymbol)` 二值；得 17,186 基因 / 118,933 对 / 中位 25 突变/样本（与官方 OmicsSomaticMutationsMatrixDamaging/Hotspot 语义对齐，HLA-A/HLA-B 已按 26Q1 从 hotspot 排除）。宽松口径 A（+VariantInfo 非沉默）用于消融/覆盖率对照（18,620 基因 / 中位 172）。**禁止**用 `MolecularConsequence` 或 `VepSomatic` 做过滤；可沿官方矩阵约定用 0/1/2（同基因 AF 求和 >0.95 → 2）。
6. **CNV**：转 `log2(linear_relative_CN)`；删除 811 个全 NaN 基因列（或保留并加缺失 mask/指示特征）；阈值化建议 log2 空间分箱（如 < -1 / [-1,-0.585) / [-0.585,0.585) / (0.585,1] / >1，或计划书 -2..2 的离散级别）；极端值按分位数（如 0.1%/99.9%）裁剪；1,118 模型（∩GE 1,105）。
7. **样本池与标签**：主训练可用 = **509 株**（三模态 ∩ GDSC2 969 株）；报告时必须给出 drop 漏斗（GDSC 969 → 可映射 968 → ∩GE 717/∩MU 959/∩CNV 513 → 三模态 509）。若做"无 CNV"消融，样本池 716 株。
8. **ID 匹配**：优先 SangerModelID；名字兜底仅对 RH-1/NCI-H748 这类边界情况人工核对（本轮 NCI-H748 已救回，RH-1 失败）。`SIDM00117`→ ACH-002172（名字+同名+COSMICID=905967 一致）；因两候选均在三模态池内，509 不受影响，但后续按 ModelID 关联时须固定选择。

## 5. 待核实项

1. **组学 release 号**：文件无版本标签；PDF 只对应 26Q1 突变流程；Model.csv（2026-11-10）疑为更晚季度。需从 DepMap 下载页/随包 release note 确认四个文件是否同一 release，并在数据卡落 release + MD5。
2. **表达 log 底**：列名 `TPMLogp1` 但文件内无定义；推断 log2(TPM+1)，需对照 DepMap 官方文档确认（影响 z-score 前的尺度报告，不影响排序）。
3. **Entrez ↔ HGNC 一致性校验**未完成（本轮 HGNC 交叉脚本中断）：需验证 19,215 个 `SYMBOL (ENTREZ)` 与 `data/raw/hgnc/Gene.csv`（protein-coding gene 19,291 条）的一一映射与冲突数。
4. **突变残留 germline 比例**：文件无 matched-normal 字段；需用 gnomAD AF/PoN 清单或 26Q1 文档口径量化（本轮只知 M2 带 germline 参数、GT 被改写）。
5. **CNV 14 个重复 ModelID** 的语义（同 MC 双 SequencingID）是否为同一模型两次 WGS run 合并；仅影响行数，不影响 default 后集合。
6. **RH-1（SIDM00427）** 是否可由 `data/raw/gdsc/Cell_Lines_Details.xlsx` 的 COSMIC/别名映射补回。
7. **811 个全 NaN CNV 基因列**与表达/突变基因空间的关系未细查（是否与 589 个非标准命名列重叠）。
8. **509 株的按药覆盖**：本任务只审细胞系；每药可用细胞数需在标签审计（GDSC2 侧）中核对，避免"细胞系数够但单药样本过少"。

## 6. 复现命令清单

```bash
cd /home/tonywu/workspace/KITE-DRP
# (1) 列数/行数（先探宽表）
for f in data/raw/depmap/DepMap_*.csv data/raw/depmap/Model.csv; do
  echo "$f"; head -1 "$f" | tr ',' '\n' | wc -l; wc -l < "$f"; done
# (2) MD5 完整性（全部 OK）
cd data/raw && grep -E 'depmap/' CHECKSUMS.md5 | md5sum -c - && cd -
# (3) 突变 26Q1 口径文档（PDF 第 100–360 行含 Variant Selection/Columns）
pdftotext -layout data/raw/depmap/26Q1_Mutation_Pipeline_Documentation.pdf - | sed -n '100,360p'
# (4) Model.csv/GDSC 匹配 + 三模态交集（核心数字，约 60 s）
.venv/bin/python - <<'PY'
import pandas as pd, re
m=pd.read_csv('data/raw/depmap/Model.csv',dtype=str,keep_default_na=False,na_values=[''])
g=pd.read_csv('data/raw/gdsc/GDSC2_fitted_dose_response_27Oct23.csv',usecols=['CELL_LINE_NAME','SANGER_MODEL_ID'],dtype=str)
norm=lambda s: re.sub(r'[^A-Za-z0-9]','',str(s)).upper()
sid2name=g.drop_duplicates('SANGER_MODEL_ID').set_index('SANGER_MODEL_ID').CELL_LINE_NAME
res=set()
for sid,grp in m[m.SangerModelID.isin(set(g.SANGER_MODEL_ID))].groupby('SangerModelID'):
    if len(grp)==1: res.add(grp.ModelID.iloc[0])
    else:
        hit=grp[grp.StrippedCellLineName.map(norm)==norm(sid2name[sid])]
        if len(hit)==1: res.add(hit.ModelID.iloc[0])
for _,r in g[~g.SANGER_MODEL_ID.isin(m.SangerModelID.dropna())].drop_duplicates('SANGER_MODEL_ID').iterrows():
    res|=set(m[(m.CellLineName.map(norm)==norm(r.CELL_LINE_NAME))|(m.StrippedCellLineName.map(norm)==norm(r.CELL_LINE_NAME))].ModelID)
mc=['ModelID','IsDefaultEntryForModel']
E=pd.read_csv('data/raw/depmap/DepMap_ExpressionTPMLogp1HumanProteinCodingGenes.csv',usecols=mc,dtype=str)
C=pd.read_csv('data/raw/depmap/DepMap_CNGeneWGS.csv',usecols=mc,dtype=str)
U=pd.read_csv('data/raw/depmap/DepMap_SomaticMutations.csv',usecols=mc,dtype=str)
EX=set(E.loc[E.IsDefaultEntryForModel=='Yes','ModelID']); CN=set(C.loc[C.IsDefaultEntryForModel=='Yes','ModelID']); MU=set(U.loc[U.IsDefaultEntryForModel=='Yes','ModelID'])
print('|EX|,|MU|,|CN| =',len(EX),len(MU),len(CN),'| triple =',len(EX&MU&CN))
print('GDSC mappable =',len(res),'| ∩EX,∩MU,∩CN =',len(res&EX),len(res&MU),len(res&CN),'| ∩triple =',len(res&EX&MU&CN))
PY
# (5) 基因列解析/重复检查（表达为例）
.venv/bin/python - <<'PY'
import re
h=open('data/raw/depmap/DepMap_ExpressionTPMLogp1HumanProteinCodingGenes.csv',encoding='utf-8-sig').readline().rstrip('\n').split(',')
g=h[6:]; p=[re.match(r'^(.+) \((\d+)\)$',x) for x in g]
print(len(g),'gene cols; parsed',sum(bool(x) for x in p),'; dup symbol',len(g)-len({x.group(1) for x in p if x}),'; dup entrez',len(g)-len({x.group(2) for x in p if x}))
PY
# (6) 突变候选口径（A/B 两口径的基因数/对数/每样本中位突变数）
.venv/bin/python - <<'PY'
import pandas as pd, re
cols=['ModelID','IsDefaultEntryForModel','VepBiotype','VariantInfo','LikelyLoF','Hotspot','HugoSymbol']
mu=pd.read_csv('data/raw/depmap/DepMap_SomaticMutations.csv',usecols=lambda c:c in cols,dtype=str)
d=mu[mu.IsDefaultEntryForModel=='Yes']; pc=d.VepBiotype.isin(['protein_coding','protein_coding_LoF'])
SIL={'synonymous_variant','intron_variant','3_prime_UTR_variant','5_prime_UTR_variant','non-coding_transcript_variant','genic_upstream_transcript_variant','genic_downstream_transcript_variant','upstream_gene_variant','downstream_gene_variant','intergenic_variant','sequence_variant'}
ns=lambda s: len(set(re.split(r'[&,]',str(s)))-SIL)>0
for nm,sub in [('A',d[pc&d.VariantInfo.map(ns)]),('B',d[pc&((d.LikelyLoF=='True')|(d.Hotspot=='True'))])]:
    pairs=sub[['ModelID','HugoSymbol']].drop_duplicates(); per=pairs.groupby('ModelID').size()
    print(nm,'rows',len(sub),'genes',sub.HugoSymbol.nunique(),'pairs',len(pairs),'median',per.median())
PY
```

---

## 7) Lead 复算补充（2026-11-17，供主规划引用与 G0-4 证据链）

> 审计员原文只到 §6；本节由 Team Lead 在写《数据预处理详细规划》时补算，用于把规划中的派生数字接到可复现命令上。
> 环境：`.venv/bin/python`（pandas 3.1.1），只读 `data/raw/`。**这些数字不属于原审计员的独立实测**，故单列。

| 派生数字 | 值 | 口径/算法 |
|---|---|---|
| U_teach_ge | **19,125** | GE symbol 19,215 ∩ CNV 可解析 symbol 19,130 = 19,130，再删 5 个全局零方差基因（DEFB106A/KRTAP23-1/KRTAP20-3/DEFB113/SCYGR7，均在交集中） |
| 突变 ∩ U_teach_ge | **A：18,449 / B：17,035** | 相对 19,125 重算（原审计的 18,495/17,074 是相对 GE 全集 19,215，已废弃） |
| 化合物级标签对（全 969 株） | **235,748** | 化合物键 = InChIKey 优先、缺键退归一化药名；双 ID 化合物合并 |
| 化合物级标签对（mapped 968 株） | **235,478** | 排除 RH-1（SIDM00427，279 行 → 270 对） |
| M-509 池标签对 | **126,803** | 509 = GDSC∩GE∩MU∩CNV（default 后）；130,372 行 − 3,569 行（双 ID 合并） |
| M-509 per-compound | min **94** / median **475** | 286 化合物全部 ≥50 |
| M-716 池 | **716**（GE∩MU∩GDSC）；标签对 **176,487** | CNV 缺失 207 株 → mask |
| RH-1 标签行 | **279** | 无 ACH，`cell_line_uid=null` |
| METH ∩ 主池 | **404**（METH 843 列 → 663 有 SangerID → 559 ∩ GDSC 969 → 404 ∩ M-509） | 经 `Cell_lines_annotations_20181226.depMapID` 映射 |

**复现命令骨架**（完整实现将落为 W1 的 `scripts/audit_pool_funnel.py`）：

```bash
cd /home/tonywu/workspace/KITE-DRP
.venv/bin/python - <<'PY'
import pandas as pd, re
g = pd.read_csv('data/raw/gdsc/GDSC2_fitted_dose_response_27Oct23.csv', dtype={'SANGER_MODEL_ID':str,'DRUG_ID':str})
m = pd.read_csv('data/raw/depmap/Model.csv', dtype=str, keep_default_na=False, na_values=[''])
norm = lambda s: re.sub(r'[^A-Za-z0-9]','',str(s)).upper()
sid2name = g.drop_duplicates('SANGER_MODEL_ID').set_index('SANGER_MODEL_ID').CELL_LINE_NAME
res = {}
for sid, grp in m[m.SangerModelID.isin(set(g.SANGER_MODEL_ID))].groupby('SangerModelID'):
    if len(grp)==1: res[sid]=grp.ModelID.iloc[0]
    else:
        hit = grp[grp.StrippedCellLineName.map(norm)==norm(sid2name[sid])]
        if len(hit)==1: res[sid]=hit.ModelID.iloc[0]
for _,r in g[~g.SANGER_MODEL_ID.isin(m.SangerModelID.dropna())].drop_duplicates('SANGER_MODEL_ID').iterrows():
    cand = m[(m.CellLineName.map(norm)==norm(r.CELL_LINE_NAME))|(m.StrippedCellLineName.map(norm)==norm(r.CELL_LINE_NAME))]
    if len(cand)==1: res[r.SANGER_MODEL_ID]=cand.ModelID.iloc[0]
g['ach'] = g.SANGER_MODEL_ID.map(res)
mc=['ModelID','IsDefaultEntryForModel']
EX=set(pd.read_csv('data/raw/depmap/DepMap_ExpressionTPMLogp1HumanProteinCodingGenes.csv',usecols=mc,dtype=str).query("IsDefaultEntryForModel=='Yes'").ModelID)
CN=set(pd.read_csv('data/raw/depmap/DepMap_CNGeneWGS.csv',usecols=mc,dtype=str).query("IsDefaultEntryForModel=='Yes'").ModelID)
MU=set(pd.read_csv('data/raw/depmap/DepMap_SomaticMutations.csv',usecols=mc,dtype=str).query("IsDefaultEntryForModel=='Yes'").ModelID)
G=set(g.ach.dropna()); pool=G&EX&MU&CN
m5=pd.read_csv('data/raw/gdsc/screened_compounds_rel_8.5_inchi.csv',dtype={'DRUG_ID':str})
g=g.merge(m5[['DRUG_ID','INCHI_KEY']],on='DRUG_ID',how='left')
ck=lambda r:('inchikey:'+r.INCHI_KEY) if isinstance(r.INCHI_KEY,str) and r.INCHI_KEY else 'name:'+re.sub(r'[^A-Z0-9]','',str(r.DRUG_NAME).upper())
g['compound']=g.apply(ck,axis=1)
print('mapped',len(G),'M509',len(pool),'M716',len(G&EX&MU),
      'pairs_mapped',len(g[g.ach.notna()].drop_duplicates(['SANGER_MODEL_ID','compound'])),
      'pairs_M509',len(g[g.ach.isin(pool)].drop_duplicates(['SANGER_MODEL_ID','compound'])))
PY
```

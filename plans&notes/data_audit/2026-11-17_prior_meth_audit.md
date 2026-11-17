# 先验网络与 CCLE 甲基化实测审计（task-3）

- 日期：2026-11-17；执行：audit-prior；数据：`data/raw/`（**只读**）；环境：`.venv/bin/python`（pandas 3.1.1），**未联网、未装包、未跑 `uv run`**。
- 范围：STRING v12.5/v12.0、HGNC Gene.csv、DGIdb v5.0.13、CCLE DNA methylation TSS1kb、GDSC2 靶点、缺失先验资源。唯一产出即本文件。
- 文件完整性：MD5 取自 `data/raw/CHECKSUMS.md5`（未重算）；审计开始时 `git status --short` 为空，审计只新增本文件（`plans&notes/data_audit/`）。

## 0. 结论摘要（先看这里）

1. **STRING links 文件每对互作存了“双向两行”，行数 ≠ 边数**：v12.5 共 10,826,054 行，唯一无向对仅 **5,413,027**；反向行存在且 score 完全相同（见 §2.1 证据）。v12.0 同构（13,715,404 行 → 推定 6,857,702 对）。**建 $A_{GG}$ 前必须按 (min,max) 去重**，否则边数/度分布全部虚高一倍。
2. 去重后 v12.5：≥700 为 **282,864 对 / 17,525 蛋白**；≥900 为 **125,344 对 / 14,301 蛋白**；折叠到基因级（ENSP→HGNC symbol）≥700 为 **282,789 边 / 17,518 基因**（中位度 15，TP53 度 872）。
3. ENSP→HGNC symbol 映射率 **97.67%**（19,191/19,649），458 个蛋白无 symbol 且无法用 ENSG 补；歧义仅 1 个（CCL3L1/CCL3L3）。
4. **CCLE_DNA_methylation_TSS1kb.txt 不是 450K beta 值，而是 2018 年 CCLE RRBS 的 hg19 TSS±窗口汇总**。证据：行名 `GENE_chr_start_end`（1,000 bp 窗口）、每窗口 `CpG_sites_hg19` 位点列表（1–131 个 CpG）+ `avg_coverage`（测序深度，中位 195）、缺失随 coverage 单调下降、92% 值落在 5 位小数格点、文件日期 2018-12-27。
5. 该 METH 矩阵：21,337 个有效窗口（另有 1 行 `NA_NA` 占位需删）→ **17,181 基因**，843 细胞系列中 **10 列全空（833 可用）**，总缺失 5.47%，值域 [0,1]（27.1% 恰 0、3.3% 恰 1、中位 0.021）。**不能按计划书 §3.3-4 的“450K beta 取启动子均值”处理；也不建议作为 VAE METH 头/患者域（TCGA HM450）补全的数据源**，默认只补 CNV；若要保留教师 METH，只能作“可选+掩码”特权模态（详见 §4）。
6. 交叉匹配：843 列 100% 可映射到 `Model.csv` ModelID（经 CCLE 注释 depMapID），但仅 663 有 SangerModelID、**559 株 ∩ GDSC2**（GDSC2 的 141,468 行 / 58.4%，286 药名全覆盖）。METH 基因符号是 2018 旧符号：95.2% 可映射当前 HGNC，余 833 个 LOC/FLJ 不可映射。
7. DGIdb 可作解释校验/兜底，但单靠它撑不起 $A_{DG}$：GDSC 286 药名规范化后 **202/286（70.6%）** 在 DGIdb 有交互；限制到 HGNC protein-coding 基因后仅 **158/286 药、5,401 唯一对、1,468 基因**。GDSC 自带 `PUTATIVE_TARGET` 可补 256/295 药（185 可解析为基因名）。
8. **缺失（只列不下载）**：Reactome/GO/Hallmark/KEGG、SMILES 源、ChEMBL/Open Targets/DrugBank、panel 清单、Edge 预印本 1,603 基因 PPI 子集清单、CCLE legacy 450K 甲基化、TCGA-BRCA/LUAD。许可提示见 §5。

## 1. 文件与方法（含复现命令）

| 文件 | 大小(B) | MD5（CHECKSUMS） | 实测行/形状 | 说明 |
|---|---|---|---|---|
| `string/9606.protein.links.v12.5.txt.gz` | 62,813,591 | 3844d2f6… | 10,826,054 数据行 | 3 列：`protein1 protein2 combined_score` |
| `string/9606.protein.aliases.v12.5.txt.gz` | 19,777,800 | 09a0a186… | 3,889,207 行 / 19,699 蛋白 | 表头 `#string_protein_id alias source`，99 种 source |
| `string/9606.protein.links.v12.0.txt` | 630,908,617 | 01eaf4f3… | 13,715,404 数据行 | 3 列同构；双向重复同 v12.5 |
| `hgnc/Gene.csv` | 17,440,425 | f7522367… | 44,083 行 × 56 列 | HGNC 全库快照（date_modified 最晚 2025-03-31） |
| `dgidb/interactions.tsv` | 19,340,837 | 3082ca32… | 122,904 行 × 16 列 | 2 行 `#` 注释头；Data version 2026-10 / v5.0.13 |
| `dgidb/drugs.tsv` / `genes.tsv` / `categories.tsv` | 8.1M/4.8M/1.7M | 9837083b…/94a2d2fa…/890fa296… | 80,483×9 / 85,738×6 / 33,889×4 | 同上 |
| `ccle/CCLE_DNA_methylation_TSS1kb.txt` | 149,292,202 | 5875461a… | 21,338 行；1+2+843 列 | 见 §2.4 |
| `gdsc/GDSC2_fitted_dose_response_27Oct23.csv` | 36,352,214 | b989a6a9… | 242,036 行 / 969 株 / 286 药名 / 295 DRUG_ID | 16 列，含 `SANGER_MODEL_ID`、`PUTATIVE_TARGET` |
| `ccle/Cell_lines_annotations_20181226.txt` / `depmap/Model.csv` | 0.34M/0.70M | — | 1,461×33 / 2,154×49 | 旧注释 fallback / 权威 ID 交叉表 |

复现命令（在本仓库根目录执行；STRING 全部流式，峰值内存 <1.5 GB）：

```bash
# --- STRING 行数 / 双向重复证据 / 阈值统计 ---
zcat data/raw/string/9606.protein.links.v12.5.txt.gz | wc -l        # 10826055（含表头）
wc -l data/raw/string/9606.protein.links.v12.0.txt                  # 13715405
# 正向边在第 2 行、反向边在第 818 行、score 均 200（v12.0：第 2 行 vs 第 7,352,103 行，均 173）
zcat data/raw/string/9606.protein.links.v12.5.txt.gz | awk -F' ' 'NR>1{if($1=="9606.ENSP00000000233"&&$2=="9606.ENSP00000000412"){fl=NR;fs=$3} if($1=="9606.ENSP00000000412"&&$2=="9606.ENSP00000000233"){rl=NR;rs=$3}} END{print fl,fs,rl,rs}'
# 唯一无向对 / 各阈值 / ENSP→symbol / gene 级折叠：用 .venv/bin/python 流式 set（脚本见 §6 片段）
# --- METH（注意 skipinitialspace=True，否则解析报错）---
.venv/bin/python - <<'PY'
import pandas as pd
f='data/raw/ccle/CCLE_DNA_methylation_TSS1kb.txt'
cols=pd.read_csv(f,sep='\t',nrows=0).columns.tolist()
m=pd.read_csv(f,sep='\t',index_col=0,skipinitialspace=True,
              dtype={c:'float32' for c in cols[3:]}|{'avg_coverage':'float32'})
print(m.shape, m['avg_coverage'].median(), m.iloc[:,2:].isna().to_numpy().mean())
PY
# --- HGNC / DGIdb / GDSC ---
.venv/bin/python -c "import pandas as pd; h=pd.read_csv('data/raw/hgnc/Gene.csv',low_memory=False); print(h.shape, h['locus_group'].value_counts().to_dict())"
.venv/bin/python -c "import pandas as pd; print(pd.read_csv('data/raw/dgidb/interactions.tsv',sep='\t',comment='#').shape)"
```

## 2. 实测数字表

### 2.1 STRING v12.5 / v12.0

| 指标 | v12.5（主版本） | v12.0（留档） |
|---|---|---|
| links 数据行 | 10,826,054 | 13,715,404 |
| **唯一无向蛋白对** | **5,413,027** | **6,857,702（推定同构；行数恰为 2×）** |
| 双向重复证据 | 正向 line 2 / 反向 line 818，score 均 200；`a>=b` 行数 = 5,413,027 = n/2；自环 0 | 正向 line 2 / 反向 line 7,352,103，score 均 173；`a>=b` = 6,857,702 = n/2 |
| 涉及蛋白数 | 19,649 | 未全量统计（量级同族） |
| 行数 vs v12.5 | — | **多 2,889,350 行（+26.7%）**；去重后对约 +26.7% |

combined_score 分布（v12.5，行数；去重后各减半）：[100,200) 3,639,026；[200,300) 3,702,836；[300,400) 1,331,744；[400,500) 861,514；[500,600) 478,718；[600,700) 246,488；[700,800) 153,838；[800,900) 161,202；[900,1000] 250,688。最低 score = 150。

| 阈值 | 行数 | 去重后唯一对 | 蛋白数 | gene 级唯一对 | gene 数 | 含无 symbol 蛋白的边 |
|---|---|---|---|---|---|---|
| ≥150（全量） | 10,826,054 | 5,413,027 | 19,649 | 5,410,662 | 19,641 | 223,530 |
| ≥400 | 2,152,448 | 1,076,224 | 19,547 | 1,075,922 | 19,540 | 33,982 |
| ≥700 | 565,728 | 282,864 | 17,525 | 282,789 | 17,518 | 10,396（3.7%） |
| ≥900 | 250,688 | 125,344 | 14,301 | 125,323 | 14,296 | 3,678 |

- ENSP→HGNC symbol：aliases 文件 source=`Ensembl_HGNC_symbol`（19,203 行 / 19,202 蛋白）。19,191/19,649 = **97.67%** 的 links 蛋白可映射；458 个无 symbol，且其 `Ensembl_HGNC_ensembl_gene_id` 也无法经 HGNC 当前表补回（可补回数=0）→ 映射率上限即 97.67%。19,191 蛋白 → 19,184 个不同 symbol（7 个 symbol 多蛋白碰撞），歧义蛋白 1 个（ENSP00000480558 → {CCL3L1, CCL3L3}）。
- aliases source 分布（前 8）：Ensembl_EMBL 487,265；Ensembl_protein_id 455,461；Ensembl_RefSeq(_short) 各 225,974；Ensembl_transcript 152,417；Ensembl_HGNC_trans_name 151,338；Ensembl_UCSC 148,513；Ensembl_Reactome_gene 110,630；`Ensembl_HGNC_prev_symbol` 8,696。
- gene 级 ≥700 度分布：q50=15、q75=38、q90=81、q95=126、q99=251、max=872（TP53；其后 H4C6 715、RPS27A 710、MAPT 616、ACTB 612——组蛋白/核糖体/泛素前体 hub 明显，解释与随机重连对照时需注意）。
- 许可与体积：v12.5 为 **CC BY 4.0（仅需署名，含商用）**（`data/raw/README.md` §4-8；计划书 §3.4 仍写“学术需签许可、商用付费”，**两处矛盾，待统一**）。gz 62.8 MB，解压约 630 MB。

### 2.2 HGNC Gene.csv

| 指标 | 实测 |
|---|---|
| 行×列 / status | 44,083 × 56；`status` **全部为 Approved（无 Withdrawn 行）** |
| locus_group | protein-coding 19,291；pseudogene 14,471；ncRNA 9,318；other 1,003 |
| 符号唯一性 | 精确重复 0、大小写不敏感重复 0（symbol 不受大小写歧义影响） |
| ENSG | 41,448 个非空；2,635 个缺失（其中 protein-coding 缺 50）；**`ensembl_gene_id` 无 `|` 多值（每行单 ENSG）**；重复 ENSG 2 处 |
| Entrez | 233 个缺失（protein-coding 缺失 0）；重复 Entrez 1 处 |
| alias_symbol | 44,400 条 (alias,symbol)；43,036 个唯一 alias；**1,083 个 alias 对应 >1 个 symbol**（PAP、U4、ASP 各 7 个） |
| prev_symbol | 15,699 条；15,564 个唯一；**231 个 prev_symbol 同时是另一基因的当前 symbol** → 直接用会误映射 |
| 其他 | `mane_select` 19,235 个非空；symbol 含 `@`/`-`/`_`（HOXA@、APOBEC3A_B、GTF2H2C_2 等）与 >20 字符 5 个 |

**“基因单一真源”建议口径**：以 HGNC 本文件 `symbol`（唯一）+ `ensembl_gene_id`/`entrez_id` 为骨架；映射顺序 = 当前 symbol → prev_symbol → alias_symbol，且**仅当该 alias/prev 唯一对应 1 个 symbol 且不与其他基因当前 symbol 冲突时才采用**；protein-coding 过滤用 `locus_group=='protein-coding gene'`；Withdrawn 检测不能用 status（本文件无此值），只能靠“目标 symbol 是否存在于当前表 + prev_symbol 表”。

### 2.3 DGIdb v5.0.13 与 GDSC 靶点

- 规模：interactions 122,904 行 ×16 列；drugs 80,483×9（26,148 个 drug_name / 29,017 concept_id）；genes 85,738×6（12,728 gene_name）；categories 33,889×4（DRUGGABLE GENOME 10,500 等）。
- 字段陷阱：**无名为 `source` 的列**，交互来源在 `interaction_source_db_name`（24 个库：DTC 23,879、ChEMBL 18,202、GuideToPharmacology 17,104、Drug Repurposing Hub 15,964、TTD 15,245…）；score 字段 4 个（`interaction_score / drug_specificity_score / gene_specificity_score / evidence_score`），**21,796 行（17.7%）全为空**；`interaction_types` 仅 37,066 行非空。
- gene_name→HGNC：6,849 个唯一名，6,804（99.3%）直接命中当前 symbol，+7 经 alias/prev，38 个未映射 → gene 侧基本干净。
- drug↔GDSC：GDSC2 286 个药名，规范化（大写去标点）后 `drug_name` 命中 161/286、`drug_claim_name` 命中 189/286、**合并 202/286（70.6%）**；未覆盖多为内部代号/机制名（Wee1 Inhibitor、CCT007093、CDK9_5576、ERK_2440…）。
- $A_{DG}$ 可行性：全库唯一 (drug,gene) 对 78,039（19,992 药 × 6,307 基因）；**限制到 GDSC 药 × HGNC coding 基因后仅 5,401 对 / 158 药 / 1,468 基因**（158/286 = 55.2%）。GDSC 自带兜底：`PUTATIVE_TARGET` 在 256/295 DRUG_ID 非空，其中 **185 个可解析为基因名**（71 个是机制词如 “Microtubule destabiliser”）；`screened_compounds_rel_8.5_inchi.csv` 给 295/295 DRUG_ID、TARGET 579/621 行、CHEMBL_ID 417 行、INCHI_KEY 501 行，**无 SMILES**。
- 许可：DGIdb 混合来源、再分发受限（内部研究用；逐源条款待核）。

### 2.4 CCLE_DNA_methylation_TSS1kb.txt —— 平台判定

| 证据项 | 实测 | 含义 |
|---|---|---|
| 行名 | `locus_id = GENE_chr_start_end`（如 `SGIP1_1_66998638_66999638`），21,337 行可解析 + 1 行 `NA_NA`；窗口长度**恒为 1,000 bp** | 基因 TSS 窗口汇总，不是 CpG 探针 ID |
| 行/基因 | 21,337 窗口 → **17,181 基因**（4,156 个额外/重叠 TSS 窗口，如 SGIP1 有 2 个窗口） | 需按基因聚合 |
| 列 | 843 列 CCLE_ID 格式 `CELLLINE_TISSUE`（如 `DMS53_LUNG`），843 个唯一 | 2018 CCLE 命名，非 ACH ID |
| 第 2 列 `CpG_sites_hg19` | 每窗口列出覆盖到的 CpG 位点（hg19 坐标，如 `1:66998970;…`）；每窗 **1–131 个**，中位 14，合计 **395,519 个 CpG 观测** | 测序覆盖位点列表，非芯片探针集 |
| 第 3 列 `avg_coverage` | 每窗口单值：min 5.03 / q25 74.94 / **中位 194.95** / q95 751.96 / max 1,722.17（1 个缺失=NA_NA 行） | 测序深度（RRBS）；450K beta 无此概念 |
| 缺失 | 总 NaN 984,182 / 17,987,934 = **5.47%**；**10 列整列全空**（OPM2、REC1、CL34、ML1、JHUEM7、OE21、MOLT3、HOP62、EKVX、UO31）；列缺失中位 2.96%，行缺失中位 2.25%（q99 60.4%） | 覆盖度依赖的缺失（MNAR），非芯片 |
| 缺失 vs coverage | 窗口 coverage [5,20) 非缺失 76.7%（1,450 行）；[20,50) 87.0%；[50,200) 95.4%；[200,1000) 98.2%；≥1000 98.7%；Pearson corr(cov, 非缺失率)=0.3458 | RRBS 覆盖-缺失机制的硬证据 |
| 值域/分布 | [0,1]；q50=0.02137、q75=0.3333、q90=0.8727、q95=0.9664；**恰 0 占 27.12%**、**恰 1 占 3.32%**、其余 (0,1)；**92.05% 值恰为 5 位小数** | 测序甲基化比例（低甲基化 CpG 岛富集），非连续 beta |
| 元数据 | 文件 mtime 2018-12-27；同批 `Cell_lines_annotations_20181226.txt`；坐标 **hg19**（当前 DepMap 为 hg38） | 2018 年 CCLE 旧资源 |

**结论（平台真相）**：该文件是 **CCLE 2018 版 RRBS（reduced representation bisulfite sequencing）的 TSS±1 kb 汇总表（hg19）**，值 = 窗口内被 RRBS 覆盖 CpG 的甲基化比例均值，非 450K/EPMC 芯片 beta 矩阵。计划书 §3.3-4 “450K beta 值取启动子区均值或高变 CpG”的假设**不成立，必须更正**（另注：任务描述的“21,338 TSS × 843 列”实际为 21,338 行/843 数据列，其中 1 行占位、10 列全空）。

### 2.5 细胞系/标签交叉匹配

| 匹配链路 | 结果 |
|---|---|
| METH 843 列 → `Cell_lines_annotations_20181226.CCLE_ID` | 843/843（注释表另 618 株无 METH） |
| → `depMapID` → `Model.csv.ModelID` | 843/843 |
| → `Model.csv.SangerModelID` | 663/843 |
| **METH ∩ GDSC2（SANGER_MODEL_ID）** | **559 株**（= 843 的 66.3%、GDSC2 969 株的 57.7%） |
| 上述 559 株上的 GDSC2 标签 | 141,468 / 242,036 行（58.4%），286 药名 / 295 DRUG_ID 全覆盖 |

## 3. 关键缺陷与陷阱（≥5 条）

1. **STRING links 双向重复（P0）**：每对互作以 (a,b) 与 (b,a) 各存一行且 score 相同；行数恰为唯一对的 2×。按行建图 → 边数、度、随机重连对照全部虚高一倍；必须 `(min(p1,p2), max(...))` 去重。`data/raw/README.md` 的“10,826,054 边”实为**行数**，须改为“行数 10,826,054 / 唯一无向边 5,413,027”。
2. **版本不可混用**：v12.0 行数（13,715,404）**多于** v12.5（10,826,054），差 26.7%，且两版都双向重复；计划书 §4.4 的“0.5–3 万条”是去重后子图量级，若误用行数会放大 2×。另 v12.0 为未压缩 630 MB，仅建议作对照。
3. **STRING 映射不是 100%**：458/19,649 蛋白无 HGNC symbol（无法用 ENSG 补回），7 个 symbol 多蛋白碰撞，1 个蛋白双 symbol 歧义；gene 级折叠会轻微合并节点（19,191 蛋白→19,184 symbol）。aliases 文件首行为 `#` 注释，直接 `read_csv` 会把它当表头。
4. **HGNC 无 Withdrawn 行**：本快照 status 全 Approved，不能用 status 过滤废弃基因；1,083 个 alias 对应多 symbol、231 个 prev_symbol 与别的基因当前 symbol 冲突 → 别名映射必须“唯一才用”，否则静默错配。
5. **METH 文件解析陷阱**：缺失值写成右对齐带前导空格的 `    NaN`/`    NA`，`pd.read_csv(dtype=float32)` 会报 `cannot safely convert`/整列 object；**必须 `skipinitialspace=True`**。另有 1 行 `NA_NA` 占位（整行缺失，需删除）与 10 列整列全空（需删列）。
6. **METH 缺失是 MNAR 且 `avg_coverage` 是窗口级单值**：缺失率随 coverage 升高而下降（76.7%→98.7%）；`avg_coverage` 是每个窗口跨样本的平均覆盖度，**不是每个细胞系的覆盖度**，无法据此恢复样本级 n_meth/n 或做样本级覆盖加权（本审计用 `v×coverage` 整数比值检验仅 42.5% 命中，证明该列非样本级）。
7. **METH 基因符号是 2018 旧符号**：17,181 个中 15,753 直接命中当前 HGNC，+579 prev +21 alias → 16,348（95.2%）；833 个 LOC/FLJ 未映射；解析后 13,691 为 protein-coding，与 STRING symbol 宇宙交集 13,480（coding 13,414）。跨版本 join 前必须建映射表。
8. **跨平台风险（METH 头）**：TCGA METH 是 HM450（manifest 312 个 `*.methylation_array.sesame.level3betas.txt`），与 CCLE RRBS 不同平台、hg19 vs hg38、不同 CpG 覆盖与不同缺失机制；用 RRBS 训练 VAE METH 头再迁移患者域不可信。
9. **DGIdb 药名未做盐/别名归一化**：本审计仅“大写去标点”，GDSC 的 286 药名里 84 个完全匹配不上 DGIdb（内部代号/机制名），真实覆盖率靠 ChEMBL ID/InChIKey 桥接后才可能更高；DGIdb 17.7% 交互无任何 score；`source` 字段名与常见预期不同。
10. **文档矛盾**：计划书 §3.4 称 STRING “学术需签许可、商用需付费”，而 `data/raw/README.md` 与 STRING v12.5 发布信息为 CC BY 4.0（含商用）；计划书 §3.3-4 的 450K 假设与实测 RRBS 冲突。两处需在数据卡/计划书更正并记录决策。

## 4. 对数据预处理的影响

- **$A_{GG}$（PPI）**：流程固定为 `读 gz 流式 → (min,max) 去重 → 阈值（建议主用 ≥700，消融 ≥900/≥400）→ ENSP→symbol（aliases 的 Ensembl_HGNC_symbol，缺符号者按 ENSG/ENSP 保留并标记）→ 与教师基因节点（2,000–3,000，D4/D5 定稿）取子图 → 边属性用 combined_score`。去重后全库 ≥700 为 282,789 gene 级边/17,518 基因，子图内边数按 2,000–3,000 节点随机抽样近似约 0.4–0.8 万条（靶点/通路/高变基因构成的子图会偏多；与计划书 §4.4 “0.5–3 万”兼容）。随机重连对照必须基于去重后边表。
- **$A_{DG}$（药物-靶点）**：DGIdb 只作**解释校验/兜底**（§4.6 命中率），主边源按计划书优先 **Open Targets（CC0）+ ChEMBL + GDSC `PUTATIVE_TARGET`**；当前本地可立即构造的 $A_{DG}$ 上限 = GDIdb 5,401 对 + GDSC 靶点 185 药，预计覆盖 286 药中的 ~200（约 70%），剩 80+ 药（内部代号/机制类）需 Open Targets/ChEMBL/文献补。
- **基因单一真源**：以 HGNC（44,083 symbol）+ STRING aliases 为权威；建 `gene_master.parquet`（hgnc_id、symbol、prev/alias 展开、ensembl、entrez、locus_group、是否 coding）；所有模态 join 前先映射到该表。METH/DGIdb/STRING 的旧符号必须落表记录 provenance。
- **METH（教师特权模态）**：若保留 RRBS，预处理口径：删 `NA_NA` 行与 10 个全空列 → 按基因聚合（多 TSS 窗口取均值，记录窗口数/平均 coverage）→ 旧符号映射到 gene_master（833 个不可映射者丢弃并记录）→ logit(clip 1e-3) → **mask-aware loss/缺失 mask 输入**（MNAR 不能按均值插补当真实值）→ 训练/验证 split 内拟合统计量。可用于教师 METH 编码器（833 株 × 13,691 coding 基因），配合 modality dropout。
- **VAE METH 头（学生补全）**：默认**不启用**（与计划书 D5“METH 仅覆盖足够时加入”一致）。若启用：仅限细胞系内部补全评估，或先替换为 450K 数据源（CCLE legacy 450K / TCGA hg19 legacy）再谈患者域迁移；L1 患者域补全仍只补 CNV。另：1,603 基因目标空间的原始清单（Edge 预印本 STRING v11）不在本地，需补或改用 STRING v12.5 重建并声明口径差异。
- **学生通路特征**：Reactome/GO/Hallmark GMT 全部缺失（P0），ssGSEA/通路均值打分无法开工；需先补 §5 资源。

## 5. 需补充资源清单（只列不下载）

| 资源 | 用途 | 本地状态 | 许可提示 |
|---|---|---|---|
| Reactome（v97，GMT/NCBI Human） | 教师通路先验、通路归因 | 缺失 | 开放许可（计划书标“开放”；CC BY 4.0 待核实原文），需署名 |
| GO（go-basic.obo + GAF/GMT） | 通路先验、可解释性 | 缺失 | CC BY 4.0，需署名 |
| MSigDB Hallmark（h.all.v*.symbols.gmt） | 学生通路打分（路线 A） | 缺失 | 学术免费但需注册；商用需许可；不得公开再分发原始 GMT |
| KEGG（hsa 通路/基因） | 可选通路先验 | 缺失 | 学术免费、**商用需购、禁止再分发**（红线） |
| Open Targets Platform（targets/drugs parquet） | $A_{DG}$ 首选开源边源 | 缺失 | **CC0 4.0**（最适合开源再分发） |
| ChEMBL v36（drug mechanisms/activities） | $A_{DG}$、SMILES、药名桥接 | 缺失 | CC BY-SA 4.0（copyleft，派生数据需同许可） |
| DrugBank 6.0 | $A_{DG}$ 补充 | 缺失 | 学术需申请、商用付费；**不可再分发** |
| SMILES 源（PubChem PUG-REST 按 INCHI_KEY 501/621；或 ChEMBL API 417/621） | 分子图 GIN、Morgan FP | 缺失（GDSC 文件只有 DRUG_NAME/SYNONYMS/TARGET/CHEMBL_ID/INCHI_KEY，无 SMILES） | PubChem 公共数据；ChEMBL CC BY-SA |
| RDKit（Python 包） | 分子图/Morgan FP 特征 | **未安装**（pyproject 仅 `graph` extra 含 rdkit>=2026.9） | BSD 3-Clause |
| panel 清单（409/410/523 基因） | 学生突变输入 | 缺失 | 内部定义/文献，需记录版本 |
| CCLE legacy 450K beta（旧 CCLE portal 的 `CCLE_DNA_methylation_450K`，~1,000 系） | 替代 RRBS 作教师 METH / VAE METH 头 | 缺失（本地只有 RRBS） | Broad/CCLE 旧资源条款待核实（见 `data/licenses/`） |
| TCGA-BRCA/LUAD（含 HM450 METH） | L1 补全保真扩展 | 仅有 CESC（METH 312 文件） | GDC 开放层（需引用） |
| Edge 预印本 1,603 基因 PPI 子集清单（STRING v11 口径） | VAE 补全目标空间 | 缺失 | 文献补充材料（引用原文） |
| GDSC1 / CTRPv2 / PRISM | 外部验证（计划书 §3.2 #2/#4/#5） | 缺失 | 各自学术条款，下载前补数据卡 |

## 6. 待核实项与复现命令

待核实：① STRING 官方发布文件是否本就双向重复（本审计只证明本地文件如此；若官方如此，则“边数”表述需改）；② STRING v12.5 许可以官网为准，并统一计划书 §3.4 与 `data/raw/README.md` 的矛盾；③ 本 METH 文件的发布出处（疑 CCLE 2018 / Ghandi et al. 2019 补充文件），需查原始下载页确认 “RRBS, TSS1kb, hg19”；④ DGIdb 药名匹配需用 ChEMBL ID/InChIKey 做二次桥接后重算覆盖；⑤ 1,603 基因清单口径；⑥ HGNC/CCLE 旧资源再分发条款（`data/licenses/README.md` 已标注“待核实”）。

```bash
# 唯一无向蛋白对 + 阈值统计（流式；去重后阈值表见 §2.1）
.venv/bin/python - <<'PY'
import gzip
base='data/raw/string/'
prots=set(); T=[400,700,900]; up={t:set() for t in T}; n=0
with gzip.open(base+'9606.protein.links.v12.5.txt.gz','rt') as f:
    f.readline()
    for line in f:
        a,b,s=line.rstrip('\n').split(' '); s=int(s); n+=1
        prots.add(a); prots.add(b)
        k=(a,b) if a<b else (b,a)
        for t in T:
            if s>=t: up[t].add(k)
print('rows',n,'proteins',len(prots),{t:(len(up[t])) for t in T})
PY
# DGIdb 交互/药名覆盖（GDSC 286 药名的规范化匹配）
.venv/bin/python - <<'PY'
import pandas as pd, re
it=pd.read_csv('data/raw/dgidb/interactions.tsv',sep='\t',comment='#',low_memory=False)
gd=pd.read_csv('data/raw/gdsc/GDSC2_fitted_dose_response_27Oct23.csv',usecols=['DRUG_NAME'],dtype=str)
norm=lambda s: re.sub(r'[^A-Z0-9]','',str(s).upper())
names=set(it['drug_name'].dropna().map(norm))|set(it['drug_claim_name'].dropna().map(norm))
g=set(gd['DRUG_NAME'].dropna().unique()); hit=[d for d in g if norm(d) in names]
print('GDSC drugs covered:',len(hit),'/',len(g))
PY
```

> 备注：以上数字均为本机 `data/raw/` 实测（2026-11-17）；凡是“推定/待核实”均已标注，引用时以本笔记表格为准并保留 MD5（§1）以便复现。

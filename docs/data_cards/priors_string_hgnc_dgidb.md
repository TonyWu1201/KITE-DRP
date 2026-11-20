# 数据卡：先验网络（STRING / HGNC；DGIdb 已停用）

- 状态：已核实（本地实测，2026-11-17）；DGIdb 于 2026-11-20 停用
- 更新日期：2026-11-20
- 负责人：数据审计 B
- 相关：计划书 §3.4、§4.4；`src/kite_drp/data/priors.py`、`genes.py`、`opentargets.py`

## 1. 基本信息

| 项 | 值 |
|---|---|
| 数据集 | STRING v12.5、HGNC Gene.csv；Open Targets 26.09（DTI 主源）；DGIdb 仅留档 |
| 提供方 | STRING Consortium / HGNC / Open Targets |
| 版本 | STRING v12.5、HGNC snapshot（最新 date_modified 2025-03-31）、Open Targets 26.09 |
| 文件 | `9606.protein.links.v12.5.txt.gz`、`9606.protein.aliases.v12.5.txt.gz`、`Gene.csv`、`opentargets/26.09/*` |
| 存储 | `data/raw/string/`、`data/raw/hgnc/`、`data/raw/opentargets/` |

## 2. 许可与引用

- STRING v12.5：**CC BY 4.0**（含商用，需署名）。
- HGNC：学术/公开使用，需引用 HGNC。
- Open Targets：**CC0 4.0**，可再分发。
- DGIdb：混合来源、再分发受限；**已停用**，仅历史留档。
- 详见 `data/licenses/{string,hgnc,opentargets,dgidb}.md`。

## 3. 内容与口径

- **STRING links 每对互作存双向两行**：10,826,054 行 = 5,413,027 唯一无向边；≥700 为 282,864 对 / 17,525 蛋白。
- ENSP→HGNC symbol 映射率 97.67%（19,191/19,649）；458 个蛋白无 symbol；1 个歧义。
- HGNC 44,083 行 / 19,291 protein-coding；1,083 个 alias 多义、231 个 prev_symbol 与当前 symbol 冲突 → 仅唯一才采用。
- Open Targets：6,092 条 MoA；药物族匹配 214/286 个 GDSC 化合物，其中 105 个有可用靶点边。
- DGIdb（停用前）：122,904 交互 / 80,483 药 / 85,738 基因；GDSC 286 药名覆盖 202/286。

## 4. 预处理

- PPI：`(min,max)` 去重 → 阈值 ≥700 → ENSP→symbol → 基因级去重 → 边属性 `combined_score`。
- 基因真源：HGNC `symbol` + `ensembl_gene_id`/`entrez_id`；映射顺序 current → prev → alias（唯一）。
- DTI：GDSC `TARGET` + Open Targets MoA；母体/盐型族 + InChIKey + 唯一名称匹配；限制到 U_teach_ge。
- 产物：`data/interim/canonical/{ppi,dti,pathways,genes}.parquet`、`data/processed/{dti.parquet,dti_coverage.json}`。

## 5. 已知缺陷与风险

- Open Targets 覆盖 213/286 药；73 个内部代号/无结构药仍无 DTI，需在评估中报告覆盖组与全量。
- DGIdb 已停用；旧 T1 权重对应的 DTI 版本不同，复现时需固定 `dti.parquet` 哈希。
- STRING 的 458 个无 symbol 蛋白无法补回；基因级折叠会轻微合并节点。
- Reactome/GO GMT 本地缺失，通路先验当前为空（学生路线 A 降级 B）。
- KEGG/DrugBank 不可再分发，未纳入。

## 6. 使用计划

- 教师：PPI 图（A_GG）、DTI 图（A_DG）与通路可解释性。
- 学生：通路分数（资源到位后）或 HVG 路线。
- 图对照：随机重连必须基于去重后的边表，并保持同度分布。

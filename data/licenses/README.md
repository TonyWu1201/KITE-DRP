# `data/licenses/` —— 数据源许可与引用要求

每个数据源一份说明（来源、许可类型、是否允许再分发、必须引用的文献、使用限制），
供论文/开源 release 前核对。**本目录只放说明文本，不放授权文件/签名字节等敏感材料。**

## 许可摘要（当前已知信息，标注"待核实"的需在发布前确认原文）

| 数据源 | 位置 | 许可 / 使用要求 | 状态 |
|---|---|---|---|
| GDSC2 | `data/raw/gdsc/` | 学术使用需引用原文献；再分发前确认官方条款 | ⚠️ 待核实原文 |
| DepMap / CCLE 公共数据 | `data/raw/depmap/` | DepMap Public 数据（CC BY 4.0 量级）需署名；以官方 terms 为准 | ⚠️ 待核实原文 |
| 旧 CCLE 资源（甲基化/注释） | `data/raw/ccle/` | Broad/CCLE 旧资源条款；非权威主表，仅 fallback | ⚠️ 待核实原文 |
| STRING v12.5 / v12.0 | `data/raw/string/` | CC BY 4.0，仅需署名（须引用 STRING 文献） | ✅ 已知 |
| PubChem（SMILES） | `data/raw/drugs/` | PubChem 公共数据；再分发前核对 NCBI 条款 | ⚠️ 待核实原文 |
| Reactome | `data/raw/reactome/` | 开放许可（CC BY 4.0 量级），需署名 | ⚠️ 待核实原文 |
| GO | `data/raw/go/` | CC BY 4.0，需署名 | ✅ 已知 |
| cBioPortal gene panels | `data/raw/panel/` | 基因列表为事实信息；门户条款待核 | ⚠️ 待核实原文 |
| Open Targets 26.09 | `data/raw/opentargets/` | CC0 4.0 | ✅ 已知 |
| ChEMBL 37 | `data/raw/chembl/` | CC BY-SA 3.0（copyleft） | ✅ 已知 |
| GDSC1 / CTRPv2 / PRISM | `data/raw/{gdsc,ctrp,prism}/` | 学术引用；再分发待核 | ⚠️ 待核实原文 |
| DGIdb v5.0.13 | `data/raw/dgidb/` | **已停用**：混合来源、再分发条款不清；DTI 改用 Open Targets CC0 | ⛔ 不再进入管线 |
| HGNC | `data/raw/hgnc/` | 引用 HGNC；再分发条款以官方为准 | ⚠️ 待核实原文 |
| TCGA（GDC 开放层） | `data/raw/tcga/` | NIH 开放获取，需引用；受控层不得下载/分发 | ✅ 开放层可用 |
| TCGA-CDR | `data/raw/tcga/cdr/` | PanCan-Clinical-2018，引用 Liu et al. 2018 | ✅ 已知 |
| Ding, Zu & Gu 2016 | `data/raw/tcga/ding2016/` | 引用原文；为人工整理的响应标签，非 TCGA 官方药敏数据 | ✅ 已知 |
| Open Targets Platform 26.09 | `data/raw/opentargets/` | CC0 4.0；可再分发、可商用 | ✅ 已核实 |
| ChEMBL 37（API 两表） | `data/raw/chembl/` | CC BY-SA 3.0；copyleft，派生库同许可 | ✅ 已核实 |
| GDSC1 release 8.5 | `data/raw/gdsc/` | 学术使用需引用；再分发前确认官方条款 | ⚠️ 待核实原文 |
| CTRPv2 v21 AUC | `data/raw/ctrp/` | CC BY 4.0；需引用 Seashore-Ludlow 2015 / Rees 2016 | ✅ 已核实 |
| PRISM Repurposing | `data/raw/prism/` | CC BY 4.0（DepMap/CCLE；以官方 terms 为准） | ⚠️ 待复核原文 |

> 逐源文件：`gdsc1.md`、`gdsc2.md`、`depmap.md`、`string.md`、`hgnc.md`、`dgidb.md`、`tcga.md`、`tcga_cdr.md`、`ding2016.md`、`pubchem.md`、`reactome.md`、`go.md`、`cbioportal.md`、`opentargets.md`、`chembl.md`、`ctrp.md`、`prism.md`。

> 更细的文件清单与版本信息见 [`../raw/README.md`](../raw/README.md)（原始数据索引入口）。

## 模板

见 [`_TEMPLATE.md`](_TEMPLATE.md)。新增数据源时复制模板为 `<source>.md`。

## 红线

- 不把受限数据（TCGA 受控层、需授权的 DrugBank/KEGG 等）提交到公开仓库；
- 公开 release 只提供下载脚本与派生特征，不打包受限原始数据（计划书附录 B.2-8）；
- 论文与仓库 README 的引用必须与这里的记录一致。

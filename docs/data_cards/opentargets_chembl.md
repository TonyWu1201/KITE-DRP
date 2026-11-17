# 数据卡：Open Targets / ChEMBL（药物-靶点与机制）

- 状态：已生成（2026-11-17）
- 更新日期：2026-11-17
- 负责人：数据管线 / B
- 相关：计划书 §3.4、§4.4、§4.6；D31；`src/kite_drp/data/priors.py`

## 1. 基本信息

| 项 | 值 |
|---|---|
| 数据源 | Open Targets Platform 26.09；ChEMBL 37 |
| 本地位置 | `data/raw/opentargets/26.09/`、`data/raw/chembl/37/` |
| 表 | Open Targets：target / drug_molecule / drug_mechanism_of_action / clinical_indication；ChEMBL：mechanism / drug_indication |
| 许可 | Open Targets CC0 4.0；ChEMBL CC BY-SA 3.0 |

## 2. 用途

- 药物-靶点边（A_DG）：补 GDSC `TARGET` 与 DGIdb 的缺口。
- 药物机制解释：`action_type`、`mechanism_of_action` 用于可解释性。
- 药物名称/InChIKey 桥接：GDSC ↔ ChEMBL/Open Targets。

## 3. 预处理

- 脚本：`src/kite_drp/data/priors.py::build_dti`。
- 匹配：GDSC InChIKey → Open Targets `drug_molecule.inchiKey`；目标 Ensembl → HGNC symbol；ChEMBL target ID 经 Open Targets `target.dbXrefs` 映射。
- 只保留 `U_teach_ge` 内的靶点；未匹配药物进入 mask。
- 输出：`data/interim/canonical/dti.parquet`（列含 source/action_type/mechanism/target_name），并并入 `compounds.parquet` 的 `target_genes`。

## 4. 当前实测

- Open Targets：78,733 target 行、19,170 drug、6,092 MOA；匹配 **173** 个 GDSC 化合物，466 条边。
- ChEMBL：7,561 mechanism；匹配 173 化合物，242 条边。
- 合并 DTI：4,395 边 / **237/286** 化合物 / 1,663 靶点基因；较 GDSC 单源（186 药、334 边）显著提升。
- 49 个化合物仍无 DTI（多为内部代号/无 InChIKey 化合物），保留 mask。

## 5. 已知缺陷与风险

- Open Targets/ChEMBL 版本更新会改变机制/靶点；需记录 release + 文件 hash。
- ChEMBL 为 CC BY-SA 3.0，派生边表的再分发需复核。
- 机制方向（agonist/antagonist）不等同于敏感性方向，模型不应把它当标签。
- 同名药物可能有多个盐型/母体 ID；当前按 InChIKey 优先。

## 6. 使用计划

- 教师 DTI 图（A_DG）与解释；
- 学生不直接使用 DTI（端侧只有 panel + 结构特征），但可用于药物特征消融。

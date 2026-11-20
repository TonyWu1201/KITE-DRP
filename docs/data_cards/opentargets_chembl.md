# 数据卡：Open Targets / ChEMBL（药物-靶点与机制）

- 状态：已更新（2026-11-20，DGIdb 停用）
- 负责人：数据管线 / B
- 相关：计划书 §3.4、§4.4、§4.6；D31；`src/kite_drp/data/opentargets.py`、`src/kite_drp/data/priors.py`

## 1. 基本信息

| 项 | 值 |
|---|---|
| 数据源 | Open Targets Platform 26.09；ChEMBL 37（可选） |
| 本地位置 | `data/raw/opentargets/26.09/`、`data/raw/chembl/37/` |
| 表 | Open Targets：target / drug_molecule / drug_mechanism_of_action / clinical_indication；ChEMBL：mechanism / drug_indication |
| 许可 | Open Targets CC0 4.0；ChEMBL CC BY-SA 3.0（默认不参与 DTI） |

## 2. 用途

- **DTI 主源**：Open Targets `drug_mechanism_of_action` → GDSC 化合物–HGNC 靶点边。
- 药物机制解释：`action_type`、`mechanism_of_action`、`targetType` 用于逐边追溯。
- 药物名称/InChIKey 桥接：GDSC ↔ Open Targets；支持母体/盐型族匹配。
- DGIdb 已于 2026-11-20 停用，不进入 `build_dti`；ChEMBL 仅作为显式可选补充。

## 3. 预处理

- 脚本：`src/kite_drp/data/opentargets.py::build_opentargets_edges`；
  `src/kite_drp/data/priors.py::build_dti` 负责与 GDSC `TARGET` 合并。
- 匹配顺序（高置信优先）：
  1. GDSC `CHEMBL_ID` ↔ Open Targets `drug_molecule.id`（含 `parentId`/`childChemblIds` 药物族）；
  2. GDSC `INCHI_KEY` ↔ Open Targets `drug_molecule.inchiKey`；
  3. GDSC 药名/别名 ↔ Open Targets `name`/`synonyms`/`tradeNames`；仅当名称只对应一个药物族时采用，同名歧义直接跳过。
- 靶点：Open Targets `targets` Ensembl ID → `target.approvedSymbol`；失败时用 `targetName` 经 HGNC 兜底；只保留 `U_teach_ge`。
- 每条边保留 `ot_drug_id`、`match_method`、`target_id`、`target_type`、`source_version`、
  `action_type`、`mechanism`，供审计。
- 输出：
  - `data/interim/canonical/dti.parquet`（canonical 合并表）；
  - `data/processed/dti.parquet`（训练/图构建自包含表）；
  - `data/processed/dti_coverage.json`（覆盖率与匹配方式统计）。

## 4. 当前实测

- Open Targets：78,733 target 行、19,170 drug、6,092 MOA。
- Open Targets 药物族匹配：214/286 个 GDSC 化合物；其中 105 个有可用 MoA 靶点边。
- 合并 DTI（GDSC `TARGET` + Open Targets，`U_teach_ge` 内）：**640 边 / 213 个化合物 / 301 个靶点基因**。
- 来源计数：GDSC 334 边、Open Targets 445 边；其中 139 条边由两源共同支持。
- 未覆盖 73 个化合物，多为纯数字名/内部代号/无 ChEMBL/InChIKey 记录；保留 mask，训练时精确回退。

> 以上数字来自 2026-11-20 重建的 canonical/processed；重建后以
> `data/processed/dti_coverage.json` 与 `manifest.json` 为准。

## 5. 已知缺陷与风险

- Open Targets 版本更新会改变机制/靶点；需记录 release + 文件 hash（已写入 `download_manifest.json`）。
- Open Targets 的 MOA 记录本身来源于 ChEMBL 等上游；本项目按 Open Targets 的 CC0 发布口径使用，公开 release 前仍建议复核上游条款。
- 机制方向（agonist/antagonist）不等同于敏感性方向，模型不应把它当标签。
- 同名药物可能有多个盐型/母体 ID；已按药物族合并，但名称歧义时不做低置信兜底。
- 与旧 DGIdb 版本相比，边数和化合物覆盖下降（旧 4,395 边 / 237 药）；这是许可合规的预期代价，覆盖率需在模型评估中显式报告。

## 6. 使用计划

- 教师 DTI 图（A_DG）与解释；
- 学生不直接使用 DTI（端侧只有 panel + 结构特征），但可用于药物特征消融；
- 后续局部图实验以 `dti.parquet` 的 `match_method`/`target_id` 作为合规靶点与证据等级依据。

# ADR-0006：DTI 主源切换为 Open Targets，停用 DGIdb

- 状态：已接受
- 日期：2026-11-20
- 相关：计划书 §3.4、§4.4、§4.6；`plans&notes/2026-11-20_教师模型复盘与图教师重构方案.md` §5.1

## 背景

旧 DTI 合并 GDSC `TARGET` + Open Targets + ChEMBL + DGIdb，共 4,395 边 / 237 个
化合物。复算显示约 97.5% 的边含 DGIdb 来源标记。DGIdb 是混合来源聚合库，
逐源再分发条款不清，不适合作为公开 release 的主 DTI 源。

Open Targets Platform 为 CC0 4.0，`drug_mechanism_of_action` 提供可追溯的
药物–靶点机制记录；项目已下载 26.09 release。计划书原本即把 Open Targets
列为首选开源边源，DGIdb 仅作兜底/解释校验。

## 选项

1. **保留 DGIdb，公开 release 时删除其边**：需要维护两套 DTI，训练/发布口径不一致；
2. **DGIdb → Open Targets 直接替换**（采用）：DTI 只由 GDSC `TARGET` 与 Open Targets
   MoA 构建；ChEMBL 默认不参与，保留显式开关；
3. **完全去掉外部 DTI，只用 GDSC `TARGET`**：覆盖 186/286，损失 Open Targets 的
   27 个增量化合物与机制注释。

## 决策

选择 2，并补充以下口径：

- `build_dti(compounds, master, *, allowed_genes, opentargets_dir, chembl_dir=None)`；
- 匹配优先级：GDSC `CHEMBL_ID`（含 `parentId`/`childChemblIds` 药物族）→
  `INCHI_KEY` → 唯一规范化药名；名称歧义不做低置信兜底；
- 每条边保留 `ot_drug_id`、`match_method`、`target_id`、`target_type`、
  `source_version`、`action_type`、`mechanism`，便于逐条追溯；
- 新增 `data/processed/dti.parquet` 与 `dti_coverage.json`，训练/图构建只读
  processed，不依赖 canonical；
- `graph_inputs.npz` 的 DTI 部分改为读取 processed `dti.parquet`；
- 配置：`priors.dti.sources=[gdsc_target, opentargets]`、`include_chembl=false`；
  `drug_features.targets.sources=[gdsc_target, opentargets]`。

## 后果

**正面**

- DTI 主源为 CC0，公开 release 的许可风险显著下降；
- 覆盖/匹配方式/未覆盖药物有独立审计文件，满足计划书 §5.1 的追溯要求；
- processed 自包含边界延伸到 DTI，clone 后图构建不再读 canonical。

**负面 / 权衡**

- 覆盖从 237/286 降至 213/286；边数从 4,395 降至 640，靶点基因从 1,663 降至 301；
- 旧 T1 权重对应旧 DTI，复现/对比时必须固定 `dti.parquet` 哈希；
- 73 个内部代号/无结构药仍无 DTI，模型需保留精确回退并在评估中分组报告；
- Open Targets 的 MoA 上游仍包含 ChEMBL 等来源，公开 release 前建议复核其 CC0
  发布口径。

**后续动作**

- 图教师实验按 `dti_coverage.json` 报告“有合规靶点/有可观测邻居/可有效重连”的
  药物数；
- 若后续需要 ChEMBL 补充，显式设置 `priors.dti.include_chembl=true` 并单独审计
  CC BY-SA 影响。

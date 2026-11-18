# ADR-0005：processed 自包含（教师训练可脱离 raw/canonical）

- 状态：已接受
- 日期：2026-11-18
- 相关：计划书 §3.8、§7.2、附录 B.2；`plans&notes/2026-11-17_数据预处理详细规划.md` §2.3

## 背景

G0 把 `data/processed/` 定义为「无 raw 也能训练的最小分发单元」，但教师训练实际需要的
信息仍有三个缺口：

1. **标签只在 canonical**：`labels.parquet` 位于 `data/interim/canonical/`（git 忽略），
   clone + `git lfs pull` 后没有标签，教师无法训练；
2. **PPI/DTI 边表只在 canonical**：processed 只有药物 targets multi-hot，缺少 T1 图需要的
   `ppi.parquet` / `dti.parquet`；
3. **药物分子图缺失**：`drug_features.npz` 只有 Morgan/targets/scaffold/SMILES，
   `drugs.py` 未实现 `molecular_graphs()`，GIN 无法训练。

另外实测发现 `manifest.json` 存的是**构建机的绝对路径**（Linux `/home/...`），
迁移到另一台机器后 `Manifest.verify()` 全部报 missing，破坏了可复现性。

## 选项

1. **让模型直接读 canonical**：改动最小，但 clone 后不可用，且训练侧会绕过 fit_stats 契约；
2. **新增三个 processed 产物**（采用）：
   - `teacher_pairs.parquet`：pair 表 + 教师池/主池标记；
   - `graph_inputs.npz`：PPI/DTI 边表 + 基因/化合物节点表；
   - `drug_features.npz` 增加分子图 ragged 数组；
   同时把 manifest 路径改为仓库相对路径。

## 决策

选择 2，并补充以下口径：

- `teacher_pairs` 保留全部 **已映射** pair（235,478 行），用 `has_teacher_omics`
  （176,487 行）与 `in_main_pool`（126,803 行）标记子集，避免为不同池重复落盘；
- `graph_inputs` 保留 `U_teach_ge` 内的**全量** PPI/DTI 边（16,197 基因 / 253,279 PPI /
  4,395 DTI），T1 训练时再按基因空间裁剪；不在此处固化 2,000–3,000 基因子图；
- 分子图采用 77 维原子特征 + 4 维键特征（计划书 D3 口径），ragged 存储
  （`graph_atom_ptr` / `graph_edge_ptr`），双向边、全局原子索引；
- `Manifest` 落盘相对路径；读取旧 manifest 时按 `data/` 后缀重定位，跨机器可校验；
- 新增 `kite-drp data preflight`：不读 canonical，构造 `TeacherPairDataset` 验证
  clone 可训练性。

## 后果

**正面**

- clone + LFS 后即可跑教师数据加载与训练（`data preflight --target teacher` 全绿）；
- 训练侧不再触碰 canonical，泄漏边界更清晰；
- manifest 跨机器可校验，`data validate --all` 在迁移后仍能通过。

**负面 / 权衡**

- `teacher_pairs` 约 235k 行（≈15 MB）与 `graph_inputs`（≈4 MB）进入 LFS；
- 分子图需要 RDKit 重建 canonical 药物特征（`data canonical --only drug_features`），
  未装 RDKit 时显式降级为 `graph_has_graph=False`；
- `graph_inputs` 保留全量边，T1 侧必须显式做基因子图裁剪，否则节点数会到 1.6 万。

**后续动作**

- T1 实现在 `models/teacher/graph.py` 中完成基因空间裁剪与保度重连；
- 若后续新增模态（METH/RPPA），沿用「processed 自包含 + manifest 相对路径」口径。

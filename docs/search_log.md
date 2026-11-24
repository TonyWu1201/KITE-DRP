# 文献检索记录（Search Log）

计划书 §1.2 所述"据本课题组检索（截至 2026-11）"的检索式与记录汇总。
由负责文献的成员/Agent 维护，每次检索追加一行。

## 记录格式

| 日期 | 数据库 | 检索式 | 命中数 | 筛选后保留 | 结论/备注 |
|---|---|---|---|---|---|
| 2026-11-09 | PubMed | `("drug response prediction" OR "drug sensitivity prediction") AND (knowledge distillation OR teacher-student) AND (graph neural network OR multi-omics)` | — | — | 计划书初稿检索，详见 `plans&notes/` 文献笔记 |
| 2026-11-23 | Europe PMC / Crossref / bioRxiv / 出版方页面 | `("drug response" OR "drug sensitivity") AND (PPI OR "protein-protein interaction" OR DTI OR "knowledge graph") AND (propagation OR pretrain* OR "auxiliary task" OR "cold drug")`；并追查 NetGP/DGDRP/CSG²A/KGDRP/DTLCDR/DGP-LA/PINNACLE/XPert/Pillbox 的全文与 ablation | 277（PPI 主式）；10（KG 主式）；33（pretrain/self-supervised 主式） | 约 20 | 结论与改进方案见 `plans&notes/2026-11-23_PPI-DTI先验学习与T2改进调研.md`；重点：NetGP 传播、KGDRP 辅助任务、DTLCDR DTI 补全、XPert/Pillbox 负结果 |
| 待补 | — | — | — | — | 补录 2026-11 前的检索式与导出文件位置 |

## 说明

- 检索式需可复现：记录数据库、日期范围、语言限制、筛选标准；
- 保留的文献全文/笔记按计划书约定存放（如 `plans&notes/papers/`，不放受限 PDF）；
- 引用到计划书/论文的结论必须能回溯到这里的检索记录。

# 模型卡：Teacher T1（图增强 P1 教师）

- 状态：草稿（v0.1；LDO 快速筛选进行中）
- 更新日期：2026-11-20
- 权重路径 / SHA256：待发布
- 相关：计划书 §4.4、D1；`configs/teacher/t1.yaml`；`src/kite_drp/models/teacher/t1.py`；
  `src/kite_drp/train/t1_graph.py`

## 1. 基本信息

| 项 | 值 |
|---|---|
| 模型类型 | 教师 T1（训练期图教师，不参与端侧部署） |
| 参数量 | 6,097,099（backbone 4,595,526 + graph 1,501,573） |
| 图节点 | 基因 ~2,770 + 药物 286 + 细胞系 716 |
| 图边（LDO fold0） | PPI 49,402 有向（~24.7k 无向）、DTI 4,395（旧 DGIdb 版）、cell-gene 213,403 |
| 输出 | 回归 `-z(LN_IC50)` + 敏感概率 + graph-only 辅助头 |

## 2. 设计

T1 是 **P1 的图增强残差版**，不是另起一套全图模型：

1. P1 的 18,396×16 基因嵌入表取子集得到 `G0`（2,770×16）；
2. `G0` 经 2 层 PPI GNN 得到 `H`，再以 gate=0 的残差写回
   `G = G0 + tanh(α)·W(H)`；
3. P1 的 target 分支与药物靶点编码器共用 `G`（替代原来的独立 gene table /
   EmbeddingBag），直接补 P1 “药物基因级、细胞通路级”的共享空间缺口；
4. DTI（gene→drug）与 cell-gene（gene→cell）是**有向边**，只更新 drug/cell
   节点，避免 gene 节点聚合全部细胞导致表示被平均；
5. `z_cell/z_drug` 以 gate=0 残差融合 `h_cell/h_drug`；graph-only 头做深监督，
   防止融合 gate 学到 0 后图分支不学习。

初始化时 `α=β=γ=0`，且 `G0` 从 P1 基因表拷贝，因此 **T1 ≡ P1**（单测与真实
LDO batch 均验证，max |Δy| ~1e-7）。图必须先“挣出”增益。

## 3. 图构建口径

- 基因空间：全部药物靶点 ∪ 训练 fold HVG top1000 ∪ 训练细胞 top500 突变基因，
  与 `graph_inputs.npz` 的 16,197 基因取交集；LDO fold0 实测 2,770 基因；
- PPI：STRING combined_score ≥0.7 的诱导子图，双向边，边特征 score；
- DTI：只保留与 286 个 GDSC 药物对齐的边；边特征 score；
- cell-gene：每个细胞取表达 top10% + 全部突变基因；边特征
  `[expr_z, mutation, cnv_z, cnv_observed]`；
- 所有 fold 内统计量（HVG、突变频率、表达均值/方差）只用 train cells；
- 随机对照：PPI/DTI 度保持双边交换；去边：`edge_drop=ppi|dti|cg`。

## 4. 训练

- 初始化：同 split/fold P1 checkpoint；`G0` 从 P1 基因表拷贝；三个 gate=0；
- 损失：Huber + 0.5×BCE + 0.3×graph-only Huber + 0.2×graph-only BCE；
- 优化：AdamW lr=1.5e-4、wd=1e-3、cosine warmup、batch=512、AMP；
- 早停：val SCC（与 P1 相同口径）；正式实验 ≥3 seeds；
- 本机耗时（RTX 4060 Ti 8GB）：空闲时 T1 ~100 s/epoch、P1 ~47 s/epoch；
  当前桌面/GPU 共享负载下实测约 5 min/epoch。按 3 折 × 2 seeds × 20 epochs
  估算约 10 h，**仍低于 48 h 本机运行门槛**。

## 5. 评估结果（LDO 快速筛选，seed 0，3 折）

| 变体 | global SCC | per-drug SCC | AUROC |
|---|---:|---:|---:|
| P1（3 折 × 2 seeds） | 0.3480 ± 0.0126 | **0.6350 ± 0.0325** | 0.6653 |
| T1 v1（embedding 残差，seed0） | 0.3519 | 0.6414 | 0.6655 |
| T1 v1 + graph-only blend w=0.3（seed0） | 0.3562* | 0.6444* | — |
| **T1 v2（预测级 gate，seed0）** | **0.3614** | **0.6437** | **0.6681** |
| T1 v2 + random PPI（fold0） | 0.3538 | 0.6233 | 0.6694 |
| T1 v2 + random DTI（fold0） | 0.3500 | 0.6240 | 0.6674 |
| T1 v2 去掉 cell-gene 边（fold0） | 0.3453 | **0.6242** | — |
| graph-only 头单独（v1） | 0.350 × 3 | 0.6325* | — |

\* blend/graph-only 为 val 后处理诊断，不是独立训练结果；测试集未评。

- T1 v2 逐折 per-drug ΔSCC vs P1（seed0）：+0.0176 / +0.0004 / +0.0196，
  均值 **+0.0125**；相对 P1 两 seed 均值 0.6350 仅 **+0.0087**，小于 P1 的 seed
  SD 0.0325 —— 即 T1 的增益在 seed 噪声范围内；
- **随机图对照复现了全部增益**：fold0 real 0.6232、random PPI 0.6233、
  random DTI 0.6240；PPI/DTI 图结构对主指标没有可检测贡献；
- fold1 的 v2 在 epoch0（P1 初始化）即最佳，5 epoch 后早停——单一学习率会把 P1
  推离最优点；v3 已加入 `freeze_p1_epochs=5` + `p1_lr_mult=0.1`，但仅在图信号
  成立时才有继续投入的意义；
- `graph_pred_gate` 在 fold0/fold2 学到 ~0.35，与后处理最优 0.3–0.4 一致；
- **结论（fold0 负对照已足够明确）**：
  - random PPI 0.6233、random DTI 0.6240、no cell-gene 0.6242，全部 ≥ real 0.6232；
  - 当前实现中没有证据表明 PPI/DTI/cell-gene 图结构对主指标（逐药 SCC）有贡献；
  - T1 相对 P1 的 +0.0125 更像“额外分支/集成/正则化”的容量效应；
  - 按计划书 R4，**P1 继续做主教师，T1 降级为附录/负对照**；若继续图方向，
    应改为 cell-specific target 子图 + 边级自监督，而不是全局 gene 图。
  - 若需要 T1 的 +0.0125 集成增益，优先用 **P1 + 第二 MLP 预测头**（无图、参数/时间
    更省），而不是 6.1M 的图分支。
  - `t1_no_dti` / `t1_no_ppi` 去边对照因时间原因中止；random PPI/DTI + no-CG
    已足以支持上述结论。
  - **Go/No-Go：未通过**（Δ per-drug SCC +0.0125 < 0.02，随机图复现增益）。

## 6. 适用范围与限制

- 图节点包含全部 716 个细胞系（transductive），LDO/scaffold 只留出药物标签；
- 73/286 个药物无靶点（OT+GDSC clean DTI）；T1 对它们退化为 Morgan + 图上下文，需按“有无 DTI”分组报告；
- 2026-11-20 已重建 clean DTI：GDSC `TARGET` + Open Targets MoA（CC0），DGIdb 不再参与；旧 T1 权重仍对应旧 DTI，复现/对比时需固定版本；
- **科研原型，不用于临床诊断或治疗决策。**

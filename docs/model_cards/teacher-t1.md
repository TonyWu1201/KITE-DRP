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
| 图边（LDO fold0） | PPI 49,402 有向（~24.7k 无向）、DTI 4,395、cell-gene 213,403 |
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

## 5. 评估结果

待补：LDO 3 折 × 2 seeds 快速筛选（P1 / T1 / random-PPI / random-DTI），
主指标为逐药 SCC（raw LN_IC50）。

## 6. 适用范围与限制

- 图节点包含全部 716 个细胞系（transductive），LDO/scaffold 只留出药物标签；
- 49/286 个药物无靶点；T1 对它们退化为 Morgan + 图上下文，需按“有无 DTI”分组报告；
- 当前 DTI 97.5% 含 DGIdb，公开 release 前需重建 clean DTI 或提供 PPI-only 版本；
- **科研原型，不用于临床诊断或治疗决策。**

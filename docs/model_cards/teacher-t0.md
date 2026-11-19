# 模型卡：Teacher T0（多模态交叉注意力教师）

- 状态：草稿（M2 教师 v0；P0/P1 变体快速验证完成）
- 版本：`v0.2.0`
- 更新日期：2026-11-19
- 权重路径 / SHA256：`experiments/<run>/fold<k>/best.pt`（未发布）
- 相关：计划书 §4.3、D1/D2；`configs/teacher/t0.yaml`；`src/kite_drp/models/teacher/t0.py`

## 1. 基本信息

| 项 | 值 |
|---|---|
| 模型类型 | 教师 T0（无图，训练期使用） |
| 参数量 | v1 4,728,260；P1 4,889,862（均 ≤5M 目标） |
| 输入模态 | GE/MU/CNV 通路分数（1,000 维）+ H3CDR 式 top-15 邻居上下文；P1 额外使用靶点基因级 GE/MU/CNV |
| 药物表示 | Morgan 2048 + 靶点基因嵌入（GIN 可选，默认关闭） |
| 输出 | 回归 `-z(LN_IC50)` + 敏感概率 |

## 2. 结构

1. 每个模态：`[CLS] + top-64 通路 token`（通路嵌入 + 分数投影 + 模态类型嵌入）；
2. 药物 token 作 query，对组学 token 做交叉注意力；
3. 拼接 `[药物, GE, MU, CNV, 邻居]` 后过 2 层 TransformerEncoder（d=256, 4 heads）；
4. `z_cell = mean(组学 token)`，`z_drug = 药物 token + 残差`，`z_pair = z_cell + z_drug`；
5. 回归 + 分类双头。

## 3. 训练

- 相似性：H3CDR 式 GE+MU 高斯核 → 归一化全核 → top-15 → 3 轮交叉融合；
  **fit 只在 train 细胞**，val/test 只连 train 锚点（`docs/adr/0005`、`data/similarity.py`）。
- 损失：Huber + 0.5 × masked BCE；AdamW lr=2e-4、wd=1e-3、dropout=0.1、
  模态 dropout=0.05、batch=512、AMP；早停 val SCC（patience 5）。

## 4. 评估结果（LCO `p1_lco_s0`，5 折，单 seed）

| 指标 | 数值（mean ± SD） | 评测集 |
|---|---|---|
| R² | 0.178 ± 0.037 | val |
| PCC | 0.440（5 折均值） | val |
| SCC | 0.427 ± 0.042 | val |
| AUROC | 0.704 ± 0.020 | val |
| RMSE | 0.914 | val |

**消融（详见 [`teacher-t0-ablations.md`](teacher-t0-ablations.md)）**：

- 5 折去掉邻居 token：SCC 0.415 vs 0.427（Δ−0.013，p=0.37），
  R² 0.170 vs 0.178（Δ−0.008，p=0.66）→ 方向为正但不显著；
- 5 折 GE-only：SCC 0.438 vs 0.427（p=0.53），R² 0.176 vs 0.178，
  参数 3.70M vs 4.73M → 多组学在教师精度上无显著增益；
- 5 折 T0 vs B0：SCC 0.427 vs 0.440（p=0.27）→ 交叉注意力结构未显著优于双线性基线。

因此论文中必须如实报告“多组学/注意力/相似性 token 的边际增益不显著”；
教师侧的差异化留给 T1 图结构或蒸馏实验检验。

### 4.1 P0 快速验证（2026-11-19）

P0（排序损失 / token-readout / 缺失 mask / 逐折 seed）的代码已落在
`feat/t0-p0-ranking-readout`；快速验证为 LCO fold 0/2 × 2 seeds × 5 epochs：

| 变体 | R² | SCC | AUROC |
|---|---:|---:|---:|
| T0 v1 | 0.155 | **0.414** | **0.701** |
| P0-lite（sum, no-rank, 8×64） | **0.166** | 0.410 | 0.698 |
| P0-lite + rank 0.2 | 0.132 | 0.403 | 0.697 |

- 完整 P0（hybrid + rank 0.5 + 4×128）在 seed0 仅 0.374 SCC，明显低于 v1；
  **hybrid readout 是主要负贡献**，4×128 grouped sampler 也因药物多样性不足掉点。
- P0-lite 与 v1 的 SCC 基本打平、R² 略好；**没有达到“排序损失提升 SCC”的目标**。
- 详细组件消融与限制见
  [`plans&notes/2026-11-19_T0-P0快速验证.md`](../../plans&notes/2026-11-19_T0-P0快速验证.md)。

### 4.2 P1 快速验证（2026-11-19）

P1 = target-aware gene branch + 低秩共享 `cls_proj`（独立 pathway 表，参数 4,889,862）。
每个 (drug, cell) pair 取药物靶点基因的细胞 GE/MU/CNV 逐基因值，经共享基因 ID 嵌入 +
1 层 target self-attention + attention pooling 得到 target token；基因 ID 嵌入与药物
靶点 `EmbeddingBag` 共享。快速验证同样为 LCO fold 0/2 × 2 seeds × 5 epochs：

| 变体 | 参数 | R² | SCC | AUROC | per-drug SCC |
|---|---:|---:|---:|---:|---:|
| T0 v1 | 4,728,260 | 0.156 | 0.415 | 0.700 | 0.411 |
| **T0 P1** | 4,889,862 | **0.170** | **0.440** | **0.710** | **0.436** |
| P1 去掉 target branch | 4,092,677 | 0.114 | 0.417 | 0.702 | 0.411 |

- P1 相对 v1：ΔSCC +0.025、ΔR² +0.014、ΔAUROC +0.009；逐 seed/fold 的 ΔSCC
  均值 +0.025（SD 0.028，t≈1.79，n=4），**方向为正但未达显著**。
- target branch 平均贡献 +0.023 SCC，但跨 fold/seed 不稳定（t≈0.91）。
- **共享 pathway embedding 在 quick screen 中低于独立表**（fold0 0.386 vs 0.430），
  因此默认关闭；共享 pathway + 完全去掉 `cls_proj` 会明显掉点。
- 低秩 `cls_proj`（1000→64→256，三模态共享）保留全局摘要并把该部分参数从 3×256k 降到 80k。
- 详细消融与限制见
  [`plans&notes/2026-11-19_T0-P1快速验证.md`](../../plans&notes/2026-11-19_T0-P1快速验证.md)。

## 5. 适用范围与限制

- v1 正式结果仅 LCO、单 seed；P0/P1 快速验证为 2 折 × 2 seeds × 5 epochs，
  LDO/scaffold 与 ≥3 seeds 待 E1/E7/E12。
- P1 的靶点表在 >64 个靶点时做固定随机截断；逐基因值未做额外归一化/插补。
- 快速验证的早停与报告都用 val SCC，指标偏乐观；未评 test、未做多重比较校正。
- 未启用 GIN：单折 E-drug 筛查中 GIN 增加 0.88M 参数但没有收益，默认保持关闭。
- **科研原型，不用于临床诊断或治疗决策**。

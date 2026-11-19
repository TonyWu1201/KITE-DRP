# 教师 T0 消融报告（E2 / E-drug / E-sim）

- 状态：筛选 + 5 折确认（单 seed）
- 更新日期：2026-11-19
- 相关：`configs/teacher/t0.yaml`、`configs/similarity/h3cdr.yaml`、`scripts/run_ablations.py`
- 运行目录（git 忽略）：`experiments/ablations/2026-11-19_*`

> 本报告是**研究筛选**，不是最终论文结果。单 seed、部分变体仅 fold0；
> 最终结论需要 ≥3 seeds + 逐折独立 RNG + LDO/scaffold。

## 1. 协议

| 项 | 设置 |
|---|---|
| 划分 | LCO `p1_lco_s0`（5 折，`source_group` 分组） |
| 标签 | `-z(LN_IC50)`；敏感/耐药分类掩码 |
| 数据 | M-716 教师池；GE/MU/CNV 各 1,000 维通路分数 |
| 优化 | AdamW，batch 512，AMP，warmup+cosine，Huber + 0.5×BCE |
| 早停 | val SCC；patience 见表 |
| 主指标 | R² / PCC / SCC / AUROC / RMSE（val） |

三阶段：

1. **筛查**：fold0，8 epochs，patience 2，13 个变体；
2. **2 折确认**：fold0+2，12 epochs，patience 3，6 个关键变体；
3. **5 折确认**：fold0–4，20 epochs，patience 5，4 个关键变体。

筛查轮很多变体在 epoch 0–1 达到最优后早停，单折数值噪声很大；
**以 5 折结果为主证据**，2 折/单折只用于方向性筛选。

## 2. E2 多组学消融

### 2.1 5 折（主证据）

| 变体 | 模态 | 参数 | R² | PCC | SCC | AUROC | RMSE |
|---|---|---|---:|---:|---:|---:|---:|
| T0 full | GE+MU+CNV | 4,728,260 | 0.178 ± 0.037 | 0.440 ± 0.033 | 0.427 ± 0.042 | 0.704 ± 0.020 | 0.914 ± 0.027 |
| T0 no-MU | GE+CNV | 4,214,468 | 0.176 ± 0.025 | 0.440 ± 0.027 | 0.425 ± 0.030 | 0.704 ± 0.018 | 0.915 ± 0.022 |
| T0 GE-only | GE | 3,700,676 | 0.176 ± 0.037 | 0.452 ± 0.032 | 0.438 ± 0.027 | 0.710 ± 0.015 | 0.915 ± 0.026 |
| B0（参照） | GE+MU+CNV | 2,383,555 | 0.180 ± 0.035 | 0.457 ± 0.036 | 0.440 ± 0.040 | 0.713 ± 0.022 | 0.913 ± 0.026 |

逐折配对差（vs T0 full，n=5）：

| 变体 | ΔR² | ΔSCC | 配对 t 检验 p（SCC） |
|---|---:|---:|---:|
| no-MU | −0.002 ± 0.031 | −0.002 ± 0.026 | 0.85 |
| GE-only | −0.002 ± 0.042 | +0.011 ± 0.035 | 0.53 |
| no-neighbor | −0.008 ± 0.038 | −0.013 ± 0.028 | 0.37 |

### 2.2 单折筛查（fold0，8 epochs）

| 变体 | 参数 | R² | SCC | AUROC | 说明 |
|---|---:|---:|---:|---:|---|
| full | 4,728,260 | 0.159 | 0.391 | 0.690 | 参照 |
| no-GE | 4,214,468 | **−0.010** | **0.203** | 0.601 | 崩塌 |
| no-MU | 4,214,468 | 0.118 | 0.391 | 0.683 | SCC 持平 |
| no-CNV | 4,214,468 | 0.123 | 0.374 | 0.674 | 小幅下降 |
| GE-only | 3,700,676 | 0.113 | 0.403 | 0.698 | SCC 略升、R² 降 |

**结论**

- **GE 是不可替代的主模态**：去掉 GE 后 R² 接近 0、SCC 从 0.39 掉到 0.20。
- **MU 的边际贡献在 5 折上不可测**：no-MU 与 full 的 SCC/R² 差异均在 ±0.002 内（p≈0.85）。
  单折曾出现 +0.021，属于早停/单折噪声。
- **CNV 可能有小幅贡献**（单折 no-CNV SCC −0.017），但 5 折未单独确认；
  其价值更可能体现在“教师特权信息”而非教师自身精度。
- **GE-only 在 5 折上与 full 等价**（SCC +0.011，p=0.53），参数少 1.0M。
  但按计划 D21，教师默认仍保留 GE+MU+CNV：MU/CNV 是端侧学生拿不到的特权信息，
  是否值得保留应由蒸馏实验（E5）回答，而不是只看教师精度。

## 3. E-drug 药物表示消融

单折筛查（fold0，8 epochs）：

| 变体 | 药物输入 | 参数 | R² | SCC | AUROC |
|---|---|---|---:|---:|---:|
| full | Morgan + targets | 4,728,260 | 0.159 | 0.391 | 0.690 |
| morgan-only | Morgan | 4,364,036 | 0.080 | 0.349 | 0.666 |
| targets-only | 靶点基因 | 4,072,388 | 0.085 | 0.348 | 0.668 |
| +GIN | Morgan + targets + 图 | 5,607,879 | 0.104 | 0.379 | 0.685 |

**结论**

- Morgan 与靶点基因**互补**：单独去掉任一路，SCC 都下降约 0.04、R² 下降约 0.07。
- **GIN 没有收益**：+0.88M 参数，SCC 反而低 0.012、R² 低 0.055。
  在 T0 上继续默认 `use_gin=false`；GIN 留给 T1（分子图与 DTI/PPI 同图）再评估。
- 该组只有单折筛查，需在最终轮补 5 折。

## 4. E-sim 相似性消融

| 变体 | 设置 | fold0 筛查 SCC / R² | 2 折确认 SCC / R² | 5 折 SCC / R² |
|---|---|---|---:|---:|---:|
| full | GE+MU 核，N=15，t=3 | 0.391 / 0.159 | 0.400 / 0.145 | 0.427 / 0.178 |
| no-neighbor | 去掉邻居 token | 0.388 / 0.113 | 0.413 / 0.148 | 0.415 / 0.170 |
| top5 | N=5 | 0.380 / 0.101 | – | – |
| top30 | N=30 | 0.384 / 0.137 | – | – |
| sigma2 | 固定 σ=2 | 0.400 / 0.130 | 0.418 / 0.165 | – |
| iter1 | 融合 t=1 | 0.396 / 0.114 | – | – |

5 折配对（no-neighbor vs full）：ΔSCC −0.013 ± 0.028（p=0.37），
ΔR² −0.008 ± 0.038（p=0.66）。

**结论**

- 邻居 token 的方向为正（full 略好），但**在 5 折上不显著**；
  单折曾出现“有邻居 SCC 0.424 vs 无邻居 0.390”，该结论不稳健，已作废。
- top-N=5 明显更差；N=30 与 N=15 接近；固定 σ=2、t=1 均在噪声内。
- **H3CDR 式相似性尚未在 T0 上转化为可测增益**。当前实现只是把邻居 GE 通路分数
  加权平均成一个 token，与细胞自身的 GE CLS 高度冗余。
  相似性更合理的用法是 T1 的细胞–细胞边 / 嵌入传播，而不是 T0 的单一上下文 token。

## 5. 与 B0 的对比

| 模型 | 参数 | R² | SCC | AUROC |
|---|---:|---:|---:|---:|
| B0 双线性 | 2,383,555 | 0.180 ± 0.035 | 0.440 ± 0.040 | 0.713 ± 0.022 |
| T0 交叉注意力 | 4,728,260 | 0.178 ± 0.037 | 0.427 ± 0.042 | 0.704 ± 0.020 |

逐折配对（T0 − B0）：ΔSCC −0.013 ± 0.022（p=0.27），ΔR² −0.002 ± 0.008（p=0.66）。
**在 LCO 上，交叉注意力结构没有显著优于双线性基线**；这与计划书风险 R2 的预判一致。
教师侧的差异化需要来自图结构（T1）或蒸馏，而不是“更多模态/更复杂注意力”。

## 6. 限制与下一步

限制：

- 单 seed；5 折结果仍是同一进程内顺序训练，fold 间共享 RNG 进程状态；
  最终论文需改为逐折 `set_seed(seed + fold)` 并跑 ≥3 seeds。
- E-drug 仅 fold0 筛查；E-sim 的 topN/σ/t 网格仅 fold0。
- p 值未做多重比较校正，仅作探索性参考。
- 只评估 val，未动 test；LDO/scaffold 尚未跑。

下一步：

1. **T1 异构图教师**：GE 为主，加入 PPI/DTI 边与细胞相似性边；
   用图结构而不是单 token 来检验相似性/先验的增益。
2. **蒸馏验证特权模态**：在 E5 中比较“full T0 vs GE-only T0”作为教师，
   看 MU/CNV 是否通过软标签/潜向量传递价值。
3. 最终消融轮：逐折独立 RNG、3 seeds、5 折、BH-FDR。

## 7. 复现

```bash
# 单折筛查（13 变体）
uv run python scripts/run_ablations.py \
  --folds 0 --epochs 8 --patience 2 \
  --out experiments/ablations/2026-11-19_screen_f0

# 2 折确认
uv run python scripts/run_ablations.py \
  --only t0_full,t0_ge_only,t0_no_mu,t0_targets_only,t0_no_neighbor,t0_sim_sigma2 \
  --folds 0,2 --epochs 12 --patience 3 \
  --out experiments/ablations/2026-11-19_confirm_f02

# 5 折确认
uv run python scripts/run_ablations.py \
  --only t0_no_mu --folds all --epochs 20 --patience 5 \
  --out experiments/ablations/2026-11-19_confirm5_no_mu
uv run python scripts/run_ablations.py \
  --only t0_ge_only,t0_no_neighbor --folds all --epochs 20 --patience 5 \
  --out experiments/ablations/2026-11-19_confirm5_ge_sim
```

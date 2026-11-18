# 模型卡：Teacher B0（双线性潜交互基线）

- 状态：草稿（M2 基线）
- 版本：`v0.1.0`
- 更新日期：2026-11-19
- 权重路径 / SHA256：`experiments/<run>/fold<k>/best.pt`（未发布）
- 相关：计划书 §4.2 B0；`configs/teacher/b0.yaml`；`src/kite_drp/models/teacher/b0.py`

## 1. 基本信息

| 项 | 值 |
|---|---|
| 模型类型 | 教师 B0（无图强基线，不部署） |
| 参数量 | 2,383,555（训练 = 部署） |
| 输入模态 | GE/MU/CNV 各 1,000 维通路分数（fold 内 z）；Morgan 2048 + 靶点基因 |
| 输出 | 药敏响应分数 `-z(LN_IC50)` + 敏感概率 |
| 训练数据 | GDSC2 + DepMap（processed M-716；LCO fold0–4） |
| 训练配置 | `configs/teacher/b0.yaml` + `configs/train/teacher.yaml`；seed=0 |
| 训练硬件 / 时间 | RTX 4060 Ti 8GB；5 折约 2 min |

## 2. 训练

- 结构：三个模态投影（MLP + missing token）→ concat → 细胞塔 256；
  Morgan MLP + 靶点 `EmbeddingBag(16)` → 药物塔 256；点积读出 + 分类头。
- 损失：Huber（δ=1，回归）+ 0.5 ×  masked BCE（分类）。
- 早停：验证集 SCC，patience 5；AdamW lr=5e-4、wd=1e-4、batch=512、AMP。

## 3. 评估结果（LCO `p1_lco_s0`，5 折，单 seed）

| 指标 | 数值（mean ± SD） | 评测集 |
|---|---|---|
| R² | 0.180 ± 0.035 | val（192 株/折） |
| PCC | 0.457（5 折均值） | val |
| SCC | 0.440 ± 0.040 | val |
| AUROC | 0.713 ± 0.022 | val（敏感/耐药） |
| RMSE | 0.913 | val |

- naive 基线（fold0，逐药中位数）：R² = −0.012、SCC = 0.034；B0 显著优于 naive。
- 该结果是**单 seed 5 折**；计划书要求的 ≥3 seeds 待 E12 补齐。

## 4. 适用范围与限制

- 用途：T0/T1 的对照基线与数据/指标链路 sanity check，不用于部署。
- B0 在 LCO 上与 T0 几乎持平（见 T0 模型卡），说明通路特征本身信息量很强；
  论文中必须保留该基线，避免把 T0 的增益归因于注意力。
- **科研原型，不用于临床诊断或治疗决策**。

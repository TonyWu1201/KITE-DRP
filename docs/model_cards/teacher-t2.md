# 教师模型卡：T2（冻结 P1 + 无图第二塔 + DTI/PPI 固定特征）

- 状态：v0.1 快速筛选；LDO fold0、seed0、12 epochs；不是正式结论
- 代码：`src/kite_drp/models/teacher/t2.py`、`src/kite_drp/train/ppi_features.py`
- 配置：`configs/teacher/t2.yaml`
- 基线：同 split/fold 的冻结 T0-P1（`init_from`）
- 相关：`plans&notes/2026-11-21_无图双塔与DTI-PPI实施.md`

## 1. 定位

T2 不是图模型，而是 **P1 的第二视角集成 + DTI/PPI 特征化**：

- P1 全部冻结，始终 `eval()` / 无梯度；
- 第二塔独立初始化（d=192、1 层、top-32、关闭相似性邻居），
  从 Morgan + targets + GE/MU/CNV 重新编码；
- DTI/PPI 只生成固定特征，通过 `pair_context` 加到第二塔的 `z_pair`：
  根/邻居集合统计、通路 profile、STRING RWR 加权组学；
- 融合：`y = y_p1 + tanh(α)·(y_t2 − y_p1)`，α 初始 0；
  也支持固定 0.5（`fusion=mean`）和逐样本 gate（`fusion=sample`）；
- 深监督：`0.3·Huber(y_t2) + 0.2·BCE(logits_t2)`，保证第二塔在 α=0 时也学习。

## 2. 参数量

| 部分 | 参数 |
|---|---:|
| P1 backbone（冻结） | 4,889,862 |
| 第二塔（可训练） | 2,522,279 |
| DTI/PPI context（可训练，全量） | 296,832 |
| **总参数** | **7,708,973** |
| **可训练参数** | **2,819,111** |

## 3. DTI/PPI 特征

| 特征块 | 内容 | 维度 |
|---|---|---:|
| `summary` | 根/邻居 × GE/MU/CNV 的 mean/max/min/std/观测率/异常率 + 节点数 | 38 |
| `rwr` | 从靶点出发的 STRING personalized PageRank，再与 GE/MU/CNV 加权 | 8 |
| `pathway` | 靶点/邻居基因在 1000 条通路上的 profile，与细胞通路分数逐元素相乘 | 2×1000 |

- 无 GNN、无消息传递、无每节点隐状态。
- 无合规 DTI 的药物零填充；`coverage` 门控。
- 缓存键覆盖 DTI/PPI/通路文件、基因空间、局部图、特征配置和代码版本。
- 随机对照：`neighbor_mode=pseudo`、`random_control=permute_drugs`、`gaussian`。

## 4. 快速验证命令

```powershell
# 同折 P1 checkpoint 已在 experiments/p1_ldo_quick/runs/p1_s0
uv run python scripts/run_t2_quick.py \
  --variants t2_base,t2_all,t2_random,t2_pseudo \
  --split ldo --folds 0 --seeds 0 --epochs 12 --patience 4 \
  --init-from experiments/p1_ldo_quick/runs/p1_s0 \
  --out experiments/t2_quick_f0

# 集成分析（P1 seed ensemble vs T1/T2）
uv run python scripts/analyze_ensembles.py \
  --run p1_s0=experiments/p1_ldo_quick/runs/p1_s0 \
  --run p1_s1=experiments/p1_ldo_quick/runs/p1_s1 \
  --run p1_s2=experiments/p1_seed2_ldo_quick/runs/p1_s2 \
  --run t1=experiments/t1v2_ldo_quick/runs/t1_s0 \
  --reference p1_s0 --out experiments/analysis/ensemble_ldo_p1x3_t1.json

# 可解释性：alpha、分层、context 遮挡/置换、top 通路/基因
uv run python scripts/analyze_t2_interpretability.py \
  --run experiments/t2_quick_f0/runs/t2_all_s0 --fold 0 --split ldo \
  --out experiments/analysis/t2_all_f0
```

## 5. 结果（fold0，seed0，12 epochs）

| 变体 | P1 | T2 单独 | 最终（α） | 固定 0.5 融合 |
|---|---:|---:|---:|---:|
| `t2_base` | 0.6065 | 0.6113 | 0.6110（0.128） | 0.6175 |
| `t2_all` | 0.6065 | 0.6144 | 0.6121（0.137） | 0.6196 |
| `t2_random` | 0.6065 | 0.6113 | 0.6113 | 0.6178 |
| `t2_pseudo` | 0.6065 | 0.6122 | 0.6115 | 0.6183 |

分层（逐药 SCC，固定 0.5 融合）：

| 方案 | GDSC-only (20) | 无 DTI (15) | OT (22) |
|---|---:|---:|---:|
| P1 s0 | 0.6263 | 0.5978 | 0.5943 |
| P1 s0+s1 | 0.6273 | 0.6036 | 0.6065 |
| `t2_base` | 0.6291 | 0.6029 | 0.6170 |
| `t2_all` | 0.6295 | 0.6027 | 0.6222 |
| `t2_random` | 0.6285 | 0.6031 | 0.6180 |
| `t2_pseudo` | 0.6292 | 0.6027 | 0.6192 |

fold1：`t2_base` P1 0.6202、T2 单独 0.6206、最终 0.6237、固定 0.5 融合
0.6277；`t2_all` P1 0.6202、T2 单独 0.6201、最终 0.6238、固定融合 0.6279；
`t2_pseudo` 最终 0.6237、固定融合 0.6274、T2 单独 0.6183。
两折均值（固定 0.5 融合）：`t2_base` 0.6226、`t2_all` 0.6238、
`t2_pseudo` 0.6229；PPI 增量全量 +0.0012（vs base）/ +0.0009（vs pseudo），
OT +0.0032/+0.0020，不稳定。

初步结论：

- **第二塔本身**把 OT 从 0.5943 提到约 0.611–0.617（+0.017 至 +0.023），
  不需要 PPI；
- **真实 DTI/PPI 特征**相对 pseudo 邻居在 fold0 OT 上约 +0.003、相对
  permuted 特征约 +0.004（0.5 融合）；但 fold1 的 PPI 增量全量仅
  +0.0002、OT +0.0012。两折全量约 +0.001，低于预注册的 +0.005 阈值，
  不能宣称稳定增益；
- 学到的 α≈0.13 明显低估第二塔；固定 0.5 融合在 fold0 高约 +0.007。

## 6. 可解释性

- OT 子集 context 遮挡（逐药 SCC 变化）：
  `zero_summary` −0.0026、`zero_all` −0.0019、`permute_all` −0.0024；
  单独 target/neighbor pathway 约 −0.0001 量级。
- 当前模型更多依赖邻居集合统计 + RWR，而不是通路 profile；绝对效应很小。
- 所有结论是“模型依赖哪些输入”，不是因果生物学结论。

## 7. 适用范围与限制

- 只在 LDO fold0、单 seed、12 epochs 上验证；不能作为正式教师结论。
- 早期快速筛选同时用 `val_per_drug_scc` 早停和报告，数字偏乐观。
- OT 子集只有 22 个药；+0.005 量级需要 3 折 × 2 seeds 确认。
- 73/286 个药物无 DTI；这些药物只能使用第二塔的无 PPI 路径。
- 科研原型，不用于临床诊断或治疗决策。

# 教师模型卡：T1-local（每药一跳局部图残差）

- 状态：v0.1；新 DTI（Open Targets 26.09 + GDSC TARGET）下的快速验证
- 代码分支：`feat/t1-local-graph`
- 基线：T0-P1（同 split/fold 冻结权重）
- 相关：`plans&notes/2026-11-20_教师模型复盘与图教师重构方案.md` §5

## 1. 定位

T1-local 不是新的端到端教师，而是 **P1 的图增强残差版**：

- P1 全部冻结，始终 `eval()` / 无梯度；
- 每个药物单独构建一张以靶点为根、STRING 一跳邻居为上下文的小图；
- 当前细胞的 GE/MU/CNV 在消息传递前写入节点；
- 图分支只输出标量残差 `r`，最终 `y = y_P1 + a_d · r`；
- 无合规靶点时 `a_d=0`，精确回退 P1。

旧 T1 的全局异构图把 PPI 传播放在细胞组学进入之前，因此同一药物的消息
对所有细胞相同；T1-local 的核心改变是 **cell-specific 节点状态 + 每药局部
拓扑**。

## 2. 图构建口径

| 项 | 设置 |
|---|---|
| 根 | `dti.parquet` 中该药的可测直接靶点；Open Targets MoA（ChEMBL/InChIKey 匹配 + 非空 action）优先，GDSC `TARGET` 文本为第二层 |
| 邻居 | `graph_inputs.npz` 的 STRING PPI，`combined_score >= 0.7`，每靶点最多 8 个，每药最多 64 个节点 |
| 排序 | 根按证据层级 + HGNC symbol；邻居按 `(-score, symbol)`；确定性裁剪 |
| 边 | 首轮等权、无向；去掉自环/重复边；保存裁剪前后覆盖 |
| 覆盖 | 主门控 `any`（OT+GDSC，213/286）；`OT` 仅用于分层报告；73 个药物无 DTI，精确回退 |
| 缓存 | `data/interim/teacher/local_graph/{hash}`；hash 含 DTI/PPI 内容、基因空间、裁剪参数、构图代码版本 |

新 DTI 下：640 条边、301 个靶点基因、213/286 个药物有任意 DTI、104 个
药物有高置信 OT 根、97 个单靶点药物。局部图节点中位数约 9（单靶点药为主），
多靶点药可到 64；212/213 个药物有邻居-邻居边，保度重连主要检验邻居间拓扑。

## 3. 模型

- 节点输入 6 维：GE/MU/CNV 值 + 三个 observed flag；根标记单独传入。
- 共享 `f_in: 6 → 64`；一层等权邻居均值；`LayerNorm(h0 + GELU(W_m m))`。
- 根/非根分别 masked pooling；拼接冻结 P1 的 `z_pair`；两层 MLP 输出残差。
- 残差末层零初始化；首步严格等于 P1。
- 无覆盖或该细胞没有观测到根时 gate=0。
- 分类沿用 P1；首版 `z_pair` 仍来自 P1，供蒸馏复用。

## 4. 快速验证命令

```powershell
# 新 DTI 下重跑 P1（少折、单 seed）
uv run python -m kite_drp.train.teacher_train teacher=t0_p1 split=ldo "fold=[0,1]" seed=0 \
  train.monitor=val_per_drug_scc train.epochs=12 train.patience=4 train.eval_test=false \
  hydra.run.dir=experiments/2026-11-20_dti_p1_ldo_quick_s0

# T1 legacy 对照 + 阶段 1 邻居集合（fold0，单 seed）
uv run python scripts/run_local_graph_quick.py \
  --variants t1,mlp,root,neighbors,pseudo --split ldo --folds 0 --seeds 0 \
  --epochs 12 --patience 4 \
  --init-from experiments/2026-11-20_dti_p1_ldo_quick_s0 \
  --out experiments/local_graph_quick_s0
```

## 5. 快速验证结果

| 变体 | 协议 | per-drug SCC | 说明 |
|---|---|---:|---|
| P1（新 DTI） | LDO fold 0–1，seed 0，12 ep | **0.6245** | fold0 0.6228 / fold1 0.6262 |
| T1 legacy | LDO fold 0，seed 0，12 ep | **0.6397** | 新 DTI 全局图；相对 P1 fold0 +0.0169，OT 子集 +0.0372 |
| P1 + MLP 头 | 同上 | **0.6233** | +0.0004；容量不能解释 T1 legacy |
| 根集合 | 同上 | **0.6231** | +0.0003；仅根不优于 P1 |
| 真邻居集合 | 同上 | **0.6232** | +0.0004；与根/MLP 基本相同 |
| 匹配伪邻居 | 同上 | **0.6232** | +0.0003；与真邻居无差异 |

阶段 1 未通过：`neighbors − root = +0.0001`，`neighbors − pseudo = +0.0001`。
真邻居没有优于仅根或匹配伪邻居，按预注册门槛不进入阶段 2。事后诊断的
可训练传播 `train_true` 为 0.6233（+0.0005），与集合编码相同。T1 legacy 的
+0.0169 不能由 55k MLP 头解释，但其 1.5M 图分支容量/深监督尚未做等容量
对照；旧随机图负对照仍是“图拓扑不成立”的主要证据。

> 这是快速筛选，不是正式结论；正式比较需要 3 折 × 2 seeds、实体分组 LDO
> 和逐药配对 bootstrap。旧 T1/P1 数字对应旧 DTI，不能与新数字直接比较。

T1-local 的可训练参数为 55,041（P1 冻结，总参数 4,944,903）；容量对照
T1-MLP 为 55,213（总参数 4,945,075），两者相差约 0.3%。

## 6. 适用范围与限制

- 局部图只覆盖有合规 DTI 的药物；73/286 个药物精确回退 P1。
- 单靶点药物（97 个）的根-邻居边固定，重连主要检验邻居间拓扑。
- 首轮不做两跳、边权、联合微调；这些留到一跳对照有信号之后。
- `z_pair` 含药物表征，残差头可能记住训练药的 drug×cell 修正；正式阶段
  需要 `condition_on=cell` / `none` 消融与残差收缩。
- 当前快速验证中，局部图分支没有超过 P1；下一步应先解决“残差头不读图”
  的问题（graph-only 头/深监督/逐药中心化），再谈正式比较。
- 科研原型，不用于临床诊断或治疗决策。

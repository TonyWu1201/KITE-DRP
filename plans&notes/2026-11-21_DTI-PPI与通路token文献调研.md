# DTI/PPI 与通路 token 文献调研（2026-11-21）

- 目的：解释 T1/T1-local 与 T2 两轮 DTI/PPI 负结果，并评估“细胞系数据压缩成低维通路 token”的依据与信息损失风险。
- 范围：药物响应预测（DRP）、图先验、通路/基因集表征、critical evaluation / benchmark。
- 结论等级：文献调研，不替代本项目预注册实验；所有“阳性/阴性”均注明原论文评估协议。

## 0. 结论摘要

1. **你的两轮负结果与当前最严格的独立评测一致，不是实现失败。**
   - DTI/PPI 的阳性报道大多来自 LPO（随机 drug–cell pair）、cell-blind 或 warm-start；在 LDO / scaffold-cold / double-cold 下，增益普遍消失，甚至低于 drug-mean。
   - 最直接的证据：Tran & Klau 的独立复现中，TGSA 在 LDO 的 RMSE 为 2.538 ± 0.280，而 mean predictor 为 2.536 ± 0.242；作者明确写“PPI networks did not improve predictive performance”。[Tran2026]
   - PathGuard-DRP 的 pathway-aware prior 在 warm 上明显高于 drug-mean（Spearman 0.7984 vs 0.7299），但 cell-cold 仅 +0.0045（95% CI 跨 0），drug-cold / scaffold-cold / double-cold 均不超过最强简单基线。[PathGuard2026]
2. **T1 的增益来自“第二套编码 + 融合”，不是 PPI/DTI 拓扑。** 这与 DrEval 的结论一致：图模型必须做“保度重连”消融；很多图收益在随机图下复现。[DrEval2026]
3. **通路 token 有依据，但证据支持“辅助/去噪/鲁棒/可解释”，不支持“无损压缩、可单独替代 gene-level”。**
   - 正向：SurvPath 的通路 token 比“全部基因一个 token”和 6 个粗基因家族更好（c-index 0.629 vs 0.586 / 0.609），但 dense gene-level MLP 也有 0.599。[SurvPath2024]
   - 正向：BioMod 的 660 个重叠生物模块在 drug-response 上 r≈0.931；shuffled-module 对照变差，说明“生物分组”本身有贡献。[BioMod2026]
   - 风险：PathwayBench 发现 rank-based 通路评分在 competitor burden 高时可丢信号、甚至反向（模拟中 40% 符号反转）；没有一种评分方法满足全部 5 项准则。[PathwayBench2026]
   - 风险：BioMod 的模块只覆盖 12,895 / 27,874 个基因；不在模块里的基因无法进入 token。[BioMod2026]
4. **最值得继续的方向不是“再加一个先验”，而是：target-aware token + gene-level 残差 + 多分辨率融合 + 严格 cold-start 对照。**
   - 先用 target-only baseline 判断 PPI 是否有增量；若 PPI 打不过“直接靶点表达/CNV/MU + 药物指纹”，就应停止。
   - 通路 token 应保留 gene-level 通道，并用可学习 tokenizer / reconstruction 约束，而不是固定 mean/GSVA 单值。

---

## 1. 你的两种方案与文献的对应

### 1.1 T1 / T1-local：把 PPI/DTI 当图结构

你的结果：
- T1 legacy 的 +0.0169 被 random PPI（+0.0169）、random DTI（+0.0150）、冻结 P1（+0.0156）、去 cell-gene（+0.0160）几乎完整复现；
- T1-local 的“真邻居 vs 仅根 vs 匹配伪邻居”差 +0.0001；
- 结论：增益来自独立第二套编码 + 门控融合，不是图拓扑。

文献对应：
- **TGSA / TGDRP** 用 STRING PPI 把 cell line 建成 gene graph，再用 GNN 编码；论文报告在 LPO/LCO/LDO 的 blind test 上优于 baselines。[TGSA2022]
- 但独立复现发现：TGSA 在 LPO 很强，LCO/LDO 明显下降；LDO 与 mean predictor 打平，PPI 没有带来可检测增益。[Tran2026]
- **DRPreter** 把 STRING PPI 限制在 34 条 KEGG 通路子图内，用 GAT 编码；主实验是随机 8:1:1 pair split，ablation 中 pathway 子图相对 template graph 的 MSE 0.8536 vs 0.8926，pathway+transformer 0.8302；没有保度随机图对照。[DRPreter2022]
- **DrEval** 明确把“graph inputs 的边打乱但保持节点度”作为标准消融；并指出 deep models 仅略优于 naive drug/cell mean，复杂模型在 relevant settings 不显著优于调好的树模型。[DrEval2026]

对你的解释：
- 你的 random-graph 对照已经比多数论文更严格；T1 的负结果应当保留为“图拓扑不成立”的证据，而不是继续调图。
- 若还要做图，必须满足两个前提之一：**cell-specific 图**（不是静态 STRING）或**图只作为辅助任务/正则**，而不是主预测路径。

### 1.2 T2：把 DTI/PPI 变成固定特征

你的结果：
- 第二塔稳定 +0.008～0.011；
- DTI/PPI 固定特征两折全量增量 +0.0012（vs base）/ +0.0009（vs pseudo），OT 子集 +0.003/+0.002，低于预注册 +0.005；
- RWR 根节点权重约 0.50，一跳邻居仅 0.002 量级；PPI 信息主要在 neighbor summary，遮挡效应很小。

文献对应：
- **Precily** 用 pathway activity + drug descriptors，cell-blind split；相对 500-gene baseline 有优势，但紧跟其后的 CaDRReS-Sc 也接近。[Precily2022]
- **PASO** 用 619 条 KEGG_MEDICUS 的“通路内 vs 通路外”差异特征，Drug-Blind PCC 0.7448 / R² 0.5253；比 Mixed-Set 的 0.9425 明显下降，说明 cold-drug 仍是难点。[PASO2025]
- **PathGuard-DRP** 用 Morgan + Reactome pathway summaries + drug-conditioned pathway gate；warm 明显好，cell-cold 仅 +0.0045 且不显著，drug-cold / scaffold-cold / double-cold 不超过简单基线；在 PRISM 新 scaffold 上还低于 Morgan-only Ridge（0.0718 vs 0.0887）。[PathGuard2026]
- **DIPK** 用 BIONIC 基因互作网络得到 512 维 cell 表征；ablation 中 DIPK 0.8461 vs 去 interactome 的 DEM 0.8604 RMSE（known），cold 设置 3.4188 vs 3.7897；有增益但幅度有限，且不是 per-drug DTI。[DIPK2024]

对你的解释：
- T2 的“第二塔有效、DTI/PPI 增量很小”与文献的“warm 有效、cold 不稳定”一致。
- RWR 根节点主导是一个实现层面的诊断：它让 PPI 特征退化成“靶点表达的平滑版”，而 target expression 本来就在 P1 的 gene branch 里。要继续，必须让邻居/传播真正有独立信息，而不是复刻根节点。

---

## 2. 类似工作怎么用 DTI/PPI：四种范式

### 2.1 全局 PPI 图 / cell-line graph

代表：PaccMann、TGDRP/TGSA、DRPreter、GCNPath。
做法：gene 为节点，STRING/Reactome 边；节点特征是 GE/CNV/MU；GNN 聚合后 pooling 成 cell embedding；再与 drug encoder 融合。
优点：能表达 gene–gene 关系；缺点：静态图、跨细胞系共享、容易退化成“基因表达平滑”；在 LDO 下药物侧信息不足。
证据：TGSA 在 LDO 不如 mean predictor [Tran2026]；GCNPath 用 GSVA pathway 特征，作者自己写“does not achieve overwhelming performance gains”，但在 unseen cell lines/drugs 上“most balanced improvements”。[GCNPath2026]

### 2.2 每药局部图 / target neighborhood

代表：你的 T1-local；DRPreter 的 pathway 子图；PDGrapher 的 PPI + 因果目标排序。
做法：以药物靶点为根，取 1-hop/2-hop 邻居，节点写入当前细胞组学，输出残差。
优点：cell-specific、target-specific；缺点：单靶点药图很小（你的中位数 9 个节点），拓扑几乎固定；多靶点药覆盖不均；73/286 无 DTI。
证据：你的 T1-local 真邻居 = 根 = pseudo；PDGrapher 在 PPI 上做 target ranking/perturbation response 有较强证据，但任务不是 IC50。[PDGrapher2025]

### 2.3 固定特征 / pathway activity

代表：Precily、PASO、PathGuard、你的 T2。
做法：把 DTI/PPI 压成 summary、RWR、pathway profile、pathway difference 等固定特征，再拼到 drug–cell 表征。
优点：便宜、可解释、易做 cold-start；缺点：压缩损失、根节点主导、与 gene-level 特征冗余。
证据：Precily/PASO 在 cell-blind/drug-blind 有正向结果；PathGuard 在严格 cold-start 下没有增量；你的 T2 两折 +0.001。[Precily2022, PASO2025, PathGuard2026]

### 2.4 预训练 / 辅助任务

代表：DIPK（BIONIC 自监督）、TGSA（auxiliary pretraining）、GEARS（gene-gene graph + perturbation embedding）。
做法：用基因网络/PPI 预训练 gene/cell embedding，或加 link prediction / masked modeling / graph reconstruction 辅助损失；主任务仍用 drug + cell 特征。
优点：先验不是硬塞进主路径，而是通过表示学习“渗入”；缺点：增益依赖预训练质量与数据规模，且很难归因到 PPI 拓扑。
证据：DIPK 的 interactome 消融有 modest gain [DIPK2024]；GEARS 在 unseen perturbation 上证明 gene-gene graph 有用，但任务是 perturbation response，不是 IC50。[GEARS2023]

### 2.5 不同端点：target ranking / perturbation response

代表：PDGrapher、GEARS。
做法：不预测 IC50，而是预测“哪些基因/通路是治疗靶点”或“扰动后的表达谱”；PPI 是因果传播的骨架。
证据：PDGrapher 在 held-out samples/disease 上 target ranking 有优势；GEARS 在 novel multigene perturbation 上优于无图 baseline。[PDGrapher2025, GEARS2023]
启示：**PPI 在“机制/靶点/扰动”端点上比在“IC50 回归”端点上更容易成立。** 如果你的课题允许，把 PPI 用在解释/靶点排序，而不是主指标。

---

## 3. 为什么 LDO 下很难看到 DTI/PPI 增益

### 3.1 评估协议：阳性多来自更容易的 split

- LPO：所有 drug/cell 都在训练集出现，模型可以记住 drug mean；PPI 只要帮助平滑 cell 侧就有增益。
- LCO：drug 出现、cell 未出现；drug mean 仍可学。
- LDO/cold-drug：drug 未出现，必须靠 drug 特征（structure/target）泛化；这是最难的。
- DrEval 指出：LPO/LCO 的“好成绩”很大程度来自 drug mean / cell mean；必须报告 normalized R² / per-drug metrics。[DrEval2026]
- Tran & Klau：LDO 下 TGSA 与 mean predictor 打平；并指出多数模型在 blind 场景下降。[Tran2026]

### 3.2 覆盖度与统计功效

- 你的 DTI：640 边 / 213 个药物 / 301 个靶点基因；73/286 无 DTI；OT 子集只有 22 个药。
- 在这种覆盖下，+0.005 的 OT 增量需要很大的 per-drug 样本才能稳定；单折 +0.005 很容易是波动。
- 文献同样受覆盖限制：PathGuard 的 pathway prior 在 cold-drug 上不超过简单基线；DIPK 的 cold 设置 MSE 3.4+，绝对值很差。[PathGuard2026, DIPK2024]

### 3.3 静态 PPI 的局限

- STRING 是 tissue-agnostic、context-agnostic 的；同一 edge 在不同细胞系里可能不表达/不功能。
- 你的 RWR 根权重 ~0.50、邻居 ~0.002，说明传播几乎没有离开根；PPI 信息被 target expression 淹没。
- 文献共识：静态 PPI 作为全局正则可以，但要产生 cell-specific 预测，需要 cell-specific 网络/上下文权重（如 BIONIC、PRECISE 类方法），而不是全库 STRING 直接传播。

### 3.4 二值 DTI 的局限

- DTI 只表示“结合/机制”，不表示 affinity、efficacy、occupancy、downstream effect；对 IC50 的解释力有限。
- 多靶点/多机制药物、prodrug、无结构药、内部代号药都会让 DTI 缺失或错配。
- Tran & Klau 直接建议：与其用 2D 分子图/二值 DTI，不如做 drug–target binding（结构/对接）或更好的 target 数据。[Tran2026]

### 3.5 图拓扑 vs 集成/容量混淆

- 你的 random PPI/DTI 复现了 T1 的增益；1.5M MLP 头只有 +0.0009；说明不是纯容量，而是“第二套编码 + 融合”的多样性。
- 文献里很多“图模型 SOTA”没有做 random-graph / degree-preserving control；DrEval 把这类消融列为标准。[DrEval2026]
- 结论：**图分支的增益可以先由无图第二塔复现**；如果第二塔已经拿到大部分增益，图拓扑的边际贡献必须用更严格的对照证明。

### 3.6 泄漏与 transductive similarity

- 多数图模型在构造 drug/cell similarity graph 时使用了全量数据（包括 test drugs/cells），所谓“unseen”并不完全 inductive。
- DrEval 的六类问题包括 data leakage、pseudoreplication、biased metrics、missing ablation；PathGuard 专门做 leakage-controlled benchmark。[DrEval2026, PathGuard2026]
- 另外，feature screening 在 CV 之前做会泄漏；Widespread data leakage 审计了 32 个 DRP 方法，发现预处理/特征筛选泄漏会显著高估性能并污染 biomarker。[Leakage2026]

---

## 4. 文献证据表（与你的问题最相关的 12 篇）

| 工作 | DTI/PPI 用法 | 评估协议 | 与你的关系 |
|---|---|---|---|
| Tran & Klau 2026, Pattern Recognit Lett | TGSA/TGDRP 的 STRING PPI + 分子图 | LPO/LCO/LDO 5-fold，mean predictor / MMLP 对照 | **最直接**：LDO 下 TGSA ≈ mean predictor；作者明确说 PPI 没有改善 |
| DrEval 2026, Nat Commun | 图输入做保度边打乱；naive mean effects；调好的树基线 | LPO/LCO/LTO/LDO | 你的 random-graph 对照是正确方向；复杂模型常被高估 |
| PathGuard-DRP 2026, Comput Biol Chem | Reactome pathway summaries + drug-conditioned gate | warm / cell-cold / tissue-heldout / drug-cold / scaffold-cold / double-cold | 与 T2 最像：warm 有效，cold 无增量 |
| TGSA 2022, Bioinformatics | STRING PPI cell graph + drug graph + similarity augmentation | LPO/LCO/LDO blind tests | 阳性报道，但独立复现 LDO 失败；SA 可能 transductive |
| DRPreter 2022, IJMS | STRING PPI 限制在 34 条 KEGG 通路子图 | 随机 8:1:1 pair split | pathway 子图有小幅 ablation 增益；无 random-graph 对照 |
| DIPK 2024, Brief Bioinform | BIONIC 基因互作网络 → 512-d cell 表征 | known + unlearned cell/drug；25-fold | interactome 有 modest gain；不是 per-drug DTI |
| PaccMann 2019, Mol Pharm | PPI + gene expression + drug structure，multimodal attention | 随机 split | 早期阳性代表；未做 LDO/cold 对照 |
| GCNPath 2026, Commun Biol | GSVA pathway 特征 + pathway crosstalk graph | 多平台、unseen cell/drug | 通路特征提升跨平台鲁棒性，但“不是压倒性增益” |
| PDGrapher 2025, Nat Biomed Eng | PPI + 因果目标排序 | held-out samples/disease | PPI 在 target ranking 端点上更强 |
| GEARS 2023, Nat Biotechnol | gene-gene graph + perturbation embedding | unseen multigene perturbation | 图先验在 perturbation response 上成立 |
| Precily 2022, Nat Commun | pathway activity + drug descriptors | cell-blind | pathway 特征可竞争 gene-level baseline |
| PASO 2025, PLoS Comput Biol | 619 条 pathway difference 特征 | Mixed / Cell-Blind / Drug-Blind | Drug-Blind 仍难；pathway difference 保留对比信息 |

---

## 5. 通路 token 的依据

### 5.1 生物学依据

- 通路/基因集是细胞功能的自然单元；药物通常作用于通路而非单个基因； pathway-level 表征更接近机制。
- PROGENy 用 perturbation-response genes 推断 14 条信号通路活性；decoupleR 把多种通路/TF 活性推断方法统一成可组合框架。[PROGENy2018, decoupleR2022]
- MSigDB Hallmark 用 50 条精简基因集降低通路冗余；Reactome/KEGG 提供更细的层级。[Hallmarks2015]

### 5.2 方法学依据

- **降噪/降维**：把 2 万基因压到几十到几百个通路分数，减少多重检验和过拟合；对跨平台/批次更鲁棒。
- **可解释性**：通路 token 可以直接对应到机制，便于 attention/IG 解释。
- **先验注入**：在样本量有限时，基因集结构相当于一种正则。
- **多模态融合**：通路 token 作为 transcriptomics 的“语义单元”，比单个 gene token 更适合与 histology/药物 token 做 cross-attention。

### 5.3 证据：正向

- **SurvPath（CVPR 2024）**：pathway token 用 sparse MLP 对通路内基因做端到端编码；tokenizer ablation 中 Reactome+Hallmarks 0.629，高于 single token 0.586、6 个 gene families 0.609；但 dense gene-level MLP 也有 0.599，说明通路 token 的增益是“中等”，不是“碾压”。[SurvPath2024]
- **BioMod（2026）**：660 个重叠模块（Hallmark 50 + Reactome 491 + Hotspot 73 + hdWGCNA 46）；module token 由 Gene2Vec + expression-weighted attention pooling 生成；在 drug-response 上 r≈0.931；shuffled-module 对照在 drug-response 两项任务上都变差，说明“生物分组”有贡献。[BioMod2026]
- **Precily（2022）**：pathway activity + drug descriptors，cell-blind split；相对 500-gene baseline 有优势，CaDRReS-Sc 紧随。[Precily2022]
- **PASO（2025）**：619 条 KEGG_MEDICUS 的“通路内 vs 通路外”差异特征；Drug-Blind PCC 0.7448；比简单 pathway mean 更保留基因级对比。[PASO2025]
- **GCNPath（2026）**：GSVA pathway 特征 + pathway crosstalk graph；跨 TPM/microarray/proteomics 更稳；作者承认不是压倒性增益，但在 unseen 场景更平衡。[GCNPath2026]
- **DRPreter（2022）**：pathway 子图 + GAT；相对 template graph 有小幅 MSE 改善；但本质是 pathway-structured gene graph，不是低维通路向量。[DRPreter2022]
- **DeePathNet（2024）**：cancer-specific pathway token + transformer；用“同尺寸随机通路”做对照，是通路 token 论文里少见的 control。[DeePathNet2024]

### 5.4 证据：负向/混合

- **Pateras 2025（Future Pharmacology）**：在 7 个药物上比较 data-driven RFE-SVR 与 KEGG/CTD pathway gene sets；**pathway-based gene sets 的表现低于 data-driven 方法**；两者结合才稳定提升。[Pateras2025]
- **PathGuard-DRP 2026**：Reactome pathway summaries + drug-conditioned gate 在 warm 上强，但 cold-drug/scaffold-cold/double-cold 不超过简单基线。[PathGuard2026]
- **Tran & Klau 2026**：综述中写“pathway information has shown limited benefit”，并指出 PPI 在其复现中没有改善。[Tran2026]
- **Two-step multi-omics 2020**：pathway activation 与 gene expression 有冗余；去掉 gene expression + pathway activation + tissue 才出现 31–32% 的大幅下降，单独 pathway 的边际贡献不清晰。[TwoStep2020]
- **Transcriptomic Profiles Carry Most... 2026**：12 个药物、509 细胞系，gene expression 捕获了大部分可预测响应；外部 PRISM 验证下降，说明 gene-level 信号本身有限，压缩后更容易丢。[Transcriptomic2026]

---

## 6. 通路 token 的信息损失风险

### 6.1 覆盖损失（coverage loss）

- 固定基因集不可能覆盖全部基因。BioMod 的 660 个模块只覆盖 12,895 / 27,874 个基因；不在模块里的基因不能进入 token。[BioMod2026]
- 药物靶点、罕见驱动基因、融合基因、非编码调控可能不在通路库里；直接压成通路分数会把它们丢掉。
- **建议**：保留 gene-level 通道；为未覆盖基因加 “other/uncategorized” token；报告 target coverage。

### 6.2 稀释/淹没（dilution）

- 通路 mean/score 会把单个高效应基因（如靶点表达、单基因突变）平均掉。
- 对 targeted drugs，target expression 往往是最强单基因特征；pathway score 可能把它稀释到无关通路成员中。
- **建议**：target-aware token（直接靶点 + 1-hop 邻居）单独成 token；pathway token 只作为上下文。

### 6.3 方法依赖与符号反转（method dependence / sign inversion）

- 通路评分方法（ssGSEA、GSVA、z-score、AUCell、UCell）结果不一致；PathwayBench 在 8 个数据集、682 donors 上发现 rank-window competition：非通路 competitor genes 占满 top-rank 窗口时，rank-based 方法会丢信号，极端情况下反向；模拟中 40% 的 replicate 出现符号反转，真实数据中 CKD 的 ECM remodeling 出现 wrong-direction。[PathwayBench2026]
- NAR scPS 基准也指出 gene set size、quality、dropout 会显著影响评分；scPS 更稳但仍有 false positives。[scPS2024]
- **建议**：不要只用一种评分；至少做 z-score/GSVA + 可学习 tokenizer 的对照；报告评分方法的敏感性。

### 6.4 定义、重叠与上下文（definition / overlap / context）

- 通路大量重叠，一个基因属于多条通路；固定定义忽略 cell-type/context-specific wiring；MSigDB 的冗余需要 Hallmark 之类精简，但仍不彻底。[Hallmarks2015]
- 同一通路在不同组织/突变背景下可能完全不同；静态通路 token 无法表达。
- **建议**：允许重叠 token（BioMod 的做法）；报告 overlap；用 cell-specific 表达/共表达加权，而不是固定成员平均。

### 6.5 非转录层信息丢失

- 通路活性从 mRNA 推断，无法直接看到蛋白丰度、磷酸化、定位、复合物组装、药物结合构象。
- 对 IC50，蛋白/磷酸化可能比 mRNA 更接近；但当前数据以 transcriptome 为主。
- **建议**：如果未来有 proteomics，优先补；否则不要把 pathway token 说成“机制完整表征”。

### 6.6 瓶颈与不可逆压缩

- 如果下游只看到 14–1000 个通路分数，任何未在 token 中保留的 gene–gene 交互都无法恢复。
- MulDR 2026 明确反对“过早压进单一 feature/relation space”：它把 pathway activity、gene expression、mutation、methylation 作为独立 view，保留各自的关系结构，再做 drug-conditioned interaction。[MulDR2026]
- **建议**：多分辨率/多 view；pathway token 是 auxiliary view，不是 sole input；加 reconstruction/contrastive 约束，保证 token 还能预测 gene-level 表达。

### 6.7 选择泄漏

- 先在全数据上选通路/基因，再做 CV，会泄漏；Widespread data leakage 2026 审计 32 个方法，发现 preprocessing/feature screening 泄漏会显著高估性能并污染 biomarker。[Leakage2026]
- **建议**：通路/基因选择、标准化、HVG 全部放进 training fold；用 nested CV。

### 6.8 解释性幻觉

- 通路 attention/importance 高，不等于因果；可能只是与 drug mean/tissue 相关。
- **建议**：把解释性分析当作“模型依赖哪些输入”，不要当作“生物学因果结论”。你的 T2 遮挡实验已经这样表述，保持。

---

## 7. 对你项目的具体建议

### 7.1 如果继续 DTI/PPI：先做“止损矩阵”

1. **Target-only baseline**：只用直接靶点的 GE/CNV/MU + 药物指纹，不加 PPI。若 full PPI 打不过它，PPI 没有独立信息。
2. **Coverage strata**：OT-only / GDSC-only / 无 DTI；single-target / multi-target；报告每层的 ΔSCC 与配对 bootstrap CI。
3. **Random controls**：保度重连 PPI、保度重连 DTI、pseudo-neighbor（匹配度/可观测性）、drug-permuted features；你已做，继续保持。
4. **Cell-specific PPI**：用 cell line 内 co-expression 给 STRING 边加权，或使用 BIONIC/PRECISE 类 personalized network；与 static PPI 直接对比。
5. **Auxiliary task**：加 PPI link prediction / DTI prediction / target membership 辅助损失；看 LDO 是否改善。
6. **不同端点**：如果 IC50 端点上仍无信号，考虑把 PPI 用于 target ranking/perturbation response（PDGrapher/GEARS 范式），而不是主指标。

**止损规则（建议）**：
- 若 target-only baseline ≥ full PPI 的 95% CI 重叠，停止 PPI 主线；
- 若 random PPI/DTI 复现 ≥80% 增益，停止图拓扑归因；
- 若 OT 子集 <30 个药，不宣称稳定增益，只报告探索性结果。

### 7.2 如果继续通路 token：从“压缩”改为“多分辨率”

1. **保留 gene-level 残差**：P1/P2 的 gene branch 不动；pathway token 作为 auxiliary context，通过 gate/cross-attention 注入。
2. **Target-aware token**：每个药单独一个 token，包含直接靶点 + 1-hop 邻居的 GE/CNV/MU；这比全局 1000 条通路更接近 DTI 的特异性。
3. **可学习 tokenizer**：用 sparse MLP / attention pooling（SurvPath、BioMod），不要固定 mean/GSVA；让 token 依赖当前细胞的表达模式。
4. **Pathway difference**：用“通路内 vs 通路外”的对比（PASO），而不是通路 mean；保留 gene-level 对比信息。
5. **Reconstruction/contrastive 约束**：从 pathway token 预测 gene-level 表达（BioMod 的 asymmetric decoder 思路），或对齐 gene-level 表征；否则压缩不可逆。
6. **Coverage audit**：报告 token 覆盖的基因比例、靶点覆盖率、未覆盖基因清单；对未覆盖基因加 residual token。
7. **Ablation**：gene-level MLP vs pathway-only vs hybrid；LDO/scaffold-cold 下比较；不要只在 warm 上比较。

### 7.3 最小实验矩阵（建议直接排进下一轮）

| # | 实验 | 目的 | 预期 |
|---|---|---|---|
| E1 | target-only（靶点 GE/CNV/MU + fingerprint） | 判断 PPI 是否有独立信息 | 若 E1 ≈ full PPI，停止 |
| E2 | static PPI vs cell-specific PPI（co-expression weighted） | 判断“静态图”是否是瓶颈 | cell-specific 可能小正 |
| E3 | T2 pathway block vs gene-level MLP vs hybrid | 判断通路 token 是否丢信息 | hybrid ≥ pathway-only |
| E4 | target-aware token + pathway token + gene residual | 最可能有效的组合 | 小正/不稳定，但可解释 |
| E5 | LDO + scaffold-cold + double-cold | 严格 cold-start | 增益大概率消失 |
| E6 | degree-preserving random PPI/DTI + pseudo neighbors | 归因 | 若复现，停止图主线 |
| E7 | pathway score 方法敏感性（z-score/GSVA/ssGSEA） | 判断 token 是否 method-dependent | 可能符号/大小变化 |
| E8 | 通路 token reconstruction / probing | 量化信息损失 | 覆盖/稀释越严重，重建越差 |

### 7.4 论文叙事建议

- 不要把负结果写成“失败”；可以写成 **“DTI/PPI 在 LDO 下的可迁移性边界”**：
  - 贡献 1：严格 random-graph / pseudo-neighbor / coverage-stratified 对照；
  - 贡献 2：证明 T1/T2 的增益来自第二塔/融合，而非 PPI/DTI 拓扑；
  - 贡献 3：通路 token 的 coverage/dilution/method-dependence 审计；
  - 贡献 4：给出“什么时候 PPI 值得用”的决策规则（target-only baseline + cold-start protocol）。
- 这与 DrEval、PathGuard、Tran & Klau 的 critical evaluation 方向一致，审稿人通常比“又一个 SOTA”更认可。

---

## 8. 参考文献

- [Tran2026] Tran NK, Klau GW. Drug response prediction: A critical systematic review of current datasets and methods. *Pattern Recognition Letters* 199 (2026) 21–26. doi:10.1016/j.patrec.2025.10.016
- [DrEval2026] Bernett J, Iversen P, Picciani M, Wilhelm M, Baum K, List M. Critical evaluation of drug response prediction models with DrEval. *Nature Communications* 17 (2026). doi:10.1038/s41467-026-72903-w; preprint doi:10.1101/2025.05.26.655288
- [PathGuard2026] Liu C, Zhou Y, Zhang C, Wu S, Ma W. PathGuard-DRP: A leakage-controlled benchmark and external audit of a pathway-aware model for cancer drug response prediction. *Computational Biology and Chemistry* 126(Pt 3):109563 (2026). doi:10.1016/j.compbiolchem.2026.109563
- [TGSA2022] Zhu Y, Ouyang Z, Chen W, Feng R, Chen DZ, Cao J, Wu J. TGSA: protein–protein association-based twin graph neural networks for drug response prediction with similarity augmentation. *Bioinformatics* 38(2):461–468 (2022). doi:10.1093/bioinformatics/btab650
- [DRPreter2022] Shin J, Piao Y, Bang D, Kim S, Jo K. DRPreter: Interpretable Anticancer Drug Response Prediction Using Knowledge-Guided Graph Neural Networks and Transformer. *Int J Mol Sci* 23(22):13919 (2022). doi:10.3390/ijms232213919
- [DIPK2024] Li P, Jiang Z, Liu T, Liu X, Qiao H, Yao X. Improving drug response prediction via integrating gene relationships with deep learning. *Briefings in Bioinformatics* 25(3):bbae153 (2024). doi:10.1093/bib/bbae153
- [PaccMann2019] Manica M, Oskooei A, Born J, Subramanian V, Sáez-Rodríguez J, Rodríguez Martínez M. Toward Explainable Anticancer Compound Sensitivity Prediction via Multimodal Attention-Based Convolutional Encoders. *Molecular Pharmaceutics* 16(12):4797–4806 (2019). doi:10.1021/acs.molpharmaceut.9b00520
- [GCNPath2026] GCNPath: introspecting drug response prediction with pathway-guided graph convolution networks. *Communications Biology* (2026). doi:10.1038/s42003-026-09957-5
- [PDGrapher2025] Gonzalez G, Lin X, Herath I, Veselkov K, Bronstein M, Zitnik M. Combinatorial prediction of therapeutic perturbations using causally inspired neural networks. *Nature Biomedical Engineering* 10:1008–1025 (2025). doi:10.1038/s41551-025-01481-x
- [GEARS2023] Roohani Y, Huang K, Leskovec J. Predicting transcriptional outcomes of novel multigene perturbations with GEARS. *Nature Biotechnology* 42:927–935 (2023). doi:10.1038/s41587-023-01905-6
- [Precily2022] Chawla S, Rockstroh A, Lehman M, et al. Gene expression based inference of cancer drug sensitivity. *Nature Communications* 13:5680 (2022). doi:10.1038/s41467-022-33291-z
- [PASO2025] Wu Y, Chen M, Qin Y. Anticancer drug response prediction integrating multi-omics pathway-based difference features and multiple deep learning techniques. *PLoS Computational Biology* 21(3):e1012905 (2025). doi:10.1371/journal.pcbi.1012905
- [SurvPath2024] Jaume G, Vaidya A, Chen RJ, Williamson DFK, Liang PP, Mahmood F. Modeling Dense Multimodal Interactions Between Biological Pathways and Histology for Survival Prediction. *CVPR 2024*. arXiv:2304.06819
- [BioMod2026] Meng L, Zhu J, Han B, et al. BioMod: Biological Module Tokenization for Single-Cell Foundation Modeling. *bioRxiv* (2026). doi:10.64898/2026.10.12.758490
- [PathwayBench2026] Alakwaa FM, et al. PathwayBench: a multi-criterion benchmark of pseudobulk pathway activity scoring methods reveals rank-window truncation as a two-sided mechanism of biological signal loss in single-cell RNA-seq. *BMC Bioinformatics* (2026). doi:10.1186/s12859-026-06632-8
- [scPS2024] Comparative analysis of single-cell pathway scoring methods and a novel approach. *NAR Genomics and Bioinformatics* 6(3):lqae124 (2024). doi:10.1093/nargab/lqae124
- [PROGENy2018] Schubert M, Klinger B, Klünemann M, et al. Perturbation-response genes reveal signaling footprints in cancer gene expression. *Nature Communications* 9:20 (2018). doi:10.1038/s41467-017-02391-6
- [decoupleR2022] Badia-i-Mompel P, Vélez Santiago J, Braunger J, et al. decoupleR: ensemble of computational methods to infer biological activities from omics data. *Bioinformatics Advances* 2:vbac016 (2022). doi:10.1093/bioadv/vbac016
- [Hallmarks2015] Liberzon A, Birger C, Thorvaldsdóttir H, Ghandi M, Mesirov JP, Tamayo P. The Molecular Signatures Database Hallmark Gene Set Collection. *Cell Systems* 1(6):417–425 (2015). doi:10.1016/j.cels.2015.12.004
- [MulDR2026] Wang Z, Li W, Xu J, Meng Y, Zhou H, Jin S. Structured multi-view relational learning for drug response prediction. *Briefings in Bioinformatics* (2026). doi:10.1093/bib/bbag619
- [TwoStep2020] Two-step multi-omics modelling of drug sensitivity in cancer cell lines to identify driving mechanisms. *PLoS One* 15(11):e0238961 (2020). doi:10.1371/journal.pone.0238961
- [Transcriptomic2026] Alfawaz SA, et al. Transcriptomic Profiles Carry Most of the Multi-Omics Signal for Drug Sensitivity Prediction in Cancer Cell Lines. *Cancers* 18(21):3503 (2026). doi:10.3390/cancers18213503
- [Leakage2026] Asiaee A, Strauch J, Azinfar L, et al. Widespread data leakage inflates accuracy and corrupts biomarker discovery in cancer drug response prediction. *bioRxiv* (2026). doi:10.64898/2026.02.05.704016
- [DeePathNet2024] Cai Z, Poulos RC, Aref A, et al. DeePathNet: A Transformer-Based Deep Learning Model Integrating Multiomic Data with Cancer Pathways. *Cancer Research Communications* 4(12):3151–3164 (2024). doi:10.1158/2767-9764.CRC-24-0285
- [DrugCell2020] Kuenzi BM, Park J, Fong SH, et al. Predicting Drug Response and Synergy Using a Deep Learning Model of Human Cancer Cells. *Cancer Cell* 38(5):672–684.e6 (2020). doi:10.1016/j.ccell.2020.09.014
- [DeepCDR2020] Liu Q, Hu Z, Jiang R, Zhou M. DeepCDR: a hybrid graph convolutional network for predicting cancer drug response. *Bioinformatics* 36(Suppl 2):i911–i918 (2020). doi:10.1093/bioinformatics/btaa822
- [Shen2023] Shen B, Feng F, Li K, Lin P, Ma L, Li H. A systematic assessment of deep learning methods for drug response prediction: from in vitro to clinical applications. *Briefings in Bioinformatics* 24(1):bbac605 (2023). doi:10.1093/bib/bbac605
- [Trust2024] Trust me if you can: a survey on reliability and interpretability of machine learning approaches for drug sensitivity prediction in cancer. *Briefings in Bioinformatics* 25(5):bbae379 (2024). doi:10.1093/bib/bbae379

## 9. 检索与筛选备注

- 检索库：Europe PMC、Crossref、bioRxiv/medRxiv、CVPR/arXiv、期刊页面。
- 关键词：drug response prediction + PPI / DTI / graph prior / cold-start / leave-drug-out；pathway token / pathway activity / gene set scoring / information loss。
- 纳入优先：有 LDO/scaffold-cold/double-cold 或 random-graph/degree-preserving 对照；有 pathway token 消融或信息损失讨论。
- 排除：纯 DTI 预测（不是 DRP）；纯分子生成；无方法细节的综述性宣传。
- 注意：部分 2026 年文献为预印本或在线优先出版，引用时需再核对卷期/页码。

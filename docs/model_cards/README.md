# 模型卡（Model Cards）

每个对外发布的模型（教师/学生/量化版/外部验证版）一张模型卡，记录**训练数据版本、
超参、量化配置、验证指标、适用癌种、使用限制**（计划书 §9.4 版本管理、附录 B.2-8）。
模型权重文件放 `checkpoints/`（Git LFS）或 `deploy/packages/`（Git LFS）。

## 计划中的模型卡

| 模型卡 | 对象 | 状态 |
|---|---|---|
| [`teacher-b0.md`](teacher-b0.md) | B0 双线性基线 | ✅ v0.1（LCO 5 折，单 seed） |
| [`teacher-t0.md`](teacher-t0.md) | T0 多模态交叉注意力教师（v1 / P1 变体） | ✅ v0.2（v1 LCO 5 折单 seed；P1 2 折 × 2 seeds 快速验证；正式 5 折待补） |
| [`teacher-t0-ablations.md`](teacher-t0-ablations.md) | T0 的 E2/E-drug/E-sim 消融报告 | ✅ 筛选 + 5 折确认（单 seed） |
| [`teacher-t1.md`](teacher-t1.md) | T1 图增强 P1 教师（主教师候选） | 🚧 v0.1（新 DTI 快筛完成；图拓扑归因阴性，增益来自第二塔/融合） |
| [`teacher-t1-local.md`](teacher-t1-local.md) | T1-local：每药一跳局部图残差（新 DTI 下的图教师主线） | ⛔ v0.1（两轮快筛完成；阶段 1 未通过，归档为负结果） |
| `student-a-int8.md` | A 档学生 + 动态 INT8（端侧交付） | 待蒸馏/QAT（M5） |
| 待补 | B 档学生 / S-latent 极限档 | — |

## 模板

见 [`_TEMPLATE.md`](_TEMPLATE.md)。

## 红线

- 模型卡必须注明**科研原型，不用于临床诊断/治疗决策**；
- 必须记录训练数据版本与 split（可复现三件套：commit + config + seed）；
- 量化模型必须记录 ΔAUC/ΔRMSE 等精度损失（§9.3 #6）；
- 发布包附 SHA256 与语义化版本号，支持签名校验与回滚（§9.3 #10–#11）。

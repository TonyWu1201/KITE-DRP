# 模型卡（Model Cards）

每个对外发布的模型（教师/学生/量化版/外部验证版）一张模型卡，记录**训练数据版本、
超参、量化配置、验证指标、适用癌种、使用限制**（计划书 §9.4 版本管理、附录 B.2-8）。
模型权重文件放 `checkpoints/`（Git LFS）或 `deploy/packages/`（Git LFS）。

## 计划中的模型卡

| 模型卡 | 对象 | 状态 |
|---|---|---|
| [`teacher-b0.md`](teacher-b0.md) | B0 双线性基线 | ✅ v0.1（LCO 5 折，单 seed） |
| [`teacher-t0.md`](teacher-t0.md) | T0 多模态交叉注意力教师 | ✅ v0.1（LCO 5 折，单 seed；GIN 默认关） |
| `teacher-t1.md` | T1 统一异构图教师（主教师） | 待训练（M4） |
| `student-a-int8.md` | A 档学生 + 动态 INT8（端侧交付） | 待蒸馏/QAT（M5） |
| 待补 | B 档学生 / S-latent 极限档 | — |

## 模板

见 [`_TEMPLATE.md`](_TEMPLATE.md)。

## 红线

- 模型卡必须注明**科研原型，不用于临床诊断/治疗决策**；
- 必须记录训练数据版本与 split（可复现三件套：commit + config + seed）；
- 量化模型必须记录 ΔAUC/ΔRMSE 等精度损失（§9.3 #6）；
- 发布包附 SHA256 与语义化版本号，支持签名校验与回滚（§9.3 #10–#11）。

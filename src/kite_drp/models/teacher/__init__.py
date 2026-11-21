"""教师模型（计划书 §4）：仅训练期存在，部署侧不依赖图或大模型。

已实现：

- ``b0.py``        B0 双线性潜交互基线（无图强基线）
- ``t0.py``        T0 多模态交叉注意力教师（P1 的 backbone）
- ``t1.py``        T1 图增强 P1 教师（旧全局异构图，附录/负对照）
- ``t1_local.py``  T1-local 每药一跳局部图残差（负结果归档）
- ``t2.py``        T2 冻结 P1 + 无图第二塔 + DTI/PPI 固定特征
- ``common.py``    batch 契约 / TeacherOutput / MLP
- ``drug_enc.py``  Morgan + targets + 可选 GIN 分子图
- ``heads.py``     回归 + 分类双头
"""

__all__ = ["b0", "common", "drug_enc", "heads", "t0", "t1", "t1_local", "t2"]

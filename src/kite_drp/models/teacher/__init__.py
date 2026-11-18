"""教师模型（计划书 §4）：仅训练期存在，部署侧不依赖图或大模型。

已实现：

- ``b0.py``        B0 双线性潜交互基线（无图强基线）
- ``t0.py``        T0 多模态交叉注意力教师（必做，M1–M2）
- ``common.py``    batch 契约 / TeacherOutput / MLP
- ``drug_enc.py``  Morgan + targets + 可选 GIN 分子图
- ``heads.py``     回归 + 分类双头

规划中（T1 轮次）：

- ``t1.py``        统一 drug-cell-gene 异构图教师（主教师，M2–M4）
- ``graph.py``     PPI/DTI/通路图构建与缓存
"""

__all__ = ["b0", "common", "drug_enc", "heads", "t0"]

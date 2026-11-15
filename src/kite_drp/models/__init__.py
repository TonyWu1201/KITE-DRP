"""模型层（计划书 §4–§5）。

- :mod:`kite_drp.models.teacher` 教师：B0 双线性基线 / T0 多模态交叉注意力 / T1 异构图
- :mod:`kite_drp.models.student` 学生：VAE 组学补全 + 轻量 MLP 预测头（部署 ≈0.4M 参数）
"""

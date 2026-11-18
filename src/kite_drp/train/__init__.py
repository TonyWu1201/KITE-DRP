"""训练入口（计划书 §6.3 五阶段流程）。

已实现：

- ``features.py``      fold-safe 特征装配（通路分数、药物特征、相似性邻居）
- ``data.py``          DataLoader / batch 拼装
- ``engine.py``        训练/验证/早停/指标落盘
- ``teacher_train.py`` 教师训练（B0/T0）

规划中：

- ``student_train.py``  学生预热 + 课程式蒸馏训练
- ``qat.py``            量化感知训练 + 校准（导出前的最后阶段）
"""

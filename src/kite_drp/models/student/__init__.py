"""学生模型（计划书 §5）：只用临床可及模态，纯 MLP + VAE，无 GNN。

规划模块：

- ``vae.py``       组学补全 VAE/CVAE（从 MU+GE 补 CNV，可选 METH）
- ``predictor.py`` 轻量 MLP 预测头（部署 ≈0.4M 参数）
- ``student.py``   学生整体（部署图 = 编码器 + 预测头 + 药嵌入）
"""

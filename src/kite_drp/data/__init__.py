"""数据层（计划书 §3）：下载、预处理、划分、dataset。

规划模块（随 G0 门禁逐步实现）：

- ``download.py``   各数据源下载与校验（raw 不提交 git）
- ``preprocess.py`` 基因对齐/去重/归一化，产物写入 ``data/interim``、``data/processed``
- ``splits.py``     LCO/LDO/scaffold 等划分协议（§7.2），产物写入 ``data/splits``
- ``dataset.py``    PyTorch Dataset/DataLoader，读取 processed 与 splits
- ``schemas.py``    样本/药物/标签的 schema 与单位约定（§2.1）
"""

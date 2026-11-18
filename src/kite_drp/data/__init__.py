"""数据层（计划书 §3）：清单、预处理、划分、fit-context 与 dataset。

模块对应规划 §8.1：

- :mod:`kite_drp.data.inventory`  L0 原始清单
- :mod:`kite_drp.data.canonical`  L1 canonical 编排
- :mod:`kite_drp.data.assembly`   L2 model-input 装配
- :mod:`kite_drp.data.splits`     P0–P4/LPO
- :mod:`kite_drp.data.fit`        fold 内统计量 / FitContext
- :mod:`kite_drp.data.validation` 产物与泄漏校验
- :mod:`kite_drp.data.dataset`    模型侧 Dataset 契约
- :mod:`kite_drp.data.pairs`      药敏 pair 表（processed 自包含）
- :mod:`kite_drp.data.graph_inputs` T1 PPI/DTI 图输入
- :mod:`kite_drp.data.preflight`  训练前置检查
"""

from __future__ import annotations

__all__ = [
    "assembly",
    "canonical",
    "config",
    "dataset",
    "drugs",
    "fit",
    "genes",
    "graph_inputs",
    "ids",
    "inventory",
    "io",
    "labels",
    "manifest",
    "pairs",
    "preflight",
    "priors",
    "registry",
    "report",
    "splits",
    "tcga",
    "validation",
]

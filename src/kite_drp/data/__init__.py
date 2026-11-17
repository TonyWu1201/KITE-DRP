"""数据层（计划书 §3）：清单、预处理、划分、fit-context 与 dataset。

模块对应规划 §8.1：

- :mod:`kite_drp.data.inventory`  L0 原始清单
- :mod:`kite_drp.data.canonical`  L1 canonical 编排
- :mod:`kite_drp.data.assembly`   L2 model-input 装配
- :mod:`kite_drp.data.splits`     P0–P4/LPO
- :mod:`kite_drp.data.fit`        fold 内统计量 / FitContext
- :mod:`kite_drp.data.validation` 产物与泄漏校验
- :mod:`kite_drp.data.dataset`    模型侧 Dataset 契约
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
    "ids",
    "inventory",
    "io",
    "labels",
    "manifest",
    "priors",
    "registry",
    "report",
    "splits",
    "tcga",
    "validation",
]

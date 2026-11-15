"""KITE-DRP：端侧肿瘤药物敏感性预测的师生迁移模型。

顶层包只做版本与公共出口管理；具体实现位于各子包：

- :mod:`kite_drp.data`     数据下载/预处理/划分（对应计划书 §3）
- :mod:`kite_drp.models`   教师（B0/T0/T1）与学生（VAE + MLP）模型（§4–§5）
- :mod:`kite_drp.distill`  四层知识蒸馏损失（§6）
- :mod:`kite_drp.train`    训练/QAT 入口（§6.3、§10）
- :mod:`kite_drp.eval`     指标、统计、OOD、临床验证（§7–§8）
- :mod:`kite_drp.explain`  可解释性（§4.6、§5.7、§8）
- :mod:`kite_drp.deploy`   训练机侧 ONNX 导出/量化/端侧基准（§9，对应根目录 deploy/ 的资产）
"""

from importlib.metadata import PackageNotFoundError, version

try:  # 以可编辑方式安装时可用；直接从源码运行时回退
    __version__ = version("kite-drp")
except PackageNotFoundError:  # pragma: no cover - 未安装场景
    __version__ = "0.1.0+unknown"

__all__ = ["__version__"]

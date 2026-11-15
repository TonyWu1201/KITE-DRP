"""训练机侧部署工具（计划书 §9）：导出、量化、基准测试。

注意与仓库根目录 ``deploy/`` 的分工：

- 本包是**训练机侧的 Python 代码**：ONNX 导出、INT8 量化、延迟/内存基准。
- 仓库根 ``deploy/`` 是**部署资产与独立服务**：``deploy/server``（本机 Web 服务，
  前端 + HTTP API + 排队）、``deploy/edge``（树莓派侧脚手架）、``deploy/packages``
  （打包产物，Git LFS）。

规划模块：

- ``export_onnx.py``   固定 batch=1、μ 确定性化后导出 opset 17
- ``quantize.py``      动态 INT8 / 静态 PTQ / QAT 校准
- ``package.py``       生成部署包（ONNX + 配置 + gene list + SHA256，≤50MB）
- ``benchmark.py``     端侧延迟/内存/功耗基准与验收报告
"""

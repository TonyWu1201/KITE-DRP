"""KITE-DRP 本机 Web 推理服务（科研原型，不用于临床决策）。

本包属于仓库根 ``deploy/server`` 的**独立 uv 项目**，刻意不依赖根项目
``kite_drp`` 包：服务需要能脱离训练环境、离线分发到医生站或树莓派上单独安装，
训练机侧的导出/量化代码仍在 ``src/kite_drp/deploy/``。

模块划分：

- :mod:`app.config`    环境变量配置（默认只绑定 127.0.0.1，fail fast）
- :mod:`app.schemas`   请求/响应 pydantic 模型（对齐计划书 §9.4 输入输出 schema）
- :mod:`app.predictor` 预测器：ONNX 优先；缺失/校验失败时确定性 demo 模式（显式标记）
- :mod:`app.queue`     asyncio 任务队列 + worker + 任务状态机（内存存储，重启清空）
- :mod:`app.api`       HTTP 路由（/health、/api/v1/schema、/api/v1/jobs*）
- :mod:`app.main`      ``create_app()`` / lifespan，托管纯本地静态前端

安全/隐私红线见 ``deploy/server/README.md`` 与 ``deploy/README.md``。
"""

__version__ = "0.1.0"

__all__ = ["__version__"]

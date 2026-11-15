"""FastAPI 应用工厂与服务入口。

启动方式（详见 deploy/server/README.md）：

.. code-block:: bash

    cd deploy/server
    uv sync                                  # 建立独立 .venv（不污染根项目环境）
    uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
    # 或：uv run python -m app.main

安全/合规要点：

- 默认只绑定 127.0.0.1（计划书 §9.4）；不提供内网穿透/上报/云调用。
- 关闭 Swagger/ReDoc 默认 UI（其静态资源来自 CDN），仅保留纯本地
  ``/api/v1/openapi.json``，保证运行期零外网请求（计划书 §9.3 #10）。
- 前端静态页全部同源本地文件（无 CDN/外部字体），由本模块挂载。
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from . import __version__
from .api import AppState, router
from .config import Settings
from .predictor import build_predictor
from .queue import JobManager

logger = logging.getLogger("kite_drp_server")

#: 纯本地前端静态资源目录（index.html/style.css/app.js，无任何外链）
STATIC_DIR = Path(__file__).resolve().parent / "static"

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def _configure_logging(settings: Settings) -> None:
    """配置日志：只输出 job_id/状态/耗时，绝不记录输入数据（计划书 §9.4）。"""
    level = getattr(logging, settings.log_level, logging.INFO)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level, format=_LOG_FORMAT)
    # 本服务使用 "app" 命名空间（app.queue/app.api/...），统一受 KITE_LOG_LEVEL 控制
    logging.getLogger("app").setLevel(level)
    logging.getLogger("kite_drp_server").setLevel(level)


def create_app(settings: Settings | None = None) -> FastAPI:
    """应用工厂：测试可注入自定义 :class:`Settings`（独立端口/队列/demo 配置）。"""
    settings = settings or Settings.from_env()
    _configure_logging(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        predictor = build_predictor(settings)
        jobs = JobManager(
            predictor,
            maxsize=settings.queue_max,
            workers=settings.workers,
            retention=settings.job_retention,
        )
        await jobs.start()
        app.state.ctx = AppState(
            settings=settings,
            predictor=predictor,
            jobs=jobs,
            started_monotonic=time.monotonic(),
            started_at=datetime.now(UTC),
        )
        logger.info(
            "kite-drp-server %s 已启动：http://%s:%d（demo=%s，workers=%d，队列容量=%d）",
            __version__,
            settings.host,
            settings.port,
            predictor.demo,
            settings.workers,
            settings.queue_max,
        )
        if not settings.is_loopback:
            logger.warning(
                "KITE_SERVER_HOST=%s 不是回环地址：仅限受控测试环境，"
                "正式部署必须保持 127.0.0.1（计划书 §9.4）",
                settings.host,
            )
        if predictor.demo:
            logger.warning(
                "当前为 demo 模式：%s（输出禁止用于科研结论/临床判断）", predictor.demo_reason
            )
        try:
            yield
        finally:
            await jobs.stop()
            logger.info("kite-drp-server 已停止")

    app = FastAPI(
        title="KITE-DRP 本机推理服务（科研原型）",
        version=__version__,
        description=(
            "端侧肿瘤药敏学生模型的本机 Web 服务：离线运行、仅 127.0.0.1、"
            "内置纯本地前端页面 + HTTP API + 后端起排队。科研原型，不用于临床决策。"
        ),
        docs_url=None,  # Swagger UI 资源来自 CDN → 关闭以维持运行期零外网请求
        redoc_url=None,  # ReDoc 同理
        openapi_url="/api/v1/openapi.json",  # 仅机器可读契约（本地 JSON）
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.include_router(router)
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="frontend")
    return app


#: uvicorn app.main:app 的默认入口（默认配置来自环境变量/.env）
app = create_app()


def run() -> None:
    """``uv run python -m app.main`` 的入口（等价于 uvicorn 命令）。"""
    import uvicorn

    settings = Settings.from_env()
    if not settings.is_loopback:
        logger.warning(
            "KITE_SERVER_HOST=%s 非回环地址：仅限受控测试环境，正式部署必须为 127.0.0.1",
            settings.host,
        )
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        access_log=False,  # 访问日志无输入数据，但默认关闭以减少泄漏面
    )


if __name__ == "__main__":  # pragma: no cover - 手动启动入口
    run()

"""HTTP 路由（契约见 deploy/server/README.md，机器可读版本见 /api/v1/openapi.json）。

接口一览：

- ``GET /health``                    健康检查 + 模型/队列状态（含 demo 标记）
- ``GET /api/v1/schema``             输入/输出 JSON Schema + 可直接 POST 的示例
- ``POST /api/v1/jobs``              提交单样本任务 → 202 + job_id
- ``GET /api/v1/jobs/{job_id}``      查询任务状态/结果 → 404 任务不存在
- ``GET /api/v1/jobs?limit=N``       最近任务列表（1–100）

错误约定（响应体统一为 ``{"detail": {"code": "...", "message": "..."}}``）：

- 422：请求体校验失败（pydantic 标准错误格式）
- 404：job_id 不存在（code=job_not_found；内存队列重启清空）
- 503：队列已满（code=queue_full）
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request, status

from . import __version__
from .config import Settings
from .predictor import Predictor
from .queue import JobManager, QueueFullError
from .schemas import (
    DISCLAIMER,
    EXAMPLE_SAMPLE,
    SCHEMA_NOTES,
    SCHEMA_VERSION,
    ErrorDetail,
    HealthOut,
    JobCreateAck,
    JobListOut,
    JobOut,
    QueueHealth,
    ResultOut,
    SampleInput,
    SchemaOut,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@dataclass
class AppState:
    """应用级运行时状态（由 ``app.main.lifespan`` 创建/销毁）。"""

    settings: Settings
    predictor: Predictor
    jobs: JobManager
    started_monotonic: float
    started_at: datetime


def get_state(request: Request) -> AppState:
    """取出 lifespan 初始化的运行时状态（缺失说明服务未就绪）。"""
    state = getattr(request.app.state, "ctx", None)
    if state is None:  # pragma: no cover - 仅在 lifespan 未执行时出现
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ErrorDetail(code="not_ready", message="服务尚未初始化，请稍后重试").model_dump(),
        )
    return state


@router.get("/health", response_model=HealthOut, summary="健康检查与模型状态", tags=["meta"])
def health(request: Request) -> HealthOut:
    """返回服务状态、模型版本/SHA256、demo 标记与队列统计（供前端状态条与运维探活）。"""
    ctx = get_state(request)
    predictor = ctx.predictor
    demo = bool(predictor.demo)
    return HealthOut(
        status="degraded" if demo else "ok",
        version=__version__,
        schema_version=SCHEMA_VERSION,
        model_version=predictor.model_version,
        model_sha256=predictor.model_sha256,
        demo=demo,
        engine=predictor.engine,
        demo_reason=predictor.demo_reason,
        model_dir=predictor.model_dir,
        host=ctx.settings.host,
        port=ctx.settings.port,
        queue=QueueHealth(**ctx.jobs.stats()),
        uptime_s=round(time.monotonic() - ctx.started_monotonic, 3),
        time=datetime.now(UTC),
        notice=DISCLAIMER,
        offline=True,
    )


@router.get(
    "/api/v1/schema", response_model=SchemaOut, summary="输入/输出 schema 与示例", tags=["meta"]
)
def describe_schema() -> SchemaOut:
    """返回 pydantic 生成的 JSON Schema 与可直接 POST 的示例（前端/脚本据此构造请求）。"""
    return SchemaOut(
        schema_version=SCHEMA_VERSION,
        description="单样本药敏响应预测输入/输出 schema（对齐计划书 §9.4）",
        input_schema=SampleInput.model_json_schema(),
        output_schema=ResultOut.model_json_schema(),
        example=EXAMPLE_SAMPLE,
        notes=SCHEMA_NOTES,
    )


@router.post(
    "/api/v1/jobs",
    response_model=JobCreateAck,
    status_code=status.HTTP_202_ACCEPTED,
    summary="提交单样本预测任务（入队，202）",
    tags=["jobs"],
    responses={
        422: {"description": "输入校验失败（字段缺失/类型错误/未提供 mutation 或 expression 等）"},
        503: {"description": "队列已满（detail.code=queue_full），请稍后重试"},
    },
)
def create_job(payload: SampleInput, request: Request) -> JobCreateAck:
    """校验入参并入队；立即返回 202 与 job_id，前端轮询任务详情获取结果。"""
    ctx = get_state(request)
    try:
        record = ctx.jobs.submit(payload)
    except QueueFullError as exc:
        logger.warning("拒绝任务：队列已满（%d/%d）", ctx.jobs.size, ctx.jobs.capacity)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ErrorDetail(
                code="queue_full", message=f"任务队列已满（容量 {ctx.jobs.capacity}），请稍后重试"
            ).model_dump(),
        ) from exc
    return JobCreateAck(
        job_id=record.job_id,
        status=record.status,
        created_at=record.created_at,
        poll_url=f"/api/v1/jobs/{record.job_id}",
        demo=bool(ctx.predictor.demo),
        notice=DISCLAIMER,
    )


@router.get(
    "/api/v1/jobs/{job_id}",
    response_model=JobOut,
    summary="查询任务状态/结果",
    tags=["jobs"],
    responses={
        404: {"description": "任务不存在（detail.code=job_not_found；服务重启后内存队列清空）"}
    },
)
def get_job(job_id: str, request: Request) -> JobOut:
    """返回任务状态机当前状态；终态（succeeded/failed）时附带 result 或 error。"""
    ctx = get_state(request)
    record = ctx.jobs.get(job_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ErrorDetail(
                code="job_not_found",
                message=f"任务不存在：{job_id}（内存队列在服务重启后清空）",
            ).model_dump(),
        )
    return record.to_out()


@router.get("/api/v1/jobs", response_model=JobListOut, summary="最近任务列表", tags=["jobs"])
def list_jobs(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=100, description="返回最近任务条数（1–100）")] = 10,
) -> JobListOut:
    """按创建时间倒序返回最近任务（仅本机内存，重启清空）。"""
    ctx = get_state(request)
    records = ctx.jobs.recent(limit)
    return JobListOut(
        count=len(records),
        total=ctx.jobs.total,
        items=[record.to_out() for record in records],
    )

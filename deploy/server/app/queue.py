"""asyncio 任务队列 + worker + 任务状态机（纯内存，重启清空）。

状态机::

    queued ──▶ running ──▶ succeeded
                  │
                  └──▶ failed

设计要点：

- 队列容量由 ``KITE_QUEUE_MAX`` 控制，满时 :meth:`JobManager.submit` 抛
  :class:`QueueFullError`（HTTP 503），避免无界内存增长。
- worker 数由 ``KITE_WORKERS`` 控制（默认 1 = 单 worker 串行，便于端侧限流/复现）。
- 预测调用通过 ``asyncio.to_thread`` 执行，阻塞推理不会卡住 ``/health`` 等接口。
- **不写磁盘、不写输入数据到日志**：日志只含 job_id/状态/耗时，满足计划书 §9.4
  「日志不含 PHI 与原始基因值」；任务在进程重启后清空（已写入文档）。
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import UTC, datetime

from .predictor import Predictor
from .schemas import (
    RESPONSE_SCALE,
    SCHEMA_VERSION,
    ConfidenceInterval,
    JobOut,
    JobStatus,
    ResultOut,
    SampleInput,
)

logger = logging.getLogger(__name__)


class QueueFullError(Exception):
    """任务队列已满（映射为 HTTP 503）。"""


def _utcnow() -> datetime:
    """当前 UTC 时间（带时区，序列化为 ISO 8601）。"""
    return datetime.now(UTC)


def _format_error(exc: BaseException) -> str:
    """把异常格式化为**不含输入数据**的简短错误摘要。"""
    message = str(exc).strip().replace("\n", " ")
    if len(message) > 300:
        message = message[:300] + "…"
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


@dataclass
class JobRecord:
    """内存中的任务记录（由 worker 修改，读取时转换为 :class:`JobOut`）。"""

    job_id: str
    sample: SampleInput
    status: JobStatus = JobStatus.QUEUED
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: ResultOut | None = None
    error: str | None = None

    def to_out(self) -> JobOut:
        """转换为不可变响应模型。"""
        return JobOut(
            schema_version=SCHEMA_VERSION,
            job_id=self.job_id,
            status=self.status,
            created_at=self.created_at,
            updated_at=self.updated_at,
            started_at=self.started_at,
            finished_at=self.finished_at,
            input=self.sample,
            result=self.result,
            error=self.error,
        )


class JobManager:
    """asyncio 队列 + 固定数量 worker + 内存任务表。"""

    def __init__(
        self,
        predictor: Predictor,
        *,
        maxsize: int = 64,
        workers: int = 1,
        retention: int = 200,
    ) -> None:
        if maxsize < 1:
            raise ValueError("queue maxsize 必须 >= 1")
        if workers < 0:
            raise ValueError("workers 必须 >= 0")
        if retention < 1:
            raise ValueError("retention 必须 >= 1")
        self.predictor = predictor
        self._queue: asyncio.Queue[JobRecord] = asyncio.Queue(maxsize=maxsize)
        self._workers = workers
        self._retention = retention
        self._jobs: OrderedDict[str, JobRecord] = OrderedDict()
        self._tasks: list[asyncio.Task[None]] = []
        self._running = 0
        self._succeeded = 0
        self._failed = 0
        self._started = False

    # ------------------------------------------------------------------ 状态查询
    @property
    def capacity(self) -> int:
        """队列容量上限。"""
        return self._queue.maxsize

    @property
    def size(self) -> int:
        """当前排队任务数（不含已被 worker 取走的执行中任务）。"""
        return self._queue.qsize()

    @property
    def total(self) -> int:
        """内存中任务总数。"""
        return len(self._jobs)

    def get(self, job_id: str) -> JobRecord | None:
        """按 ID 查询任务（不存在返回 None → HTTP 404）。"""
        return self._jobs.get(job_id)

    def recent(self, limit: int = 10) -> list[JobRecord]:
        """最近任务列表（最新在前）。"""
        records = list(self._jobs.values())
        records.reverse()
        return records[:limit]

    def stats(self) -> dict[str, int]:
        """队列健康统计（供 /health 使用）。"""
        return {
            "size": self.size,
            "capacity": self.capacity,
            "active_workers": len(self._tasks),
            "running": self._running,
            "succeeded": self._succeeded,
            "failed": self._failed,
        }

    # ------------------------------------------------------------------ 生命周期
    async def start(self) -> None:
        """启动 worker（幂等；workers=0 时只入队不执行，仅供测试/维护）。"""
        if self._started:  # pragma: no cover - lifespan 保证只调用一次
            return
        self._started = True
        if self._workers == 0:
            logger.warning("KITE_WORKERS=0：任务只入队不执行（仅供测试/维护模式）")
            return
        for index in range(self._workers):
            task = asyncio.create_task(self._worker_loop(index), name=f"kite-worker-{index}")
            self._tasks.append(task)
        logger.info("已启动 %d 个推理 worker（队列容量 %d）", len(self._tasks), self.capacity)

    async def stop(self) -> None:
        """停止 worker，并把未完成任务标记为失败（内存队列不持久化）。"""
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self._started = False
        for record in self._jobs.values():
            if record.status in (JobStatus.QUEUED, JobStatus.RUNNING):
                record.status = JobStatus.FAILED
                record.error = "服务关闭：任务未完成（内存队列不持久化，重启后需重新提交）"
                record.updated_at = _utcnow()
                record.finished_at = record.updated_at
        logger.info("推理队列已停止")

    # ------------------------------------------------------------------ 提交/执行
    def submit(self, sample: SampleInput) -> JobRecord:
        """创建任务并入队；队列满时抛 :class:`QueueFullError`（HTTP 503）。"""
        record = JobRecord(job_id=uuid.uuid4().hex, sample=sample)
        try:
            self._queue.put_nowait(record)
        except asyncio.QueueFull as exc:
            raise QueueFullError(f"任务队列已满（容量 {self.capacity}）") from exc
        self._jobs[record.job_id] = record
        self._prune()
        logger.info("任务 %s 已入队（排队 %d/%d）", record.job_id, self.size, self.capacity)
        return record

    def _prune(self) -> None:
        """只淘汰已进入终态的旧任务，保持内存占用有界。"""
        if len(self._jobs) <= self._retention:
            return
        for job_id in list(self._jobs):
            if len(self._jobs) <= self._retention:
                break
            record = self._jobs[job_id]
            if record.status in (JobStatus.QUEUED, JobStatus.RUNNING):
                continue
            del self._jobs[job_id]

    async def _worker_loop(self, worker_index: int) -> None:
        """worker 主循环：FIFO 取任务并串行执行。"""
        while True:
            record = await self._queue.get()
            try:
                await self._execute(record, worker_index)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - worker 绝不能被单个任务拖垮
                logger.exception(
                    "worker %d 处理任务 %s 时出现未捕获异常", worker_index, record.job_id
                )
                record.status = JobStatus.FAILED
                record.error = "内部错误：worker 未捕获异常（详情见服务端日志）"
                record.updated_at = _utcnow()
                record.finished_at = record.updated_at
            finally:
                self._queue.task_done()

    async def _execute(self, record: JobRecord, worker_index: int) -> None:
        """执行单个任务并推进状态机（异常不允许逃逸到队列）。"""
        record.status = JobStatus.RUNNING
        record.started_at = _utcnow()
        record.updated_at = record.started_at
        self._running += 1
        logger.info("任务 %s 开始推理（worker=%d）", record.job_id, worker_index)
        started = time.perf_counter()
        try:
            prediction = await asyncio.to_thread(self.predictor.predict, record.sample)
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
            record.result = ResultOut(
                schema_version=SCHEMA_VERSION,
                sample_id=record.sample.sample_id,
                drug=record.sample.drug,
                response=prediction.response,
                response_scale=RESPONSE_SCALE,
                ci=ConfidenceInterval(
                    level=prediction.ci_level,
                    low=prediction.ci_low,
                    high=prediction.ci_high,
                ),
                model_version=prediction.model_version,
                model_sha256=prediction.model_sha256,
                demo=prediction.demo,
                engine=prediction.engine,
                warnings=list(prediction.warnings),
                explanation=prediction.explanation,
                elapsed_ms=elapsed_ms,
            )
            record.status = JobStatus.SUCCEEDED
            self._succeeded += 1
            logger.info(
                "任务 %s 成功（%.2f ms，engine=%s，demo=%s）",
                record.job_id,
                elapsed_ms,
                prediction.engine,
                prediction.demo,
            )
        except Exception as exc:  # noqa: BLE001 - 单任务失败不应影响服务
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
            record.status = JobStatus.FAILED
            record.error = _format_error(exc)
            self._failed += 1
            logger.warning("任务 %s 失败（%.2f ms）：%s", record.job_id, elapsed_ms, record.error)
        finally:
            record.finished_at = _utcnow()
            record.updated_at = record.finished_at
            self._running -= 1

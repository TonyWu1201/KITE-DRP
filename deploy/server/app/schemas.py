"""请求/响应模型（对齐计划书 §9.4 输入输出 schema）。

- 输入单样本 JSON：``schema_version/sample_id/assay/mutation/expression/drug/meta``
- 输出：``response/ci/model_version/model_sha256``（另含 demo/engine/warnings 等可审计字段）
- 隐私：``sample_id`` 只允许伪名化 ID（正则限制为字母数字/``._-``），接口文档明确
  禁止填写姓名、住院号等 PHI；服务端日志不记录任何输入内容。
- 合规：不输出"敏感/耐药"硬分类与临床阈值（计划书附录 C：临床呈现以分数+置信区间为主，
  阈值仅作参考）；``response`` 是科研用未校准分数，不是临床概率。
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

#: API/数据 schema 版本（破坏性变更时递增，见 deploy/server/README.md「接口版本策略」）
SCHEMA_VERSION = "1.0"
#: 响应分数刻度标识：0≈更可能耐药、1≈更可能敏感（未校准科研分数）
RESPONSE_SCALE = "kite_response_score_v1"
#: 全站统一声明：科研原型，不用于临床决策（计划书 §9.4、附录 B.2-9）
DISCLAIMER = "科研原型，不用于临床决策"
#: 基因/样本 ID 允许字符（避免空格与特殊字符进入日志/文件名）
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
#: 单样本允许的最大基因条目数（防滥用；临床 panel 量级为 ~500–2000）
MAX_GENES = 4096
#: 表达值上限（防止 NaN/Inf 与极端值；训练侧 log2(TPM+1) 量级远小于该值）
MAX_EXPRESSION_ABS = 1.0e6


def _validate_gene_map(entries: dict[str, Any], field_name: str) -> None:
    if len(entries) > MAX_GENES:
        raise ValueError(f"{field_name} 条目数不能超过 {MAX_GENES}（当前 {len(entries)}）")
    for gene in entries:
        if not _ID_RE.match(gene):
            raise ValueError(f"{field_name} 的基因名非法：{gene!r}（允许字母数字/._-，≤64 字符）")


class SampleInput(BaseModel):
    """单样本输入（主形态；批量 CSV 为一期扩展项，见 deploy/README.md）。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    schema_version: Literal["1.0"] = Field(
        default=SCHEMA_VERSION, description="输入 schema 版本，当前固定为 1.0"
    )
    sample_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=_ID_RE.pattern,
        description="伪名化样本 ID（例如 DEMO-0001）；禁止真实姓名/住院号等 PHI",
    )
    assay: str | None = Field(
        default=None, max_length=128, description="检测平台/方法，例如 'DNA panel + RNA-seq'"
    )
    mutation: dict[str, int] = Field(
        default_factory=dict,
        description="突变：基因名 → 0/1；缺失基因请省略（不要用 0 冒充缺失，计划书 §9.2 #3）",
    )
    expression: dict[str, float] = Field(
        default_factory=dict,
        description="表达：基因名 → 数值（log2(TPM+1) 或通路分数，需与训练侧一致）",
    )
    drug: str = Field(
        min_length=1, max_length=128, description="药物名称（须与部署包 manifest 药表一致）"
    )
    meta: dict[str, str | int | float | bool] = Field(
        default_factory=dict,
        description="可选临床变量（癌种/分期等），只放低敏字段，不含 PHI",
    )

    @field_validator("mutation")
    @classmethod
    def _check_mutation(cls, value: dict[str, int]) -> dict[str, int]:
        _validate_gene_map(value, "mutation")
        for gene, flag in value.items():
            if flag not in (0, 1):
                raise ValueError(f"mutation[{gene}] 必须为 0（未突变）或 1（突变），当前 {flag!r}")
        return value

    @field_validator("expression")
    @classmethod
    def _check_expression(cls, value: dict[str, float]) -> dict[str, float]:
        _validate_gene_map(value, "expression")
        for gene, level in value.items():
            if not math.isfinite(level):
                raise ValueError(f"expression[{gene}] 必须是有限数值")
            if abs(level) > MAX_EXPRESSION_ABS:
                raise ValueError(f"expression[{gene}] 超出允许范围 ±{MAX_EXPRESSION_ABS:g}")
        return value

    @field_validator("meta")
    @classmethod
    def _check_meta(cls, value: dict[str, str | int | float | bool]) -> dict[str, Any]:
        if len(value) > 64:
            raise ValueError("meta 字段数不能超过 64")
        for key in value:
            if not _ID_RE.match(str(key)):
                raise ValueError(f"meta 的键非法：{key!r}（允许字母数字/._-，≤64 字符）")
        return value

    @model_validator(mode="after")
    def _require_signal(self) -> SampleInput:
        if not self.mutation and not self.expression:
            raise ValueError("至少提供 mutation 或 expression 之一（否则无法预测）")
        return self


class ConfidenceInterval(BaseModel):
    """区间估计（科研用；demo 模式为确定性伪区间）。"""

    model_config = ConfigDict(extra="forbid")

    level: float = Field(ge=0.5, le=0.999, description="置信水平，例如 0.9")
    low: float = Field(ge=0.0, le=1.0, description="区间下界")
    high: float = Field(ge=0.0, le=1.0, description="区间上界")

    @model_validator(mode="after")
    def _ordered(self) -> ConfidenceInterval:
        if self.low > self.high:
            raise ValueError("ci.low 不能大于 ci.high")
        return self


class ResultOut(BaseModel):
    """单样本推理结果。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = Field(default=SCHEMA_VERSION, description="输出 schema 版本")
    sample_id: str = Field(description="回显的伪名化样本 ID")
    drug: str = Field(description="回显的药物名称")
    response: float = Field(
        ge=0.0,
        le=1.0,
        description="药敏响应分数：0≈更可能耐药、1≈更可能敏感；科研用未校准分数，非临床概率",
    )
    response_scale: str = Field(default=RESPONSE_SCALE, description="响应分数刻度标识")
    ci: ConfidenceInterval = Field(description="区间估计（demo 模式为伪区间）")
    model_version: str = Field(description="模型/部署包版本（demo 模式为 demo-*）")
    model_sha256: str = Field(
        min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$", description="模型文件 SHA256"
    )
    demo: bool = Field(description="是否 demo 模式（true 时输出为确定性伪分数，禁止用于任何判断）")
    engine: Literal["demo", "onnxruntime"] = Field(description="推理引擎")
    warnings: list[str] = Field(default_factory=list, description="必须原样展示给用户的警示")
    explanation: dict[str, Any] | None = Field(
        default=None, description="可解释性输出（通路/基因归因，demo 模式为伪归因）"
    )
    elapsed_ms: float = Field(ge=0.0, description="本次推理耗时（毫秒），不含排队时间")

    @model_validator(mode="after")
    def _ci_contains_response(self) -> ResultOut:
        if not (self.ci.low - 1e-9 <= self.response <= self.ci.high + 1e-9):
            raise ValueError("response 必须落在 ci 区间内")
        return self


class JobStatus(StrEnum):
    """任务状态机：queued → running → succeeded / failed。"""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class JobOut(BaseModel):
    """任务详情（创建响应与轮询/列表共用同一任务对象）。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = Field(default=SCHEMA_VERSION)
    job_id: str = Field(description="任务 ID（uuid4 hex）")
    status: JobStatus = Field(description="queued/running/succeeded/failed")
    created_at: datetime = Field(description="创建时间（UTC，ISO 8601）")
    updated_at: datetime = Field(description="最近一次状态变更时间（UTC）")
    started_at: datetime | None = Field(default=None, description="开始推理时间（UTC）")
    finished_at: datetime | None = Field(default=None, description="终态时间（UTC）")
    input: SampleInput = Field(description="提交的输入（内存回显，服务重启后清空）")
    result: ResultOut | None = Field(default=None, description="成功时的推理结果")
    error: str | None = Field(default=None, description="失败时的错误摘要（不含输入数据）")


class JobCreateAck(BaseModel):
    """POST /api/v1/jobs 的 202 响应体。"""

    job_id: str = Field(description="任务 ID，用于轮询")
    status: JobStatus = Field(default=JobStatus.QUEUED)
    created_at: datetime = Field(description="创建时间（UTC）")
    poll_url: str = Field(description="轮询地址（同源相对路径）")
    demo: bool = Field(description="服务当前是否 demo 模式")
    notice: str = Field(default=DISCLAIMER, description="固定合规声明")


class JobListOut(BaseModel):
    """GET /api/v1/jobs 的响应体（最近任务，最新在前）。"""

    count: int = Field(ge=0, description="本次返回条数")
    total: int = Field(ge=0, description="内存中任务总数（受 retention 限制）")
    items: list[JobOut] = Field(description="任务列表，最新在前")


class QueueHealth(BaseModel):
    """队列健康信息。"""

    size: int = Field(ge=0, description="排队中的任务数（不含正在执行的 worker）")
    capacity: int = Field(ge=1, description="队列容量上限")
    active_workers: int = Field(ge=0, description="worker 数")
    running: int = Field(ge=0, description="正在推理的任务数")
    succeeded: int = Field(ge=0, description="本次进程生命周期内成功任务数")
    failed: int = Field(ge=0, description="本次进程生命周期内失败任务数")


class HealthOut(BaseModel):
    """GET /health 的响应体。"""

    status: Literal["ok", "degraded"] = Field(
        description="ok=已加载真实 ONNX 模型；degraded=demo 模式（或模型校验失败）"
    )
    service: str = Field(default="kite-drp-server")
    version: str = Field(description="服务版本")
    schema_version: str = Field(description="API schema 版本")
    model_version: str = Field(description="模型版本（demo 模式为 demo-*）")
    model_sha256: str = Field(description="模型文件 SHA256（demo 模式为 64 个 0）")
    demo: bool = Field(description="是否 demo 模式；true 时所有结果都不可用于任何判断")
    engine: Literal["demo", "onnxruntime"] = Field(description="推理引擎")
    demo_reason: str | None = Field(default=None, description="demo/降级原因（便于运维排查）")
    model_dir: str | None = Field(default=None, description="模型目录（未配置时为 null）")
    host: str = Field(description="配置的监听地址（正式部署必须为 127.0.0.1）")
    port: int = Field(description="配置的监听端口")
    queue: QueueHealth = Field(description="队列状态")
    uptime_s: float = Field(ge=0.0, description="进程运行时长（秒）")
    time: datetime = Field(description="服务器当前时间（UTC）")
    notice: str = Field(default=DISCLAIMER, description="固定合规声明")
    offline: bool = Field(
        default=True, description="运行期零外网请求承诺（前端与后端均无外部调用）"
    )


class SchemaOut(BaseModel):
    """GET /api/v1/schema 的响应体（机器可读契约 + 示例）。"""

    schema_version: str = Field(description="schema 版本")
    description: str = Field(description="用途说明")
    input_schema: dict[str, Any] = Field(description="SampleInput 的 JSON Schema")
    output_schema: dict[str, Any] = Field(description="ResultOut 的 JSON Schema")
    example: dict[str, Any] = Field(description="可直接 POST 的示例输入")
    notes: list[str] = Field(description="字段语义/合规注意事项")


class ErrorDetail(BaseModel):
    """统一错误响应体（HTTPException.detail）。"""

    code: str = Field(description="机器可读错误码，如 queue_full / job_not_found")
    message: str = Field(description="中文错误说明（不含输入数据）")


#: 可直接 POST 的示例（/api/v1/schema 返回，测试与前端示例按钮复用）
EXAMPLE_SAMPLE: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "sample_id": "DEMO-0001",
    "assay": "DNA panel + RNA-seq",
    "mutation": {"EGFR": 1, "TP53": 1, "KRAS": 0},
    "expression": {"EGFR": 5.2, "TP53": 4.1, "MYC": 6.0},
    "drug": "Osimertinib",
    "meta": {"cancer_type": "LUAD", "stage": "IV"},
}

#: /api/v1/schema 的说明条目（字段语义与合规边界）
SCHEMA_NOTES: list[str] = [
    "schema_version 当前固定为 1.0；破坏性变更时递增并在 deploy/server/README.md 记录。",
    "sample_id 必须是伪名化 ID；接口与日志均不接受/记录姓名、住院号等 PHI（计划书 §9.4）。",
    "mutation 取值为 0（未突变）或 1（突变）；缺失基因请省略，由部署包内基因列表与 mask 处理，"
    "不要用 0 冒充缺失（计划书 §9.2 #3）。",
    "expression 数值须与训练侧归一化方式一致（log2(TPM+1) 或通路分数）；缺失基因同样请省略。",
    "至少提供 mutation 或 expression 之一（降级输入：仅突变 panel 亦可，见计划书 §9.5）。",
    "response 是 0–1 的科研用未校准分数（越高越可能敏感），不是临床概率；服务不输出"
    "敏感/耐药硬分类与临床阈值（计划书附录 C）。",
    "结果中的 demo=true 表示输出为演示用确定性伪分数，必须在界面上明确展示，禁止用于任何判断。",
    "服务仅监听 127.0.0.1、运行期零外网请求；任务与结果只存内存，重启清空。",
]

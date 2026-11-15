"""预测器：ONNX 优先，缺失/校验失败时进入确定性 demo 模式（显式标记）。

部署包契约（由训练机侧 ``src/kite_drp/deploy`` 导出，详见 deploy/server/README.md）::

    deploy/packages/<版本>/
    ├── manifest.json   # 版本、SHA256、基因列表、药表、输入输出名
    └── model.onnx      # 固定 batch=1 的 ONNX（opset 17，CPU EP；计划书 §9.2）

合规要求（计划书 §9.3 #10）：**模型 SHA256 校验通过才加载**；校验失败或不满足契约时
拒绝加载真实模型，回退 demo 模式，并在 ``/health`` 与所有结果中显式标记 ``demo=true``。

本模块不打印任何输入特征到日志（隐私红线，计划书 §9.4）。
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .config import Settings
from .schemas import DISCLAIMER, SampleInput

logger = logging.getLogger(__name__)

#: demo 模式的伪模型标识（64 个 0 一眼可辨，避免被误当作真实校验和）
DEMO_MODEL_VERSION = "demo-0.0.0"
DEMO_MODEL_SHA256 = "0" * 64
DEMO_CI_LEVEL = 0.9
#: demo 模式必带的警示语（结果 warnings 中必须原样展示）
DEMO_WARNING = (
    "demo 模式：未加载真实 ONNX 模型，response/ci 为确定性伪分数，禁止用于科研结论或任何临床判断。"
)


class PredictorLoadError(RuntimeError):
    """ONNX 部署包不满足契约（缺文件/校验失败/依赖缺失等），需回退 demo。"""


@dataclass(frozen=True, slots=True)
class Prediction:
    """单样本预测输出（由队列层包装为 :class:`~app.schemas.ResultOut`）。"""

    response: float
    ci_low: float
    ci_high: float
    ci_level: float
    model_version: str
    model_sha256: str
    demo: bool
    engine: str
    warnings: tuple[str, ...]
    explanation: dict[str, Any] | None = None


class Predictor(Protocol):
    """预测器接口（同步调用；队列层用 ``asyncio.to_thread`` 执行，不阻塞事件循环）。"""

    demo: bool
    engine: str
    model_version: str
    model_sha256: str
    demo_reason: str | None
    model_dir: str | None

    def predict(self, sample: SampleInput) -> Prediction:  # pragma: no cover - 协议定义
        ...


def _sample_digest(sample: SampleInput) -> bytes:
    """对输入做**顺序无关**的规范化哈希（demo 分数确定性来源）。"""
    canonical = {
        "sample_id": sample.sample_id,
        "drug": sample.drug,
        "assay": sample.assay or "",
        "mutation": sorted(sample.mutation.items()),
        "expression": sorted((gene, round(value, 6)) for gene, value in sample.expression.items()),
        "meta": sorted((key, str(value)) for key, value in sample.meta.items()),
    }
    encoded = json.dumps(
        canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(encoded.encode("utf-8")).digest()


class DemoPredictor:
    """确定性 demo 预测器：同一输入永远得到同一输出，所有输出显式 mark demo。"""

    demo = True
    engine = "demo"
    model_version = DEMO_MODEL_VERSION
    model_sha256 = DEMO_MODEL_SHA256

    def __init__(self, reason: str, model_dir: str | None = None) -> None:
        self.demo_reason: str | None = reason
        self.model_dir: str | None = model_dir

    def predict(self, sample: SampleInput) -> Prediction:
        digest = _sample_digest(sample)
        primary = int.from_bytes(digest[0:8], "big") / float(1 << 64)
        spread = int.from_bytes(digest[8:10], "big") / float(1 << 16)
        response = round(0.05 + 0.90 * primary, 4)
        half_width = round(0.04 + 0.08 * spread, 4)
        ci_low = round(max(0.0, response - half_width), 4)
        ci_high = round(min(1.0, response + half_width), 4)
        if ci_low > ci_high:  # pragma: no cover - 数值上不可能，防御性处理
            ci_low, ci_high = ci_high, ci_low
        return Prediction(
            response=response,
            ci_low=ci_low,
            ci_high=ci_high,
            ci_level=DEMO_CI_LEVEL,
            model_version=DEMO_MODEL_VERSION,
            model_sha256=DEMO_MODEL_SHA256,
            demo=True,
            engine="demo",
            warnings=(DISCLAIMER, DEMO_WARNING),
            explanation=_demo_explanation(digest, sample),
        )


def _demo_explanation(digest: bytes, sample: SampleInput) -> dict[str, Any]:
    """demo 模式的伪归因（仅为验证前端可解释性展示链路，不代表任何真实原因）。"""
    genes = sorted(set(sample.mutation) | set(sample.expression))
    scored: list[dict[str, Any]] = []
    for gene in genes[:50]:
        gene_digest = hashlib.sha256(digest + gene.encode("utf-8")).digest()
        weight = int.from_bytes(gene_digest[0:4], "big") / float(1 << 32)
        scored.append({"gene": gene, "contribution": round(0.1 + 0.9 * weight, 3)})
    scored.sort(key=lambda item: item["contribution"], reverse=True)
    return {
        "method": "demo-hash（伪归因，仅用于界面联通性验证，不代表真实生物学原因）",
        "top_features": scored[:5],
    }


def _sha256_file(path: Path) -> str:
    """流式计算文件 SHA256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OnnxPredictor:
    """ONNX Runtime CPU 预测器（可选依赖 onnxruntime，通过 ``uv sync --extra onnx`` 安装）。"""

    demo = False
    engine = "onnxruntime"
    demo_reason: str | None = None

    def __init__(
        self,
        model_dir: Path,
        session: Any,
        manifest: dict[str, Any],
        model_sha256: str,
    ) -> None:
        self.model_dir = str(model_dir.resolve())
        self._session = session
        self._manifest = manifest
        self.model_version = str(manifest["model_version"])
        self.model_sha256 = model_sha256
        self._mutation_genes = [str(gene) for gene in manifest.get("mutation_genes") or []]
        self._expression_genes = [str(gene) for gene in manifest.get("expression_genes") or []]
        self._drug_index = {
            str(key): int(value) for key, value in (manifest.get("drug_index") or {}).items()
        }
        inputs = manifest.get("inputs") or {}
        self._inputs = {str(key): str(value) for key, value in inputs.items()}
        outputs = manifest.get("outputs") or {}
        self._outputs = {str(key): str(value) for key, value in outputs.items()}
        self._ci_level = float(manifest.get("ci_level", 0.9))
        self._ci_half_width = float(manifest.get("ci_half_width", 0.05))
        if not self._mutation_genes and not self._expression_genes:
            raise PredictorLoadError(
                "manifest.json 至少需要 mutation_genes 或 expression_genes 之一"
            )
        if "response" not in self._outputs:
            raise PredictorLoadError("manifest.json 的 outputs 必须包含 response 输出名")

    @classmethod
    def from_dir(cls, model_dir: Path) -> OnnxPredictor:
        """按部署包契约加载并校验 ONNX 模型（校验失败抛 :class:`PredictorLoadError`）。"""
        manifest_path = model_dir / "manifest.json"
        if not manifest_path.is_file():
            raise PredictorLoadError(f"缺少模型清单：{manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PredictorLoadError(f"模型清单不可解析：{manifest_path}（{exc}）") from exc
        if not isinstance(manifest, dict):
            raise PredictorLoadError("manifest.json 顶层必须是 JSON 对象")
        if not manifest.get("model_version"):
            raise PredictorLoadError("manifest.json 缺少 model_version")
        model_file = str(manifest.get("model_file") or "model.onnx")
        model_path = model_dir / model_file
        if not model_path.is_file():
            raise PredictorLoadError(f"模型文件不存在：{model_path}")
        actual_sha256 = _sha256_file(model_path)
        expected_sha256 = str(manifest.get("model_sha256") or "").strip().lower()
        if not expected_sha256:
            raise PredictorLoadError(
                "manifest.json 缺少 model_sha256（合规要求模型 SHA256 校验通过才加载，"
                "计划书 §9.3 #10）"
            )
        if expected_sha256 != actual_sha256:
            raise PredictorLoadError(
                f"模型 SHA256 校验失败：manifest={expected_sha256}，实际={actual_sha256}"
            )
        manifest["model_sha256"] = actual_sha256
        try:
            import onnxruntime as ort  # 可选依赖：未安装时回退 demo
        except ImportError as exc:
            raise PredictorLoadError(
                "未安装 onnxruntime（安装命令：uv sync --extra onnx）"
            ) from exc
        options = ort.SessionOptions()
        options.log_severity_level = 3  # ERROR：避免运行期向外部/日志刷屏
        session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        return cls(
            model_dir=model_dir, session=session, manifest=manifest, model_sha256=actual_sha256
        )

    def predict(self, sample: SampleInput) -> Prediction:
        import numpy as np  # onnxruntime 的依赖；仅 ONNX 模式需要

        feeds: dict[str, Any] = {}
        mutation_genes = self._mutation_genes
        if mutation_genes:
            # 缺失基因用显式 mask 标记；0 只表示"未突变"（计划书 §9.2 #3）
            feeds[self._inputs.get("mutation", "mutation")] = np.asarray(
                [[sample.mutation.get(gene, 0) for gene in mutation_genes]], dtype=np.int64
            )
            feeds[self._inputs.get("mutation_mask", "mutation_mask")] = np.asarray(
                [[1 if gene in sample.mutation else 0 for gene in mutation_genes]], dtype=np.int64
            )
        expression_genes = self._expression_genes
        if expression_genes:
            feeds[self._inputs.get("expression", "expression")] = np.asarray(
                [[float(sample.expression.get(gene, 0.0)) for gene in expression_genes]],
                dtype=np.float32,
            )
            feeds[self._inputs.get("expression_mask", "expression_mask")] = np.asarray(
                [[1 if gene in sample.expression else 0 for gene in expression_genes]],
                dtype=np.int64,
            )
        if self._drug_index:
            if sample.drug not in self._drug_index:
                raise PredictorLoadError(
                    f"药物不在部署包药表中：{sample.drug}（请检查 manifest.json 的 drug_index）"
                )
            feeds[self._inputs.get("drug", "drug_id")] = np.asarray(
                [[self._drug_index[sample.drug]]], dtype=np.int64
            )

        requested = [self._outputs["response"]]
        has_ci_low = "ci_low" in self._outputs
        has_ci_high = "ci_high" in self._outputs
        if has_ci_low:
            requested.append(self._outputs["ci_low"])
        if has_ci_high:
            requested.append(self._outputs["ci_high"])
        raw = self._session.run(requested, feeds)
        raw_response = float(np.asarray(raw[0]).reshape(-1)[0])
        raw_ci_low = (
            float(np.asarray(raw[1]).reshape(-1)[0])
            if has_ci_low
            else raw_response - self._ci_half_width
        )
        raw_ci_high = (
            float(np.asarray(raw[2]).reshape(-1)[0])
            if has_ci_high
            else raw_response + self._ci_half_width
        )

        warnings: list[str] = [DISCLAIMER]
        if not 0.0 <= raw_response <= 1.0:
            warnings.append(
                "模型原始输出不在 [0,1] 内（已钳制），请检查导出侧 response 定标与 manifest 契约。"
            )
        response = min(1.0, max(0.0, raw_response))
        ci_low = min(1.0, max(0.0, min(raw_ci_low, response)))
        ci_high = min(1.0, max(response, raw_ci_high))
        return Prediction(
            response=response,
            ci_low=ci_low,
            ci_high=ci_high,
            ci_level=self._ci_level,
            model_version=self.model_version,
            model_sha256=self.model_sha256,
            demo=False,
            engine="onnxruntime",
            warnings=tuple(warnings),
            explanation=None,
        )


def build_predictor(settings: Settings) -> Predictor:
    """按配置加载预测器；任何加载问题都回退 demo 模式并记录显式原因。"""
    model_dir = settings.model_dir
    model_dir_text = str(model_dir) if model_dir is not None else None
    if model_dir is None:
        reason = "KITE_MODEL_DIR 未配置或为空（显式 demo 模式）"
        logger.warning("预测器：%s", reason)
        return DemoPredictor(reason)
    if not model_dir.is_dir():
        reason = f"模型目录不存在：{model_dir}"
        logger.warning("预测器：%s → demo 模式", reason)
        return DemoPredictor(reason, model_dir=model_dir_text)
    try:
        predictor = OnnxPredictor.from_dir(model_dir)
    except PredictorLoadError as exc:
        logger.error("预测器：ONNX 模型加载失败，回退 demo 模式：%s", exc)
        return DemoPredictor(str(exc), model_dir=model_dir_text)
    logger.info(
        "预测器：已加载 ONNX 模型 %s（sha256=%s，目录=%s）",
        predictor.model_version,
        predictor.model_sha256,
        predictor.model_dir,
    )
    return predictor

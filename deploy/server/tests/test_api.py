"""deploy/server 接口测试（真实运行 uvicorn 应用对象，不 mock 队列/预测器）。

覆盖 Lead 验收点：

- /health 状态、模型版本、demo 标记
- /api/v1/schema 契约与示例
- 创建任务 → 轮询到终态 → 结果字段（response/ci/model_version/model_sha256）
- 非法输入 422、未知任务 404、队列满 503
- 前端静态页 200 且**无任何外链/CDN**（含 CSP、Swagger UI 关闭）
- demo 模式标记与确定性
- 隐私：日志不含样本 ID/基因值；失败任务状态机

运行：``cd deploy/server && uv sync && uv run pytest -q``
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path

import pytest
from app.config import DEFAULT_ENV_FILE, DEFAULT_MODEL_DIR, ConfigError, Settings
from app.main import STATIC_DIR, create_app
from app.predictor import DEMO_MODEL_SHA256, DemoPredictor, build_predictor
from app.schemas import (
    DISCLAIMER,
    EXAMPLE_SAMPLE,
    SCHEMA_VERSION,
    JobStatus,
    SampleInput,
)
from fastapi.testclient import TestClient

#: 轮询终态的最长等待时间（测试环境 demo 推理为毫秒级）
TERMINAL_TIMEOUT_S = 5.0


# --------------------------------------------------------------------------- fixtures
@pytest.fixture()
def clean_kite_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """清空进程中的 KITE_* 变量，保证配置测试不受外部环境影响。"""
    for name in list(os.environ):
        if name.startswith("KITE_"):
            monkeypatch.delenv(name, raising=False)


@pytest.fixture()
def settings() -> Settings:
    """测试默认使用 demo 模式 + 单 worker + 小队列，不接触任何真实模型目录。"""
    return Settings(model_dir=None, workers=1, queue_max=8, job_retention=50)


@pytest.fixture()
def client(settings: Settings):
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


def wait_terminal(client: TestClient, job_id: str) -> dict:
    """轮询任务直到 succeeded/failed，返回最终任务 JSON。"""
    deadline = time.monotonic() + TERMINAL_TIMEOUT_S
    payload: dict = {}
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200, response.text
        payload = response.json()
        if payload["status"] in {JobStatus.SUCCEEDED, JobStatus.FAILED}:
            return payload
        time.sleep(0.01)
    raise AssertionError(f"任务 {job_id} 未在 {TERMINAL_TIMEOUT_S}s 内进入终态：{payload}")


def create_job(client: TestClient, payload: dict | None = None) -> dict:
    response = client.post("/api/v1/jobs", json=payload or EXAMPLE_SAMPLE)
    assert response.status_code == 202, response.text
    return response.json()


# --------------------------------------------------------------------------- 健康/契约
def test_health_reports_demo_mode(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"  # 未加载 ONNX → degraded
    assert body["demo"] is True
    assert body["engine"] == "demo"
    assert body["model_version"].startswith("demo-")
    assert body["model_sha256"] == DEMO_MODEL_SHA256
    assert body["demo_reason"]
    assert body["schema_version"] == SCHEMA_VERSION
    assert body["host"] == "127.0.0.1"
    assert body["offline"] is True
    assert body["notice"] == DISCLAIMER
    assert body["queue"]["capacity"] == 8
    assert body["queue"]["active_workers"] == 1
    assert body["uptime_s"] >= 0


def test_schema_endpoint_matches_pydantic_models(client: TestClient) -> None:
    response = client.get("/api/v1/schema")
    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == SCHEMA_VERSION
    properties = body["input_schema"]["properties"]
    expected = {"schema_version", "sample_id", "assay", "mutation", "expression", "drug", "meta"}
    assert expected <= set(properties)
    output_properties = body["output_schema"]["properties"]
    assert {"response", "ci", "model_version", "model_sha256"} <= set(output_properties)
    # 示例必须可直接通过校验并提交
    example = SampleInput.model_validate(body["example"])
    assert example.sample_id == "DEMO-0001"
    assert any("PHI" in note for note in body["notes"])
    assert any("demo=true" in note for note in body["notes"])


# --------------------------------------------------------------------------- 任务主流程
def test_create_job_poll_to_success(client: TestClient) -> None:
    ack = create_job(client)
    assert re.fullmatch(r"[0-9a-f]{32}", ack["job_id"])
    assert ack["status"] == "queued"
    assert ack["poll_url"] == f"/api/v1/jobs/{ack['job_id']}"
    assert ack["demo"] is True
    assert ack["notice"] == DISCLAIMER

    final = wait_terminal(client, ack["job_id"])
    assert final["status"] == "succeeded"
    assert final["error"] is None
    assert final["input"]["sample_id"] == EXAMPLE_SAMPLE["sample_id"]
    result = final["result"]
    assert 0.0 <= result["response"] <= 1.0
    assert result["ci"]["low"] <= result["response"] <= result["ci"]["high"]
    assert result["ci"]["level"] == pytest.approx(0.9)
    assert result["model_version"].startswith("demo-")
    assert re.fullmatch(r"[0-9a-f]{64}", result["model_sha256"])
    assert result["demo"] is True
    assert result["engine"] == "demo"
    assert result["response_scale"] == "kite_response_score_v1"
    assert result["elapsed_ms"] >= 0
    assert any(DISCLAIMER in warning for warning in result["warnings"])
    assert any("demo" in warning for warning in result["warnings"])
    # 状态机时间戳：created <= updated，且 started/finished 存在
    assert final["created_at"] <= final["updated_at"]
    assert final["started_at"] and final["finished_at"]
    assert final["started_at"] <= final["finished_at"]


def test_mutation_only_input_is_accepted(client: TestClient) -> None:
    """降级输入（仅突变 panel，计划书 §9.5）应可正常推理。"""
    payload = {
        "sample_id": "DEMO-MUT-ONLY",
        "drug": "Osimertinib",
        "mutation": {"EGFR": 1, "TP53": 1},
    }
    ack = create_job(client, payload)
    final = wait_terminal(client, ack["job_id"])
    assert final["status"] == "succeeded"


def test_recent_jobs_list_and_limit(client: TestClient) -> None:
    first = create_job(client, {**EXAMPLE_SAMPLE, "sample_id": "DEMO-LIST-1"})
    second = create_job(client, {**EXAMPLE_SAMPLE, "sample_id": "DEMO-LIST-2"})
    wait_terminal(client, first["job_id"])
    wait_terminal(client, second["job_id"])

    response = client.get("/api/v1/jobs", params={"limit": 10})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert body["total"] >= 2
    assert [item["job_id"] for item in body["items"][:2]] == [second["job_id"], first["job_id"]]

    limited = client.get("/api/v1/jobs", params={"limit": 1}).json()
    assert limited["count"] == 1
    assert limited["items"][0]["job_id"] == second["job_id"]

    assert client.get("/api/v1/jobs", params={"limit": 0}).status_code == 422
    assert client.get("/api/v1/jobs", params={"limit": 101}).status_code == 422


def test_demo_output_is_deterministic(client: TestClient) -> None:
    shuffled = {
        **EXAMPLE_SAMPLE,
        "mutation": {"KRAS": 0, "TP53": 1, "EGFR": 1},
        "expression": {"MYC": 6.0, "EGFR": 5.2, "TP53": 4.1},
    }
    first = wait_terminal(client, create_job(client, EXAMPLE_SAMPLE)["job_id"])
    second = wait_terminal(client, create_job(client, shuffled)["job_id"])
    assert first["result"]["response"] == second["result"]["response"]
    assert first["result"]["ci"] == second["result"]["ci"]


# --------------------------------------------------------------------------- 错误码
@pytest.mark.parametrize(
    "payload",
    [
        {},  # 全空
        {  # 缺 drug/sample_id
            "mutation": {"EGFR": 1},
        },
        {  # schema 版本错误
            "schema_version": "2.0",
            "sample_id": "X",
            "drug": "Osimertinib",
            "mutation": {"EGFR": 1},
        },
        {  # mutation 与 expression 同时缺失
            "sample_id": "X",
            "drug": "Osimertinib",
        },
        {  # sample_id 含非 ASCII（疑似真实姓名 → 拒绝）
            "sample_id": "张三",
            "drug": "Osimertinib",
            "mutation": {"EGFR": 1},
        },
        {  # 突变值必须是 0/1
            "sample_id": "X",
            "drug": "Osimertinib",
            "mutation": {"EGFR": 2},
        },
        {  # 未知字段（契约严格）
            "sample_id": "X",
            "drug": "Osimertinib",
            "mutation": {"EGFR": 1},
            "unknown_field": True,
        },
        {  # 表达值超出允许范围
            "sample_id": "X",
            "drug": "Osimertinib",
            "expression": {"EGFR": 1.0e7},
        },
        {  # 药物名为空
            "sample_id": "X",
            "drug": "",
            "mutation": {"EGFR": 1},
        },
    ],
)
def test_invalid_input_returns_422(client: TestClient, payload: dict) -> None:
    response = client.post("/api/v1/jobs", json=payload)
    assert response.status_code == 422, response.text
    assert "detail" in response.json()


def test_unknown_job_returns_404(client: TestClient) -> None:
    response = client.get("/api/v1/jobs/does-not-exist")
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert detail["code"] == "job_not_found"
    assert "不存在" in detail["message"]


def test_queue_full_returns_503() -> None:
    """workers=0 + 容量 1：第一个任务占满队列，第二个必须收到 503。"""
    app = create_app(Settings(model_dir=None, workers=0, queue_max=1, job_retention=10))
    with TestClient(app) as client:
        first = client.post("/api/v1/jobs", json=EXAMPLE_SAMPLE)
        assert first.status_code == 202, first.text
        second = client.post("/api/v1/jobs", json=EXAMPLE_SAMPLE)
        assert second.status_code == 503, second.text
        detail = second.json()["detail"]
        assert detail["code"] == "queue_full"
        assert "已满" in detail["message"]


def test_failed_job_enters_failed_state(client: TestClient) -> None:
    class BrokenPredictor:
        demo = True
        engine = "demo"
        model_version = "broken-0.0.0"
        model_sha256 = "f" * 64
        demo_reason = "测试用故障预测器"
        model_dir = None

        def predict(self, sample: SampleInput):  # noqa: ANN201 - 测试桩
            raise RuntimeError("模拟推理故障")

    client.app.state.ctx.jobs.predictor = BrokenPredictor()
    ack = create_job(client)
    final = wait_terminal(client, ack["job_id"])
    assert final["status"] == "failed"
    assert final["result"] is None
    assert "模拟推理故障" in final["error"]
    assert final["started_at"] and final["finished_at"]


# --------------------------------------------------------------------------- 隐私
def test_logs_do_not_leak_sample_id_or_gene_values(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    sample_id = "PSEUDO-PATIENT-42"
    payload = {
        "sample_id": sample_id,
        "drug": "Osimertinib",
        "mutation": {"EGFR": 1},
        "expression": {"MYC": 9876.54321},
    }
    ack = create_job(client, payload)
    wait_terminal(client, ack["job_id"])

    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert ack["job_id"] in messages  # 只记录任务 ID 用于审计
    assert sample_id not in messages  # 样本 ID 不落日志
    assert "9876.54321" not in messages  # 原始表达值不落日志
    assert "EGFR" not in messages  # 基因名不落日志


# --------------------------------------------------------------------------- 前端静态页
def test_frontend_is_fully_local(client: TestClient) -> None:
    index = client.get("/")
    assert index.status_code == 200
    html = index.text
    assert "科研原型，不用于临床决策" in html
    assert 'href="style.css"' in html
    assert 'src="app.js"' in html
    assert "Content-Security-Policy" in html
    assert "default-src 'self'" in html

    assert client.get("/style.css").status_code == 200
    js = client.get("/app.js")
    assert js.status_code == 200
    assert "fetch(" in js.text

    static_files = sorted(path.name for path in STATIC_DIR.iterdir() if path.is_file())
    assert static_files == ["app.js", "index.html", "style.css"]
    for name in static_files:
        text = (STATIC_DIR / name).read_text(encoding="utf-8")
        assert "http://" not in text, f"{name} 含绝对外链"
        assert "https://" not in text, f"{name} 含绝对外链"
        assert "@import" not in text.lower(), f"{name} 含 CSS @import"
        for match in re.findall(r"""(?:href|src)\s*=\s*["']([^"']*)["']""", text):
            assert not match.startswith(("http:", "https:", "//")), f"{name} 引用了外部资源 {match}"


def test_swagger_ui_disabled_to_keep_offline(client: TestClient) -> None:
    """Swagger/ReDoc 默认 UI 会从 CDN 拉资源，必须关闭；只保留本地 openapi.json。"""
    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404
    assert client.get("/openapi.json").status_code == 404
    spec_response = client.get("/api/v1/openapi.json")
    assert spec_response.status_code == 200
    paths = spec_response.json()["paths"]
    assert {"/health", "/api/v1/schema", "/api/v1/jobs", "/api/v1/jobs/{job_id}"} <= set(paths)


# --------------------------------------------------------------------------- 预测器/配置单测
def test_demo_predictor_is_order_independent() -> None:
    predictor = DemoPredictor("test")
    sample_a = SampleInput.model_validate(EXAMPLE_SAMPLE)
    reordered = {
        **EXAMPLE_SAMPLE,
        "mutation": {"KRAS": 0, "TP53": 1, "EGFR": 1},
        "expression": {"MYC": 6.0, "EGFR": 5.2, "TP53": 4.1},
    }
    sample_b = SampleInput.model_validate(reordered)
    first = predictor.predict(sample_a)
    second = predictor.predict(sample_b)
    assert first.response == second.response
    assert (first.ci_low, first.ci_high) == (second.ci_low, second.ci_high)
    assert first.demo is True
    assert first.engine == "demo"
    assert first.model_sha256 == "0" * 64
    explanation = first.explanation or {}
    assert explanation["top_features"], "demo 模式也应返回可解释性占位数据"


def test_predictor_falls_back_to_demo() -> None:
    unconfigured = build_predictor(Settings(model_dir=None))
    assert isinstance(unconfigured, DemoPredictor)
    assert "KITE_MODEL_DIR" in (unconfigured.demo_reason or "")

    missing_dir = build_predictor(Settings(model_dir=Path("/nonexistent/kite-model")))
    assert isinstance(missing_dir, DemoPredictor)
    assert "不存在" in (missing_dir.demo_reason or "")


def test_predictor_rejects_checksum_mismatch(tmp_path: Path) -> None:
    """合规要求：SHA256 校验失败必须拒绝加载真实模型（计划书 §9.3 #10）。"""
    model_dir = tmp_path / "pkg"
    model_dir.mkdir()
    (model_dir / "model.onnx").write_bytes(b"fake-onnx-bytes")
    manifest = {
        "schema_version": "1.0",
        "model_version": "0.1.0",
        "model_file": "model.onnx",
        "model_sha256": "f" * 64,
        "mutation_genes": ["EGFR"],
        "expression_genes": ["EGFR"],
        "inputs": {},
        "outputs": {"response": "response"},
    }
    (model_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    predictor = build_predictor(Settings(model_dir=model_dir))
    assert isinstance(predictor, DemoPredictor)
    assert "SHA256" in (predictor.demo_reason or "")


def test_settings_defaults_and_env_override(
    clean_kite_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    empty_env = tmp_path / "empty.env"
    empty_env.write_text("", encoding="utf-8")
    monkeypatch.setenv("KITE_ENV_FILE", str(empty_env))

    defaults = Settings.from_env()
    assert defaults.host == "127.0.0.1"
    assert defaults.port == 8000
    assert defaults.is_loopback is True
    assert defaults.model_dir == DEFAULT_MODEL_DIR

    monkeypatch.setenv("KITE_WORKERS", "3")
    monkeypatch.setenv("KITE_MODEL_DIR", "")
    overridden = Settings.from_env()
    assert overridden.workers == 3
    assert overridden.model_dir is None


def test_settings_env_file_precedence(
    clean_kite_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_file = tmp_path / "server.env"
    env_file.write_text(
        "KITE_SERVER_PORT=8111\nKITE_MODEL_DIR=../packages/current\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KITE_ENV_FILE", str(env_file))

    from_file = Settings.from_env()
    assert from_file.port == 8111
    # 相对路径以 deploy/server 为基准
    assert from_file.model_dir == (DEFAULT_ENV_FILE.parent / "../packages/current").resolve()

    monkeypatch.setenv("KITE_SERVER_PORT", "8222")
    assert Settings.from_env().port == 8222


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("KITE_SERVER_PORT", "abc"),
        ("KITE_SERVER_PORT", "0"),
        ("KITE_QUEUE_MAX", "0"),
        ("KITE_WORKERS", "9"),
        ("KITE_JOB_RETENTION", "-1"),
        ("KITE_LOG_LEVEL", "LOUD"),
    ],
)
def test_settings_invalid_values_fail_fast(
    clean_kite_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str, value: str
) -> None:
    empty_env = tmp_path / "empty.env"
    empty_env.write_text("", encoding="utf-8")
    monkeypatch.setenv("KITE_ENV_FILE", str(empty_env))
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigError):
        Settings.from_env()

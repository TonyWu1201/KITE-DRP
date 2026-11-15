"""服务配置（环境变量驱动，无隐藏状态）。

设计约束（计划书 §9.3 #10、§9.4）：

- 默认 ``KITE_SERVER_HOST=127.0.0.1``：只允许医生站本机浏览器访问；
  本服务不实现任何内网穿透、上报或云端调用。
- 所有配置来自环境变量；另支持从 ``deploy/server/.env`` 读取（仅接受 ``KITE_``
  前缀的键，显式导出的环境变量优先），便于 Linux/Windows 本地运维。
- 配置非法时在启动阶段直接报错（fail fast），而不是运行期静默降级。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

#: 默认只绑定回环地址（计划书 §9.4「本机 Web UI/FastAPI，仅 127.0.0.1」）
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_QUEUE_MAX = 64
DEFAULT_WORKERS = 1
DEFAULT_JOB_RETENTION = 200
DEFAULT_LOG_LEVEL = "INFO"

#: 默认模型目录：``deploy/packages/current``（不存在时自动进入 demo 模式）
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "packages" / "current"
#: ``.env`` 默认位置：``deploy/server/.env``（不入库，已被根 .gitignore 忽略）
DEFAULT_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
#: 只加载此前缀的环境变量，避免 .env 文件意外注入其他进程级配置
ENV_PREFIX = "KITE_"


class ConfigError(ValueError):
    """环境变量配置非法（启动期直接失败）。"""


def _read_env_file(path: Path) -> dict[str, str]:
    """读取极简 ``KEY=VALUE`` 格式的 .env 文件（不引入额外依赖）。"""
    if not path.is_file():
        return {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:  # pragma: no cover - 权限/编码异常场景
        raise ConfigError(f"无法读取环境文件 {path}：{exc}") from exc
    values: dict[str, str] = {}
    for line in lines:
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        if item.startswith("export "):
            item = item[len("export ") :].strip()
        key, sep, value = item.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key.startswith(ENV_PREFIX):
            values[key] = value
    return values


def _parse_int(name: str, value: str | None, *, default: int, minimum: int, maximum: int) -> int:
    """解析整数配置并做范围校验（非法时抛 :class:`ConfigError`）。"""
    if value is None:
        return default
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} 必须是整数（当前值：{value!r}）") from exc
    if not minimum <= parsed <= maximum:
        raise ConfigError(f"{name} 必须在 [{minimum}, {maximum}] 范围内（当前值：{parsed}）")
    return parsed


def _parse_model_dir(value: str | None) -> Path | None:
    """解析模型目录：``None`` 表示未配置（走默认），空字符串表示显式 demo 模式。"""
    if value is None:
        return DEFAULT_MODEL_DIR
    text = value.strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        # 相对路径以 deploy/server 为基准，避免受启动时工作目录影响
        path = (DEFAULT_ENV_FILE.parent / path).resolve()
    return path


@dataclass(frozen=True, slots=True)
class Settings:
    """服务运行配置；测试可直接构造，生产通过 :meth:`from_env` 读取环境变量。"""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    model_dir: Path | None = DEFAULT_MODEL_DIR
    queue_max: int = DEFAULT_QUEUE_MAX
    workers: int = DEFAULT_WORKERS
    job_retention: int = DEFAULT_JOB_RETENTION
    log_level: str = DEFAULT_LOG_LEVEL

    @classmethod
    def from_env(cls) -> Settings:
        """从环境变量（叠加可选 .env 文件）构造配置。"""
        env_file = os.environ.get("KITE_ENV_FILE", "").strip()
        file_path = Path(env_file).expanduser() if env_file else DEFAULT_ENV_FILE
        if env_file and not file_path.is_file():
            raise ConfigError(f"KITE_ENV_FILE 指向的文件不存在：{file_path}")
        file_values = _read_env_file(file_path)

        def raw(name: str, default: str | None = None) -> str | None:
            if name in os.environ:
                return os.environ[name]
            return file_values.get(name, default)

        host = (raw("KITE_SERVER_HOST", DEFAULT_HOST) or DEFAULT_HOST).strip() or DEFAULT_HOST
        port = _parse_int(
            "KITE_SERVER_PORT",
            raw("KITE_SERVER_PORT", str(DEFAULT_PORT)),
            default=DEFAULT_PORT,
            minimum=1,
            maximum=65535,
        )
        queue_max = _parse_int(
            "KITE_QUEUE_MAX",
            raw("KITE_QUEUE_MAX", str(DEFAULT_QUEUE_MAX)),
            default=DEFAULT_QUEUE_MAX,
            minimum=1,
            maximum=10_000,
        )
        workers = _parse_int(
            "KITE_WORKERS",
            raw("KITE_WORKERS", str(DEFAULT_WORKERS)),
            default=DEFAULT_WORKERS,
            minimum=0,
            maximum=8,
        )
        job_retention = _parse_int(
            "KITE_JOB_RETENTION",
            raw("KITE_JOB_RETENTION", str(DEFAULT_JOB_RETENTION)),
            default=DEFAULT_JOB_RETENTION,
            minimum=1,
            maximum=10_000,
        )
        log_level = (raw("KITE_LOG_LEVEL", DEFAULT_LOG_LEVEL) or DEFAULT_LOG_LEVEL).strip().upper()
        if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ConfigError(
                f"KITE_LOG_LEVEL 非法：{log_level!r}（可选 DEBUG/INFO/WARNING/ERROR/CRITICAL）"
            )
        model_dir = _parse_model_dir(raw("KITE_MODEL_DIR"))
        return cls(
            host=host,
            port=port,
            model_dir=model_dir,
            queue_max=queue_max,
            workers=workers,
            job_retention=job_retention,
            log_level=log_level,
        )

    @property
    def is_loopback(self) -> bool:
        """当前监听地址是否为回环地址（正式部署必须为 True）。"""
        host = self.host.strip().lower()
        return host in {"localhost", "::1"} or host.startswith("127.")

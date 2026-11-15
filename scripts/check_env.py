#!/usr/bin/env python3
"""KITE-DRP 环境自检脚本。

特点：
  - 只依赖 Python 标准库（可在 ``uv sync`` 之前先跑），缺失第三方包时给安装提示、不抛 traceback；
  - 覆盖：Python/平台、uv 项目配置与锁文件、核心依赖、可选 extras、PyTorch+CUDA 12.6、
    NVIDIA 驱动、仓库结构与 data/ 目录、uv/git/git-lfs/pre-commit；
  - 退出码：0 = 没有 FAIL（WARN 不阻塞）；1 = 存在 FAIL 或 ``--strict`` 下有 WARN。

用法::

    uv run python scripts/check_env.py              # 日常自检（CPU/数据开发环境应通过）
    uv run python scripts/check_env.py --strict     # CI/验收：WARN 也视为失败
    uv run python scripts/check_env.py --require-train   # GPU 训练机：要求 cu126 torch + CUDA 可用
    uv run kite-drp doctor                          # 等价于 uv run python scripts/check_env.py

设计说明与验收命令清单见 ``docs/environment.md``；CUDA 12.6 的由来见
``docs/adr/0002-python-environment-cuda126.md``。
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.metadata as metadata
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import unicodedata
from pathlib import Path

# ---------------------------------------------------------------------------
# 常量（与 pyproject.toml / docs/environment.md 保持一致）
# ---------------------------------------------------------------------------
MIN_PYTHON: tuple[int, int] = (3, 12)
SUPPORTED_PLATFORMS: dict[str, str] = {"linux": "Linux", "win32": "Windows"}
EXPECTED_TORCH: str = "2.14.1"
EXPECTED_CUDA: str = "12.6"
MIN_GLIBC: tuple[int, int] = (2, 28)  # manylinux_2_28（cu126 torch / 主流科学计算 wheel）
RECOMMENDED_DRIVER_MAJOR: int = (
    560  # CUDA 12.6 GA 驱动；>=525 亦可经 minor version compatibility 运行
)
MIN_CUDA12_DRIVER_MAJOR: int = 525  # CUDA 12.x 最低驱动（Linux 525.60.13 / Windows 527.41）
MIN_UV_VERSION: tuple[int, int] = (0, 12)

OK, WARN, FAIL, SKIP = "OK", "WARN", "FAIL", "SKIP"

#: 核心依赖：Windows/Linux 均有 cp312 wheel，``uv sync`` 默认安装（含 dev 组）。
CORE_PACKAGES: list[tuple[str, str]] = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("scikit-learn", "sklearn"),
    ("xgboost", "xgboost"),
    ("pyarrow", "pyarrow"),
    ("pyyaml", "yaml"),
    ("hydra-core", "hydra"),
    ("omegaconf", "omegaconf"),
    ("tqdm", "tqdm"),
    ("rich", "rich"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("optuna", "optuna"),
]

#: 可选 extras 组：发行名 -> 导入名；缺整组时给出一条安装提示。
EXTRA_GROUPS: dict[str, list[tuple[str, str]]] = {
    "train": [
        ("torch", "torch"),
    ],
    "graph": [
        ("torch", "torch"),
        ("torch-geometric", "torch_geometric"),
        ("rdkit", "rdkit"),
    ],
    "explain": [
        ("torch", "torch"),
        ("captum", "captum"),
        ("shap", "shap"),
        ("gseapy", "gseapy"),
    ],
    "deploy": [
        ("onnx", "onnx"),
        ("onnxruntime", "onnxruntime"),
    ],
    "clinical": [
        ("lifelines", "lifelines"),
        ("scikit-survival", "sksurv"),
    ],
}

#: data/ 子目录（计划书 §3.8）
DATA_SUBDIRS: tuple[str, ...] = ("raw", "interim", "processed", "splits", "licenses")
DATA_GITKEEPS: tuple[str, ...] = ("raw", "interim", "processed", "splits")


# ---------------------------------------------------------------------------
# 输出与结果记录
# ---------------------------------------------------------------------------
class Checker:
    """收集检查结果并打印带状态的报告。"""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self.counts: dict[str, int] = {OK: 0, WARN: 0, FAIL: 0, SKIP: 0}
        self.problems: list[tuple[str, str, str]] = []  # (status, name, detail)

    def section(self, title: str) -> None:
        print(f"\n== {title} ==")

    def add(self, status: str, name: str, detail: str) -> None:
        self.counts[status] += 1
        if status in (WARN, FAIL):
            self.problems.append((status, name, detail))
        tag = {OK: " OK ", WARN: "WARN", FAIL: "FAIL", SKIP: "SKIP"}[status]
        print(f"[{tag}] {_pad(name, 22)} {detail}")

    def ok(self, name: str, detail: str) -> None:
        self.add(OK, name, detail)

    def warn(self, name: str, detail: str) -> None:
        self.add(WARN, name, detail)

    def fail(self, name: str, detail: str) -> None:
        self.add(FAIL, name, detail)

    def skip(self, name: str, detail: str) -> None:
        self.add(SKIP, name, detail)


def _run_cmd(cmd: list[str], timeout: float = 15.0) -> tuple[int, str]:
    """运行外部命令，返回 (returncode, stdout+stderr)；任何异常都转成 (-1, 原因)。"""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, output.strip()
    except FileNotFoundError:
        return -1, "命令不存在"
    except subprocess.TimeoutExpired:
        return -1, f"命令超时（>{timeout:.0f}s）"
    except OSError as exc:
        return -1, f"命令无法执行：{exc}"


def _version_of(dist_name: str) -> str | None:
    """返回已安装发行版的版本号；未安装返回 None。"""
    try:
        return metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        return None
    except Exception:  # noqa: BLE001 - 损坏的 dist-info 也不应让自检崩掉
        return None


def _import_ok(module_name: str) -> tuple[bool, str]:
    """尝试导入模块，返回 (成功?, 错误信息)。"""
    try:
        importlib.import_module(module_name)
        return True, ""
    except Exception as exc:  # noqa: BLE001 - 缺依赖/DLL 加载失败都归为不可用
        return False, f"{type(exc).__name__}: {exc}"


def _parse_major(version: str) -> int | None:
    match = re.match(r"(\d+)", version.strip())
    return int(match.group(1)) if match else None


def _display_width(text: str) -> int:
    """按终端显示宽度计算字符串长度（中文等全角字符占 2 列）。"""
    return sum(2 if unicodedata.east_asian_width(char) in ("W", "F") else 1 for char in text)


def _pad(text: str, width: int) -> str:
    """按显示宽度右侧补空格，保证中英文混排的输出列对齐。"""
    return text + " " * max(0, width - _display_width(text))


# ---------------------------------------------------------------------------
# 单个检查
# ---------------------------------------------------------------------------
def check_python_and_platform(c: Checker) -> None:
    c.section("Python 与平台")

    py_ver = platform.python_version()
    if sys.version_info >= MIN_PYTHON:
        c.ok("Python 版本", f"{py_ver}（要求 >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}）")
    else:
        c.fail(
            "Python 版本",
            f"{py_ver} 低于要求 {MIN_PYTHON[0]}.{MIN_PYTHON[1]}；"
            "请用 `uv python install 3.12` 并由 `uv run` 启动本脚本",
        )

    if struct.calcsize("P") * 8 == 64:
        c.ok("解释器位数", "64-bit")
    else:
        c.fail("解释器位数", "32-bit 解释器不受支持，请安装 64-bit Python 3.12")

    platform_label = SUPPORTED_PLATFORMS.get(sys.platform)
    if platform_label is None:
        c.fail(
            "操作系统",
            f"sys.platform={sys.platform} 不受支持；本项目只支持 Linux 与 Windows"
            "（macOS 无 cu126 torch wheel，见 docs/adr/0002-python-environment-cuda126.md）",
        )
    else:
        machine = platform.machine() or "unknown"
        cpu = machine.lower()
        if cpu in {"x86_64", "amd64"}:
            c.ok("操作系统", f"{platform_label} / {machine}（x86_64）")
        elif sys.platform == "linux" and cpu in {"aarch64", "arm64"}:
            c.warn(
                "操作系统",
                f"{platform_label} / {machine}：可做 CPU 推理/端侧（onnxruntime），"
                "cu126 训练未在本项目验证；训练机请用 x86_64",
            )
        elif sys.platform == "win32" and cpu in {"arm64", "aarch64"}:
            c.warn("操作系统", f"{platform_label} / {machine}：cu126 无 win_arm64 wheel，请用 x64")
        else:
            c.warn("操作系统", f"{platform_label} / {machine}：非主流架构，wheel 覆盖可能不全")

    if sys.platform == "linux":
        glibc = _glibc_version()
        if glibc is None:
            c.warn(
                "glibc", "无法识别 glibc 版本（非 glibc 系统？cu126 manylinux wheel 可能不可用）"
            )
        elif glibc >= MIN_GLIBC:
            c.ok("glibc", f"{glibc[0]}.{glibc[1]}（cu126 manylinux_2_28 要求 >= 2.28）")
        else:
            c.fail(
                "glibc",
                f"{glibc[0]}.{glibc[1]} 低于 2.28；cu126 torch manylinux_2_28 wheel 无法安装，"
                "请升级到 Ubuntu 22.04+ 等较新发行版",
            )

    venv = c.repo_root / ".venv"
    try:
        in_project_venv = Path(sys.prefix).resolve() == venv.resolve() and venv.is_dir()
    except OSError:
        in_project_venv = False
    if in_project_venv:
        c.ok("虚拟环境", f"{venv}（uv 管理，推荐 `uv run ...`）")
    else:
        c.warn(
            "虚拟环境",
            f"当前解释器不是项目 .venv（sys.prefix={sys.prefix}）；"
            "请先 `uv sync`，再用 `uv run python scripts/check_env.py`",
        )


def _glibc_version() -> tuple[int, int] | None:
    try:
        raw = os.confstr("CS_GNU_LIBC_VERSION")  # 形如 "glibc 2.39"
    except (AttributeError, OSError, ValueError):
        return None
    if not raw:
        return None
    match = re.search(r"(\d+)\.(\d+)", raw)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def check_uv_project(c: Checker) -> None:
    c.section("uv 项目与锁文件")

    pyproject = c.repo_root / "pyproject.toml"
    if pyproject.is_file():
        c.ok("pyproject.toml", str(pyproject.relative_to(c.repo_root)))
    else:
        c.fail("pyproject.toml", "缺失；请在仓库根目录执行本脚本（或检查 clone 是否完整）")
        return

    lock = c.repo_root / "uv.lock"
    if not lock.is_file():
        c.fail("uv.lock", "缺失；请运行 `uv lock` 生成并提交（CI 会校验 uv lock --check）")
    else:
        lock_text = ""
        try:
            lock_text = lock.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            c.warn("uv.lock", f"存在但无法读取：{exc}")
        if lock_text:
            size_mb = lock.stat().st_size / (1024 * 1024)
            c.ok("uv.lock", f"存在（{size_mb:.1f} MB）")
            if "https://download.pytorch.org/whl/cu126" in lock_text:
                c.ok(
                    "torch 来源",
                    "uv.lock 中的 torch 来自 cu126 索引（download.pytorch.org/whl/cu126）",
                )
            else:
                c.warn(
                    "torch 来源",
                    "uv.lock 未出现 cu126 索引；请检查 pyproject [tool.uv.sources]"
                    " 的 torch 配置并重跑 uv lock",
                )
            if f'version = "{EXPECTED_TORCH}+cu126"' in lock_text:
                c.ok("torch 版本", f"uv.lock 锁定 {EXPECTED_TORCH}+cu126")
            else:
                c.warn(
                    "torch 版本",
                    f"uv.lock 中未找到 {EXPECTED_TORCH}+cu126（可能已升级，请核对 ADR 0002）",
                )

    py_version_file = c.repo_root / ".python-version"
    if py_version_file.is_file():
        content = ""
        with contextlib.suppress(OSError):
            content = py_version_file.read_text(encoding="utf-8", errors="replace").strip()
        if content.startswith(f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]}"):
            c.ok(".python-version", content)
        else:
            c.warn(
                ".python-version", f"内容为 {content!r}，与 pyproject requires-python 3.12 不一致"
            )
    else:
        c.warn(".python-version", "缺失（建议保留 3.12，让 uv 自动选择解释器）")

    _check_pyproject_config(c, pyproject)

    try:
        free_gb = shutil.disk_usage(str(c.repo_root)).free / (1024**3)
    except OSError:
        free_gb = -1.0
    if free_gb < 0:
        c.skip("磁盘空间", "无法读取；训练环境安装 cu126 torch 约需 6–8 GB")
    elif free_gb < 8:
        c.warn(
            "磁盘空间",
            f"仓库所在磁盘仅剩 {free_gb:.1f} GB；`uv sync --extra train` "
            "约需 3–4 GB 下载、6–8 GB 磁盘",
        )
    else:
        c.ok("磁盘空间", f"剩余 {free_gb:.1f} GB（训练环境约需 6–8 GB）")


def _check_pyproject_config(c: Checker, pyproject: Path) -> None:
    """校验 pyproject 中的关键配置（cu126 index/sources、平台限制、CLI 入口）。"""
    try:
        import tomllib
    except ImportError:  # pragma: no cover - 仅 Python<3.11（本项目不支持）会走到
        c.skip("pyproject 配置", "当前 Python 无 tomllib，跳过 pyproject 语义检查")
        return
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        c.fail("pyproject 配置", f"TOML 解析失败：{exc}")
        return

    project = data.get("project", {})
    requires_python = project.get("requires-python", "")
    if "3.12" in requires_python:
        c.ok("requires-python", requires_python)
    else:
        c.warn("requires-python", f"{requires_python!r} 未锁定 3.12 下界")

    scripts = project.get("scripts", {})
    if scripts.get("kite-drp") == "kite_drp.cli:main":
        c.ok("CLI 入口", "kite-drp = kite_drp.cli:main")
    else:
        c.warn("CLI 入口", f"project.scripts 中的 kite-drp 为 {scripts.get('kite-drp')!r}")

    tool_uv = data.get("tool", {}).get("uv", {})
    envs = tool_uv.get("environments", [])
    if envs and any("linux" in e for e in envs) and any("win32" in e for e in envs):
        c.ok("uv environments", f"已限定 {len(envs)} 个平台（linux / win32，不含 macOS）")
    else:
        c.warn(
            "uv environments", "未同时限定 linux 与 win32；macOS 无 cu126 wheel 会导致 lock 失败"
        )

    indexes = {item.get("name"): item for item in tool_uv.get("index", [])}
    cu126 = indexes.get("pytorch-cu126", {})
    sources = tool_uv.get("sources", {}).get("torch", [])
    index_ok = cu126.get("url") == "https://download.pytorch.org/whl/cu126" and cu126.get(
        "explicit"
    )
    source_ok = any(isinstance(s, dict) and s.get("index") == "pytorch-cu126" for s in sources)
    if index_ok and source_ok:
        c.ok("cu126 index/sources", "torch 通过显式索引 pytorch-cu126 解析（explicit=true）")
    else:
        c.fail(
            "cu126 index/sources",
            "配置不完整：需要 [[tool.uv.index]] name=pytorch-cu126, url=.../cu126, explicit=true；"
            "以及 [tool.uv.sources] torch = [{ index = 'pytorch-cu126' }]",
        )


def check_core_packages(c: Checker) -> None:
    c.section("核心依赖（uv sync 默认安装）")
    missing: list[str] = []
    broken: list[str] = []
    for dist_name, import_name in CORE_PACKAGES:
        version = _version_of(dist_name)
        if version is None:
            missing.append(dist_name)
            continue
        ok, err = _import_ok(import_name)
        if ok:
            c.ok(dist_name, version)
        else:
            broken.append(dist_name)
            c.fail(dist_name, f"{version} 已安装但导入失败：{err}；请重新 `uv sync --reinstall`")
    if missing:
        c.fail(
            "核心依赖缺失",
            f"{', '.join(missing)}；请运行 `uv sync` 安装（CI 与本机应完全一致）",
        )
    if not missing and not broken:
        c.ok("核心依赖汇总", f"{len(CORE_PACKAGES)} 个核心包全部可用")


def check_optional_groups(c: Checker) -> None:
    c.section("可选依赖组（extras）")
    for group, packages in EXTRA_GROUPS.items():
        missing: list[str] = []
        present: list[str] = []
        import_errors: list[str] = []
        for dist_name, import_name in packages:
            version = _version_of(dist_name)
            if version is None:
                missing.append(dist_name)
                continue
            ok, err = _import_ok(import_name)
            if ok:
                present.append(f"{dist_name} {version}")
            else:
                present.append(dist_name)
                import_errors.append(f"{dist_name}: {err}")
        hint = f"uv sync --extra {group}"
        if not missing and not import_errors:
            c.ok(f"extra: {group}", f"已安装：{'，'.join(present)}")
        elif present and missing:
            c.warn(f"extra: {group}", f"部分缺失 {missing}；安装：`{hint}`")
        elif import_errors:
            c.warn(f"extra: {group}", f"导入失败 {import_errors}；重装：`{hint}`")
        else:
            c.skip(f"extra: {group}", f"未安装（minimal 环境属正常）；需要时：`{hint}`")


def check_torch_and_cuda(c: Checker, require_train: bool) -> None:
    c.section("PyTorch / CUDA 12.6")
    version = _version_of("torch")
    if version is None:
        if require_train:
            c.fail(
                "torch", "未安装；GPU 训练机请运行 `uv sync --extra train`（--require-train 要求）"
            )
        else:
            c.skip("torch", "未安装（CPU/数据开发正常）；GPU 训练机请 `uv sync --extra train`")
        return

    c.ok("torch 版本", version)
    if not version.startswith(EXPECTED_TORCH):
        c.warn(
            "torch 版本",
            f"已安装 {version}，pyproject 钉住 {EXPECTED_TORCH}；如需升级请同步修改 pyproject 并 "
            "`uv lock`（见 ADR 0002）",
        )

    try:
        import torch  # noqa: PLC0415 - 延迟导入，避免无 torch 时的硬依赖
    except Exception as exc:  # noqa: BLE001
        c.fail(
            "torch 导入", f"{type(exc).__name__}: {exc}；重装：`uv sync --extra train --reinstall`"
        )
        return

    ok, err = _torch_cuda_report(c, torch)
    if ok:
        c.ok("CUDA 运行时", f"torch.version.cuda = {err}（要求 {EXPECTED_CUDA}，cu126）")
    else:
        c.fail("CUDA 运行时", err)

    if torch.cuda.is_available():
        try:
            count = torch.cuda.device_count()
            names = [torch.cuda.get_device_name(i) for i in range(count)]
            caps = [torch.cuda.get_device_capability(i) for i in range(count)]
            detail = "，".join(
                f"{name}（sm_{cap[0]}{cap[1]}）" for name, cap in zip(names, caps, strict=False)
            )
            c.ok("CUDA 可用", f"{count} 张 GPU：{detail}")
        except Exception as exc:  # noqa: BLE001
            c.ok("CUDA 可用", f"True（获取设备信息失败：{type(exc).__name__}: {exc}）")
    elif require_train:
        c.fail(
            "CUDA 可用",
            "torch.cuda.is_available() = False；请检查 NVIDIA 驱动 R560+、"
            "`nvidia-smi` 与 CUDA_VISIBLE_DEVICES",
        )
    else:
        c.warn(
            "CUDA 可用",
            "torch.cuda.is_available() = False（无 GPU 或驱动不匹配；CPU 开发可忽略，"
            "训练机加 --require-train 复查）",
        )


def _torch_cuda_report(c: Checker, torch: object) -> tuple[bool, str]:
    """返回 torch 构建的 CUDA 版本检查结果。"""
    cuda_version = getattr(torch, "version", None)
    cuda_version = getattr(cuda_version, "cuda", None) if cuda_version is not None else None
    if cuda_version is None:
        return False, (
            "torch 是 CPU-only 构建（torch.version.cuda = None）；"
            "请删除环境后 `uv sync --extra train`（torch 必须来自 cu126 索引）"
        )
    if str(cuda_version) == EXPECTED_CUDA:
        return True, str(cuda_version)
    return False, (
        f"torch.version.cuda = {cuda_version}，应为 {EXPECTED_CUDA}；"
        "说明 torch 装成了别的 CUDA 版本，请运行 `uv sync --extra train --reinstall`"
    )


def check_nvidia_driver(c: Checker, require_train: bool) -> None:
    c.section("NVIDIA 驱动")
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        if require_train:
            c.fail(
                "nvidia-smi",
                "未找到 nvidia-smi；训练机需安装 NVIDIA 驱动 R560+（cu126）并加入 PATH",
            )
        else:
            c.skip(
                "nvidia-smi",
                "未找到：本机无 NVIDIA GPU 或驱动未安装（CPU 开发正常；训练机需 R560+）",
            )
        return

    rc, output = _run_cmd([nvidia_smi, "--query-gpu=name,driver_version", "--format=csv,noheader"])
    if rc != 0 or not output:
        message = f"nvidia-smi 查询失败（{output or '无输出'}）"
        if require_train:
            c.fail("NVIDIA GPU", message)
        else:
            c.warn("NVIDIA GPU", message)
        return

    gpus: list[str] = []
    driver_versions: list[str] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        gpus.append(parts[0])
        driver_versions.append(parts[1])
    c.ok("NVIDIA GPU", f"{len(gpus)} 张：{'，'.join(gpus)}")

    max_cuda = _nvidia_max_cuda_version(nvidia_smi)
    if max_cuda:
        c.ok("驱动支持的 CUDA", f"{max_cuda}（须 >= {EXPECTED_CUDA} 才能跑 cu126）")

    bad = False
    for driver in driver_versions:
        major = _parse_major(driver)
        if major is None:
            c.warn("驱动版本", f"无法解析 {driver!r}")
            continue
        if major >= RECOMMENDED_DRIVER_MAJOR:
            c.ok(
                "驱动版本", f"{driver}（满足 R{RECOMMENDED_DRIVER_MAJOR}+，对应 CUDA 12.6 GA 驱动）"
            )
        elif major >= MIN_CUDA12_DRIVER_MAJOR:
            c.warn(
                "驱动版本",
                f"{driver} 低于推荐的 R{RECOMMENDED_DRIVER_MAJOR}；cu126 可经 CUDA "
                "minor version compatibility 运行，但建议升级驱动（见 docs/environment.md）",
            )
        else:
            bad = True
            c.warn(
                "驱动版本",
                f"{driver} 低于 CUDA 12.x 最低驱动（Linux 525.60.13 / Windows 527.41），"
                "cu126 无法运行；请升级到 R560+",
            )
    if bad and require_train:
        c.fail("驱动版本", "训练机驱动过旧，无法运行 cu126，请升级 NVIDIA 驱动")


def _nvidia_max_cuda_version(nvidia_smi: str) -> str | None:
    """从 nvidia-smi 头部解析 "CUDA Version: X.Y"，即驱动支持的最高 CUDA 版本。"""
    rc, output = _run_cmd([nvidia_smi])
    if rc != 0:
        return None
    match = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)", output)
    return match.group(1) if match else None


def check_repo_layout(c: Checker) -> None:
    c.section("仓库结构与数据目录")
    required_dirs = {
        "src/kite_drp": "FAIL",
        "tests": "WARN",
        "docs": "WARN",
        "configs": "WARN",
        "scripts": "WARN",
    }
    for relative, severity in required_dirs.items():
        path = c.repo_root / relative
        if path.is_dir():
            c.ok(relative, "存在")
        elif severity == "FAIL":
            c.fail(relative, "缺失；仓库结构不符合计划书附录 B.1，请检查 clone")
        else:
            c.warn(relative, "缺失（计划书附录 B.1 目录结构的一部分）")

    data_dir = c.repo_root / "data"
    missing_dirs = [name for name in DATA_SUBDIRS if not (data_dir / name).is_dir()]
    if not data_dir.is_dir():
        c.warn("data/", "缺失；数据目录约定见计划书 §3.8（可由数据准备脚本创建）")
    elif missing_dirs:
        c.warn("data/ 子目录", f"缺少：{', '.join(missing_dirs)}（计划书 §3.8）")
    else:
        c.ok("data/ 子目录", f"{', '.join(DATA_SUBDIRS)} 均存在")

    missing_keep = [name for name in DATA_GITKEEPS if not (data_dir / name / ".gitkeep").is_file()]
    if missing_keep:
        c.warn(".gitkeep", f"data/ 下缺少占位文件：{missing_keep}（空目录无法被 git 跟踪）")
    else:
        c.ok(".gitkeep", "data/raw|interim|processed|splits 占位文件齐全")

    raw_meta = [data_dir / "raw" / "README.md", data_dir / "raw" / "CHECKSUMS.md5"]
    missing_meta = [p.name for p in raw_meta if not p.is_file()]
    if missing_meta:
        c.warn("data/raw 元数据", f"缺少 {missing_meta}（数据卡/校验和清单）")
    else:
        c.ok("data/raw 元数据", "README.md 与 CHECKSUMS.md5 存在")

    tool_files = [
        ".pre-commit-config.yaml",
        ".github/workflows/ci.yml",
        "docs/environment.md",
        "docs/adr/0002-python-environment-cuda126.md",
    ]
    missing_tools = [name for name in tool_files if not (c.repo_root / name).is_file()]
    if missing_tools:
        c.warn("工程配置", f"缺少：{', '.join(missing_tools)}")
    else:
        c.ok("工程配置", "pre-commit / CI / environment 文档 / ADR 齐全")


def check_tool_chain(c: Checker) -> None:
    c.section("工具链（uv / git / git-lfs / pre-commit）")

    uv_path = shutil.which("uv")
    if uv_path is None:
        c.warn(
            "uv",
            "未找到；请安装 https://docs.astral.sh/uv/getting-started/installation/"
            "（本脚本通常由 `uv run` 启动，一般不会走到这里）",
        )
    else:
        rc, output = _run_cmd([uv_path, "--version"])
        match = re.search(r"(\d+)\.(\d+)\.(\d+)", output or "")
        if rc == 0 and match:
            version = match.group(0)
            if (int(match.group(1)), int(match.group(2))) >= MIN_UV_VERSION:
                c.ok("uv", f"{version}（{uv_path}）")
            else:
                c.warn(
                    "uv",
                    f"{version} 偏旧；本项目按 uv >= {MIN_UV_VERSION[0]}.{MIN_UV_VERSION[1]} 验证",
                )
        else:
            c.warn("uv", f"无法获取版本：{output or rc}")

    if (c.repo_root / ".git").exists():
        rc, output = _run_cmd(["git", "--version"])
        if rc == 0:
            c.ok("git", output.splitlines()[0] if output else "可用")
        else:
            c.warn("git", output or "git 不可用")
    else:
        c.warn("git", "当前目录不是 git 仓库（.git 缺失）")

    rc, output = _run_cmd(["git", "lfs", "version"])
    if rc == 0:
        c.ok("git-lfs", output.splitlines()[0] if output else "已安装")
    else:
        c.warn(
            "git-lfs",
            "未安装：data/processed、checkpoints、deploy/packages 走 Git LFS；"
            "安装见 https://git-lfs.com（Windows: winget install GitHub.GitLFS）",
        )

    pre_commit_version = _version_of("pre-commit")
    if pre_commit_version is None:
        c.skip("pre-commit", "未安装（dev 组依赖；`uv sync` 会安装，可选 `pre-commit install`）")
    else:
        c.ok(
            "pre-commit", f"{pre_commit_version}（仓库内运行 `uv run pre-commit install` 启用钩子）"
        )


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def resolve_repo_root() -> Path:
    """优先 KITE_DRP_ROOT，其次从脚本位置向上查找 pyproject.toml + src/kite_drp。"""
    env_root = os.environ.get("KITE_DRP_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    origin = Path(__file__).resolve()
    for candidate in [origin, *origin.parents]:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "kite_drp").is_dir():
            return candidate
    return origin.parent.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="check_env.py",
        description="KITE-DRP 环境自检：Python/uv/核心依赖/CUDA 12.6/驱动/仓库结构/工具链",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="WARN 也视为失败（CI/验收用）",
    )
    parser.add_argument(
        "--require-train",
        action="store_true",
        help="要求 GPU 训练环境：torch 2.14.1+cu126、CUDA 可用、NVIDIA 驱动 R560+（缺失即 FAIL）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        # Windows 重定向到 GBK 文件时避免 UnicodeEncodeError
        with contextlib.suppress(AttributeError, OSError, ValueError):
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]

    repo_root = resolve_repo_root()
    c = Checker(repo_root)
    print("KITE-DRP 环境自检")
    print(f"仓库根目录：{repo_root}")
    print(f"Python：{sys.executable}（{platform.python_version()}）")
    print(f"平台：{platform.platform()}")

    check_python_and_platform(c)
    check_uv_project(c)
    check_core_packages(c)
    check_optional_groups(c)
    check_torch_and_cuda(c, require_train=args.require_train)
    check_nvidia_driver(c, require_train=args.require_train)
    check_repo_layout(c)
    check_tool_chain(c)

    print("\n== 汇总 ==")
    print(
        f"OK {c.counts[OK]} | WARN {c.counts[WARN]} | FAIL {c.counts[FAIL]} | SKIP {c.counts[SKIP]}"
    )
    failures = [(name, detail) for status, name, detail in c.problems if status == FAIL]
    warnings = [(name, detail) for status, name, detail in c.problems if status == WARN]
    if failures:
        print("\n必须修复（FAIL）：")
        for name, detail in failures:
            print(f"  - {name}: {detail}")
    if warnings and not failures:
        if args.strict:
            print("\n--strict 下需处理（WARN）：")
            for name, detail in warnings:
                print(f"  - {name}: {detail}")
        else:
            print(f"\n提示：{len(warnings)} 个 WARN 项（可选组件/硬件/数据目录），未阻塞自检。")

    if failures or (args.strict and warnings):
        print("\n结论：环境存在问题，请修复后重跑本脚本（说明见 docs/environment.md）。")
        return 1
    if args.require_train:
        print("\n结论：GPU 训练环境就绪（torch cu126 + CUDA + 驱动）。")
    else:
        print("\n结论：环境可用（核心 + dev）。GPU 训练机请加 --require-train 复查。")
    return 0


def entrypoint() -> int:
    """入口：任何未捕获异常都转成清晰的 FAIL 输出，绝不抛 traceback。"""
    try:
        return main()
    except KeyboardInterrupt:
        print("\n[FAIL] 自检被中断", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - 自检脚本自身不应崩溃
        print(f"\n[FAIL] 自检脚本内部错误：{type(exc).__name__}: {exc}", file=sys.stderr)
        print("请把上面的错误信息与操作系统/Python 版本反馈给环境维护者。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(entrypoint())

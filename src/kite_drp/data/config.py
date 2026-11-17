"""配置加载与路径解析（数据管线专用）。

所有预处理参数来自 ``configs/data/*.yaml``，代码中不允许写死数据口径。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from kite_drp.common.paths import REPO_ROOT

DEFAULT_CONFIG = REPO_ROOT / "configs" / "data" / "gdsc2.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并两个配置字典（override 优先）。"""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_yaml(path: str | Path) -> dict[str, Any]:
    """读取 YAML；文件不存在时返回空 dict。"""
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    if not resolved.is_file():
        return {}
    with resolved.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件顶层必须是 mapping：{resolved}")
    return data


def load_data_config(path: str | Path | None = None) -> dict[str, Any]:
    """加载数据配置，并把相对路径解析为绝对路径。

    ``path`` 为空时使用 ``configs/data/gdsc2.yaml``。配置中的 ``raw_files``
    与 ``paths`` 字段相对仓库根目录解析。
    """
    config = load_yaml(DEFAULT_CONFIG)
    if path is not None:
        config = _deep_merge(config, load_yaml(path))

    config.setdefault("name", "gdsc2")
    config.setdefault("label", {})
    config.setdefault("omics", {})
    config.setdefault("genes", {})
    config.setdefault("drug_features", {})
    config.setdefault("priors", {})
    config.setdefault("splits", {})

    label = config["label"]
    label.setdefault("compound_key", ["inchikey", "normalized_name"])
    label.setdefault("dual_id_aggregation", "mean")
    label.setdefault("min_obs_per_compound", 30)
    label.setdefault("target", "neg_z_ln_ic50")
    label.setdefault("classification_threshold", "median")

    omics = config["omics"]
    omics.setdefault("expression", {})
    omics["expression"].setdefault("drop_zero_variance", True)
    omics.setdefault("mutations", {})
    omics["mutations"].setdefault("teacher_filter", "B")
    omics["mutations"].setdefault("student_filter", "A")
    omics.setdefault("cnv", {})
    omics["cnv"].setdefault("transform", "log2")
    omics["cnv"].setdefault("clip_percentile", [0.1, 99.9])
    omics.setdefault("methylation", {"enabled": False})

    genes = config["genes"]
    genes.setdefault("universe", "protein_coding")
    genes.setdefault("mapping_alias_policy", "unique_only")

    priors = config["priors"]
    priors.setdefault("ppi", {"version": "v12.5", "cutoff": 700, "dedupe": "undirected"})
    priors["ppi"].setdefault("cutoff", 700)
    priors["ppi"].setdefault("version", "v12.5")
    priors.setdefault("pathways", {"sources": ["reactome", "go"], "min_genes": 10})
    priors["pathways"].setdefault("min_genes", 10)
    priors["pathways"].setdefault("max_genes", 500)
    priors["pathways"].setdefault("max_pathways", 1000)

    splits = config["splits"]
    splits.setdefault(
        "protocols",
        ["p0_random", "p1_lco", "p2_ldo", "p3_scaffold", "p4_lto", "lpo"],
    )
    splits.setdefault("seeds", [0, 1, 2])

    # 相对路径统一转绝对路径
    for section in ("raw_files", "local_files"):
        for key, value in list(config.get(section, {}).items()):
            if value is None:
                continue
            p = Path(str(value)).expanduser()
            if not p.is_absolute():
                p = REPO_ROOT / p
            config[section][key] = p
    return config


@dataclass(frozen=True)
class DataPaths:
    """数据目录集合（避免各模块重复拼路径）。"""

    root: Path = REPO_ROOT
    raw: Path = REPO_ROOT / "data" / "raw"
    interim: Path = REPO_ROOT / "data" / "interim"
    canonical: Path = REPO_ROOT / "data" / "interim" / "canonical"
    audit: Path = REPO_ROOT / "data" / "interim" / "audit"
    processed: Path = REPO_ROOT / "data" / "processed"
    splits: Path = REPO_ROOT / "data" / "splits"
    licenses: Path = REPO_ROOT / "data" / "licenses"
    extra: dict[str, Path] = field(default_factory=dict)

    def ensure(self) -> DataPaths:
        for path in (self.audit, self.canonical, self.processed, self.splits):
            path.mkdir(parents=True, exist_ok=True)
        return self


__all__ = ["DEFAULT_CONFIG", "DataPaths", "load_data_config", "load_yaml"]

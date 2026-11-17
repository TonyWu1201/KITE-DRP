"""质量报告生成（markdown）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from kite_drp.common.paths import REPO_ROOT
from kite_drp.data import io
from kite_drp.data.config import DataPaths


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        if pd.isna(value):
            return "NA"
        return f"{value:.4g}"
    return str(value)


def build_report(paths: DataPaths | None = None) -> Path:
    paths = paths or DataPaths()
    canonical = paths.canonical
    audit = paths.audit
    stats_path = audit / "canonical_stats.json"
    payload = io.read_json(stats_path) if stats_path.is_file() else {}
    stats = payload.get("stats", payload)
    labels = (
        io.read_parquet(canonical / "labels.parquet")
        if (canonical / "labels.parquet").is_file()
        else pd.DataFrame()
    )
    cell_lines = (
        io.read_parquet(canonical / "cell_lines.parquet")
        if (canonical / "cell_lines.parquet").is_file()
        else pd.DataFrame()
    )
    drug_cov_path = canonical / "drug_feature_coverage.json"
    drug_cov = io.read_json(drug_cov_path) if drug_cov_path.is_file() else {}
    universes_path = canonical / "gene_universes.json"
    universes = io.read_json(universes_path) if universes_path.is_file() else {}
    pool_path = paths.processed / "pool_index.json"
    pools = io.read_json(pool_path) if pool_path.is_file() else {}
    route_path = paths.processed / "student_inputs" / "route_a_meta.json"
    route = io.read_json(route_path) if route_path.is_file() else {}
    tcga_index_path = paths.processed / "tcga_inputs" / "index.json"
    tcga_index = io.read_json(tcga_index_path) if tcga_index_path.is_file() else {}

    lines: list[str] = []
    lines.append("# KITE-DRP 数据质量报告")
    lines.append("")
    lines.append(
        "> 自动生成；数字来自本次 pipeline 实测。"
        "规划口径见 `plans&notes/2026-11-17_数据预处理详细规划.md`。"
    )
    lines.append("")
    lines.append("## 1. 样本漏斗")
    lines.append("")
    lines.append("| 阶段 | 数量 |")
    lines.append("|---|---:|")
    if not cell_lines.empty:
        mapped = cell_lines["mapped"].astype(bool)
        has_ge = mapped & cell_lines["has_ge"].astype(bool)
        has_mu = mapped & cell_lines["has_mu"].astype(bool)
        has_cnv = mapped & cell_lines["has_cnv"].astype(bool)
        main = int(cell_lines["in_main_pool"].sum())
        lines.append(f"| Model.csv 全部模型 | {len(cell_lines)} |")
        lines.append(f"| GDSC2 可映射细胞系 | {int(mapped.sum())} |")
        lines.append(f"| GDSC2 ∩ GE | {int(has_ge.sum())} |")
        lines.append(f"| GDSC2 ∩ MU | {int(has_mu.sum())} |")
        lines.append(f"| GDSC2 ∩ CNV | {int(has_cnv.sum())} |")
        lines.append(f"| **M-509 主池（GE∩MU∩CNV, eligible）** | **{main}** |")
        if pools:
            lines.append(f"| M-716 教师矩阵（GE∩MU） | {len(pools.get('M-716', []))} |")
    if stats.get("labels"):
        ls = stats["labels"]
        lines.append(f"| 标签原始行 | {ls.get('raw_rows')} |")
        lines.append(f"| 化合物层标签对（全 969 株） | {ls.get('compound_level_rows')} |")
        lines.append(f"| 化合物层标签对（映射 968 株） | {ls.get('compound_level_rows_mapped')} |")
        lines.append(f"| 双 ID 合并移除行 | {ls.get('dual_id_rows_removed')} |")
        lines.append(f"| 主池标签对 | {stats.get('main_pool_labels')} |")
    lines.append("")
    lines.append("## 2. 标签质量")
    lines.append("")
    if not labels.empty:
        lines.append("| 指标 | 值 |")
        lines.append("|---|---:|")
        lines.append(f"| 细胞系 | {labels['cell_line_uid'].nunique()} |")
        lines.append(f"| 化合物 | {labels['compound_uid'].nunique()} |")
        median_labels = labels.groupby("compound_uid").size().median()
        lines.append(f"| 每化合物标签数中位 | {median_labels:.0f} |")
        min_labels = labels.groupby("compound_uid").size().min()
        lines.append(f"| 每化合物标签数最小 | {min_labels} |")
        lines.append(f"| 外推标签占比 | {labels['extrapolated'].mean():.4f} |")
        lines.append(f"| suspect 标签占比 | {labels['suspect'].mean():.4f} |")
        dual_ids = labels.loc[labels["n_source_ids"] > 1, "compound_uid"].nunique()
        lines.append(f"| 双 ID 化合物数 | {dual_ids} |")
    lines.append("")
    lines.append("## 3. 基因宇宙")
    lines.append("")
    lines.append("| 宇宙 | 维度 |")
    lines.append("|---|---:|")
    for key, value in universes.items():
        if isinstance(value, list):
            lines.append(f"| {key} | {len(value)} |")
    lines.append("")
    lines.append("## 4. 药物特征覆盖")
    lines.append("")
    if drug_cov:
        for key in ("n_compounds", "has_inchikey", "has_target", "has_smiles", "has_scaffold"):
            lines.append(f"- {key}: {drug_cov.get(key)}")
        lines.append(f"- RDKit: {drug_cov.get('rdkit_available')}")
    lines.append("")
    lines.append("## 5. 学生/TCGA 输入")
    lines.append("")
    route_name = route.get("route", "NA")
    n_pathways = route.get("n_pathways", 0)
    lines.append(f"- 学生表达路线：**{route_name}**；通路数 {n_pathways}")
    if route.get("degraded"):
        lines.append(f"- 降级原因：{route.get('reason', '')}")
    if tcga_index:
        lines.append(f"- TCGA 评估交集：{tcga_index.get('n_patients')} 例（GE+CNV+MU）")
    lines.append("")
    lines.append("## 6. naive 基线（G0-9）")
    lines.append("")
    baseline_dir = REPO_ROOT / "experiments" / "data_baselines"
    baseline_files = sorted(baseline_dir.glob("naive_*.json")) if baseline_dir.is_dir() else []
    if baseline_files:
        lines.append("| split | model | RMSE | R² | PCC | SCC | n |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for path in baseline_files:
            baseline = io.read_json(path)
            for fold, fold_result in baseline.get("folds", {}).items():
                for kind in ("per_drug", "per_cell", "global"):
                    metrics = fold_result.get(f"{kind}_raw", {})
                    lines.append(
                        f"| {path.stem.replace('naive_', '')} | fold{fold}:{kind} | "
                        f"{_fmt(metrics.get('rmse'))} | {_fmt(metrics.get('r2'))} | "
                        f"{_fmt(metrics.get('pcc'))} | {_fmt(metrics.get('scc'))} | "
                        f"{metrics.get('n')} |"
                    )
    else:
        lines.append("- 尚未运行（`kite-drp eval naive --split p1_lco_s0`）。")
    lines.append("")
    lines.append("## 7. 降级与缺口")
    lines.append("")
    warnings = payload.get("warnings", stats.get("warnings", []))
    if warnings:
        for item in warnings:
            lines.append(f"- {item}")
    else:
        lines.append("- 无")
    lines.append("")
    lines.append("## 8. 与计划书对照")
    lines.append("")
    lines.append("| 项 | 规划/审计值 | 本次实测 |")
    lines.append("|---|---|---|")
    if not cell_lines.empty:
        lines.append(f"| 主池细胞系 | 509 | {int(cell_lines['in_main_pool'].sum())} |")
    if not labels.empty:
        lines.append(f"| 化合物数 | 286 | {labels['compound_uid'].nunique()} |")
    if universes.get("teach_ge"):
        lines.append(f"| U_teach_ge | ≈19,130 | {len(universes['teach_ge'])} |")
    lines.append("")
    if universes.get("teach_ge") and len(universes["teach_ge"]) < 19000:
        lines.append(
            "> U_teach_ge 低于规划估计：CNV 的 811 个全 NaN 列中有 753 个是可解析基因列，"
            "删除后有效基因 18,613（规划按 19,144 估计）；以构建脚本 manifest 为准。"
        )
        lines.append("")
    lines.append("## 9. 复现命令")
    lines.append("")
    lines.append("```bash")
    lines.append(".venv/bin/python -m kite_drp data pipeline --profile core")
    lines.append(".venv/bin/python -m kite_drp data validate --all")
    lines.append("```")
    lines.append("")
    target = audit / "quality_report.md"
    io.atomic_write_text(target, "\n".join(lines) + "\n")
    return target


__all__ = ["build_report"]

"""产物校验、split 合规检查与泄漏扰动测试。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kite_drp.data import io, registry
from kite_drp.data.config import DataPaths
from kite_drp.data.fit import fit_stats_for_fold, load_fit_scope
from kite_drp.data.manifest import Manifest
from kite_drp.data.splits import fold_masks, load_split, validate_split


def _check(condition: bool, message: str, *, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate_canonical(paths: DataPaths) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, bool] = {}
    canonical = paths.canonical

    required = [
        "cell_lines.parquet",
        "genes.parquet",
        "compounds.parquet",
        "labels.parquet",
        "id_mapping.parquet",
        "expression.parquet",
        "cnv.parquet",
        "mutations_A.npz",
        "mutations_B.npz",
        "ppi.parquet",
        "pathways.parquet",
        "drug_features.npz",
        "gene_universes.json",
    ]
    for name in required:
        exists = (canonical / name).is_file()
        checks[f"exists:{name}"] = exists
        _check(exists, f"缺少 canonical 产物：{name}", errors=errors)
    if errors:
        return {"checks": checks, "errors": errors, "warnings": warnings}

    labels = io.read_parquet(canonical / "labels.parquet")
    cell_lines = io.read_parquet(canonical / "cell_lines.parquet")
    compounds = io.read_parquet(canonical / "compounds.parquet")
    expression = io.read_parquet(canonical / "expression.parquet")
    cnv = io.read_parquet(canonical / "cnv.parquet")
    universes = io.read_json(canonical / "gene_universes.json")

    checks["labels_unique_pair"] = not labels.duplicated(["cell_line_uid", "compound_uid"]).any()
    _check(checks["labels_unique_pair"], "labels 存在重复 (cell,compound)", errors=errors)
    checks["labels_286_compounds"] = labels["compound_uid"].nunique() == 286
    _check(checks["labels_286_compounds"], "化合物数不是 286", errors=errors)
    checks["main_pool_509"] = int(cell_lines["in_main_pool"].sum()) == 509
    if not checks["main_pool_509"]:
        warnings.append(f"M-509 实测 {int(cell_lines['in_main_pool'].sum())}（预期 509，需在报告中说明）")
    checks["expression_no_inf"] = bool(np.isfinite(expression.to_numpy()).all())
    _check(checks["expression_no_inf"], "表达矩阵含 NaN/Inf", errors=errors)
    checks["cnv_no_inf"] = bool(np.isfinite(cnv.to_numpy()).all())
    _check(checks["cnv_no_inf"], "CNV 矩阵含 NaN/Inf", errors=errors)
    checks["expression_gene_order"] = expression.columns.tolist() == sorted(expression.columns.tolist())
    checks["cnv_gene_order"] = cnv.columns.tolist() == sorted(cnv.columns.tolist())

    for name in ("teach_ge", "teach_mu_A", "teach_mu_B", "cnv", "vae", "panel"):
        if name in universes:
            checks[f"universe:{name}"] = len(universes[name]) > 0
            if not universes[name]:
                warnings.append(f"基因宇宙 {name} 为空（降级项）")

    manifest_path = canonical / "manifest.json"
    if manifest_path.is_file():
        problems = Manifest(manifest_path).verify()
        checks["canonical_manifest"] = not problems
        errors.extend(problems)
    return {
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
        "stats": {
            "cells": int(len(cell_lines)),
            "main_pool": int(cell_lines["in_main_pool"].sum()),
            "labels": int(len(labels)),
            "compounds": int(len(compounds)),
            "expression_shape": list(expression.shape),
            "cnv_shape": list(cnv.shape),
            "universes": {key: len(value) for key, value in universes.items() if isinstance(value, list)},
        },
    }


def validate_processed(paths: DataPaths) -> dict[str, Any]:
    errors: list[str] = []
    checks: dict[str, bool] = {}
    required = {
        "teacher_inputs": paths.processed / "teacher_inputs.npz",
        "teacher_index": paths.processed / "teacher_index.json",
        "pool_index": paths.processed / "pool_index.json",
        "panel_mutations": paths.processed / "student_inputs" / "panel_mutations_A.npz",
        "mut_burden": paths.processed / "student_inputs" / "mut_burden.parquet",
        "vae_target_cnv": paths.processed / "vae_target_cnv.parquet",
        "vae_target_mask": paths.processed / "vae_target_mask.parquet",
        "manifest": paths.processed / "manifest.json",
    }
    for name, path in required.items():
        exists = path.is_file()
        checks[f"exists:{name}"] = exists
        _check(exists, f"缺少 processed 产物：{path}", errors=errors)
    if errors:
        return {"checks": checks, "errors": errors, "warnings": []}

    data = io.read_npz(paths.processed / "teacher_inputs.npz")
    pool = [str(x) for x in data["cell_line_uid"]]
    checks["teacher_rows_716"] = len(pool) == 716
    _check(checks["teacher_rows_716"], f"teacher_inputs 行数 {len(pool)} != 716", errors=errors)
    checks["teacher_cnv_mask"] = int(data["cnv_observed_mask"].sum()) == 509
    _check(checks["teacher_cnv_mask"], "teacher CNV observed mask != 509", errors=errors)
    checks["finite_ge"] = bool(np.isfinite(data["ge"]).all())
    checks["finite_cnv"] = bool(np.isfinite(data["cnv"]).all())
    _check(checks["finite_ge"], "teacher GE 含 NaN/Inf", errors=errors)
    _check(checks["finite_cnv"], "teacher CNV 含 NaN/Inf（未观测应填 0 + mask）", errors=errors)
    problems = Manifest(paths.processed / "manifest.json").verify()
    checks["processed_manifest"] = not problems
    errors.extend(problems)
    return {"checks": checks, "errors": errors, "warnings": []}


def validate_splits(paths: DataPaths) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []
    checks: dict[str, bool] = {}
    labels = io.read_parquet(paths.canonical / "labels.parquet")
    cell_lines = io.read_parquet(paths.canonical / "cell_lines.parquet")
    compounds = io.read_parquet(paths.canonical / "compounds.parquet")
    split_files = sorted(paths.splits.glob("*_s*.json"))
    checks["split_files_present"] = bool(split_files)
    if not split_files:
        return {"checks": checks, "errors": ["data/splits 下没有 split 文件"], "warnings": []}
    summaries: dict[str, Any] = {}
    for path in split_files:
        split = load_split(path)
        for fold in split["folds"]:
            fold["_seed"] = split["seed"]
        report = validate_split(split, labels=labels, cell_lines=cell_lines, compounds=compounds)
        summaries[path.name] = report
        for key, value in report["checks"].items():
            if not value:
                errors.append(f"{path.name}:{key} 校验失败")
        notes.extend(f"{path.name}: {message}" for message in report["warnings"])
    checks["split_validation"] = not errors
    return {"checks": checks, "errors": errors, "warnings": warnings, "notes": notes, "summaries": summaries}


def leakage_perturbation_test(
    paths: DataPaths,
    *,
    protocols: tuple[str, ...] = ("p1_lco", "p2_ldo", "p3_scaffold", "p4_lto"),
) -> dict[str, Any]:
    """扰动测试 fold 的测试标签/组学，断言 fit_stats 逐字节不变。"""
    labels = io.read_parquet(paths.canonical / "labels.parquet")
    expression = io.read_parquet(paths.canonical / "expression.parquet")
    cnv = io.read_parquet(paths.canonical / "cnv.parquet")
    out_root = paths.audit / "leakage_test"
    results: dict[str, Any] = {}
    for protocol in protocols:
        path = paths.splits / f"{protocol}_s0.json"
        if not path.is_file():
            results[protocol] = {"status": "skipped", "reason": "split missing"}
            continue
        split = load_split(path)
        fold = dict(split["folds"][0])
        fold["_seed"] = split["seed"]
        base_dir = out_root / protocol / "base"
        perturbed_dir = out_root / protocol / "perturbed"
        base = fit_stats_for_fold(
            split=split,
            fold=fold,
            labels=labels,
            expression=expression,
            cnv=cnv,
            out_dir=base_dir,
            seed=split["seed"],
        )
        masks = fold_masks(fold, labels, protocol)
        labels_perturbed = labels.copy()
        test_index = labels.index[masks["test"]]
        if len(test_index):
            values = labels.loc[test_index, "ln_ic50"].to_numpy()
            labels_perturbed.loc[test_index, "ln_ic50"] = np.roll(values, 1)
        expression_perturbed = expression.copy()
        test_cells = sorted(labels.loc[masks["test"], "cell_line_uid"].unique().tolist())
        test_cells = [cell for cell in test_cells if cell in expression_perturbed.index]
        if test_cells:
            block = expression_perturbed.loc[test_cells].to_numpy()
            expression_perturbed.loc[test_cells] = np.roll(block, 1, axis=0)
        perturbed = fit_stats_for_fold(
            split=split,
            fold=fold,
            labels=labels_perturbed,
            expression=expression_perturbed,
            cnv=cnv,
            out_dir=perturbed_dir,
            seed=split["seed"],
        )
        unchanged = base["fit_scope"]["fingerprint"] == perturbed["fit_scope"]["fingerprint"]
        results[protocol] = {
            "status": "ok" if unchanged else "failed",
            "base_fingerprint": base["fit_scope"]["fingerprint"],
            "perturbed_fingerprint": perturbed["fit_scope"]["fingerprint"],
        }
    return results


def validate_all(
    *,
    paths: DataPaths | None = None,
    strict: bool = False,
    run_leakage: bool = True,
) -> dict[str, Any]:
    paths = paths or DataPaths()
    canonical = validate_canonical(paths)
    processed = validate_processed(paths)
    splits = validate_splits(paths)
    leakage = (
        leakage_perturbation_test(paths)
        if run_leakage and (paths.canonical / "labels.parquet").is_file()
        else {"status": "skipped"}
    )
    errors = [*canonical["errors"], *processed["errors"], *splits["errors"]]
    warnings = [*canonical.get("warnings", [])]
    notes = [*splits.get("notes", [])]
    expected_markers = (
        "SMILES",
        "RDKit",
        "Reactome/GO",
        "TCGA file",
        "外部突变 panel",
        "M-509",
        "P3 scaffold",
    )
    degradations = [
        message for message in warnings if any(marker in message for marker in expected_markers)
    ]
    warnings = [message for message in warnings if message not in degradations]
    for protocol, result in leakage.items():
        if isinstance(result, dict) and result.get("status") == "failed":
            errors.append(f"泄漏扰动测试失败：{protocol}")
    if strict and warnings:
        errors.extend(f"strict warning: {message}" for message in warnings)
    return {
        "status": "ok" if not errors else "failed",
        "checks": {
            "canonical": canonical["checks"],
            "processed": processed["checks"],
            "splits": splits["checks"],
        },
        "errors": errors,
        "warnings": warnings,
        "degradations": degradations,
        "notes": notes,
        "stats": canonical.get("stats", {}),
        "leakage": leakage,
    }


__all__ = [
    "leakage_perturbation_test",
    "validate_all",
    "validate_canonical",
    "validate_processed",
    "validate_splits",
]

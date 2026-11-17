"""L2 model-input 装配：teacher/student/VAE 产物。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kite_drp.data import io, registry
from kite_drp.data.manifest import Manifest, code_hash, config_hash


def _load_universes(canonical: Path) -> dict[str, Any]:
    payload = io.read_json(canonical / "gene_universes.json")
    return payload


def _main_pool(cell_lines: pd.DataFrame) -> list[str]:
    return sorted(cell_lines.loc[cell_lines["in_main_pool"], "cell_line_uid"].tolist())


def build_teacher_inputs(canonical_dir: str | Path, processed_dir: str | Path) -> dict[str, Any]:
    """M-716 教师输入 + observed masks（v1.1；M-509 为其子集）。"""
    canonical = Path(canonical_dir)
    processed = Path(processed_dir)
    processed.mkdir(parents=True, exist_ok=True)
    cell_lines = io.read_parquet(canonical / "cell_lines.parquet")
    expression = io.read_parquet(canonical / "expression.parquet")
    cnv = io.read_parquet(canonical / "cnv.parquet")
    mu_a, mu_models, mu_genes = io.load_packed_matrix(canonical / "mutations_A.npz", name="matrix")
    mu_b, _, _ = io.load_packed_matrix(canonical / "mutations_B.npz", name="matrix")

    def select_pool(required: list[str], *, eligible_only: bool = False) -> list[str]:
        mask = cell_lines["mapped"].copy()
        if eligible_only:
            mask &= cell_lines["eligible"]
        for column in required:
            mask &= cell_lines[column]
        return sorted(cell_lines.loc[mask, "cell_line_uid"].tolist())

    pool_716 = select_pool(["has_ge", "has_mu"])
    pool_509 = select_pool(["has_ge", "has_mu", "has_cnv"], eligible_only=True)
    pool = pool_716
    ge = expression.loc[pool].to_numpy(dtype=np.float32)

    cnv_values = np.zeros((len(pool), cnv.shape[1]), dtype=np.float32)
    cnv_mask = np.zeros(len(pool), dtype=bool)
    cnv_index = {cell: i for i, cell in enumerate(cnv.index)}
    for row, cell in enumerate(pool):
        if cell in cnv_index:
            cnv_values[row] = cnv.iloc[cnv_index[cell]].to_numpy(dtype=np.float32)
            cnv_mask[row] = True

    mu_index = {cell: i for i, cell in enumerate(mu_models)}
    mu_rows = [mu_index[cell] for cell in pool]
    mu_a_values = mu_a[mu_rows]
    mu_b_values = mu_b[mu_rows]

    io.write_npz(
        processed / "teacher_inputs.npz",
        {
            "ge": ge,
            "cnv": cnv_values,
            "mu_A": mu_a_values,
            "mu_B": mu_b_values,
            "ge_observed_mask": np.ones(len(pool), dtype=bool),
            "cnv_observed_mask": cnv_mask,
            "mu_observed_mask": np.ones(len(pool), dtype=bool),
            "eligible_mask": cell_lines.set_index("cell_line_uid").loc[pool, "eligible"].to_numpy(dtype=bool),
            "cell_line_uid": np.asarray(pool, dtype=np.str_),
            "ge_genes": np.asarray(expression.columns.tolist(), dtype=np.str_),
            "cnv_genes": np.asarray(cnv.columns.tolist(), dtype=np.str_),
            "mu_A_genes": np.asarray(mu_genes, dtype=np.str_),
        },
    )
    index_payload = {
        "rows": pool,
        "ge_genes": expression.columns.tolist(),
        "cnv_genes": cnv.columns.tolist(),
        "mu_A_genes": mu_genes,
        "teacher_mu_filter": "B",
        "student_mu_filter": "A",
        "shapes": {
            "ge": list(ge.shape),
            "cnv": list(cnv_values.shape),
            "mu_A": list(mu_a_values.shape),
            "mu_B": list(mu_b_values.shape),
        },
        "masks": {
            "ge_observed": "all True (M-716 requires GE)",
            "cnv_observed": int(cnv_mask.sum()),
            "mu_observed": "all True (M-716 requires MU)",
        },
        "scale": {
            "ge": "raw log2(TPM+1)",
            "cnv": "log2 relative CN (0 where unobserved; use cnv_observed_mask)",
            "mu": "0/1",
        },
    }
    io.write_json(processed / "teacher_index.json", index_payload)
    pool_index = {
        "M-716": pool_716,
        "M-509": pool_509,
        "description": {
            "M-716": "GE+MU available, CNV missing for some cells (teacher matrix + observed mask)",
            "M-509": "GE+MU+CNV all available (primary evaluation subset)",
        },
    }
    io.write_json(processed / "pool_index.json", pool_index)
    return {
        "n_cells": len(pool),
        "n_cells_cnv_observed": int(cnv_mask.sum()),
        "n_ge_genes": int(ge.shape[1]),
        "n_cnv_genes": int(cnv_values.shape[1]),
        "n_mu_genes": int(mu_a_values.shape[1]),
    }


def build_student_inputs(canonical_dir: str | Path, processed_dir: str | Path) -> dict[str, Any]:
    canonical = Path(canonical_dir)
    processed = Path(processed_dir)
    out = processed / "student_inputs"
    out.mkdir(parents=True, exist_ok=True)
    cell_lines = io.read_parquet(canonical / "cell_lines.parquet")
    pool_mask = cell_lines["mapped"] & cell_lines["has_ge"] & cell_lines["has_mu"]
    pool = sorted(cell_lines.loc[pool_mask, "cell_line_uid"].tolist())
    universes = _load_universes(canonical)

    mu, mu_models, mu_genes = io.load_packed_matrix(canonical / "mutations_A.npz", name="matrix")
    model_index = {cell: i for i, cell in enumerate(mu_models)}
    rows = [model_index[cell] for cell in pool]
    panel_genes = [gene for gene in universes.get("panel", []) if gene in set(mu_genes)]
    gene_index = {gene: i for i, gene in enumerate(mu_genes)}
    cols = [gene_index[gene] for gene in panel_genes]
    panel = mu[np.ix_(rows, cols)]
    io.save_packed_matrix(out / "panel_mutations_A.npz", panel, pool, panel_genes)
    burden = io.read_parquet(canonical / "mut_burden.parquet").set_index("cell_line_uid")
    burden_pool = burden.reindex(pool).reset_index()
    io.write_parquet(burden_pool, out / "mut_burden.parquet")

    # 通路权重：只依赖外部通路库（无标签），可全局冻结；缺失时显式降级
    pathways = io.read_parquet(canonical / "pathways.parquet")
    ge_genes = universes.get("teach_ge", [])
    gene_index_ge = {gene: i for i, gene in enumerate(ge_genes)}
    if not pathways.empty:
        pathway_ids = sorted(pathways["pathway_id"].unique().tolist())
        weights = np.zeros((len(pathway_ids), len(ge_genes)), dtype=np.float16)
        path_index = {pathway: i for i, pathway in enumerate(pathway_ids)}
        for row in pathways.itertuples(index=False):
            col = gene_index_ge.get(row.gene)
            if col is not None:
                weights[path_index[row.pathway_id], col] = 1.0
        # 每通路行归一化（mean-z 打分：权重 1/n_genes）
        counts = weights.sum(axis=1, keepdims=True)
        weights = np.divide(weights, counts, out=np.zeros_like(weights), where=counts > 0)
        io.write_npz(
            out / "pathway_weights.npz",
            {
                "weights": weights,
                "pathway_ids": np.asarray(pathway_ids, dtype=np.str_),
                "genes": np.asarray(ge_genes, dtype=np.str_),
                "n_genes_per_pathway": counts.ravel().astype(np.int32),
            },
        )
        route = "A"
        meta = {
            "route": "A",
            "n_pathways": len(pathway_ids),
            "n_genes": len(ge_genes),
            "scoring": "mean z over pathway genes (weights normalized)",
        }
    else:
        io.write_npz(
            out / "pathway_weights.npz",
            {
                "weights": np.zeros((0, len(ge_genes)), dtype=np.float16),
                "pathway_ids": np.asarray([], dtype=np.str_),
                "genes": np.asarray(ge_genes, dtype=np.str_),
                "n_genes_per_pathway": np.asarray([], dtype=np.int32),
            },
        )
        route = "B"
        meta = {
            "route": "B",
            "degraded": True,
            "reason": "Reactome/GO GMT missing",
            "fallback": "HVG top-2000 per training fold (see fit_stats hvg.json)",
        }
    io.write_json(out / "route_a_meta.json", meta)
    return {
        "n_panel_genes": len(panel_genes),
        "n_cells": len(pool),
        "route": route,
        "panel_source": universes.get("panel_source"),
    }


def build_vae_target(canonical_dir: str | Path, processed_dir: str | Path) -> dict[str, Any]:
    """M-716 × U_vae 的目标 CNV；207 株无 CNV 以 NaN + mask 表示（v1.1）。"""
    canonical = Path(canonical_dir)
    processed = Path(processed_dir)
    processed.mkdir(parents=True, exist_ok=True)
    cell_lines = io.read_parquet(canonical / "cell_lines.parquet")
    cnv = io.read_parquet(canonical / "cnv.parquet")
    universes = _load_universes(canonical)
    genes = [gene for gene in universes.get("vae", []) if gene in cnv.columns]
    pool_mask = cell_lines["mapped"] & cell_lines["has_ge"] & cell_lines["has_mu"]
    pool = sorted(cell_lines.loc[pool_mask, "cell_line_uid"].tolist())
    target = cnv.reindex(index=pool, columns=genes)
    observed = target.notna()
    target = target.astype(np.float32)
    io.write_parquet(target, processed / "vae_target_cnv.parquet", index=True)
    io.write_parquet(observed, processed / "vae_target_mask.parquet", index=True)
    io.write_json(
        processed / "vae_target_genes.json",
        {
            "genes": genes,
            "n_genes": len(genes),
            "n_cells": len(pool),
            "n_observed": int(observed.to_numpy().any(axis=1).sum()),
            "source": "STRING degree ∩ CNV ∩ U_teach_ge; rows=M-716",
        },
    )
    return {
        "n_cells": len(pool),
        "n_observed": int(observed.to_numpy().any(axis=1).sum()),
        "n_genes": len(genes),
    }


def build_tcga_inputs(canonical_dir: str | Path, processed_dir: str | Path) -> dict[str, Any] | None:
    """装配 TCGA 评估输入（purpose=eval）；缺少 file→case 映射时跳过。"""
    canonical = Path(canonical_dir)
    ge_path = canonical / "tcga_ge.npz"
    cnv_path = canonical / "tcga_cnv.npz"
    mu_path = canonical / "tcga_mu.npz"
    if not (ge_path.is_file() and cnv_path.is_file() and mu_path.is_file()):
        return None
    out = Path(processed_dir) / "tcga_inputs"
    out.mkdir(parents=True, exist_ok=True)
    ge = io.read_npz(ge_path)
    cnv = io.read_npz(cnv_path)
    mu, mu_patients, mu_genes = io.load_packed_matrix(mu_path, name="matrix")
    patients = sorted(
        set(map(str, ge["patient_ids"]))
        & set(map(str, cnv["patient_ids"]))
        & set(mu_patients)
    )
    ge_index = {str(patient): i for i, patient in enumerate(ge["patient_ids"])}
    cnv_index = {str(patient): i for i, patient in enumerate(cnv["patient_ids"])}
    mu_index = {patient: i for i, patient in enumerate(mu_patients)}
    io.write_npz(
        out / "tcga_inputs.npz",
        {
            "patients": np.asarray(patients, dtype=np.str_),
            "ge": ge["values"][[ge_index[p] for p in patients]],
            "cnv": cnv["values"][[cnv_index[p] for p in patients]],
            "mu": mu[[mu_index[p] for p in patients]],
            "ge_genes": ge["genes"],
            "cnv_genes": cnv["genes"],
            "mu_genes": np.asarray(mu_genes, dtype=np.str_),
        },
    )
    for source, target in [
        ("tcga_clinical.parquet", "clinical.parquet"),
        ("tcga_treatments.parquet", "treatments.parquet"),
        ("tcga_cdr_cesc.parquet", "cdr.parquet"),
        ("tcga_ding2016.parquet", "ding2016.parquet"),
    ]:
        if (canonical / source).is_file():
            io.atomic_copy(canonical / source, out / target)
    io.write_json(
        out / "index.json",
        {
            "purpose": "eval",
            "patients": patients,
            "n_patients": len(patients),
            "shapes": {
                "ge": list(ge["values"].shape),
                "cnv": list(cnv["values"].shape),
                "mu": list(mu.shape),
            },
            "red_line": "训练侧禁止加载；仅评估脚本可读",
        },
    )
    return {"n_patients": len(patients), "n_ge_genes": int(ge["values"].shape[1]), "n_cnv_genes": int(cnv["values"].shape[1])}


def build_processed(
    *,
    canonical_dir: str | Path,
    processed_dir: str | Path,
    config: dict[str, Any] | None = None,
    stage: str = "all",
) -> dict[str, Any]:
    canonical = Path(canonical_dir)
    processed = Path(processed_dir)
    processed.mkdir(parents=True, exist_ok=True)
    from kite_drp.data.drug_assembly import build_drug_inputs

    result: dict[str, Any] = {"teacher": None, "drug": None, "student": None, "vae": None, "tcga": None}
    if stage in {"all", "teacher", "drug"}:
        result["drug"] = build_drug_inputs(canonical, processed)
    if stage in {"all", "teacher"}:
        result["teacher"] = build_teacher_inputs(canonical, processed)
    if stage in {"all", "student"}:
        result["student"] = build_student_inputs(canonical, processed)
    if stage in {"all", "vae"}:
        result["vae"] = build_vae_target(canonical, processed)
    if stage in {"all", "tcga"}:
        result["tcga"] = build_tcga_inputs(canonical, processed)

    manifest = Manifest(processed / "manifest.json")
    manifest.artifacts = {}
    for name in [
        "teacher_inputs",
        "teacher_index",
        "pool_index",
        "drug_features_processed",
        "compounds_processed",
        "panel_mutations",
        "mut_burden",
        "pathway_weights",
        "route_a_meta",
        "vae_target_cnv",
        "vae_target_mask",
        "vae_target_genes",
        "tcga_inputs",
    ]:
        spec = registry.ARTIFACTS[name]
        if spec.path.is_file():
            manifest.register(name, spec.path, layer="processed")
    manifest.save(
        code=code_hash(),
        config=config_hash(config or {}),
        extra={"degraded": [], "notes": ["processed artifacts are LFS-tracked; see docs/data_cards"]},
    )
    result["manifest"] = sorted(manifest.artifacts)
    return result


__all__ = [
    "build_processed",
    "build_student_inputs",
    "build_teacher_inputs",
    "build_tcga_inputs",
    "build_vae_target",
]

"""数据产物登记表：名称 → 路径 / 层 / schema / 生产者与消费者。

代码与文档引用产物时应使用 :func:`resolve`，避免路径散落。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kite_drp.common.paths import REPO_ROOT


@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    relpath: str
    layer: str  # audit | canonical | split | processed
    producer: str
    consumers: tuple[str, ...] = ()
    description: str = ""

    @property
    def path(self) -> Path:
        return REPO_ROOT / self.relpath


ARTIFACTS: dict[str, ArtifactSpec] = {
    # L0
    "inventory": ArtifactSpec(
        "inventory",
        "data/interim/audit/inventory.json",
        "audit",
        "data.inventory",
        description="原始数据清单/MD5",
    ),
    "quality_report": ArtifactSpec(
        "quality_report", "data/interim/audit/quality_report.md", "audit", "data.report"
    ),
    "id_conflicts": ArtifactSpec(
        "id_conflicts", "data/interim/canonical/id_conflicts.json", "canonical", "data.ids"
    ),
    "gene_universes": ArtifactSpec(
        "gene_universes", "data/interim/canonical/gene_universes.json", "canonical", "data.genes"
    ),
    # L1 canonical
    "cell_lines": ArtifactSpec(
        "cell_lines", "data/interim/canonical/cell_lines.parquet", "canonical", "data.ids"
    ),
    "genes": ArtifactSpec(
        "genes", "data/interim/canonical/genes.parquet", "canonical", "data.genes"
    ),
    "compounds": ArtifactSpec(
        "compounds", "data/interim/canonical/compounds.parquet", "canonical", "data.ids"
    ),
    "id_mapping": ArtifactSpec(
        "id_mapping", "data/interim/canonical/id_mapping.parquet", "canonical", "data.ids"
    ),
    "labels": ArtifactSpec(
        "labels", "data/interim/canonical/labels.parquet", "canonical", "data.labels"
    ),
    "expression": ArtifactSpec(
        "expression",
        "data/interim/canonical/expression.parquet",
        "canonical",
        "data.omics.expression",
    ),
    "mutations_A": ArtifactSpec(
        "mutations_A", "data/interim/canonical/mutations_A.npz", "canonical", "data.omics.mutations"
    ),
    "mutations_B": ArtifactSpec(
        "mutations_B", "data/interim/canonical/mutations_B.npz", "canonical", "data.omics.mutations"
    ),
    "mutation_long": ArtifactSpec(
        "mutation_long",
        "data/interim/canonical/mutation_long.parquet",
        "canonical",
        "data.omics.mutations",
    ),
    "cnv": ArtifactSpec("cnv", "data/interim/canonical/cnv.parquet", "canonical", "data.omics.cnv"),
    "methylation_rrbs": ArtifactSpec(
        "methylation_rrbs",
        "data/interim/canonical/methylation_rrbs.parquet",
        "canonical",
        "data.omics.methylation",
    ),
    "drug_features": ArtifactSpec(
        "drug_features", "data/interim/canonical/drug_features.npz", "canonical", "data.drugs"
    ),
    "drug_coverage": ArtifactSpec(
        "drug_coverage",
        "data/interim/canonical/drug_feature_coverage.json",
        "canonical",
        "data.drugs",
    ),
    "ppi": ArtifactSpec("ppi", "data/interim/canonical/ppi.parquet", "canonical", "data.priors"),
    "dti": ArtifactSpec("dti", "data/interim/canonical/dti.parquet", "canonical", "data.priors"),
    "pathways": ArtifactSpec(
        "pathways", "data/interim/canonical/pathways.parquet", "canonical", "data.priors"
    ),
    "tcga_file_index": ArtifactSpec(
        "tcga_file_index", "data/interim/tcga/file_index.parquet", "canonical", "data.tcga"
    ),
    "tcga_mu": ArtifactSpec(
        "tcga_mu", "data/interim/canonical/tcga_mu.npz", "canonical", "data.tcga"
    ),
    "tcga_ge": ArtifactSpec(
        "tcga_ge", "data/interim/canonical/tcga_ge.npz", "canonical", "data.tcga"
    ),
    "tcga_cnv": ArtifactSpec(
        "tcga_cnv", "data/interim/canonical/tcga_cnv.npz", "canonical", "data.tcga"
    ),
    "tcga_meth_stats": ArtifactSpec(
        "tcga_meth_stats",
        "data/interim/canonical/tcga_meth_stats.parquet",
        "canonical",
        "data.tcga",
    ),
    "tcga_clinical": ArtifactSpec(
        "tcga_clinical", "data/interim/canonical/tcga_clinical.parquet", "canonical", "data.tcga"
    ),
    "tcga_treatments": ArtifactSpec(
        "tcga_treatments",
        "data/interim/canonical/tcga_treatments.parquet",
        "canonical",
        "data.tcga",
    ),
    "tcga_cdr_cesc": ArtifactSpec(
        "tcga_cdr_cesc", "data/interim/canonical/tcga_cdr_cesc.parquet", "canonical", "data.tcga"
    ),
    "tcga_ding2016": ArtifactSpec(
        "tcga_ding2016", "data/interim/canonical/tcga_ding2016.parquet", "canonical", "data.tcga"
    ),
    # L3 splits
    "split_manifest": ArtifactSpec(
        "split_manifest", "data/splits/split_manifest.json", "split", "data.splits"
    ),
    "external_validation_manifest": ArtifactSpec(
        "external_validation_manifest",
        "data/splits/external_validation_manifest.json",
        "split",
        "data.external",
    ),
    "external_gdsc1": ArtifactSpec(
        "external_gdsc1", "data/interim/external/gdsc1_labels.parquet", "external", "data.external"
    ),
    "external_ctrp": ArtifactSpec(
        "external_ctrp", "data/interim/external/ctrpv2_auc.parquet", "external", "data.external"
    ),
    "external_prism": ArtifactSpec(
        "external_prism", "data/interim/external/prism_auc.parquet", "external", "data.external"
    ),
    "external_prism_lfc": ArtifactSpec(
        "external_prism_lfc",
        "data/interim/external/prism_lfc.parquet",
        "external",
        "data.external",
    ),
    # L2 processed
    "teacher_inputs": ArtifactSpec(
        "teacher_inputs", "data/processed/teacher_inputs.npz", "processed", "data.assembly"
    ),
    "drug_features_processed": ArtifactSpec(
        "drug_features_processed", "data/processed/drug_features.npz", "processed", "data.assembly"
    ),
    "compounds_processed": ArtifactSpec(
        "compounds_processed", "data/processed/compounds.parquet", "processed", "data.assembly"
    ),
    "teacher_index": ArtifactSpec(
        "teacher_index", "data/processed/teacher_index.json", "processed", "data.assembly"
    ),
    "pool_index": ArtifactSpec(
        "pool_index", "data/processed/pool_index.json", "processed", "data.assembly"
    ),
    "panel_mutations": ArtifactSpec(
        "panel_mutations",
        "data/processed/student_inputs/panel_mutations_A.npz",
        "processed",
        "data.assembly",
    ),
    "mut_burden": ArtifactSpec(
        "mut_burden",
        "data/processed/student_inputs/mut_burden.parquet",
        "processed",
        "data.assembly",
    ),
    "pathway_weights": ArtifactSpec(
        "pathway_weights",
        "data/processed/student_inputs/pathway_weights.npz",
        "processed",
        "data.assembly",
    ),
    "route_a_meta": ArtifactSpec(
        "route_a_meta",
        "data/processed/student_inputs/route_a_meta.json",
        "processed",
        "data.assembly",
    ),
    "vae_target_cnv": ArtifactSpec(
        "vae_target_cnv", "data/processed/vae_target_cnv.parquet", "processed", "data.assembly"
    ),
    "vae_target_genes": ArtifactSpec(
        "vae_target_genes", "data/processed/vae_target_genes.json", "processed", "data.assembly"
    ),
    "vae_target_mask": ArtifactSpec(
        "vae_target_mask", "data/processed/vae_target_mask.parquet", "processed", "data.assembly"
    ),
    "tcga_inputs": ArtifactSpec(
        "tcga_inputs", "data/processed/tcga_inputs/tcga_inputs.npz", "processed", "data.assembly"
    ),
    "processed_manifest": ArtifactSpec(
        "processed_manifest", "data/processed/manifest.json", "processed", "data.assembly"
    ),
}


def resolve(name: str) -> Path:
    if name not in ARTIFACTS:
        raise KeyError(f"未知产物：{name}（可用：{', '.join(sorted(ARTIFACTS))}）")
    return ARTIFACTS[name].path


def by_layer(layer: str) -> list[ArtifactSpec]:
    return [spec for spec in ARTIFACTS.values() if spec.layer == layer]


__all__ = ["ARTIFACTS", "ArtifactSpec", "by_layer", "resolve"]

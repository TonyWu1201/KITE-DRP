"""L1 canonical 构建：ID、标签、组学、药物特征、先验与 TCGA。"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kite_drp.common.paths import REPO_ROOT
from kite_drp.data import ids, io, registry
from kite_drp.data import tcga as tcga_module
from kite_drp.data.config import DataPaths, load_data_config
from kite_drp.data.drugs import build_drug_features
from kite_drp.data.genes import (
    GeneMaster,
    build_gene_artifact,
    build_gene_master,
    build_universes,
    resolve_columns,
)
from kite_drp.data.labels import build_labels
from kite_drp.data.manifest import code_hash, config_hash, input_hashes
from kite_drp.data.omics import cnv as cnv_module
from kite_drp.data.omics import expression as expression_module
from kite_drp.data.omics import methylation as methylation_module
from kite_drp.data.omics import mutations as mutations_module
from kite_drp.data.priors import build_dti, build_pathways, build_ppi, find_gmt_files


def _raw_path(config: dict[str, Any], key: str, default: str) -> Path:
    value = config.get("raw_files", {}).get(key)
    return Path(value) if value else REPO_ROOT / default


def _raw_labels_minimal(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        usecols=["CELL_LINE_NAME", "SANGER_MODEL_ID", "CANCER_TYPE", "DRUG_ID", "DRUG_NAME"],
        dtype=str,
    )


def _full_labels_raw(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"SANGER_MODEL_ID": str, "DRUG_ID": str})


def _panel_from_file(path: Path, master: GeneMaster) -> list[str]:
    if not path.is_file():
        return []
    genes: list[str] = []
    if path.suffix.lower() in {".tsv", ".csv"}:
        separator = "\t" if path.suffix.lower() == ".tsv" else ","
        frame = pd.read_csv(path, sep=separator, header=None, dtype=str)
        values = frame.to_numpy().ravel().tolist()
    else:
        values = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for value in values:
        symbol, _ = master.resolve(value)
        if symbol:
            genes.append(symbol)
    return sorted(set(genes))


def _frequency_panel(
    mutations: mutations_module.MutationData,
    *,
    min_frequency: float = 0.01,
    max_genes: int = 500,
) -> list[str]:
    if mutations.matrix_A.size == 0:
        return []
    counts = mutations.matrix_A.sum(axis=0)
    frequency = counts / float(mutations.matrix_A.shape[0])
    order = np.argsort(-frequency)
    selected = [
        mutations.genes_A[index]
        for index in order
        if frequency[index] >= min_frequency
    ]
    return sorted(selected[:max_genes])


def _pathway_sources(config: dict[str, Any]) -> list[Path]:
    files: list[Path] = []
    configured = config.get("priors", {}).get("pathways", {}).get("files", [])
    for value in configured:
        path = Path(value)
        if not path.is_absolute():
            path = REPO_ROOT / path
        files.extend(find_gmt_files(path))
    for base in [
        REPO_ROOT / "data" / "raw" / "pathways",
        REPO_ROOT / "data" / "raw" / "reactome",
        REPO_ROOT / "data" / "raw" / "go",
    ]:
        if base.exists():
            files.extend(find_gmt_files(base))
    # 默认只用 GO 全量注释；experimental GMT 留给消融
    if any(path.name == "goa_human.gmt" for path in files):
        files = [path for path in files if path.name != "goa_human_experimental.gmt"]
    return sorted(set(files))


def _gene_mapping_frame(
    raw_columns: Iterable[str], master: GeneMaster, *, source_table: str
) -> pd.DataFrame:
    mapping = resolve_columns(raw_columns, master)
    frame = pd.DataFrame(
        {
            "source_table": source_table,
            "source_id": mapping["raw"],
            "source_name": mapping["raw"],
            "target_type": "gene",
            "target_id": mapping["hgnc_symbol"],
            "method": mapping["method"],
            "status": np.where(mapping["hgnc_symbol"].notna(), "ok", "failed"),
            "evidence": np.where(mapping["par_y"], "par_y", ""),
        }
    )
    return frame


def build_canonical(
    *,
    config_path: str | Path | None = None,
    only: str | None = None,
    force: bool = False,
    fetch_file_index: bool = False,
) -> dict[str, Any]:
    config = load_data_config(config_path)
    paths = DataPaths().ensure()
    code = code_hash()
    cfg_hash = config_hash(config)
    raw_paths = {
        key: str(value)
        for key, value in config.get("raw_files", {}).items()
        if isinstance(value, Path)
    }
    inputs = input_hashes(raw_paths)
    result: dict[str, Any] = {"status": "ok", "warnings": [], "artifacts": [], "stats": {}}

    if only in {"tcga"}:
        return _build_tcga_only(config, paths, result, fetch_file_index=fetch_file_index)
    if only in {"drug_features"}:
        return _build_drug_features_only(config, paths, result)

    # ------------------------------------------------------------------
    # 1. 基因真源与三个组学矩阵
    # ------------------------------------------------------------------
    hgnc_path = _raw_path(config, "hgnc", "data/raw/hgnc/Gene.csv")
    model_path = _raw_path(config, "depmap_model", "data/raw/depmap/Model.csv")
    expr_path = _raw_path(
        config,
        "depmap_expression",
        "data/raw/depmap/DepMap_ExpressionTPMLogp1HumanProteinCodingGenes.csv",
    )
    cnv_path = _raw_path(config, "depmap_cnv", "data/raw/depmap/DepMap_CNGeneWGS.csv")
    mu_path = _raw_path(config, "depmap_mutations", "data/raw/depmap/DepMap_SomaticMutations.csv")
    labels_path = _raw_path(
        config, "dose_response", "data/raw/gdsc/GDSC2_fitted_dose_response_27Oct23.csv"
    )
    compounds_path = _raw_path(
        config, "compounds", "data/raw/gdsc/screened_compounds_rel_8.5_inchi.csv"
    )

    for key, path in {
        "hgnc": hgnc_path,
        "model": model_path,
        "expression": expr_path,
        "cnv": cnv_path,
        "mutations": mu_path,
        "labels": labels_path,
        "compounds": compounds_path,
    }.items():
        if not Path(path).is_file():
            raise FileNotFoundError(f"缺少原始文件（{key}）：{path}")

    master = build_gene_master(hgnc_path)
    result["stats"]["hgnc_rows"] = int(len(master.frame))

    expression = expression_module.load_expression(
        expr_path,
        master,
        drop_zero_variance=bool(config["omics"]["expression"].get("drop_zero_variance", True)),
        universe=None,
    )
    result["stats"]["expression"] = expression.info

    cnv = cnv_module.load_cnv(
        cnv_path,
        master,
        clip_percentile=tuple(config["omics"]["cnv"].get("clip_percentile", [0.1, 99.9])),
        universe=None,
    )
    result["stats"]["cnv"] = cnv.info

    mutations = mutations_module.load_mutations(mu_path)
    result["stats"]["mutations"] = mutations.info

    # ------------------------------------------------------------------
    # 2. 先验（PPI / 通路）
    # ------------------------------------------------------------------
    string_dir = REPO_ROOT / "data" / "raw" / "string"
    links_path = _raw_path(
        config, "string_links", str(string_dir / "9606.protein.links.v12.5.txt.gz")
    )
    aliases_path = _raw_path(
        config, "string_aliases", str(string_dir / "9606.protein.aliases.v12.5.txt.gz")
    )
    ppi = None
    if Path(links_path).is_file() and Path(aliases_path).is_file():
        ppi = build_ppi(
            links_path,
            aliases_path,
            cutoff=int(config["priors"]["ppi"].get("cutoff", 700)),
            universe=None,
        )
        result["stats"]["ppi"] = ppi.info
    else:
        result["warnings"].append(f"STRING 文件缺失，PPI 降级为空：{links_path}")

    pathway_files = _pathway_sources(config)
    pathway_genes: list[str] = []
    pathways = pd.DataFrame(columns=["pathway_id", "source", "gene", "n_genes"])
    if pathway_files:
        pathways, pathway_ids = build_pathways(
            pathway_files,
            master,
            universe=set(expression.frame.columns),
            min_genes=int(config["priors"]["pathways"].get("min_genes", 10)),
            max_genes=int(config["priors"]["pathways"].get("max_genes", 500)),
            max_pathways=int(config["priors"]["pathways"].get("max_pathways", 1000)),
        )
        pathway_genes = sorted(pathways["gene"].unique().tolist()) if not pathways.empty else []
        result["stats"]["pathways"] = {
            "files": [str(path) for path in pathway_files],
            "n_pathways": len(pathway_ids),
            "n_genes": len(pathway_genes),
        }
    else:
        result["warnings"].append(
            "Reactome/GO GMT 未找到：学生通路分数（路线 A）降级；U_pathway 为空。"
        )
        result["stats"]["pathways"] = {"files": [], "n_pathways": 0, "n_genes": 0}

    # Panel：外部清单优先，否则频率定义（仅用无标签的突变携带率）
    panel_path = config.get("raw_files", {}).get("mutation_panel")
    panel_genes = _panel_from_file(Path(panel_path), master) if panel_path else []
    panel_source = "external"
    if not panel_genes:
        panel_genes = _frequency_panel(
            mutations,
            min_frequency=float(config.get("genes", {}).get("panel_min_frequency", 0.01)),
            max_genes=int(config.get("genes", {}).get("panel_max_genes", 500)),
        )
        panel_source = "frequency_defined"
        result["warnings"].append(
            "外部突变 panel 清单缺失：U_panel 使用频率定义"
            "（非沉默突变携带率 >=1%，最多 500 基因）。"
        )

    universes = build_universes(
        master=master,
        expression_genes=expression.frame.columns,
        cnv_genes=cnv.frame.columns,
        mutation_genes_A=mutations.genes_A,
        mutation_genes_B=mutations.genes_B,
        ppi_genes=ppi.degree if ppi else (),
        ppi_degree=ppi.degree if ppi else {},
        pathway_genes=pathway_genes,
        panel_genes=panel_genes,
        vae_size=int(config.get("genes", {}).get("vae_size", 1600)),
    )
    result["stats"]["universes"] = {key: len(value) for key, value in universes.items()}
    result["stats"]["panel_source"] = panel_source

    # ------------------------------------------------------------------
    # 3. 细胞系 / 化合物 / 标签
    # ------------------------------------------------------------------
    labels_raw_min = _raw_labels_minimal(labels_path)
    model_raw = pd.read_csv(model_path, dtype=str, low_memory=False)
    cell_mapping, conflicts = ids.map_gdsc_cell_lines(labels_raw_min, model_raw)
    mapped_cells = set(cell_mapping.loc[cell_mapping["status"] == "ok", "cell_line_uid"].dropna())

    non_cancer_sids = set(
        labels_raw_min.loc[
            labels_raw_min["CANCER_TYPE"].astype(str).eq("Non-Cancerous"), "SANGER_MODEL_ID"
        ]
    )
    non_cancer_cells = set(
        cell_mapping.loc[
            cell_mapping["source_id"].isin(non_cancer_sids) & cell_mapping["status"].eq("ok"),
            "cell_line_uid",
        ].dropna()
    )

    cell_lines = ids.build_cell_lines(
        model_path,
        has_ge=set(expression.frame.index),
        has_mu=set(mutations.model_ids),
        has_cnv=set(cnv.frame.index),
        has_meth=set(),
        mapped_cell_lines=mapped_cells,
        excluded_cell_lines=non_cancer_cells,
    )

    compound_meta = pd.read_csv(compounds_path, dtype=str, low_memory=False)
    compounds = ids.build_compounds(
        _full_labels_raw(labels_path), compound_meta, master=master
    )
    labels, label_stats = build_labels(
        _full_labels_raw(labels_path), cell_mapping, compounds, cell_lines
    )
    result["stats"]["labels"] = label_stats
    result["stats"]["main_pool_cells"] = int(cell_lines["in_main_pool"].sum())
    result["stats"]["main_pool_labels"] = int(
        labels[
            labels["cell_line_uid"].isin(
                cell_lines.loc[cell_lines["in_main_pool"], "cell_line_uid"]
            )
        ].shape[0]
    )

    # ------------------------------------------------------------------
    # 4. 落盘 canonical
    # ------------------------------------------------------------------
    io.write_parquet(cell_lines, paths.canonical / "cell_lines.parquet")
    io.write_parquet(labels, paths.canonical / "labels.parquet")
    expression_subset = expression_module.ExpressionData(
        frame=expression.frame[sorted(universes["teach_ge"])],
        info={**expression.info, "n_genes_after_universe": len(universes["teach_ge"])},
    )
    cnv_subset = cnv_module.CnvData(
        frame=cnv.frame[sorted(universes["cnv"])],
        info={**cnv.info, "n_genes_after_universe": len(universes["cnv"])},
    )
    expression_module.write_expression(expression_subset, paths.canonical / "expression.parquet")
    cnv_module.write_cnv(cnv_subset, paths.canonical / "cnv.parquet")
    mutations_module.write_mutations(mutations, paths.canonical)
    io.write_parquet(
        ppi.edges
        if ppi
        else pd.DataFrame(columns=["gene_a", "gene_b", "combined_score"]),
        paths.canonical / "ppi.parquet",
    )
    io.write_parquet(pathways, paths.canonical / "pathways.parquet")

    gene_frame, _ = build_gene_artifact(
        hgnc_path,
        paths.canonical / "genes.parquet",
        universes=universes,
        inputs=inputs,
        config=config.get("genes", {}),
    )

    compound_mapping = ids.build_compound_mapping(compound_meta, compounds)
    expr_header = expression_module.read_expression_header(expr_path)
    cnv_header = pd.read_csv(cnv_path, nrows=0).columns.tolist()
    cnv_header = [c for c in cnv_header if not str(c).startswith("Unnamed:")]
    gene_mappings = [
        _gene_mapping_frame(expr_header, master, source_table="depmap_expression"),
        _gene_mapping_frame(cnv_header, master, source_table="depmap_cnv"),
    ]
    id_mapping = ids.build_id_mapping(
        cell_mapping, compounds, gene_mappings=gene_mappings, compound_mapping=compound_mapping
    )
    io.write_parquet(id_mapping, paths.canonical / "id_mapping.parquet")
    ids.write_id_conflicts(
        paths.canonical / "id_conflicts.json",
        conflicts,
        extra={
            "n_gdsc_cell_lines": int(len(cell_mapping)),
            "n_mapped": int(cell_mapping["status"].eq("ok").sum()),
            "non_cancerous_cell_lines": sorted(non_cancer_cells),
            "n_compounds": int(len(compounds)),
            "n_drug_ids": int(compound_meta["DRUG_ID"].nunique()),
        },
    )

    # ------------------------------------------------------------------
    # 5. DTI（GDSC + Open Targets + ChEMBL + DGIdb）→ 药物特征
    # ------------------------------------------------------------------
    dgidb_path = REPO_ROOT / "data" / "raw" / "dgidb" / "interactions.tsv"
    opentargets_dir = config.get("raw_files", {}).get(
        "opentargets_dir", REPO_ROOT / "data" / "raw" / "opentargets" / "26.09"
    )
    chembl_dir = config.get("raw_files", {}).get(
        "chembl_dir", REPO_ROOT / "data" / "raw" / "chembl" / "37"
    )
    if dgidb_path.is_file():
        dti, dti_info = build_dti(
            compounds,
            dgidb_path,
            master,
            allowed_genes=universes["teach_ge"],
            opentargets_dir=opentargets_dir,
            chembl_dir=chembl_dir,
        )
        io.write_parquet(dti, paths.canonical / "dti.parquet")
        result["stats"]["dti"] = dti_info
        # 把 DTI 靶点并入 compounds，供药物靶点 multi-hot 使用
        compounds["gdsc_target_genes"] = compounds["target_genes"].map(list)
        dti_genes: dict[str, set[str]] = {}
        for row in dti.itertuples(index=False):
            dti_genes.setdefault(row.compound_uid, set()).add(row.hgnc_symbol)
        base_genes = dict(zip(compounds["compound_uid"], compounds["target_genes"], strict=False))
        compounds["target_genes"] = compounds["compound_uid"].map(
            lambda uid: sorted(set(base_genes.get(uid, [])) | dti_genes.get(uid, set()))
        )
        compounds["has_gdsc_target"] = compounds["gdsc_target_genes"].map(len) > 0
        compounds["has_target"] = compounds["target_genes"].map(len) > 0
        compounds["has_dti"] = compounds["compound_uid"].isin(dti_genes)
    else:
        dti = pd.DataFrame(
            columns=[
                "compound_uid",
                "hgnc_symbol",
                "source",
                "score",
                "action_type",
                "mechanism",
                "target_name",
            ]
        )
        io.write_parquet(dti, paths.canonical / "dti.parquet")
        result["warnings"].append("DGIdb interactions.tsv 缺失，DTI 仅保留 GDSC 靶点文本。")

    drug_arrays, drug_coverage = build_drug_features(
        compounds,
        smiles_path=config.get("local_files", {}).get("drug_smiles"),
        target_genes=universes["teach_ge"],
        radius=int(config.get("drug_features", {}).get("morgan", {}).get("radius", 2)),
        bits=int(config.get("drug_features", {}).get("morgan", {}).get("bits", 2048)),
    )
    drug_coverage["has_gdsc_target"] = (
        int(compounds["has_gdsc_target"].sum()) if "has_gdsc_target" in compounds else 0
    )
    drug_coverage["has_dti"] = (
        int(compounds["has_dti"].sum()) if "has_dti" in compounds else 0
    )
    io.write_npz(paths.canonical / "drug_features.npz", drug_arrays)
    io.write_json(paths.canonical / "drug_feature_coverage.json", drug_coverage)
    compounds["smiles"] = [
        value if value else None for value in drug_arrays["smiles"].tolist()
    ]
    compounds["scaffold"] = [
        value if value else None for value in drug_arrays["scaffold"].tolist()
    ]
    compounds["has_smiles"] = compounds["smiles"].notna()
    compounds["has_scaffold"] = compounds["scaffold"].notna()
    io.write_parquet(compounds, paths.canonical / "compounds.parquet")
    if not drug_coverage["rdkit_available"]:
        result["warnings"].append("RDKit 未安装：Morgan FP/分子图/scaffold 降级为空 + mask。")
    if drug_coverage["has_smiles"] == 0:
        result["warnings"].append("SMILES 缺失：P3 scaffold 将降级为 target-family split。")

    # ------------------------------------------------------------------
    # 6. TCGA（离线可用部分）
    # ------------------------------------------------------------------
    tcga_result = _build_tcga_offline(
        config,
        paths,
        result,
        fetch_file_index=fetch_file_index,
        universe_genes=universes["teach_ge"],
    )
    result["stats"]["tcga"] = tcga_result

    # ------------------------------------------------------------------
    # 7. 可选 METH
    # ------------------------------------------------------------------
    if bool(config["omics"]["methylation"].get("enabled", False)):
        meth_path = REPO_ROOT / "data" / "raw" / "ccle" / "CCLE_DNA_methylation_TSS1kb.txt"
        if meth_path.is_file():
            meth = methylation_module.load_methylation(meth_path, model_path, master)
            methylation_module.write_methylation(meth, paths.canonical)
            result["stats"]["methylation"] = meth.info
        else:
            result["warnings"].append("METH 已启用但原始文件缺失。")

    # ------------------------------------------------------------------
    # 8. manifest / stats
    # ------------------------------------------------------------------
    stats_payload = {
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "code_hash": code,
        "config_hash": cfg_hash,
        "inputs": inputs,
        "stats": result["stats"],
        "warnings": result["warnings"],
        "universes": {key: len(value) for key, value in universes.items()},
    }
    io.write_json(paths.audit / "canonical_stats.json", stats_payload)
    io.write_json(
        paths.canonical / "gene_universes.json",
        {**universes, "panel_source": panel_source},
    )
    # manifest
    from kite_drp.data.manifest import Manifest

    manifest = Manifest(paths.canonical / "manifest.json")
    manifest.artifacts = {}
    for name in [
        "cell_lines",
        "compounds",
        "labels",
        "expression",
        "cnv",
        "mutations_A",
        "mutations_B",
        "mutation_long",
        "genes",
        "id_mapping",
        "ppi",
        "pathways",
        "drug_features",
    ]:
        spec = registry.ARTIFACTS.get(name)
        if spec and spec.path.is_file():
            manifest.register(name, spec.path, layer="canonical", inputs=inputs)
    for extra in [
        "drug_feature_coverage.json",
        "id_conflicts.json",
        "dti.parquet",
        "mut_burden.parquet",
        "tcga_mu.npz",
        "tcga_mu_B.npz",
        "tcga_ge.npz",
        "tcga_cnv.npz",
        "tcga_meth_stats.parquet",
        "tcga_clinical.parquet",
        "tcga_treatments.parquet",
        "tcga_cdr_cesc.parquet",
        "tcga_ding2016.parquet",
    ]:
        candidate = paths.canonical / extra
        if candidate.is_file():
            manifest.register(extra, candidate, layer="canonical", inputs=inputs)
    manifest.save(code=code, config=cfg_hash, extra={"warnings": result["warnings"]})
    result["artifacts"] = sorted(manifest.artifacts)
    return result


def _build_tcga_offline(
    config: dict[str, Any],
    paths: DataPaths,
    result: dict[str, Any],
    *,
    fetch_file_index: bool,
    universe_genes: Iterable[str] | None = None,
) -> dict[str, Any]:
    tcga_root = REPO_ROOT / "data" / "raw" / "tcga"
    index_path = Path(
        config.get("raw_files", {}).get(
            "tcga_file_index", paths.interim / "tcga" / "file_index.parquet"
        )
    )
    if fetch_file_index and not index_path.is_file():
        manifest = tcga_root / "manifests" / "manifest_cesc_4modalities.tsv"
        if manifest.is_file():
            try:
                file_ids = tcga_module.file_ids_from_manifest(manifest)
                tcga_module.fetch_file_index(file_ids, index_path)
                result.setdefault("warnings", []).append(f"GDC file index 已生成：{index_path}")
            except Exception as exc:  # pragma: no cover - 网络相关
                result.setdefault("warnings", []).append(f"GDC file index 拉取失败：{exc}")
    file_index = tcga_module.load_file_index(index_path)
    if file_index is None:
        result.setdefault("warnings", []).append(
            "TCGA file→case 映射缺失：GE/CNV/METH 患者级矩阵未装配"
            "（MU/clinical/CDR/Ding 可离线构建）。"
        )

    stats: dict[str, Any] = {"file_index": None if file_index is None else len(file_index)}
    cesc_dir = tcga_root / "CESC"
    if cesc_dir.is_dir():
        mu = tcga_module.build_tcga_mutations(cesc_dir, universe_genes=None)
        tcga_module.write_tcga_mutations(mu, paths.interim / "canonical")
        stats["mu"] = mu.info
    if file_index is not None and cesc_dir.is_dir():
        hgnc_path = _raw_path(config, "hgnc", "data/raw/hgnc/Gene.csv")
        master = build_gene_master(hgnc_path)
        ge = tcga_module.build_tcga_expression(
            cesc_dir, file_index, master, universe_genes=universe_genes
        )
        cnv = tcga_module.build_tcga_cnv(
            cesc_dir, file_index, master, universe_genes=universe_genes
        )
        if not ge.frame.empty:
            tcga_module.write_tcga_omics(ge, paths.interim / "canonical" / "tcga_ge.npz")
        if not cnv.frame.empty:
            tcga_module.write_tcga_omics(cnv, paths.interim / "canonical" / "tcga_cnv.npz")
        stats["ge"] = ge.info
        stats["cnv"] = cnv.info
        meth_stats = tcga_module.build_tcga_meth_stats(cesc_dir, file_index)
        if not meth_stats.empty:
            io.write_parquet(
                meth_stats, paths.interim / "canonical" / "tcga_meth_stats.parquet"
            )
            stats["meth_stats"] = {
                "n_files": int(len(meth_stats)),
                "missing_fraction_median": float(meth_stats["missing_fraction"].median()),
            }
    clinical_path = next((tcga_root / "clinical").glob("*.tar.gz"), None)
    if clinical_path:
        clinical, treatments, info = tcga_module.build_tcga_clinical(clinical_path)
        io.write_parquet(clinical, paths.interim / "canonical" / "tcga_clinical.parquet")
        io.write_parquet(treatments, paths.interim / "canonical" / "tcga_treatments.parquet")
        stats["clinical"] = info
    cdr_path = tcga_root / "cdr" / "TCGA-CDR-SupplementalTableS1.xlsx"
    if cdr_path.is_file():
        cdr, info = tcga_module.build_tcga_cdr(cdr_path)
        io.write_parquet(cdr, paths.interim / "canonical" / "tcga_cdr_cesc.parquet")
        stats["cdr"] = info
    ding_zip = next((tcga_root / "ding2016").glob("*.zip"), None)
    if ding_zip:
        ding, info = tcga_module.build_tcga_ding(ding_zip)
        io.write_parquet(ding, paths.interim / "canonical" / "tcga_ding2016.parquet")
        stats["ding2016"] = info
    return stats


def _build_drug_features_only(
    config: dict[str, Any],
    paths: DataPaths,
    result: dict[str, Any],
) -> dict[str, Any]:
    """只重建 canonical 药物特征（SMILES/Morgan/分子图/targets/scaffold）。

    用于给已冻结的 canonical 补字段，避免重跑全量组学（约 15 min）。输入只读
    ``compounds.parquet`` 与 ``gene_universes.json``，不依赖 raw 大文件。
    """
    from kite_drp.data.manifest import Manifest

    compounds_path = paths.canonical / "compounds.parquet"
    universe_path = paths.canonical / "gene_universes.json"
    if not compounds_path.is_file() or not universe_path.is_file():
        raise FileNotFoundError(
            f"缺少 canonical 输入：{compounds_path} / {universe_path}；请先跑 data canonical"
        )
    compounds = io.read_parquet(compounds_path)
    universes = io.read_json(universe_path)
    arrays, coverage = build_drug_features(
        compounds,
        smiles_path=config.get("local_files", {}).get("drug_smiles"),
        target_genes=universes["teach_ge"],
        radius=int(config.get("drug_features", {}).get("morgan", {}).get("radius", 2)),
        bits=int(config.get("drug_features", {}).get("morgan", {}).get("bits", 2048)),
    )
    io.write_npz(paths.canonical / "drug_features.npz", arrays)
    io.write_json(paths.canonical / "drug_feature_coverage.json", coverage)
    compounds["smiles"] = [value if value else None for value in arrays["smiles"].tolist()]
    compounds["scaffold"] = [value if value else None for value in arrays["scaffold"].tolist()]
    compounds["has_smiles"] = compounds["smiles"].notna()
    compounds["has_scaffold"] = compounds["scaffold"].notna()
    io.write_parquet(compounds, compounds_path)

    manifest = Manifest(paths.canonical / "manifest.json")
    manifest.register("drug_features", paths.canonical / "drug_features.npz", layer="canonical")
    manifest.register("compounds", compounds_path, layer="canonical", df=compounds)
    manifest.register(
        "drug_feature_coverage.json",
        paths.canonical / "drug_feature_coverage.json",
        layer="canonical",
    )
    manifest.artifacts.pop("drug_coverage", None)
    manifest.save(extra={"partial_rebuild": "drug_features"})

    result["artifacts"].extend(
        [str(paths.canonical / "drug_features.npz"), str(compounds_path)]
    )
    result["stats"]["drug_features"] = coverage
    if not coverage["rdkit_available"]:
        result["warnings"].append("RDKit 未安装：分子图/Morgan/scaffold 降级为空 + mask。")
    if coverage["graphs"]["coverage"] < 1.0:
        result["warnings"].append(
            f"分子图覆盖 {coverage['graphs']['n_graphs']}/{coverage['n_compounds']}"
        )
    return result


def _build_tcga_only(
    config: dict[str, Any],
    paths: DataPaths,
    result: dict[str, Any],
    *,
    fetch_file_index: bool,
) -> dict[str, Any]:
    universe_path = paths.canonical / "gene_universes.json"
    universe_genes = None
    if universe_path.is_file():
        universe_genes = io.read_json(universe_path).get("teach_ge")
    result["stats"]["tcga"] = _build_tcga_offline(
        config,
        paths,
        result,
        fetch_file_index=fetch_file_index,
        universe_genes=universe_genes,
    )
    return result


__all__ = ["build_canonical"]

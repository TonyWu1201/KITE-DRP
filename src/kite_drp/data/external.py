"""外部验证库预处理（P5）：GDSC1、CTRPv2、PRISM。

外部库只用于评估，不进入主训练/选模/早停。产物写入 ``data/interim/external/``，
并生成 ``data/splits/external_validation_manifest.json`` 记录版本与映射率。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from kite_drp.common.paths import REPO_ROOT
from kite_drp.data import ids, io

try:  # 可选依赖：SMILES -> InChIKey
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    RDKIT_AVAILABLE = True
except Exception:  # pragma: no cover
    Chem = None  # type: ignore
    RDKIT_AVAILABLE = False


@dataclass
class ExternalResult:
    frame: pd.DataFrame
    info: dict[str, Any]


def _model_name_index(model: pd.DataFrame) -> tuple[dict[str, str], list[str]]:
    """归一化细胞系名 -> ACH；歧义名称不自动采用。"""
    buckets: dict[str, set[str]] = {}
    for row in model.itertuples(index=False):
        for value in [
            getattr(row, "CellLineName", None),
            getattr(row, "StrippedCellLineName", None),
            getattr(row, "CCLEName", None),
        ]:
            key = ids.normalize_name(value)
            if key:
                buckets.setdefault(key, set()).add(str(row.ModelID))
    index = {key: next(iter(values)) for key, values in buckets.items() if len(values) == 1}
    ambiguous = sorted(key for key, values in buckets.items() if len(values) > 1)
    return index, ambiguous


def _map_by_names(names: pd.Series, index: dict[str, str]) -> pd.Series:
    return names.map(lambda value: index.get(ids.normalize_name(value)))


def _inchi_key_from_smiles(value: object) -> str | None:
    if not RDKIT_AVAILABLE or not isinstance(value, str) or not value.strip():
        return None
    first = re.split(r"[,;]", value)[0].strip()
    molecule = Chem.MolFromSmiles(first)
    if molecule is None:
        return None
    return Chem.MolToInchiKey(molecule)


def _all_compounds(compound_meta: pd.DataFrame, gdsc2: pd.DataFrame) -> pd.DataFrame:
    """621 个 GDSC 元数据 ID → compound_uid（与 canonical 同口径）。"""
    gdsc2_keys = set(gdsc2["compound_uid"])
    rows = []
    for row in compound_meta.itertuples(index=False):
        inchikey = getattr(row, "INCHI_KEY", None)
        inchikey = inchikey if isinstance(inchikey, str) and inchikey else None
        uid = (
            f"inchikey:{inchikey}"
            if inchikey
            else f"gdscname:{ids.normalize_drug_name(row.DRUG_NAME)}"
        )
        rows.append(
            {
                "drug_id": str(row.DRUG_ID),
                "compound_uid": uid,
                "drug_name": row.DRUG_NAME,
                "inchikey": inchikey,
                "in_gdsc2": uid in gdsc2_keys,
            }
        )
    return pd.DataFrame(rows)


def build_gdsc1(
    gdsc1_path: str | Path,
    compound_meta_path: str | Path,
    model_path: str | Path,
    gdsc2_compounds: pd.DataFrame,
) -> ExternalResult:
    raw = pd.read_csv(gdsc1_path, dtype=str)
    model = pd.read_csv(model_path, dtype=str, low_memory=False)
    mapping, _ = ids.map_gdsc_cell_lines(
        raw.rename(columns={"TCGA_DESC": "CANCER_TYPE"}), model
    )
    sid_to_cell = (
        mapping.loc[mapping["status"] == "ok", ["source_id", "cell_line_uid"]]
        .drop_duplicates("source_id")
        .set_index("source_id")["cell_line_uid"]
        .to_dict()
    )
    meta = pd.read_csv(compound_meta_path, dtype=str, low_memory=False)
    all_compounds = _all_compounds(meta, gdsc2_compounds)
    id_to_uid = dict(zip(all_compounds["drug_id"], all_compounds["compound_uid"], strict=False))
    id_to_in_gdsc2 = dict(zip(all_compounds["drug_id"], all_compounds["in_gdsc2"], strict=False))

    frame = raw.copy()
    frame["SANGER_MODEL_ID"] = frame["SANGER_MODEL_ID"].astype(str).str.strip()
    frame["DRUG_ID"] = frame["DRUG_ID"].astype(str).str.strip()
    frame["cell_line_uid"] = frame["SANGER_MODEL_ID"].map(sid_to_cell)
    frame["mapped"] = frame["cell_line_uid"].notna()
    frame["compound_uid"] = frame["DRUG_ID"].map(id_to_uid)
    frame["compound_in_gdsc2"] = frame["DRUG_ID"].map(id_to_in_gdsc2).fillna(False).astype(bool)
    frame["dataset"] = "GDSC1"
    frame = frame.dropna(subset=["compound_uid"]).copy()
    for column in ("LN_IC50", "AUC", "Z_SCORE", "RMSE", "MAX_CONC", "MIN_CONC"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    # 同 (细胞系, 化合物) 的多个曲线取均值
    key = frame["cell_line_uid"].fillna("SID:" + frame["SANGER_MODEL_ID"])
    frame["_cell_key"] = key
    grouped = frame.groupby(["_cell_key", "compound_uid"], sort=True)
    rows = []
    for (_cell_key, compound), group in grouped:
        rows.append(
            {
                "cell_line_uid": group["cell_line_uid"].iloc[0],
                "mapped": bool(group["mapped"].iloc[0]),
                "sanger_model_id": group["SANGER_MODEL_ID"].iloc[0],
                "cosmic_id": group["COSMIC_ID"].iloc[0],
                "compound_uid": compound,
                "compound_in_gdsc2": bool(group["compound_in_gdsc2"].iloc[0]),
                "drug_id": group["DRUG_ID"].iloc[0],
                "drug_name": group["DRUG_NAME"].iloc[0],
                "ln_ic50": float(group["LN_IC50"].mean()),
                "sensitivity_score": -float(group["LN_IC50"].mean()),
                "auc": float(group["AUC"].mean()),
                "z_score": float(group["Z_SCORE"].mean()),
                "rmse": float(group["RMSE"].mean()),
                "max_conc": float(group["MAX_CONC"].max()),
                "min_conc": float(group["MIN_CONC"].min()),
                "dataset": "GDSC1",
            }
        )
    output = pd.DataFrame(rows).sort_values(
        ["cell_line_uid", "compound_uid"], na_position="last"
    ).reset_index(drop=True)
    info = {
        "rows_raw": int(len(raw)),
        "rows_compound_level": int(len(output)),
        "cells": int(output["cell_line_uid"].nunique()),
        "cells_mapped": int(output.loc[output["mapped"], "cell_line_uid"].nunique()),
        "compounds": int(output["compound_uid"].nunique()),
        "compounds_in_gdsc2": int(
            output.loc[output["compound_in_gdsc2"], "compound_uid"].nunique()
        ),
        "rows_in_gdsc2_compounds": int(output["compound_in_gdsc2"].sum()),
    }
    return ExternalResult(output, info)


def build_ctrp(
    ctrp_dir: str | Path,
    model_path: str | Path,
    gdsc2_compounds: pd.DataFrame,
    *,
    gdsc2_labels: pd.DataFrame | None = None,
) -> ExternalResult:
    root = Path(ctrp_dir)
    auc = pd.read_csv(root / "v21_auc_sensitivities.tsv", sep="\t", dtype=str)
    treatment = pd.read_csv(root / "treatment.tsv", sep="\t", dtype=str, low_memory=False)
    sample = pd.read_csv(root / "sample.tsv", sep="\t", dtype=str, low_memory=False)
    model = pd.read_csv(model_path, dtype=str, low_memory=False)
    name_index, ambiguous = _model_name_index(model)

    cell_names = sample["CELL_NAME"]
    sample["cell_line_uid"] = _map_by_names(cell_names, name_index)
    if "AnnotationDB.CellLine.Name" in sample:
        missing = sample["cell_line_uid"].isna()
        sample.loc[missing, "cell_line_uid"] = _map_by_names(
            sample.loc[missing, "AnnotationDB.CellLine.Name"], name_index
        )
    sample["mapped"] = sample["cell_line_uid"].notna()

    gdsc_ik = set(gdsc2_compounds["inchikey"].dropna())
    ik_to_uid = dict(
        zip(gdsc2_compounds["inchikey"], gdsc2_compounds["compound_uid"], strict=False)
    )
    name_to_uid = {}
    for row in gdsc2_compounds.itertuples(index=False):
        for name in [row.drug_name, *getattr(row, "gdsc_names", [])]:
            name_to_uid.setdefault(ids.normalize_drug_name(name), row.compound_uid)

    treatment["computed_inchikey"] = treatment["SMILES"].map(_inchi_key_from_smiles)
    treatment["inchikey"] = treatment["computed_inchikey"]
    for column in ("AnnotationDB.InChIKey", "AnnotationGx.InChIKey"):
        if column in treatment:
            treatment["inchikey"] = treatment["inchikey"].fillna(treatment[column])
    treatment["compound_uid"] = treatment["inchikey"].map(ik_to_uid)
    unresolved = treatment["compound_uid"].isna()
    treatment.loc[unresolved, "compound_uid"] = treatment.loc[unresolved, "Compound.Name"].map(
        lambda value: name_to_uid.get(ids.normalize_drug_name(value))
    )
    treatment["compound_in_gdsc2"] = treatment["compound_uid"].notna()
    treatment["in_gdsc2_by_inchikey"] = treatment["inchikey"].isin(gdsc_ik)

    cpd = treatment.set_index("master_cpd_id")[
        ["Compound.Name", "compound_uid", "compound_in_gdsc2", "in_gdsc2_by_inchikey", "inchikey"]
    ]
    output = auc.copy()
    output["master_cpd_id"] = output["master_cpd_id"].astype(str)
    output = output.join(cpd, on="master_cpd_id", how="left")
    output = output.merge(
        sample[["master_ccl_id", "CELL_NAME", "cell_line_uid", "mapped"]].rename(
            columns={"CELL_NAME": "cell_name"}
        ),
        on="master_ccl_id",
        how="left",
    )
    output["auc"] = pd.to_numeric(output["area_under_curve"], errors="coerce")
    output["sensitivity_score"] = -output["auc"]
    output = output.rename(
        columns={
            "experiment_id": "experiment_id",
            "master_cpd_id": "external_compound_id",
            "master_ccl_id": "external_cell_id",
            "source_release": "source_release",
        }
    )
    output["dataset"] = "CTRPv2"
    output = output[
        [
            "cell_line_uid",
            "mapped",
            "external_cell_id",
            "cell_name",
            "external_compound_id",
            "Compound.Name",
            "compound_uid",
            "compound_in_gdsc2",
            "in_gdsc2_by_inchikey",
            "inchikey",
            "auc",
            "sensitivity_score",
            "source_release",
            "dataset",
        ]
    ].rename(columns={"Compound.Name": "compound_name"})
    output = output.sort_values(["cell_line_uid", "compound_uid"], na_position="last").reset_index(
        drop=True
    )
    info = {
        "rows": int(len(output)),
        "cells": int(output["cell_line_uid"].nunique()),
        "cells_mapped": int(output.loc[output["mapped"], "cell_line_uid"].nunique()),
        "compounds": int(output["compound_uid"].nunique()),
        "compounds_in_gdsc2": int(
            output.loc[output["compound_in_gdsc2"], "compound_uid"].nunique()
        ),
        "rows_in_gdsc2_compounds": int(output["compound_in_gdsc2"].sum()),
        "ambiguous_cell_names": len(ambiguous),
        "auc_direction": "higher_auc_more_resistant; sensitivity_score=-auc",
    }
    return ExternalResult(output, info)


def build_prism(
    prism_dir: str | Path,
    model_path: str | Path,
    gdsc2_compounds: pd.DataFrame,
) -> ExternalResult:
    root = Path(prism_dir)
    curve = pd.read_csv(
        root / "secondary-screen-dose-response-curve-parameters.csv",
        usecols=[
            "broad_id",
            "depmap_id",
            "ccle_name",
            "screen_id",
            "auc",
            "ec50",
            "ic50",
            "name",
            "target",
            "smiles",
            "passed_str_profiling",
        ],
        low_memory=False,
    )
    model = pd.read_csv(model_path, dtype=str, low_memory=False)
    model_ids = set(model["ModelID"].astype(str))
    treatment = pd.read_csv(
        root / "secondary-screen-replicate-collapsed-treatment-info.csv",
        usecols=["broad_id", "name", "smiles", "target", "moa"],
        dtype=str,
        low_memory=False,
    ).drop_duplicates("broad_id")
    treatment["computed_inchikey"] = treatment["smiles"].map(_inchi_key_from_smiles)
    gdsc_ik = set(gdsc2_compounds["inchikey"].dropna())
    ik_to_uid = dict(
        zip(gdsc2_compounds["inchikey"], gdsc2_compounds["compound_uid"], strict=False)
    )
    name_to_uid = {}
    for row in gdsc2_compounds.itertuples(index=False):
        for value in [row.drug_name, *getattr(row, "gdsc_names", [])]:
            name_to_uid.setdefault(ids.normalize_drug_name(value), row.compound_uid)
    treatment["compound_uid"] = treatment["computed_inchikey"].map(ik_to_uid)
    unresolved = treatment["compound_uid"].isna()
    treatment.loc[unresolved, "compound_uid"] = treatment.loc[unresolved, "name"].map(
        lambda value: name_to_uid.get(ids.normalize_drug_name(value))
    )
    treatment["compound_in_gdsc2"] = treatment["compound_uid"].notna()
    treatment["in_gdsc2_by_inchikey"] = treatment["computed_inchikey"].isin(gdsc_ik)
    cpd = treatment.set_index("broad_id")[
        ["compound_uid", "compound_in_gdsc2", "in_gdsc2_by_inchikey", "computed_inchikey", "moa"]
    ]
    output = curve.copy()
    output["cell_line_uid"] = output["depmap_id"].astype(str)
    output["mapped"] = output["cell_line_uid"].isin(model_ids)
    output = output.join(cpd, on="broad_id", how="left")
    output["auc"] = pd.to_numeric(output["auc"], errors="coerce")
    output["sensitivity_score"] = -output["auc"]
    output["ec50"] = pd.to_numeric(output["ec50"], errors="coerce")
    output["ic50"] = pd.to_numeric(output["ic50"], errors="coerce")
    output["dataset"] = "PRISM"
    output = output[
        [
            "cell_line_uid",
            "mapped",
            "depmap_id",
            "ccle_name",
            "screen_id",
            "broad_id",
            "compound_uid",
            "compound_in_gdsc2",
            "in_gdsc2_by_inchikey",
            "name",
            "moa",
            "target",
            "auc",
            "sensitivity_score",
            "ec50",
            "ic50",
            "passed_str_profiling",
            "dataset",
        ]
    ].rename(columns={"name": "compound_name"})
    output = output.sort_values(["cell_line_uid", "compound_uid"], na_position="last").reset_index(
        drop=True
    )
    info = {
        "rows": int(len(output)),
        "cells": int(output["cell_line_uid"].nunique()),
        "cells_mapped": int(output.loc[output["mapped"], "cell_line_uid"].nunique()),
        "compounds": int(output["compound_uid"].nunique()),
        "compounds_in_gdsc2": int(
            output.loc[output["compound_in_gdsc2"], "compound_uid"].nunique()
        ),
        "rows_in_gdsc2_compounds": int(output["compound_in_gdsc2"].sum()),
        "treatments": int(len(treatment)),
        "auc_direction": "higher_auc_more_resistant; sensitivity_score=-auc",
    }
    return ExternalResult(output, info)


def build_prism_lfc(
    prism_dir: str | Path,
    model_path: str | Path,
    gdsc2_compounds: pd.DataFrame,
) -> ExternalResult:
    """PRISM LFC 宽表 → long；用于固定剂量/排序分析。"""
    root = Path(prism_dir)
    wide = pd.read_csv(
        root / "secondary-screen-replicate-collapsed-logfold-change.csv",
        index_col=0,
        low_memory=False,
    )
    treatment = pd.read_csv(
        root / "secondary-screen-replicate-collapsed-treatment-info.csv",
        usecols=["broad_id", "name", "smiles"],
        dtype=str,
        low_memory=False,
    ).drop_duplicates("broad_id")
    treatment["computed_inchikey"] = treatment["smiles"].map(_inchi_key_from_smiles)
    ik_to_uid = dict(
        zip(gdsc2_compounds["inchikey"], gdsc2_compounds["compound_uid"], strict=False)
    )
    name_to_uid = {}
    for row in gdsc2_compounds.itertuples(index=False):
        for value in [row.drug_name, *getattr(row, "gdsc_names", [])]:
            name_to_uid.setdefault(ids.normalize_drug_name(value), row.compound_uid)
    treatment["compound_uid"] = treatment["computed_inchikey"].map(ik_to_uid)
    unresolved = treatment["compound_uid"].isna()
    treatment.loc[unresolved, "compound_uid"] = treatment.loc[unresolved, "name"].map(
        lambda value: name_to_uid.get(ids.normalize_drug_name(value))
    )
    treatment["compound_in_gdsc2"] = treatment["compound_uid"].notna()
    broad_map = treatment.set_index("broad_id")[["compound_uid", "compound_in_gdsc2"]]

    long = wide.reset_index().melt(
        id_vars=wide.index.name or "index",
        var_name="treatment",
        value_name="lfc",
    )
    parsed = long["treatment"].str.extract(
        r"^(?P<broad_id>.+?)::(?P<dose>[^:]+)::(?P<screen_id>.+)$"
    )
    long = pd.concat([long.drop(columns=["treatment"]), parsed], axis=1)
    long = long.rename(columns={wide.index.name or "index": "cell_line_uid"})
    long["cell_line_uid"] = long["cell_line_uid"].astype(str)
    long = long.join(broad_map, on="broad_id", how="left")
    long["mapped"] = True
    long["dose"] = pd.to_numeric(long["dose"], errors="coerce")
    long["dataset"] = "PRISM"
    long = long[
        [
            "cell_line_uid",
            "mapped",
            "broad_id",
            "compound_uid",
            "compound_in_gdsc2",
            "dose",
            "screen_id",
            "lfc",
            "dataset",
        ]
    ]
    info = {
        "rows": int(len(long)),
        "cells": int(long["cell_line_uid"].nunique()),
        "treatments": int(long["broad_id"].nunique()),
        "compounds": int(long["compound_uid"].nunique()),
        "compounds_in_gdsc2": int(
            long.loc[long["compound_in_gdsc2"], "compound_uid"].nunique()
        ),
        "rows_in_gdsc2_compounds": int(long["compound_in_gdsc2"].sum()),
    }
    return ExternalResult(long, info)


def build_external(
    *,
    raw_dir: str | Path = REPO_ROOT / "data" / "raw",
    canonical_dir: str | Path = REPO_ROOT / "data" / "interim" / "canonical",
    out_dir: str | Path = REPO_ROOT / "data" / "interim" / "external",
    splits_dir: str | Path = REPO_ROOT / "data" / "splits",
) -> dict[str, Any]:
    raw = Path(raw_dir)
    canonical = Path(canonical_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    compounds = io.read_parquet(canonical / "compounds.parquet")
    labels = io.read_parquet(canonical / "labels.parquet")
    cell_lines = io.read_parquet(canonical / "cell_lines.parquet")
    main_pool = set(cell_lines.loc[cell_lines["in_main_pool"], "cell_line_uid"])
    model_path = raw / "depmap" / "Model.csv"
    results: dict[str, Any] = {}

    gdsc1_path = raw / "gdsc" / "GDSC1_fitted_dose_response_27Oct23.csv"
    meta_path = raw / "gdsc" / "screened_compounds_rel_8.5_inchi.csv"
    if gdsc1_path.is_file() and meta_path.is_file():
        gdsc1 = build_gdsc1(gdsc1_path, meta_path, model_path, compounds)
        io.write_parquet(gdsc1.frame, out / "gdsc1_labels.parquet")
        gdsc1.info["cells_in_main_pool"] = int(
            gdsc1.frame.loc[gdsc1.frame["cell_line_uid"].isin(main_pool), "cell_line_uid"].nunique()
        )
        gdsc1.info["rows_in_main_pool_cells"] = int(
            gdsc1.frame["cell_line_uid"].isin(main_pool).sum()
        )
        results["gdsc1"] = gdsc1.info

    ctrp_dir = raw / "ctrp"
    if (ctrp_dir / "v21_auc_sensitivities.tsv").is_file():
        ctrp = build_ctrp(ctrp_dir, model_path, compounds, gdsc2_labels=labels)
        io.write_parquet(ctrp.frame, out / "ctrpv2_auc.parquet")
        ctrp.info["cells_in_main_pool"] = int(
            ctrp.frame.loc[ctrp.frame["cell_line_uid"].isin(main_pool), "cell_line_uid"].nunique()
        )
        ctrp.info["rows_in_main_pool_cells"] = int(
            ctrp.frame["cell_line_uid"].isin(main_pool).sum()
        )
        results["ctrpv2"] = ctrp.info

    prism_dir = raw / "prism"
    if (prism_dir / "secondary-screen-dose-response-curve-parameters.csv").is_file():
        prism = build_prism(prism_dir, model_path, compounds)
        io.write_parquet(prism.frame, out / "prism_auc.parquet")
        prism.info["cells_in_main_pool"] = int(
            prism.frame.loc[prism.frame["cell_line_uid"].isin(main_pool), "cell_line_uid"].nunique()
        )
        prism.info["rows_in_main_pool_cells"] = int(
            prism.frame["cell_line_uid"].isin(main_pool).sum()
        )
        results["prism"] = prism.info
        lfc_path = prism_dir / "secondary-screen-replicate-collapsed-logfold-change.csv"
        if lfc_path.is_file():
            lfc = build_prism_lfc(prism_dir, model_path, compounds)
            io.write_parquet(lfc.frame, out / "prism_lfc.parquet")
            lfc.info["cells_in_main_pool"] = int(
                lfc.frame.loc[lfc.frame["cell_line_uid"].isin(main_pool), "cell_line_uid"].nunique()
            )
            lfc.info["rows_in_main_pool_cells"] = int(
                lfc.frame["cell_line_uid"].isin(main_pool).sum()
            )
            results["prism_lfc"] = lfc.info

    manifest = {
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "purpose": "eval_only",
        "libraries": results,
        "rules": [
            "外部验证不参与训练/选模/早停",
            "GDSC1 零样本按 compound_uid 交集评估；CTRP/PRISM 以细胞系泛化为主",
            "AUC 方向在评估脚本中按 library 元数据统一",
        ],
    }
    io.write_json(out / "external_manifest.json", manifest)
    split_manifest = {
        "schema_version": "1.0",
        "protocol": "p5_external",
        "n_folds": 0,
        "libraries": results,
        "notes": "P5 不生成折；只定义外部验证样本清单与映射率",
    }
    Path(splits_dir).mkdir(parents=True, exist_ok=True)
    io.write_json(Path(splits_dir) / "external_validation_manifest.json", split_manifest)
    return manifest


__all__ = [
    "ExternalResult",
    "build_ctrp",
    "build_external",
    "build_gdsc1",
    "build_prism",
    "build_prism_lfc",
]

"""TCGA-CESC 解析：file→case 映射、MU/clinical/CDR/Ding2016。

离线时只有 MU 可直接从 MAF barcode 得到患者真值；GE/CNV/METH 必须依赖
``tcga_file_index.parquet``（GDC API 一次性生成后冻结）。
"""

from __future__ import annotations

import io as stdlib_io
import json
import tarfile
import urllib.request
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kite_drp.data import io

GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_FIELDS = [
    "file_id",
    "file_name",
    "data_type",
    "platform",
    "cases.case_id",
    "cases.submitter_id",
    "associated_entities.entity_submitter_id",
    "associated_entities.entity_type",
    "associated_entities.case_id",
]
SILENT_CLASSES = {
    "Silent",
    "Intron",
    "3'UTR",
    "5'UTR",
    "3'Flank",
    "5'Flank",
    "IGR",
    "RNA",
    "lincRNA",
    "Noncoding",
}
DRIVER_CLASSES = {
    "Nonsense_Mutation",
    "Frame_Shift_Del",
    "Frame_Shift_Ins",
    "Splice_Site",
    "Translation_Start_Site",
    "Nonstop_Mutation",
    "In_Frame_Del",
    "In_Frame_Ins",
}


# ---------------------------------------------------------------------------
# GDC file index
# ---------------------------------------------------------------------------
def _post_gdc(payload: dict[str, Any], *, timeout: int = 60) -> dict[str, Any]:
    request = urllib.request.Request(
        GDC_FILES_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _sample_type_from_barcode(barcode: str | None) -> str | None:
    """TCGA barcode 第 4 段（如 01A/06A）即 sample type。"""
    if not barcode:
        return None
    parts = str(barcode).split("-")
    return parts[3] if len(parts) > 3 else None


def fetch_file_index(
    file_ids: Iterable[str],
    out_path: str | Path,
    *,
    batch_size: int = 100,
) -> pd.DataFrame:
    """调用 GDC API 拉取 file_id→case 映射并冻结为 parquet。"""
    ids = sorted({str(value).strip() for value in file_ids if str(value).strip()})
    rows: list[dict[str, Any]] = []
    for start in range(0, len(ids), batch_size):
        batch = ids[start : start + batch_size]
        payload = {
            "filters": {"op": "in", "content": {"field": "file_id", "value": batch}},
            "fields": ",".join(GDC_FIELDS),
            "size": len(batch),
            "format": "JSON",
        }
        response = _post_gdc(payload)
        for hit in response.get("data", {}).get("hits", []):
            cases = hit.get("cases") or [{}]
            entities = hit.get("associated_entities") or [{}]
            rows.append(
                {
                    "file_id": hit.get("file_id"),
                    "file_name": hit.get("file_name"),
                    "data_type": hit.get("data_type"),
                    "platform": hit.get("platform"),
                    "case_id": cases[0].get("case_id"),
                    "case_submitter_id": cases[0].get("submitter_id"),
                    "sample_type": _sample_type_from_barcode(
                        entities[0].get("entity_submitter_id")
                    ),
                    "entity_submitter_id": entities[0].get("entity_submitter_id"),
                    "entity_type": entities[0].get("entity_type"),
                }
            )
    frame = pd.DataFrame(rows).sort_values("file_id").reset_index(drop=True)
    io.write_parquet(frame, out_path)
    return frame


def file_ids_from_manifest(manifest_path: str | Path) -> list[str]:
    frame = pd.read_csv(manifest_path, sep="\t", dtype=str)
    column = "id" if "id" in frame.columns else frame.columns[0]
    return frame[column].dropna().astype(str).tolist()


def load_file_index(path: str | Path) -> pd.DataFrame | None:
    candidate = Path(path)
    return pd.read_parquet(candidate) if candidate.is_file() else None


# ---------------------------------------------------------------------------
# GE / CNV / METH（需要 file→case 映射表）
# ---------------------------------------------------------------------------
@dataclass
class TcgaOmicsData:
    frame: pd.DataFrame  # index=patient barcode, columns=HGNC symbol
    info: dict[str, Any]


def _file_to_case(file_index: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """file_id → (case_submitter_id, sample_type)。"""
    mapping: dict[str, tuple[str, str]] = {}
    for row in file_index.itertuples(index=False):
        file_id = getattr(row, "file_id", None)
        case = getattr(row, "case_submitter_id", None)
        sample = getattr(row, "sample_type", None) or ""
        if file_id and case:
            mapping[str(file_id)] = (str(case), str(sample))
    return mapping


def build_tcga_expression(
    cesc_dir: str | Path,
    file_index: pd.DataFrame,
    master,
    *,
    universe_genes: Iterable[str] | None = None,
) -> TcgaOmicsData:
    files = sorted(Path(cesc_dir).rglob("*.rna_seq.augmented_star_gene_counts.tsv"))
    mapping = _file_to_case(file_index)
    allowed = set(universe_genes) if universe_genes is not None else None
    rows: list[pd.Series] = []
    patients: list[str] = []
    sample_types: dict[str, str] = {}
    skipped = 0
    for path in files:
        case = mapping.get(path.parent.name)
        if not case:
            skipped += 1
            continue
        patient, sample = case
        if not str(sample).startswith(("01", "06")):
            skipped += 1
            continue
        frame = pd.read_csv(
            path,
            sep="\t",
            skiprows=1,
            usecols=["gene_id", "gene_name", "tpm_unstranded"],
            low_memory=False,
        )
        frame = frame[~frame["gene_id"].astype(str).str.startswith("N_")].copy()
        symbols = []
        for symbol, ensembl in zip(frame["gene_name"], frame["gene_id"], strict=False):
            resolved, _ = master.resolve(symbol)
            if resolved is None:
                resolved, _ = master.resolve(str(ensembl).split(".")[0])
            symbols.append(resolved)
        frame["symbol"] = symbols
        frame = frame.dropna(subset=["symbol"])
        frame = frame.drop_duplicates("symbol")
        values = pd.to_numeric(frame["tpm_unstranded"], errors="coerce")
        series = pd.Series(
            np.log2(values.to_numpy(dtype=float) + 1.0), index=frame["symbol"].to_numpy()
        )
        if allowed is not None:
            series = series[series.index.isin(allowed)]
        rows.append(series.rename(patient))
        patients.append(patient)
        sample_types[patient] = sample
    if not rows:
        return TcgaOmicsData(pd.DataFrame(), {"n_files": len(files), "n_patients": 0})
    frame = pd.concat(rows, axis=1).T
    frame = frame.groupby(level=0).mean()
    frame.index.name = "patient_id"
    info = {
        "n_files": len(files),
        "n_patients": int(frame.shape[0]),
        "n_genes": int(frame.shape[1]),
        "skipped_files": skipped,
        "sample_types": sample_types,
        "transform": "log2(tpm_unstranded+1)",
    }
    return TcgaOmicsData(frame.sort_index(), info)


def build_tcga_cnv(
    cesc_dir: str | Path,
    file_index: pd.DataFrame,
    master,
    *,
    universe_genes: Iterable[str] | None = None,
    normal_like_fraction: float = 0.90,
) -> TcgaOmicsData:
    files = sorted(Path(cesc_dir).rglob("*.ascat3.gene_level_copy_number.v36.tsv"))
    mapping = _file_to_case(file_index)
    allowed = set(universe_genes) if universe_genes is not None else None
    rows: list[pd.Series] = []
    skipped_normal_like = 0
    skipped_unmapped = 0
    for path in files:
        case = mapping.get(path.parent.name)
        if not case:
            skipped_unmapped += 1
            continue
        patient, _ = case
        frame = pd.read_csv(path, sep="\t", usecols=["gene_name", "copy_number"], low_memory=False)
        copy_number = pd.to_numeric(frame["copy_number"], errors="coerce")
        if float((copy_number == 2).mean()) > normal_like_fraction:
            skipped_normal_like += 1
            continue
        symbols = [master.resolve(symbol)[0] for symbol in frame["gene_name"]]
        frame["symbol"] = symbols
        frame["copy_number"] = copy_number
        frame = frame.dropna(subset=["symbol", "copy_number"]).drop_duplicates("symbol")
        values = np.log2(np.maximum(frame["copy_number"].to_numpy(dtype=float), 0.5) / 2.0)
        series = pd.Series(values, index=frame["symbol"].to_numpy())
        if allowed is not None:
            series = series[series.index.isin(allowed)]
        rows.append(series.rename(patient))
    if not rows:
        return TcgaOmicsData(pd.DataFrame(), {"n_files": len(files), "n_patients": 0})
    frame = pd.concat(rows, axis=1).T.groupby(level=0).mean()
    frame.index.name = "patient_id"
    info = {
        "n_files": len(files),
        "n_patients": int(frame.shape[0]),
        "n_genes": int(frame.shape[1]),
        "skipped_normal_like": skipped_normal_like,
        "skipped_unmapped": skipped_unmapped,
        "transform": "log2(max(copy_number,0.5)/2)",
    }
    return TcgaOmicsData(frame.sort_index(), info)


def build_tcga_meth_stats(cesc_dir: str | Path, file_index: pd.DataFrame) -> pd.DataFrame:
    """只计算 HM450 每个文件的 missing/mean/median，不建患者矩阵。"""
    files = sorted(Path(cesc_dir).rglob("*.methylation_array.sesame.level3betas.txt"))
    mapping = _file_to_case(file_index)
    records = []
    for path in files:
        case = mapping.get(path.parent.name)
        if not case:
            continue
        frame = pd.read_csv(
            path, sep="\t", header=None, names=["probe", "beta"], dtype={"probe": str}
        )
        beta = pd.to_numeric(frame["beta"], errors="coerce")
        records.append(
            {
                "file_id": path.parent.name,
                "patient_id": case[0],
                "sample_type": case[1],
                "n_probes": int(len(beta)),
                "missing_fraction": float(beta.isna().mean()),
                "mean_beta": float(beta.mean()),
                "median_beta": float(beta.median()),
            }
        )
    return pd.DataFrame(records)


def write_tcga_omics(data: TcgaOmicsData, path: str | Path) -> Path:
    """保存为 npz（values/patient_ids/genes）并返回路径。"""
    return io.write_npz(
        path,
        {
            "values": data.frame.to_numpy(dtype=np.float32),
            "patient_ids": np.asarray(data.frame.index.tolist(), dtype=np.str_),
            "genes": np.asarray(data.frame.columns.tolist(), dtype=np.str_),
        },
    )


# ---------------------------------------------------------------------------
# MU
# ---------------------------------------------------------------------------
@dataclass
class TcgaMutationData:
    matrix_A: np.ndarray
    matrix_B: np.ndarray
    genes_A: list[str]
    genes_B: list[str]
    patient_ids: list[str]
    tmb: pd.Series
    sample_types: pd.DataFrame
    long_A: pd.DataFrame
    info: dict[str, Any]


def _read_maf(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", comment="#", dtype=str, low_memory=False)
    required = {"Hugo_Symbol", "Variant_Classification", "IMPACT", "Tumor_Sample_Barcode"}
    if not required <= set(frame.columns):
        raise ValueError(f"MAF 缺列：{required - set(frame.columns)}")
    return frame


def build_tcga_mutations(
    cesc_dir: str | Path,
    *,
    universe_genes: Iterable[str] | None = None,
    tmb_mb: float = 30.0,
) -> TcgaMutationData:
    files = sorted(Path(cesc_dir).rglob("*.wxs.aliquot_ensemble_masked.maf.gz"))
    if not files:
        raise FileNotFoundError(f"未找到 MAF：{cesc_dir}")
    allowed = set(universe_genes) if universe_genes is not None else None
    frames: list[pd.DataFrame] = []
    for path in files:
        frame = _read_maf(path)
        frame["_file"] = path.name
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    raw["patient_id"] = raw["Tumor_Sample_Barcode"].astype(str).str[:12]
    if "case_id" not in raw.columns:
        raw["case_id"] = None
    classification = raw["Variant_Classification"].astype("string")
    mask_silent = classification.isin(SILENT_CLASSES)
    mask_A = raw["IMPACT"].isin({"HIGH", "MODERATE"}) & ~mask_silent
    hotspot = (
        raw["hotspot"].astype("string").str.upper().isin({"TRUE", "1", "YES"})
        if "hotspot" in raw.columns
        else pd.Series(False, index=raw.index)
    )
    mask_B = mask_A & (hotspot | classification.isin(DRIVER_CLASSES))
    if allowed is not None:
        mask_A &= raw["Hugo_Symbol"].isin(allowed)
        mask_B &= raw["Hugo_Symbol"].isin(allowed)

    long_A = raw.loc[mask_A].copy()
    long_B = raw.loc[mask_B].copy()
    pairs_A = long_A[["patient_id", "Hugo_Symbol"]].dropna().drop_duplicates()
    pairs_B = long_B[["patient_id", "Hugo_Symbol"]].dropna().drop_duplicates()
    patient_ids = sorted(raw["patient_id"].unique().tolist())
    genes_A = sorted(pairs_A["Hugo_Symbol"].unique().tolist())
    genes_B = sorted(pairs_B["Hugo_Symbol"].unique().tolist())

    def matrix(pairs: pd.DataFrame, genes: list[str]) -> np.ndarray:
        patient_index = {patient: i for i, patient in enumerate(patient_ids)}
        gene_index = {gene: j for j, gene in enumerate(genes)}
        output = np.zeros((len(patient_ids), len(genes)), dtype=bool)
        rows = pairs["patient_id"].map(patient_index).to_numpy()
        cols = pairs["Hugo_Symbol"].map(gene_index).to_numpy()
        valid = ~(pd.isna(rows) | pd.isna(cols))
        output[rows[valid].astype(int), cols[valid].astype(int)] = True
        return output

    matrix_A = matrix(pairs_A, genes_A)
    matrix_B = matrix(pairs_B, genes_B)
    counts = long_A.groupby("patient_id").size().reindex(patient_ids, fill_value=0)
    tmb = counts.astype(float) / float(tmb_mb)
    sample_type = raw.groupby("patient_id")["Tumor_Sample_Barcode"].agg(list).map(
        lambda barcodes: sorted(
            {
                str(barcode).split("-")[3][:3]
                for barcode in barcodes
                if len(str(barcode).split("-")) > 3
            }
        )
    )
    sample_types = pd.DataFrame({"patient_id": patient_ids})
    sample_types["sample_type"] = sample_types["patient_id"].map(
        lambda patient: ",".join(sample_type.get(patient, []))
    )
    sample_types["metastatic"] = sample_types["sample_type"].str.contains("06", regex=False)
    long_keep = [
        column
        for column in [
            "patient_id",
            "Tumor_Sample_Barcode",
            "Hugo_Symbol",
            "Variant_Classification",
            "IMPACT",
            "hotspot",
            "case_id",
            "_file",
        ]
        if column in long_A.columns
    ]
    info = {
        "n_files": len(files),
        "raw_rows": int(len(raw)),
        "patients": len(patient_ids),
        "rows_A": int(len(long_A)),
        "genes_A": len(genes_A),
        "rows_B": int(len(long_B)),
        "genes_B": len(genes_B),
        "metastatic_patients": int(sample_types["metastatic"].sum()),
        "median_mutations_A": float(counts.median()),
        "tmb_mb": float(tmb_mb),
    }
    return TcgaMutationData(
        matrix_A=matrix_A,
        matrix_B=matrix_B,
        genes_A=genes_A,
        genes_B=genes_B,
        patient_ids=patient_ids,
        tmb=tmb,
        sample_types=sample_types,
        long_A=long_A[long_keep].sort_values(["patient_id", "Hugo_Symbol"]).reset_index(drop=True),
        info=info,
    )


def write_tcga_mutations(data: TcgaMutationData, directory: str | Path) -> dict[str, Path]:
    directory = Path(directory)
    outputs = {
        "A": io.save_packed_matrix(
            directory / "tcga_mu.npz", data.matrix_A, data.patient_ids, data.genes_A
        ),
        "B": io.save_packed_matrix(
            directory / "tcga_mu_B.npz",
            data.matrix_B,
            data.patient_ids,
            data.genes_B,
        ),
        "tmb": io.write_parquet(
            data.tmb.rename("tmb").rename_axis("patient_id").reset_index(),
            directory / "tcga_tmb.parquet",
        ),
        "sample_types": io.write_parquet(
            data.sample_types, directory / "tcga_sample_types.parquet"
        ),
        "long": io.write_parquet(data.long_A, directory / "tcga_mutation_long.parquet"),
    }
    return outputs


# ---------------------------------------------------------------------------
# clinical
# ---------------------------------------------------------------------------
def build_tcga_clinical(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    with tarfile.open(path, "r:gz") as archive:
        member = next(
            (item for item in archive.getmembers() if item.name.endswith("clinical.tsv")), None
        )
        if member is None:
            raise FileNotFoundError("clinical.tar.gz 内未找到 clinical.tsv")
        handle = archive.extractfile(member)
        if handle is None:
            raise FileNotFoundError("无法解压 clinical.tsv")
        raw = pd.read_csv(handle, sep="\t", dtype=str, low_memory=False)

    id_column = "cases.submitter_id"
    raw[id_column] = raw[id_column].astype(str)
    for column in raw.columns:
        raw[column] = io.normalize_na(raw[column])

    treatment_columns = {
        "treatment_id": "treatments.treatment_id",
        "therapeutic_agents": "treatments.therapeutic_agents",
        "treatment_dose": "treatments.treatment_dose",
        "days_to_treatment_start": "treatments.days_to_treatment_start",
        "days_to_treatment_end": "treatments.days_to_treatment_end",
        "treatment_outcome": "treatments.treatment_outcome",
        "number_of_cycles": "treatments.number_of_cycles",
        "treatment_type": "treatments.treatment_type",
    }
    available = {
        name: source for name, source in treatment_columns.items() if source in raw.columns
    }
    treatments = raw[[id_column, *available.values()]].copy()
    treatments = treatments.rename(columns={v: k for k, v in available.items()})
    treatments = treatments.rename(columns={id_column: "patient_id"})
    treatments = treatments[treatments["therapeutic_agents"].notna()].copy()
    treatments = treatments.drop_duplicates()
    treatments = treatments.sort_values(["patient_id", "treatment_id"]).reset_index(drop=True)

    clinical_columns = {
        "patient_id": id_column,
        "case_id": "cases.case_id",
        "project": "project.project_id",
        "age_at_index": "demographic.age_at_index",
        "sex_at_birth": "demographic.sex_at_birth",
        "vital_status": "demographic.vital_status",
        "days_to_death": "demographic.days_to_death",
        "prior_malignancy": "diagnoses.prior_malignancy",
        "prior_treatment": "diagnoses.prior_treatment",
        "ajcc_pathologic_stage": "diagnoses.ajcc_pathologic_stage",
        "figo_stage": "diagnoses.figo_stage",
        "primary_diagnosis": "diagnoses.primary_diagnosis",
        "tumor_grade": "diagnoses.tumor_grade",
        "histological_type": "diagnoses.morphology",
    }
    available_clinical = {
        name: source for name, source in clinical_columns.items() if source in raw.columns
    }
    clinical = raw[list(available_clinical.values())].rename(
        columns={v: k for k, v in available_clinical.items()}
    )
    clinical = (
        clinical.drop_duplicates("patient_id")
        .sort_values("patient_id")
        .reset_index(drop=True)
    )
    info = {
        "patients": int(clinical["patient_id"].nunique()),
        "clinical_rows": int(len(raw)),
        "treatment_rows": int(len(treatments)),
        "treatment_patients": int(treatments["patient_id"].nunique()),
        "columns": int(len(raw.columns)),
    }
    return clinical, treatments, info


# ---------------------------------------------------------------------------
# CDR / Ding2016
# ---------------------------------------------------------------------------
def build_tcga_cdr(
    path: str | Path, *, cohort: str = "CESC"
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = io.xlsx_sheet_frame(path, "TCGA-CDR")
    frame.columns = [str(column).strip() for column in frame.columns]
    if "type" in frame.columns:
        cohort_frame = frame[frame["type"].astype(str).str.upper() == cohort.upper()].copy()
    else:  # 兼容没有 type 列的情况
        cohort_frame = frame.copy()
    cohort_frame = cohort_frame.drop(
        columns=[column for column in cohort_frame.columns if column.startswith("col_")]
    )
    cohort_frame = io.coerce_frame_for_parquet(cohort_frame)
    info = {
        "rows": int(len(frame)),
        "cohort_rows": int(len(cohort_frame)),
        "pfi_available": int(cohort_frame["PFI"].notna().sum()) if "PFI" in cohort_frame else 0,
        "pfi_events": int(
            pd.to_numeric(cohort_frame.get("PFI"), errors="coerce").fillna(0).eq(1).sum()
        ),
        "os_available": int(cohort_frame["OS"].notna().sum()) if "OS" in cohort_frame else 0,
    }
    return cohort_frame.reset_index(drop=True), info


def build_tcga_ding(
    zip_path: str | Path, *, cohort: str = "CESC"
) -> tuple[pd.DataFrame, dict[str, Any]]:
    with zipfile.ZipFile(zip_path) as archive:
        xlsx_names = [name for name in archive.namelist() if name.endswith(".xlsx")]
        if not xlsx_names:
            raise FileNotFoundError("Ding2016 zip 内无 xlsx")
        payload = archive.read(xlsx_names[0])
    with stdlib_io.BytesIO(payload) as buffer:
        frame = io.xlsx_sheet_frame(buffer, "Table S2", header_row=2)
    frame.columns = [str(column).strip() for column in frame.columns]
    if "bcr_patient_barcode" not in frame.columns:
        raise ValueError("Ding Table S2 缺 bcr_patient_barcode")
    frame = frame[frame["bcr_patient_barcode"].astype(str).str.startswith("TCGA-")].copy()
    cancer = frame["Cancer"].astype(str)
    cohort_frame = frame[cancer.str.contains(rf"\({cohort}\)", regex=True)].copy()
    cohort_frame["response_raw"] = cohort_frame["measure_of_response"].astype(str)
    response = cohort_frame["response_raw"].str.lower()
    cohort_frame["sensitive"] = np.where(
        response.str.contains("complete response|partial response"), 1, np.where(
            response.str.contains("progressive|stable disease"), 0, np.nan
        )
    )
    cohort_frame["is_cisplatin"] = cohort_frame["drug_name"].astype(str).str.lower().eq("cisplatin")
    cohort_frame = io.coerce_frame_for_parquet(cohort_frame)
    info = {
        "rows_total": int(len(frame)),
        "cohort_rows": int(len(cohort_frame)),
        "cohort_patients": int(cohort_frame["bcr_patient_barcode"].nunique()),
        "cisplatin_rows": int(cohort_frame["is_cisplatin"].sum()),
        "cisplatin_patients": int(
            cohort_frame.loc[
                cohort_frame["is_cisplatin"], "bcr_patient_barcode"
            ].nunique()
        ),
        "response_counts": cohort_frame["response_raw"].value_counts().to_dict(),
        "cisplatin_response_counts": cohort_frame.loc[
            cohort_frame["is_cisplatin"], "response_raw"
        ].value_counts().to_dict(),
        "sensitive_ratio_cisplatin": float(
            cohort_frame.loc[cohort_frame["is_cisplatin"], "sensitive"].mean()
        )
        if cohort_frame["is_cisplatin"].any()
        else float("nan"),
    }
    return cohort_frame.reset_index(drop=True), info


__all__ = [
    "TcgaMutationData",
    "build_tcga_cdr",
    "build_tcga_clinical",
    "build_tcga_ding",
    "build_tcga_mutations",
    "fetch_file_index",
    "file_ids_from_manifest",
    "load_file_index",
    "write_tcga_mutations",
]

"""药物特征：SMILES/Morgan FP/分子图/靶点 multi-hot/scaffold。

SMILES 与 RDKit 均为外部依赖；缺失时显式降级并写 mask，禁止静默 0 填充。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kite_drp.data import io

try:  # 可选依赖
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import rdFingerprintGenerator  # type: ignore
    from rdkit.Chem.Scaffolds import MurckoScaffold  # type: ignore

    RDKIT_AVAILABLE = True
except Exception:  # pragma: no cover - 无 rdkit 环境
    Chem = None  # type: ignore
    rdFingerprintGenerator = None  # type: ignore
    MurckoScaffold = None  # type: ignore
    RDKIT_AVAILABLE = False


def load_smiles_table(path: str | Path | None) -> dict[str, str]:
    """读取本地 SMILES 表（列：compound_uid/inchikey/drug_name/smiles）。"""
    if path is None:
        return {}
    candidate = Path(path)
    if not candidate.is_file():
        return {}
    frame = pd.read_csv(candidate, sep=None, engine="python", dtype=str)
    frame.columns = [column.strip().lower() for column in frame.columns]
    result: dict[str, str] = {}
    for row in frame.itertuples(index=False):
        uid = getattr(row, "compound_uid", None)
        inchikey = getattr(row, "inchikey", None)
        smiles = getattr(row, "smiles", None)
        if not smiles:
            continue
        if uid:
            result[str(uid)] = str(smiles)
        elif inchikey:
            result[f"inchikey:{inchikey}"] = str(smiles)
    return result


def morgan_fingerprints(
    compounds: pd.DataFrame, smiles_by_uid: dict[str, str], *, radius: int = 2, bits: int = 2048
) -> tuple[np.ndarray, dict[str, Any]]:
    """返回 ``(n_compounds, bits)`` 的 0/1 矩阵；缺失 SMILES 行全 0 + mask。"""
    matrix = np.zeros((len(compounds), bits), dtype=bool)
    mask = np.zeros(len(compounds), dtype=bool)
    if not RDKIT_AVAILABLE:
        return matrix, {"available": False, "reason": "rdkit not installed", "coverage": 0.0}
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    for index, uid in enumerate(compounds["compound_uid"]):
        smiles = smiles_by_uid.get(uid)
        if not smiles:
            continue
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            continue
        fingerprint = generator.GetFingerprint(molecule)
        matrix[index] = np.frombuffer(fingerprint.ToBitString().encode(), dtype=np.uint8) - ord("0")
        mask[index] = True
    return matrix, {
        "available": True,
        "radius": radius,
        "bits": bits,
        "coverage": float(mask.mean()) if len(mask) else 0.0,
    }


def murcko_scaffolds(compounds: pd.DataFrame, smiles_by_uid: dict[str, str]) -> dict[str, str]:
    """计算 Bemis–Murcko scaffold；RDKit 缺失时返回空。"""
    if not RDKIT_AVAILABLE:
        return {}
    result: dict[str, str] = {}
    for uid in compounds["compound_uid"]:
        smiles = smiles_by_uid.get(uid)
        if not smiles:
            continue
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            continue
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
        except Exception:  # pragma: no cover
            continue
        if scaffold:
            result[str(uid)] = scaffold
    return result


def targets_matrix(
    compounds: pd.DataFrame, genes: list[str]
) -> tuple[np.ndarray, list[str]]:
    """药物 × 靶点基因 0/1 矩阵（列限制在 U_teach_ge）。"""
    gene_index = {gene: i for i, gene in enumerate(genes)}
    matrix = np.zeros((len(compounds), len(genes)), dtype=bool)
    for row_index, genes_for_compound in enumerate(compounds["target_genes"]):
        for gene in genes_for_compound:
            col = gene_index.get(gene)
            if col is not None:
                matrix[row_index, col] = True
    return matrix, genes


def compound_split_groups(compounds: pd.DataFrame) -> tuple[pd.Series, bool]:
    """P3 分组：优先 scaffold，其次 target family，最后化合物本身。"""
    groups = []
    degraded = False
    for row in compounds.itertuples(index=False):
        scaffold = getattr(row, "scaffold", None)
        targets = getattr(row, "target_genes", [])
        if isinstance(scaffold, str) and scaffold:
            groups.append(f"scaffold:{scaffold}")
        elif len(targets) > 0:
            groups.append(f"target:{sorted(targets)[0]}")
            degraded = True
        else:
            groups.append(f"compound:{row.compound_uid}")
            degraded = True
    return pd.Series(groups, index=compounds.index), degraded


def build_drug_features(
    compounds: pd.DataFrame,
    *,
    smiles_path: str | Path | None = None,
    target_genes: list[str] | None = None,
    radius: int = 2,
    bits: int = 2048,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    smiles_by_uid = load_smiles_table(smiles_path)
    # 如果 compounds 表本身已有 smiles，优先使用
    for row in compounds.itertuples(index=False):
        if isinstance(getattr(row, "smiles", None), str) and row.smiles:
            smiles_by_uid.setdefault(row.compound_uid, row.smiles)
    morgan, morgan_info = morgan_fingerprints(compounds, smiles_by_uid, radius=radius, bits=bits)
    scaffolds = murcko_scaffolds(compounds, smiles_by_uid)
    genes = sorted(target_genes or [])
    targets, genes = targets_matrix(compounds, genes)
    arrays = {
        "compound_uid": compounds["compound_uid"].to_numpy(dtype=np.str_),
        "morgan": morgan,
        "targets": targets,
        "target_genes": np.asarray(genes, dtype=np.str_),
        "scaffold": np.asarray(
            [scaffolds.get(uid, "") for uid in compounds["compound_uid"]], dtype=np.str_
        ),
        "has_smiles": np.asarray(
            [uid in smiles_by_uid for uid in compounds["compound_uid"]], dtype=bool
        ),
        "smiles": np.asarray(
            [smiles_by_uid.get(uid, "") for uid in compounds["compound_uid"]], dtype=np.str_
        ),
    }
    coverage = {
        "n_compounds": int(len(compounds)),
        "has_inchikey": int(compounds["has_inchikey"].sum()),
        "has_target": int(compounds["has_target"].sum()),
        "has_smiles": int(arrays["has_smiles"].sum()),
        "has_scaffold": int((arrays["scaffold"] != "").sum()),
        "morgan": morgan_info,
        "rdkit_available": RDKIT_AVAILABLE,
        "target_genes": len(genes),
    }
    return arrays, coverage


__all__ = [
    "RDKIT_AVAILABLE",
    "build_drug_features",
    "compound_split_groups",
    "load_smiles_table",
    "morgan_fingerprints",
    "murcko_scaffolds",
    "targets_matrix",
]

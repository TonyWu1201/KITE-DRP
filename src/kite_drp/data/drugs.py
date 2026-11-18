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


#: 原子特征维度（计划书 D3：43 原子类型 + 11 度 + 7 隐式价 + 1 电荷 + 1 自由基
#: + 5 杂化 + 1 芳香 + 5 氢数 + 1 环 + 2 手性 = 77）。
ATOM_FEATURE_DIM = 77
#: 键特征维度（单/双/三/芳香）。
BOND_FEATURE_DIM = 4

ATOM_TYPES: tuple[str, ...] = (
    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca", "Fe",
    "As", "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag", "Pd", "Co",
    "Se", "Ti", "Zn", "H", "Li", "Ge", "Cu", "Au", "Ni", "Cd", "In", "Mn",
    "Zr", "Cr", "Pt", "Hg", "Pb",
)  # fmt: skip


def _one_hot(index: int, size: int) -> np.ndarray:
    vector = np.zeros(size, dtype=np.float32)
    if 0 <= index < size:
        vector[index] = 1.0
    else:  # 越界值落到最后一格（misc/other）
        vector[size - 1] = 1.0
    return vector


def _atom_feature(atom: Any) -> np.ndarray:
    """RDKit 原子 → 77 维特征（DeepCDR 口径，计划书 §4.3 D3）。"""
    from rdkit import Chem  # type: ignore

    symbol = atom.GetSymbol()
    atom_type = ATOM_TYPES.index(symbol) if symbol in ATOM_TYPES else len(ATOM_TYPES) - 1
    degree = min(int(atom.GetDegree()), 10)
    try:  # RDKit >= 2026 推荐 GetValence；旧版本回退
        valence_raw = int(atom.GetValence(Chem.ValenceType.IMPLICIT))
    except AttributeError:  # pragma: no cover - 旧 RDKit
        valence_raw = int(atom.GetImplicitValence())
    valence = min(valence_raw, 6)
    charge = float(np.clip(atom.GetFormalCharge(), -5, 5)) / 5.0
    radical = float(np.clip(atom.GetNumRadicalElectrons(), 0, 4)) / 4.0
    hybridization = atom.GetHybridization()
    hybridization_buckets = (
        Chem.HybridizationType.SP,
        Chem.HybridizationType.SP2,
        Chem.HybridizationType.SP3,
        Chem.HybridizationType.SP3D,
    )
    if hybridization in hybridization_buckets:
        hybridization_index = hybridization_buckets.index(hybridization)
    else:
        hybridization_index = 4  # SP3D2 / 其他统一落最后一格
    aromatic = float(bool(atom.GetIsAromatic()))
    num_h = min(int(atom.GetTotalNumHs()), 4)
    in_ring = float(bool(atom.IsInRing()))
    chirality = 0 if atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED else 1

    return np.concatenate(
        [
            _one_hot(atom_type, len(ATOM_TYPES)),
            _one_hot(degree, 11),
            _one_hot(valence, 7),
            np.asarray([charge, radical], dtype=np.float32),
            _one_hot(hybridization_index, 5),
            np.asarray([aromatic], dtype=np.float32),
            _one_hot(num_h, 5),
            np.asarray([in_ring], dtype=np.float32),
            _one_hot(chirality, 2),
        ]
    ).astype(np.float32)


def _bond_feature(bond: Any) -> np.ndarray:
    order = float(bond.GetBondTypeAsDouble())
    feature = np.zeros(BOND_FEATURE_DIM, dtype=np.float32)
    if order == 1.0:
        feature[0] = 1.0
    elif order == 2.0:
        feature[1] = 1.0
    elif order == 3.0:
        feature[2] = 1.0
    elif bond.GetIsAromatic() or order == 1.5:
        feature[3] = 1.0
    else:
        feature[0] = 1.0  # 未知键型按单键处理
    return feature


def molecular_graphs(
    compounds: pd.DataFrame, smiles_by_uid: dict[str, str]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """构建药物分子图（ragged 存储，供 GIN 使用）。

    返回的数组按 ``compound_uid`` 行对齐；每个化合物的原子/边通过
    ``graph_atom_ptr`` / ``graph_edge_ptr`` 切片（边为双向，全局原子索引）。
    RDKit 缺失时返回空数组 + ``has_graph=False``，禁止静默造图。
    """
    n_compounds = len(compounds)
    node_features: list[np.ndarray] = []
    edge_index: list[tuple[int, int]] = []
    edge_features: list[np.ndarray] = []
    atom_ptr = [0]
    edge_ptr = [0]
    has_graph = np.zeros(n_compounds, dtype=bool)

    if RDKIT_AVAILABLE:
        for index, uid in enumerate(compounds["compound_uid"]):
            smiles = smiles_by_uid.get(str(uid))
            molecule = Chem.MolFromSmiles(smiles) if smiles else None
            if molecule is None:
                atom_ptr.append(atom_ptr[-1])
                edge_ptr.append(edge_ptr[-1])
                continue
            base = atom_ptr[-1]
            for atom in molecule.GetAtoms():
                node_features.append(_atom_feature(atom))
            for bond in molecule.GetBonds():
                left = base + bond.GetBeginAtomIdx()
                right = base + bond.GetEndAtomIdx()
                feature = _bond_feature(bond)
                edge_index.extend([(left, right), (right, left)])
                edge_features.extend([feature, feature])
            atom_ptr.append(atom_ptr[-1] + molecule.GetNumAtoms())
            edge_ptr.append(edge_ptr[-1] + 2 * molecule.GetNumBonds())
            has_graph[index] = True

    arrays = {
        "graph_atom_ptr": np.asarray(atom_ptr, dtype=np.int32),
        "graph_edge_ptr": np.asarray(edge_ptr, dtype=np.int32),
        "graph_node_feat": (
            np.asarray(node_features, dtype=np.float32)
            if node_features
            else np.zeros((0, ATOM_FEATURE_DIM), dtype=np.float32)
        ),
        "graph_edge_index": (
            np.asarray(edge_index, dtype=np.int32).T
            if edge_index
            else np.zeros((2, 0), dtype=np.int32)
        ),
        "graph_edge_feat": (
            np.asarray(edge_features, dtype=np.float32)
            if edge_features
            else np.zeros((0, BOND_FEATURE_DIM), dtype=np.float32)
        ),
        "graph_has_graph": has_graph,
    }
    info = {
        "available": RDKIT_AVAILABLE,
        "coverage": float(has_graph.mean()) if n_compounds else 0.0,
        "n_graphs": int(has_graph.sum()),
        "node_feat_dim": ATOM_FEATURE_DIM,
        "bond_feat_dim": BOND_FEATURE_DIM,
        "n_atoms": int(arrays["graph_node_feat"].shape[0]),
        "n_edges": int(arrays["graph_edge_index"].shape[1]),
    }
    return arrays, info


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
    """P3 分组：优先 scaffold；完全缺失才降级 target-family/compound。"""
    scaffolds = [
        value
        for value in compounds.get("scaffold", pd.Series(dtype=object)).tolist()
        if isinstance(value, str) and value
    ]
    if not scaffolds:
        degraded = True
    else:
        degraded = False
    groups = []
    for row in compounds.itertuples(index=False):
        scaffold = getattr(row, "scaffold", None)
        targets = getattr(row, "target_genes", [])
        if isinstance(scaffold, str) and scaffold:
            groups.append(f"scaffold:{scaffold}")
        elif len(targets) > 0:
            groups.append(f"target:{sorted(targets)[0]}")
        else:
            groups.append(f"compound:{row.compound_uid}")
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
    graphs, graph_info = molecular_graphs(compounds, smiles_by_uid)
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
        **graphs,
    }
    coverage = {
        "n_compounds": int(len(compounds)),
        "has_inchikey": int(compounds["has_inchikey"].sum()),
        "has_target": int(compounds["has_target"].sum()),
        "has_smiles": int(arrays["has_smiles"].sum()),
        "has_scaffold": int((arrays["scaffold"] != "").sum()),
        "morgan": morgan_info,
        "graphs": graph_info,
        "rdkit_available": RDKIT_AVAILABLE,
        "target_genes": len(genes),
    }
    return arrays, coverage


__all__ = [
    "ATOM_FEATURE_DIM",
    "BOND_FEATURE_DIM",
    "RDKIT_AVAILABLE",
    "build_drug_features",
    "compound_split_groups",
    "load_smiles_table",
    "molecular_graphs",
    "morgan_fingerprints",
    "murcko_scaffolds",
    "targets_matrix",
]

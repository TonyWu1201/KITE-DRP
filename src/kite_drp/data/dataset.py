"""模型侧 Dataset 契约（G0 冻结接口）。

torch 不是数据管线的必需依赖；有 torch 时这些类可直接作为 Dataset 使用，
无 torch 时退化为普通可迭代对象，便于 CI 与数据侧单测。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kite_drp.data import io
from kite_drp.data.pairs import load_pairs, sensitivity_class, sensitivity_labels
from kite_drp.data.splits import fold_masks, load_split

try:  # pragma: no cover - 可选依赖
    from torch.utils.data import Dataset as _TorchDataset

    _BASE = _TorchDataset
except Exception:  # pragma: no cover
    _BASE = object


@dataclass
class FitStats:
    label_scaler: pd.DataFrame
    expr_scaler: pd.DataFrame
    cnv_scaler: pd.DataFrame
    hvg: list[str]
    scope: dict[str, Any]
    cnv_clip: dict[str, float] | None = None

    @classmethod
    def load(cls, fold_dir: str | Path) -> FitStats:
        directory = Path(fold_dir)
        clip_path = directory / "cnv_clip.json"
        return cls(
            label_scaler=io.read_parquet(directory / "label_scaler.parquet"),
            expr_scaler=io.read_parquet(directory / "expr_scaler.parquet"),
            cnv_scaler=io.read_parquet(directory / "cnv_scaler.parquet"),
            hvg=list(io.read_json(directory / "hvg.json")["hvg"]),
            scope=io.read_json(directory / "fit_scope.json")["fit_scope"],
            cnv_clip=(
                {
                    "low": float(io.read_json(clip_path)["low"]),
                    "high": float(io.read_json(clip_path)["high"]),
                }
                if clip_path.is_file()
                else None
            ),
        )

    def standardize(self, values: np.ndarray, genes: list[str], kind: str = "expr") -> np.ndarray:
        scaler = self.expr_scaler if kind == "expr" else self.cnv_scaler
        frame = scaler.set_index("gene").reindex(genes)
        mean = frame["mean"].to_numpy(dtype=np.float32)
        std = frame["std"].to_numpy(dtype=np.float32)
        valid = frame["valid"].fillna(False).to_numpy(dtype=bool)
        if kind == "cnv" and self.cnv_clip is not None:
            values = np.clip(values, self.cnv_clip["low"], self.cnv_clip["high"])
        result = (values - mean) / np.where(std == 0, np.nan, std)
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        result[:, ~valid] = 0.0
        return result


class TeacherDataset(_BASE):  # type: ignore[misc]
    """读取 processed/teacher_inputs.npz + 对应 fold 的 fit_stats。"""

    def __init__(
        self,
        processed_dir: str | Path,
        *,
        split_id: str,
        fold: int,
        pool: str = "M-716",
        fit_stats_root: str | Path | None = None,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.pool = pool
        data = io.read_npz(self.processed_dir / "teacher_inputs.npz")
        index = io.read_json(self.processed_dir / "teacher_index.json")
        pool_index = io.read_json(self.processed_dir / "pool_index.json")
        pool_ids = set(pool_index[pool])
        keep = [i for i, cell in enumerate(data["cell_line_uid"]) if str(cell) in pool_ids]
        self.cell_line_uid = [str(data["cell_line_uid"][i]) for i in keep]
        self.ge = data["ge"][keep]
        self.cnv = data["cnv"][keep]
        self.mu_A = data["mu_A"][keep]
        self.mu_B = data["mu_B"][keep]
        self.cnv_observed_mask = data["cnv_observed_mask"][keep]
        self.eligible_mask = data["eligible_mask"][keep]
        self.ge_genes = [str(gene) for gene in index["ge_genes"]]
        self.cnv_genes = [str(gene) for gene in index["cnv_genes"]]
        self.mu_genes = [str(gene) for gene in index["mu_A_genes"]]
        root = Path(fit_stats_root or self.processed_dir / "fit_stats")
        self.fit_stats = FitStats.load(root / split_id / f"fold{fold}")
        self.ge = self.fit_stats.standardize(self.ge, self.ge_genes, kind="expr")
        self.cnv = self.fit_stats.standardize(self.cnv, self.cnv_genes, kind="cnv")

    def __len__(self) -> int:
        return len(self.cell_line_uid)

    def __getitem__(self, index: int) -> dict[str, Any]:
        cell = self.cell_line_uid[index]
        return {
            "cell_line_uid": cell,
            "ge": self.ge[index],
            "cnv": self.cnv[index],
            "mu_A": self.mu_A[index],
            "mu_B": self.mu_B[index],
            "cnv_observed_mask": self.cnv_observed_mask[index],
            "eligible_mask": self.eligible_mask[index],
        }


class StudentDataset(_BASE):  # type: ignore[misc]
    """读取学生输入（panel 突变/TMB/通路权重）；表达分数在运行时由 fit_stats 提供。"""

    def __init__(
        self,
        processed_dir: str | Path,
        *,
        split_id: str,
        fold: int,
        fit_stats_root: str | Path | None = None,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        panel, panel_ids, panel_genes = io.load_packed_matrix(
            self.processed_dir / "student_inputs" / "panel_mutations_A.npz", name="matrix"
        )
        self.panel = panel
        self.cell_line_uid = panel_ids
        self.panel_genes = panel_genes
        burden = io.read_parquet(
            self.processed_dir / "student_inputs" / "mut_burden.parquet"
        ).set_index("cell_line_uid")
        self.mut_burden = burden.reindex(self.cell_line_uid)["mut_burden_A"].to_numpy(
            dtype=np.float32
        )
        weights = io.read_npz(self.processed_dir / "student_inputs" / "pathway_weights.npz")
        self.pathway_weights = weights["weights"]
        self.pathway_ids = [str(value) for value in weights["pathway_ids"]]
        self.pathway_genes = [str(value) for value in weights["genes"]]
        root = Path(fit_stats_root or self.processed_dir / "fit_stats")
        self.fit_stats = FitStats.load(root / split_id / f"fold{fold}")
        panel_path = root / split_id / f"fold{fold}" / "panel_genes.json"
        if panel_path.is_file():
            fold_panel = set(io.read_json(panel_path)["genes"])
            self.panel_columns = [
                index for index, gene in enumerate(self.panel_genes) if gene in fold_panel
            ]
        else:
            self.panel_columns = list(range(len(self.panel_genes)))

    def __len__(self) -> int:
        return len(self.cell_line_uid)

    def pathway_scores(self, ge: np.ndarray) -> np.ndarray:
        """GE 先 fold 内 z，再按冻结权重做 mean-z 通路打分。"""
        ge_z = self.fit_stats.standardize(ge, self.pathway_genes, kind="expr")
        return ge_z @ self.pathway_weights.T.astype(np.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "cell_line_uid": self.cell_line_uid[index],
            "mu": self.panel[index][self.panel_columns].astype(np.float32),
            "mut_burden": np.asarray([self.mut_burden[index]], dtype=np.float32),
        }


@dataclass
class DrugFeatureStore:
    """`processed/drug_features.npz` 的只读视图（含 GIN 分子图 ragged 数组）。"""

    compound_uid: list[str]
    morgan: np.ndarray
    targets: np.ndarray
    target_genes: list[str]
    smiles: list[str]
    has_smiles: np.ndarray
    scaffold: list[str]
    graph_atom_ptr: np.ndarray | None = None
    graph_edge_ptr: np.ndarray | None = None
    graph_node_feat: np.ndarray | None = None
    graph_edge_index: np.ndarray | None = None
    graph_edge_feat: np.ndarray | None = None
    graph_has_graph: np.ndarray | None = None

    @classmethod
    def load(cls, processed_dir: str | Path) -> DrugFeatureStore:
        path = Path(processed_dir) / "drug_features.npz"
        if not path.is_file():
            raise FileNotFoundError(
                f"缺少 {path}；请运行 `kite-drp data build --stage drug`"
            )
        arrays = io.read_npz(path)
        return cls(
            compound_uid=[str(value) for value in arrays["compound_uid"]],
            morgan=arrays["morgan"],
            targets=arrays["targets"],
            target_genes=[str(value) for value in arrays["target_genes"]],
            smiles=[str(value) for value in arrays["smiles"]],
            has_smiles=arrays["has_smiles"].astype(bool),
            scaffold=[str(value) for value in arrays["scaffold"]],
            graph_atom_ptr=arrays.get("graph_atom_ptr"),
            graph_edge_ptr=arrays.get("graph_edge_ptr"),
            graph_node_feat=arrays.get("graph_node_feat"),
            graph_edge_index=arrays.get("graph_edge_index"),
            graph_edge_feat=arrays.get("graph_edge_feat"),
            graph_has_graph=(
                arrays["graph_has_graph"].astype(bool)
                if "graph_has_graph" in arrays
                else None
            ),
        )

    @property
    def index(self) -> dict[str, int]:
        return {uid: position for position, uid in enumerate(self.compound_uid)}

    @property
    def has_molecular_graphs(self) -> bool:
        return self.graph_has_graph is not None and bool(self.graph_has_graph.any())


class TeacherPairDataset(_BASE):  # type: ignore[misc]
    """教师 pair 数据集：只读 processed + splits + fit_stats（不碰 canonical）。

    这是「clone 下来也能训练」的入口：``teacher_pairs.parquet`` 提供标签，
    ``teacher_inputs.npz`` 提供细胞系组学，``drug_features.npz`` 提供药物特征，
    ``fit_stats`` 提供 fold 内标签/组学统计量。
    """

    def __init__(
        self,
        processed_dir: str | Path,
        *,
        split_id: str,
        fold: int,
        pool: str = "M-716",
        split_dir: str | Path | None = None,
        fit_stats_root: str | Path | None = None,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        pairs = load_pairs(self.processed_dir)
        if pool == "M-509":
            pairs = pairs[pairs["in_main_pool"].astype(bool)]
        else:
            pairs = pairs[pairs["has_teacher_omics"].astype(bool)]
        self.all_pairs = pairs.reset_index(drop=True)

        root = Path(fit_stats_root or self.processed_dir / "fit_stats")
        self.fit_stats = FitStats.load(root / split_id / f"fold{fold}")
        self.split_id = split_id
        self.fold = int(fold)
        self.pool = pool

        splits_dir = Path(split_dir or Path(self.processed_dir).parent / "splits")
        split_path = splits_dir / f"{split_id}.json"
        if not split_path.is_file():
            raise FileNotFoundError(f"缺少 split 文件：{split_path}")
        self.split = load_split(split_path)
        fold_def = next(
            (item for item in self.split["folds"] if int(item["fold"]) == int(fold)),
            None,
        )
        if fold_def is None:
            raise KeyError(f"split {split_id} 缺少 fold {fold}")
        fold_def = dict(fold_def)
        fold_def["_seed"] = self.split["seed"]
        masks = fold_masks(fold_def, self.all_pairs, self.split["protocol"])
        self.split_label = np.full(len(self.all_pairs), "unused", dtype=object)
        for name in ("train", "val", "test"):
            self.split_label[masks[name].to_numpy()] = name

        # 细胞/药物索引（与 teacher_inputs / drug_features 行对齐）
        teacher = io.read_npz(self.processed_dir / "teacher_inputs.npz")
        cell_uids = [str(value) for value in teacher["cell_line_uid"]]
        self.cell_index = {uid: position for position, uid in enumerate(cell_uids)}
        drugs = DrugFeatureStore.load(self.processed_dir)
        self.drug_store = drugs
        self.drug_index = drugs.index

        keep = self.all_pairs["cell_line_uid"].isin(self.cell_index).to_numpy(dtype=bool)
        keep = keep & self.all_pairs["compound_uid"].isin(self.drug_index).to_numpy(dtype=bool)
        if not keep.any():
            raise ValueError("pair 表与 teacher_inputs/drug_features 无交集")
        self.pairs = self.all_pairs.loc[keep].reset_index(drop=True)
        self.split_label = self.split_label[keep]

        self.y = sensitivity_labels(self.pairs, self.fit_stats.label_scaler)
        y_cls, cls_mask = sensitivity_class(self.pairs, self.fit_stats.label_scaler)
        self.y_cls = y_cls
        self.cls_mask = cls_mask
        self.cell_idx = self.pairs["cell_line_uid"].map(self.cell_index).to_numpy(dtype=np.int64)
        self.drug_idx = self.pairs["compound_uid"].map(self.drug_index).to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.pairs)

    def indices(self, split: str) -> np.ndarray:
        if split == "all":
            return np.arange(len(self.pairs), dtype=np.int64)
        return np.flatnonzero(self.split_label == split)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "cell_line_uid": str(self.pairs.at[index, "cell_line_uid"]),
            "compound_uid": str(self.pairs.at[index, "compound_uid"]),
            "cell_idx": int(self.cell_idx[index]),
            "drug_idx": int(self.drug_idx[index]),
            "y": float(self.y[index]),
            "y_cls": float(self.y_cls[index]),
            "cls_mask": bool(self.cls_mask[index]),
            "split": str(self.split_label[index]),
        }


class TcgaEvalDataset(_BASE):  # type: ignore[misc]
    """TCGA 评估集（purpose=eval 红线；训练侧禁止加载）。"""

    def __init__(self, mu_path: str | Path, *, patient_ids: list[str] | None = None) -> None:
        matrix, rows, genes = io.load_packed_matrix(mu_path, name="matrix")
        if patient_ids is not None:
            index = {patient: i for i, patient in enumerate(rows)}
            keep = [index[patient] for patient in patient_ids if patient in index]
            matrix = matrix[keep]
            rows = [rows[i] for i in keep]
        self.mu = matrix
        self.patient_ids = rows
        self.genes = genes

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {"patient_id": self.patient_ids[index], "mu": self.mu[index]}


__all__ = [
    "DrugFeatureStore",
    "FitStats",
    "StudentDataset",
    "TcgaEvalDataset",
    "TeacherDataset",
    "TeacherPairDataset",
]

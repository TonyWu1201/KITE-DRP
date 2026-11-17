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


class TcgaEvalDataset(_BASE):  # type: ignore[misc]
    """TCGA 评估集（purpose=eval 红线；训练侧禁止加载）。"""

    purpose = "eval"

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


__all__ = ["FitStats", "StudentDataset", "TcgaEvalDataset", "TeacherDataset"]

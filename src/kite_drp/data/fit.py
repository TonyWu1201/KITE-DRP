"""FitContext：fold 内统计量（标签/表达/CNV/HVG）的唯一入口。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Self

import numpy as np
import pandas as pd

from kite_drp.data import io
from kite_drp.data.labels import compound_stats
from kite_drp.data.omics.expression import select_hvg
from kite_drp.data.splits import fold_masks


@dataclass(frozen=True)
class FitScope:
    split_id: str
    fold_id: int
    train_index_hash: str
    label_hash: str

    def fingerprint(self) -> str:
        return io.hash_bytes(
            io.canonical_json(
                {
                    "split_id": self.split_id,
                    "fold_id": self.fold_id,
                    "train_index_hash": self.train_index_hash,
                    "label_hash": self.label_hash,
                }
            ).encode()
        )


class FoldSafeTransform(Protocol):
    def fit(self, frame: pd.DataFrame, scope: FitScope) -> Self: ...

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame: ...

    def fingerprint(self) -> str: ...


class Standardizer:
    """逐列 mean/std 标准化；fit 只接受训练 fold 的样本。"""

    def __init__(self, *, ddof: int = 1) -> None:
        self.ddof = ddof
        self.mean: pd.Series | None = None
        self.std: pd.Series | None = None
        self.valid: pd.Series | None = None
        self.scope: FitScope | None = None

    def fit(self, frame: pd.DataFrame, scope: FitScope) -> Self:
        self.scope = scope
        self.mean = frame.mean(axis=0)
        self.std = frame.std(axis=0, ddof=self.ddof)
        self.valid = self.std > 1e-8
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.mean is None or self.std is None or self.valid is None:
            raise RuntimeError("Standardizer 尚未 fit")
        columns = [column for column in frame.columns if column in self.mean.index]
        standardized = (frame[columns] - self.mean[columns]) / self.std[columns].replace(0, np.nan)
        standardized = standardized.fillna(0.0).astype(np.float32)
        invalid = [column for column in columns if not bool(self.valid[column])]
        if invalid:
            standardized[invalid] = 0.0
        return standardized

    def fingerprint(self) -> str:
        if self.mean is None or self.std is None:
            raise RuntimeError("Standardizer 尚未 fit")
        payload = pd.DataFrame({"mean": self.mean, "std": self.std}).reset_index()
        return io.hash_dataframe(payload, sort_by=["index"])

    def scope_dict(self) -> dict[str, Any]:
        if self.scope is None:
            return {}
        return {
            "split_id": self.scope.split_id,
            "fold_id": self.scope.fold_id,
            "train_index_hash": self.scope.train_index_hash,
            "label_hash": self.scope.label_hash,
        }


def _train_labels(fold: dict[str, Any], labels: pd.DataFrame, protocol: str, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    local = dict(fold)
    local["_seed"] = seed
    masks = fold_masks(local, labels, protocol)
    if "mapped" in labels.columns:
        masks = {key: value & labels["mapped"].astype(bool) for key, value in masks.items()}
    return labels.loc[masks["train"]].copy(), masks["train"]


def _write_scaler(stats: pd.DataFrame, path: Path) -> None:
    io.write_parquet(stats, path)


def fit_stats_for_fold(
    *,
    split: dict[str, Any],
    fold: dict[str, Any],
    labels: pd.DataFrame,
    expression: pd.DataFrame,
    cnv: pd.DataFrame,
    out_dir: str | Path,
    seed: int,
    min_obs_per_compound: int = 30,
    n_hvg: int = 2000,
    ddof: int = 1,
    mutation_matrix: np.ndarray | None = None,
    mutation_model_ids: list[str] | None = None,
    mutation_genes: list[str] | None = None,
    panel_min_frequency: float = 0.01,
    panel_max_genes: int = 500,
    external_panel: list[str] | None = None,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    protocol = str(split["protocol"])
    train_labels, _ = _train_labels(fold, labels, protocol, seed)
    train_ids = sorted(train_labels["cell_line_uid"].unique().tolist())
    train_index_hash = io.hash_bytes("\n".join(train_ids).encode())
    label_hash = io.hash_dataframe(
        train_labels[["cell_line_uid", "compound_uid", "ln_ic50"]],
        sort_by=["cell_line_uid", "compound_uid"],
    )
    scope = FitScope(
        split_id=f"{protocol}_s{seed}",
        fold_id=int(fold["fold"]),
        train_index_hash=train_index_hash,
        label_hash=label_hash,
    )

    label_stats = compound_stats(train_labels)
    label_stats = label_stats[label_stats["n_train"] >= min_obs_per_compound].copy()
    label_stats["std"] = label_stats["std"].replace(0, np.nan).fillna(1.0)
    label_stats["fit_scope_hash"] = scope.fingerprint()
    _write_scaler(label_stats, out / "label_scaler.parquet")

    expr_ids = sorted(set(train_ids) & set(expression.index))
    cnv_ids = sorted(set(train_ids) & set(cnv.index))
    expr_scaler = Standardizer(ddof=ddof).fit(expression.loc[expr_ids], scope)
    cnv_train = cnv.loc[cnv_ids]
    clip_low, clip_high = (
        np.nanpercentile(cnv_train.to_numpy(), [0.1, 99.9]) if len(cnv_ids) else (np.nan, np.nan)
    )
    cnv_clipped = cnv_train.clip(lower=float(clip_low), upper=float(clip_high))
    cnv_scaler = Standardizer(ddof=ddof).fit(cnv_clipped, scope)
    io.write_json(
        out / "cnv_clip.json",
        {
            "low": float(clip_low),
            "high": float(clip_high),
            "percentile": [0.1, 99.9],
            "n_cells": len(cnv_ids),
            "fit_scope": scope.__dict__,
        },
    )
    expr_frame = pd.DataFrame(
        {
            "gene": expr_scaler.mean.index,
            "mean": expr_scaler.mean.to_numpy(),
            "std": expr_scaler.std.to_numpy(),
            "valid": expr_scaler.valid.to_numpy(),
        }
    )
    cnv_frame = pd.DataFrame(
        {
            "gene": cnv_scaler.mean.index,
            "mean": cnv_scaler.mean.to_numpy(),
            "std": cnv_scaler.std.to_numpy(),
            "valid": cnv_scaler.valid.to_numpy(),
        }
    )
    _write_scaler(expr_frame, out / "expr_scaler.parquet")
    _write_scaler(cnv_frame, out / "cnv_scaler.parquet")

    hvg = select_hvg(expression, expr_ids, n_top=n_hvg)
    io.write_json(
        out / "hvg.json",
        {"hvg": hvg, "n": len(hvg), "fit_scope": scope.__dict__},
    )

    if external_panel:
        panel_genes = sorted(external_panel)
        panel_source = "external"
    elif mutation_matrix is not None and mutation_model_ids and mutation_genes:
        model_index = {model: i for i, model in enumerate(mutation_model_ids)}
        rows = [model_index[cell] for cell in train_ids if cell in model_index]
        if rows:
            frequency = mutation_matrix[rows].mean(axis=0)
            order = np.argsort(-frequency)
            panel_genes = [
                mutation_genes[index]
                for index in order
                if frequency[index] >= panel_min_frequency
            ][:panel_max_genes]
            panel_genes = sorted(panel_genes)
        else:
            panel_genes = []
        panel_source = "frequency_train_fold"
    else:
        panel_genes = []
        panel_source = "missing"
    io.write_json(
        out / "panel_genes.json",
        {
            "genes": panel_genes,
            "n": len(panel_genes),
            "source": panel_source,
            "fit_scope": scope.__dict__,
        },
    )
    meta = {
        "fit_scope": {
            "split_id": scope.split_id,
            "fold_id": scope.fold_id,
            "train_index_hash": scope.train_index_hash,
            "label_hash": scope.label_hash,
            "fingerprint": scope.fingerprint(),
        },
        "n_train_cells": len(train_ids),
        "n_expr_cells": len(expr_ids),
        "n_cnv_cells": len(cnv_ids),
        "n_train_labels": int(len(train_labels)),
        "n_compounds_scaled": int(len(label_stats)),
        "n_compounds_below_min_obs": int(
            train_labels["compound_uid"].nunique() - len(label_stats)
        ),
        "expr_fingerprint": expr_scaler.fingerprint(),
        "cnv_fingerprint": cnv_scaler.fingerprint(),
        "cnv_clip": {"low": float(clip_low), "high": float(clip_high)},
        "hvg_fingerprint": io.hash_bytes("\n".join(hvg).encode()),
        "panel_source": panel_source,
        "panel_genes": len(panel_genes),
        "panel_fingerprint": io.hash_bytes("\n".join(panel_genes).encode()),
    }
    io.write_json(out / "fit_scope.json", meta)
    return meta


def fit_stats_for_split(
    *,
    split: dict[str, Any],
    labels: pd.DataFrame,
    expression: pd.DataFrame,
    cnv: pd.DataFrame,
    root: str | Path,
    seed: int | None = None,
    min_obs_per_compound: int = 30,
    n_hvg: int = 2000,
    force: bool = False,
    mutation_matrix: np.ndarray | None = None,
    mutation_model_ids: list[str] | None = None,
    mutation_genes: list[str] | None = None,
    external_panel: list[str] | None = None,
) -> dict[str, Any]:
    seed = int(seed if seed is not None else split["seed"])
    protocol = str(split["protocol"])
    split_dir = Path(root) / f"{protocol}_s{seed}"
    summary: dict[str, Any] = {"split_id": f"{protocol}_s{seed}", "folds": {}}
    for fold in split["folds"]:
        fold_dir = split_dir / f"fold{int(fold['fold'])}"
        if fold_dir.is_dir() and (fold_dir / "fit_scope.json").is_file() and not force:
            summary["folds"][int(fold["fold"])] = "skipped"
            continue
        meta = fit_stats_for_fold(
            split=split,
            fold=fold,
            labels=labels,
            expression=expression,
            cnv=cnv,
            out_dir=fold_dir,
            seed=seed,
            min_obs_per_compound=min_obs_per_compound,
            n_hvg=n_hvg,
            mutation_matrix=mutation_matrix,
            mutation_model_ids=mutation_model_ids,
            mutation_genes=mutation_genes,
            external_panel=external_panel,
        )
        summary["folds"][int(fold["fold"])] = {
            "fingerprint": meta["fit_scope"]["fingerprint"],
            "n_train_cells": meta["n_train_cells"],
            "n_compounds_scaled": meta["n_compounds_scaled"],
        }
    io.write_json(split_dir / "summary.json", summary)
    return summary


def load_fit_scope(fold_dir: str | Path) -> FitScope:
    payload = io.read_json(Path(fold_dir) / "fit_scope.json")["fit_scope"]
    return FitScope(
        split_id=payload["split_id"],
        fold_id=int(payload["fold_id"]),
        train_index_hash=payload["train_index_hash"],
        label_hash=payload["label_hash"],
    )


__all__ = [
    "FitScope",
    "FoldSafeTransform",
    "Standardizer",
    "fit_stats_for_fold",
    "fit_stats_for_split",
    "load_fit_scope",
]

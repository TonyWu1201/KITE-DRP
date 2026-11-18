"""教师蒸馏缓存导出（计划书 §6.3 S0 产出）。

只缓存 **train pair** 的软标签与潜向量；cell/drug embedding 可含全部细胞/药物，
但 val/test 的软标签不落盘（避免学生训练时误用）。

用法：
    uv run python -m kite_drp.train.cache_teacher \
        --run experiments/2026-11-19_01-22-54_teacher_train --fold 0 \
        --split p1_lco_s0 --out data/interim/teacher_cache/t0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from kite_drp.common.paths import REPO_ROOT
from kite_drp.data import io
from kite_drp.data.dataset import TeacherPairDataset
from kite_drp.data.similarity import SimilarityConfig
from kite_drp.train.data import build_pair_tensors, make_loader
from kite_drp.train.engine import move_batch, resolve_device
from kite_drp.train.features import (
    build_cell_features,
    build_drug_features,
    build_fold_similarity,
)
from kite_drp.train.teacher_train import _build_model


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "resolved_config.yaml"
    if not path.is_file():
        return {}
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    return payload if isinstance(payload, dict) else {}


@torch.no_grad()
def export_cache(
    *,
    run_dir: str | Path,
    fold: int,
    split_id: str,
    out_dir: str | Path,
    processed_dir: str | Path | None = None,
    splits_dir: str | Path | None = None,
    batch_size: int = 512,
) -> dict[str, Any]:
    run = Path(run_dir)
    processed = Path(processed_dir or REPO_ROOT / "data" / "processed")
    splits = Path(splits_dir or REPO_ROOT / "data" / "splits")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    spec = io.read_json(run / f"fold{fold}" / "model_spec.json")
    config = _load_run_config(run)
    teacher_cfg = dict(spec["teacher"])
    teacher_cfg["name"] = spec["model_class"]

    base = TeacherPairDataset(
        processed,
        split_id=split_id,
        fold=fold,
        pool=str(config.get("pool", "M-716")),
        split_dir=splits,
    )
    features = build_cell_features(processed, split_id=split_id, fold=fold)
    drugs = build_drug_features(processed)

    similarity_cfg = SimilarityConfig.from_dict(config.get("similarity", {}))
    similarity_enabled = (
        bool(config.get("similarity", {}).get("enabled", True))
        and teacher_cfg["name"] == "t0"
        and bool(teacher_cfg.get("use_neighbor", True))
    )
    if similarity_enabled:
        fold_def = next(item for item in base.split["folds"] if int(item["fold"]) == fold)
        neighbors, weights, _, _ = build_fold_similarity(
            features,
            train_cells=list(fold_def["train"]),
            config=similarity_cfg,
            processed_dir=processed,
            split_id=split_id,
            fold=fold,
        )
        tensors = build_pair_tensors(features, drugs, neighbors=neighbors, weights=weights)
    else:
        tensors = build_pair_tensors(features, drugs)

    loader = make_loader(
        base,
        tensors,
        drugs,
        split="train",
        batch_size=batch_size,
        shuffle=False,
        use_gin=bool(teacher_cfg.get("use_gin", False)),
        seed=0,
    )
    model = _build_model(
        {"teacher": teacher_cfg, "similarity": config.get("similarity", {})},
        pathway_dim=int(spec["pathway_dim"]),
        morgan_dim=int(spec["morgan_dim"]),
        n_target_genes=int(spec["n_target_genes"]),
    )
    device = resolve_device("auto")
    checkpoint = torch.load(
        run / f"fold{fold}" / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    rows: list[np.ndarray] = []
    y_soft: list[np.ndarray] = []
    z_pair: list[np.ndarray] = []
    cell_sum: np.ndarray | None = None
    cell_count = np.zeros(len(features.cell_ids), dtype=np.float64)
    drug_sum: np.ndarray | None = None
    drug_count = np.zeros(len(drugs.compound_uid), dtype=np.float64)

    for batch in loader:
        batch = move_batch(batch, device)
        output = model(batch)
        index = batch.row_index.cpu().numpy()
        rows.append(index)
        y_soft.append(output.y_hat.float().cpu().numpy().astype(np.float16))
        pair = output.z_pair.float().cpu().numpy()
        z_pair.append(pair.astype(np.float16))
        if cell_sum is None:
            cell_sum = np.zeros((len(features.cell_ids), pair.shape[1]), dtype=np.float64)
            drug_sum = np.zeros((len(drugs.compound_uid), pair.shape[1]), dtype=np.float64)
        cells = base.cell_idx[index]
        drugs_index = base.drug_idx[index]
        np.add.at(cell_sum, cells, pair)
        np.add.at(cell_count, cells, 1.0)
        np.add.at(drug_sum, drugs_index, pair)
        np.add.at(drug_count, drugs_index, 1.0)

    if cell_sum is None or drug_sum is None:
        raise RuntimeError("train loader 为空，无法导出缓存")

    row_index = np.concatenate(rows).astype(np.int64)
    cell_idx = base.cell_idx[row_index].astype(np.int32)
    drug_idx = base.drug_idx[row_index].astype(np.int32)
    cell_embedding = np.divide(
        cell_sum, np.maximum(cell_count, 1.0)[:, None]
    ).astype(np.float16)
    drug_embedding = np.divide(
        drug_sum, np.maximum(drug_count, 1.0)[:, None]
    ).astype(np.float16)

    io.write_npz(
        out / "train_pairs.npz",
        {
            "row_index": row_index,
            "cell_idx": cell_idx,
            "drug_idx": drug_idx,
            "y_soft": np.concatenate(y_soft),
            "z_pair": np.concatenate(z_pair),
        },
    )
    io.write_npz(
        out / "cell_emb.npz",
        {"cell_ids": np.asarray(features.cell_ids, dtype=np.str_), "z_cell": cell_embedding},
    )
    io.write_npz(
        out / "drug_emb.npz",
        {
            "compound_uid": np.asarray(drugs.compound_uid, dtype=np.str_),
            "z_drug": drug_embedding,
        },
    )
    meta = {
        "teacher": teacher_cfg["name"],
        "run_dir": str(run),
        "fold": int(fold),
        "split_id": split_id,
        "n_train_pairs": int(row_index.size),
        "n_cells_with_embedding": int((cell_count > 0).sum()),
        "n_drugs_with_embedding": int((drug_count > 0).sum()),
        "z_dim": int(z_pair[0].shape[1]),
        "dtype": "float16",
        "red_line": "只含 train pair 软标签；val/test 不落软标签",
        "checkpoint": str(run / f"fold{fold}" / "best.pt"),
        "config": {"teacher": teacher_cfg, "similarity": config.get("similarity", {})},
    }
    io.write_json(out / "meta.json", meta)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="导出教师蒸馏缓存（train pair 软标签 + 潜向量）")
    parser.add_argument(
        "--run",
        required=True,
        help="Hydra 运行目录（含 foldN/best.pt 与 model_spec.json）",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split", required=True, help="split id，如 p1_lco_s0")
    parser.add_argument("--out", required=True)
    parser.add_argument("--processed", default=None)
    parser.add_argument("--splits", default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    meta = export_cache(
        run_dir=args.run,
        fold=args.fold,
        split_id=args.split,
        out_dir=args.out,
        processed_dir=args.processed,
        splits_dir=args.splits,
        batch_size=args.batch_size,
    )
    print(json.dumps({"status": "ok", **meta}, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()


__all__ = ["export_cache"]

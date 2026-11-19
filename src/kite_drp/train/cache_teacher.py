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
from kite_drp.train.data import build_pair_tensors, make_loader
from kite_drp.train.engine import move_batch, resolve_device
from kite_drp.train.features import (
    build_cell_features,
    build_drug_features,
    build_fold_similarity,
    resolve_neighbor_source,
    resolve_similarity_config,
)
from kite_drp.train.t1_graph import build_t1_graph, graph_tensors
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
    graph_bundle = None
    if teacher_cfg["name"] in {"t1", "t1_graph_only"}:
        graph_bundle = build_t1_graph(
            processed,
            split_id=split_id,
            fold=fold,
            config=teacher_cfg.get("graph", {}),
            split_dir=splits,
        )

    active_modalities = list(teacher_cfg.get("modalities", ["GE", "MU", "CNV"]))
    similarity_cfg = resolve_similarity_config(config, active_modalities)
    neighbor_source = resolve_neighbor_source(teacher_cfg, active_modalities)
    similarity_enabled = (
        bool(config.get("similarity", {}).get("enabled", True))
        and teacher_cfg["name"] in {"t0", "t0_p0", "t0_p1", "t1", "t1_graph_only"}
        and bool(teacher_cfg.get("use_neighbor", True))
    )
    target_max_tokens = (
        int(teacher_cfg.get("target_max_tokens", 64))
        if bool(teacher_cfg.get("target_branch", False))
        else 0
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
        tensors = build_pair_tensors(
            features,
            drugs,
            neighbors=neighbors,
            weights=weights,
            neighbor_source=neighbor_source,
            target_max_tokens=target_max_tokens,
            target_modalities=active_modalities,
        )
    else:
        tensors = build_pair_tensors(
            features,
            drugs,
            target_max_tokens=target_max_tokens,
            target_modalities=active_modalities,
        )

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
        n_graph_genes=(
            int(len(graph_bundle["arrays"]["gene_ids"])) if graph_bundle is not None else None
        ),
        cell_in_dim=(
            int(graph_bundle["arrays"]["cell_features"].shape[1])
            if graph_bundle is not None
            else None
        ),
    )
    device = resolve_device("auto")
    checkpoint = torch.load(
        run / f"fold{fold}" / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    if graph_bundle is not None:
        model.set_graph(graph_tensors(graph_bundle["arrays"]))  # type: ignore[attr-defined]

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
    cell_embedding = np.divide(cell_sum, np.maximum(cell_count, 1.0)[:, None]).astype(np.float16)
    drug_embedding = np.divide(drug_sum, np.maximum(drug_count, 1.0)[:, None]).astype(np.float16)

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
        "graph_spec": (graph_bundle["spec"] if graph_bundle is not None else None),
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

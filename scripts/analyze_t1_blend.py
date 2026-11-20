#!/usr/bin/env python
"""T1 后处理：融合预测与 graph-only 头的逐药 SCC 扫描。

用于回答“graph-only 头是否比融合头更好、简单的预测级 ensemble 能否再提升”。
只在 val 上做诊断；最终报告仍以 test（或预注册的 val 口径）为准。

用法：

    uv run python scripts/analyze_t1_blend.py \
        --run experiments/p1_ldo_quick/runs/t1_s0 --fold 0 \
        --split p2_ldo_s0 --out experiments/analysis/t1_blend_fold0
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
from kite_drp.eval.metrics import per_drug_metrics, regression_metrics
from kite_drp.train.data import build_pair_tensors, make_loader
from kite_drp.train.engine import evaluate, resolve_device
from kite_drp.train.features import (
    build_cell_features,
    build_drug_features,
    build_fold_similarity,
    resolve_neighbor_source,
    resolve_similarity_config,
)
from kite_drp.train.t1_graph import build_t1_graph, graph_tensors
from kite_drp.train.teacher_train import _build_model


def _load_config(run: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(run / "resolved_config.yaml"), resolve=True)
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="T1 blend diagnostics")
    parser.add_argument("--run", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    run = Path(args.run)
    if not run.is_absolute():
        run = REPO_ROOT / run
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    spec = io.read_json(run / f"fold{args.fold}" / "model_spec.json")
    teacher_cfg = dict(spec["teacher"])
    teacher_cfg["name"] = spec["model_class"]
    config = _load_config(run)
    processed = REPO_ROOT / "data" / "processed"
    splits = REPO_ROOT / "data" / "splits"

    base = TeacherPairDataset(
        processed,
        split_id=args.split,
        fold=args.fold,
        pool=str(config.get("pool", "M-716")),
        split_dir=splits,
    )
    features = build_cell_features(processed, split_id=args.split, fold=args.fold)
    drugs = build_drug_features(processed)
    active_modalities = list(teacher_cfg.get("modalities", ["GE", "MU", "CNV"]))
    similarity_cfg = resolve_similarity_config(config, active_modalities)
    neighbor_source = resolve_neighbor_source(teacher_cfg, active_modalities)
    similarity_enabled = (
        bool(config.get("similarity", {}).get("enabled", True))
        and teacher_cfg["name"] in {"t1", "t1_graph_only"}
        and bool(teacher_cfg.get("use_neighbor", True))
    )
    if similarity_enabled:
        fold_def = next(item for item in base.split["folds"] if int(item["fold"]) == args.fold)
        neighbors, weights, _, _ = build_fold_similarity(
            features,
            train_cells=list(fold_def["train"]),
            config=similarity_cfg,
            processed_dir=processed,
            split_id=args.split,
            fold=args.fold,
        )
        tensors = build_pair_tensors(
            features,
            drugs,
            neighbors=neighbors,
            weights=weights,
            neighbor_source=neighbor_source,
            target_max_tokens=int(teacher_cfg.get("target_max_tokens", 64)),
            target_modalities=active_modalities,
        )
    else:
        tensors = build_pair_tensors(
            features,
            drugs,
            target_max_tokens=int(teacher_cfg.get("target_max_tokens", 64)),
            target_modalities=active_modalities,
        )
    graph_bundle = build_t1_graph(
        processed,
        split_id=args.split,
        fold=args.fold,
        config=teacher_cfg.get("graph", {}),
        split_dir=splits,
    )
    graph = graph_bundle["arrays"]

    model = _build_model(
        {"teacher": teacher_cfg, "similarity": config.get("similarity", {})},
        pathway_dim=int(spec["pathway_dim"]),
        morgan_dim=int(spec["morgan_dim"]),
        n_target_genes=int(spec["n_target_genes"]),
        n_graph_genes=int(len(graph["gene_ids"])),
        cell_in_dim=int(graph["cell_features"].shape[1]),
    )
    device = resolve_device("auto")
    checkpoint = torch.load(
        run / f"fold{args.fold}" / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)
    model.set_graph(graph_tensors(graph))  # type: ignore[attr-defined]
    model.eval()

    loader = make_loader(
        base,
        tensors,
        drugs,
        split="val",
        batch_size=args.batch_size,
        shuffle=False,
        use_gin=bool(teacher_cfg.get("use_gin", False)),
    )
    _, frame = evaluate(model, loader, device=device, base=base)
    if "y_graph_pred" not in frame.columns:
        raise RuntimeError("frame 缺少 y_graph_pred：请用新版 engine.evaluate")

    y_true = frame["y_true"].to_numpy(dtype=float)
    mask = frame["label_mask"].to_numpy(dtype=bool)
    fused = frame["y_pred"].to_numpy(dtype=float)
    graph = frame["y_graph_pred"].to_numpy(dtype=float)
    rows = []
    for weight in np.linspace(0.0, 1.0, 11):
        blend = (1.0 - weight) * fused + weight * graph
        temp = frame.copy()
        temp["y_pred"] = blend
        metrics = regression_metrics(y_true[mask], blend[mask])
        per_drug = per_drug_metrics(temp)
        rows.append(
            {
                "graph_weight": float(weight),
                "scc": metrics["scc"],
                "per_drug_scc": per_drug.get("scc_mean"),
                "pcc": metrics["pcc"],
            }
        )
    payload = {
        "run": str(run),
        "fold": args.fold,
        "split": args.split,
        "rows": rows,
    }
    (out / "blend_scan.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    lines = [
        "# T1 blend scan",
        "",
        f"- run: `{run}`",
        f"- fold: {args.fold}；split: `{args.split}`",
        "",
        "| graph weight | SCC | per-drug SCC | PCC |",
        "|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['graph_weight']:.1f} | {row['scc']:.4f} "
            f"| {row['per_drug_scc']:.4f} | {row['pcc']:.4f} |"
        )
    (out / "blend_scan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

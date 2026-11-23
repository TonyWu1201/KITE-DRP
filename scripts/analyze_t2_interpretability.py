#!/usr/bin/env python
"""T2 可解释性分析：融合 gate、分层指标、DTI/PPI context 遮挡与置换。

示例：

    uv run python scripts/analyze_t2_interpretability.py \
        --run experiments/t2_quick/runs/t2_all_s0 --fold 0 --split ldo \
        --out experiments/analysis/t2_all_f0

输出：

- ``summary.json``：alpha、coverage、P1/T2/final 指标、按 tier 分层；
- ``per_drug.csv``：逐药 final/P1/T2 SCC、tier、context 遮挡差；
- ``occlusion_by_tier.csv``：zero/permute summary、rwr、pathway、all 的逐药差；
- ``pathway_attribution.csv``：pathway 遮挡影响最大的药物的 top 通路；
- ``gene_attribution.csv``：这些药物的 RWR top 基因。

说明：遮挡/置换只在验证折上做诊断，不参与训练或选模；所有结论都是
“模型依赖哪些输入”，不是因果生物学结论。
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from scipy import stats

from kite_drp.common.paths import REPO_ROOT
from kite_drp.data.dataset import TeacherPairDataset
from kite_drp.eval.metrics import per_drug_metrics
from kite_drp.models.teacher.common import TeacherBatch
from kite_drp.train.data import build_pair_tensors, make_loader
from kite_drp.train.engine import evaluate, move_batch, resolve_device, set_seed
from kite_drp.train.features import (
    build_cell_features,
    build_drug_features,
    build_fold_similarity,
    resolve_neighbor_source,
    resolve_similarity_config,
)
from kite_drp.train.netgp_features import build_netgp_features
from kite_drp.train.ppi_features import build_ppi_features
from kite_drp.train.teacher_train import _build_model

BLOCKS = (
    "zero_target_pathway",
    "zero_neighbor_pathway",
    "zero_summary",
    "zero_rwr",
    "zero_pathway",
    "zero_netgp",
    "zero_meta",
    "zero_all",
    "permute_all",
)


def _load_config(run: Path) -> dict[str, Any]:
    path = run / "resolved_config.yaml"
    if not path.is_file():
        raise SystemExit(f"缺少 {path}")
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def _per_drug_scc(
    frame: pd.DataFrame, *, min_samples: int = 10, column: str = "y_pred"
) -> pd.Series:
    values: dict[str, float] = {}
    for drug, group in frame.groupby("compound_uid", sort=True):
        if len(group) < min_samples:
            continue
        y_true = group["y_true"].to_numpy(dtype=float)
        y_pred = group[column].to_numpy(dtype=float)
        if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
            continue
        values[str(drug)] = float(stats.spearmanr(y_true, y_pred).statistic)
    return pd.Series(values, dtype=float)


def _occlusion_frames(
    model: torch.nn.Module,
    loader,
    *,
    device: torch.device,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    model.eval()

    def _column(output, key: str) -> np.ndarray | None:
        value = output.aux.get(key)
        return value.float().cpu().numpy() if value is not None else None

    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            base_output = model(batch)
            base = base_output.y_hat.float().cpu().numpy()
            variants: dict[str, TeacherBatch] = {"baseline": batch}
            has_prior = any(
                value is not None
                for value in (
                    batch.ppi_summary,
                    batch.ppi_meta,
                    batch.netgp_gene_ids,
                )
            )
            if not has_prior:
                rows.append(
                    pd.DataFrame(
                        {
                            "row_index": batch.row_index.cpu().numpy(),
                            "block": "baseline",
                            "y_pred": base,
                            "y_pred_t2": _column(base_output, "y_t2"),
                            "y_pred_ctx": _column(base_output, "y_ctx"),
                        }
                    )
                )
                continue
            batch_size = int(batch.ge.size(0))
            device_ = batch.ge.device
            permutation = torch.randperm(batch_size, device=device_)
            for block in BLOCKS:
                variant = copy.copy(batch)
                if block == "zero_summary":
                    if batch.ppi_summary is None:
                        continue
                    variant.ppi_summary = torch.zeros_like(batch.ppi_summary)
                elif block == "zero_rwr":
                    if batch.ppi_rwr is None:
                        continue
                    variant.ppi_rwr = torch.zeros_like(batch.ppi_rwr)
                elif block == "zero_target_pathway":
                    if batch.ppi_target_pathway is None:
                        continue
                    variant.ppi_target_pathway = torch.zeros_like(batch.ppi_target_pathway)
                elif block == "zero_neighbor_pathway":
                    if batch.ppi_neighbor_pathway is None:
                        continue
                    variant.ppi_neighbor_pathway = torch.zeros_like(batch.ppi_neighbor_pathway)
                elif block == "zero_pathway":
                    if batch.ppi_target_pathway is None or batch.ppi_neighbor_pathway is None:
                        continue
                    variant.ppi_target_pathway = torch.zeros_like(batch.ppi_target_pathway)
                    variant.ppi_neighbor_pathway = torch.zeros_like(batch.ppi_neighbor_pathway)
                elif block == "zero_netgp":
                    if batch.netgp_gene_weight is None:
                        continue
                    variant.netgp_gene_weight = torch.zeros_like(batch.netgp_gene_weight)
                    variant.netgp_gene_mask = torch.zeros_like(
                        batch.netgp_gene_mask, dtype=torch.bool
                    )
                    variant.netgp_gene_values = torch.zeros_like(batch.netgp_gene_values)
                    if batch.netgp_pathway is not None:
                        variant.netgp_pathway = torch.zeros_like(batch.netgp_pathway)
                elif block == "zero_meta":
                    if batch.ppi_meta is None:
                        continue
                    variant.ppi_meta = torch.zeros_like(batch.ppi_meta)
                elif block == "zero_all":
                    if batch.ppi_summary is not None:
                        variant.ppi_summary = torch.zeros_like(batch.ppi_summary)
                    if batch.ppi_rwr is not None:
                        variant.ppi_rwr = torch.zeros_like(batch.ppi_rwr)
                    if batch.ppi_target_pathway is not None:
                        variant.ppi_target_pathway = torch.zeros_like(batch.ppi_target_pathway)
                    if batch.ppi_neighbor_pathway is not None:
                        variant.ppi_neighbor_pathway = torch.zeros_like(batch.ppi_neighbor_pathway)
                    if batch.ppi_meta is not None:
                        variant.ppi_meta = torch.zeros_like(batch.ppi_meta)
                    if batch.netgp_gene_weight is not None:
                        variant.netgp_gene_weight = torch.zeros_like(batch.netgp_gene_weight)
                        variant.netgp_gene_mask = torch.zeros_like(
                            batch.netgp_gene_mask, dtype=torch.bool
                        )
                        variant.netgp_gene_values = torch.zeros_like(batch.netgp_gene_values)
                        if batch.netgp_pathway is not None:
                            variant.netgp_pathway = torch.zeros_like(batch.netgp_pathway)
                elif block == "permute_all":
                    if batch.ppi_summary is not None:
                        variant.ppi_summary = batch.ppi_summary[permutation]
                    if batch.ppi_rwr is not None:
                        variant.ppi_rwr = batch.ppi_rwr[permutation]
                    if batch.ppi_target_pathway is not None:
                        variant.ppi_target_pathway = batch.ppi_target_pathway[permutation]
                    if batch.ppi_neighbor_pathway is not None:
                        variant.ppi_neighbor_pathway = batch.ppi_neighbor_pathway[permutation]
                    if batch.ppi_meta is not None:
                        variant.ppi_meta = batch.ppi_meta[permutation]
                    if batch.netgp_gene_ids is not None:
                        variant.netgp_gene_ids = batch.netgp_gene_ids[permutation]
                        variant.netgp_gene_weight = batch.netgp_gene_weight[permutation]
                        variant.netgp_gene_values = batch.netgp_gene_values[permutation]
                        variant.netgp_gene_mask = batch.netgp_gene_mask[permutation]
                    if batch.netgp_pathway is not None:
                        variant.netgp_pathway = batch.netgp_pathway[permutation]
                variants[block] = variant
            for block, variant in variants.items():
                output = base_output if block == "baseline" else model(variant)
                rows.append(
                    pd.DataFrame(
                        {
                            "row_index": batch.row_index.cpu().numpy(),
                            "block": block,
                            "y_pred": output.y_hat.float().cpu().numpy(),
                            "y_pred_t2": _column(output, "y_t2"),
                            "y_pred_ctx": _column(output, "y_ctx"),
                        }
                    )
                )
    return pd.concat(rows, ignore_index=True)


def _tier_table(per_drug: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tier, frame in per_drug.groupby("tier", sort=True):
        rows.append(
            {
                "tier": tier,
                "n_drugs": int(len(frame)),
                "final_scc": float(frame["final_scc"].mean()),
                "p1_scc": float(frame["p1_scc"].mean()),
                "t2_scc": float(frame["t2_scc"].mean()),
                "delta_final_p1": float((frame["final_scc"] - frame["p1_scc"]).mean()),
                "delta_t2_p1": float((frame["t2_scc"] - frame["p1_scc"]).mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="T2 interpretability")
    parser.add_argument("--run", required=True, help="T2 run 目录（含 foldK/best.pt）")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split", default="ldo")
    parser.add_argument("--pool", default="M-716")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    run = Path(args.run)
    if not run.is_absolute():
        run = REPO_ROOT / run
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    config = _load_config(run)
    seed = int(config.get("seed", 0))
    set_seed(seed * 100 + args.fold)
    paths = config["paths"]
    processed = REPO_ROOT / paths["processed_dir"]
    splits = REPO_ROOT / paths["splits_dir"]
    split_protocol = str(config["split"].get("protocol", args.split))
    split_id = f"{split_protocol}_s{int(config['split']['seed'])}"
    teacher_cfg = config["teacher"]
    teacher_name = str(teacher_cfg.get("name", "t2"))
    if teacher_name != "t2":
        raise SystemExit(f"该脚本只支持 teacher=t2，收到 {teacher_name}")

    base = TeacherPairDataset(
        processed,
        split_id=split_id,
        fold=args.fold,
        pool=args.pool,
        split_dir=splits,
    )
    features = build_cell_features(processed, split_id=split_id, fold=args.fold)
    drugs = build_drug_features(processed)
    active_modalities = list(teacher_cfg.get("modalities", ["GE", "MU", "CNV"]))
    similarity_cfg = resolve_similarity_config(config, active_modalities)
    neighbor_source = resolve_neighbor_source(teacher_cfg, active_modalities)
    similarity_enabled = bool(config.get("similarity", {}).get("enabled", True)) and bool(
        teacher_cfg.get("use_neighbor", True)
    )
    target_max_tokens = int(teacher_cfg.get("target_max_tokens", 64))
    if similarity_enabled:
        fold_def = next(item for item in base.split["folds"] if int(item["fold"]) == args.fold)
        neighbors, weights, _, _ = build_fold_similarity(
            features,
            train_cells=list(fold_def["train"]),
            config=similarity_cfg,
            processed_dir=processed,
            split_id=split_id,
            fold=args.fold,
            cache_dir=REPO_ROOT / paths["interim_dir"] / "teacher" / "similarity",
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

    ppi_cfg = teacher_cfg.get("ppi", {}) or {}
    ppi_bundle = (
        build_ppi_features(
            processed,
            features,
            drugs,
            config=ppi_cfg,
            cache_dir=REPO_ROOT / paths["interim_dir"] / "teacher" / "ppi_features",
            fold_key=f"{split_id}_fold{args.fold}",
        )
        if bool(ppi_cfg.get("enabled", False))
        else None
    )
    netgp_cfg = ppi_cfg.get("netgp", {}) or {}
    netgp_bundle = (
        build_netgp_features(
            processed,
            features,
            drugs,
            config=netgp_cfg,
            cache_dir=REPO_ROOT / paths["interim_dir"] / "teacher" / "netgp_features",
        )
        if bool(netgp_cfg.get("enabled", False))
        else None
    )
    # 即使模型不用 PPI（如 t2_base），也构建一份特征用于 tier 分层与审计。
    tier_bundle = ppi_bundle if ppi_bundle is not None else netgp_bundle
    if tier_bundle is None:
        try:
            tier_bundle = build_ppi_features(
                processed,
                features,
                drugs,
                config=ppi_cfg,
                cache_dir=REPO_ROOT / paths["interim_dir"] / "teacher" / "ppi_features",
                fold_key=f"{split_id}_fold{args.fold}",
            )
        except (FileNotFoundError, ValueError):
            tier_bundle = None
    model = _build_model(
        config,
        pathway_dim=int(features.ge.shape[1]),
        morgan_dim=int(drugs.morgan.shape[1]),
        n_target_genes=drugs.target_dim,
        ppi_summary_dim=(ppi_bundle.summary_dim if ppi_bundle is not None else 0),
        ppi_rwr_dim=(ppi_bundle.rwr_dim if ppi_bundle is not None else 0),
        ppi_meta_dim=(
            ppi_bundle.meta_dim
            if ppi_bundle is not None and bool(ppi_cfg.get("use_meta", False))
            else (
                netgp_bundle.meta_dim
                if netgp_bundle is not None and bool(netgp_cfg.get("use_meta", True))
                else 0
            )
        ),
        netgp_top_k=(netgp_bundle.top_k if netgp_bundle is not None else 0),
        netgp_pathway_dim=(
            netgp_bundle.pathway_dim
            if netgp_bundle is not None and bool(netgp_cfg.get("use_pathway", False))
            else 0
        ),
        netgp_reliability_dim=(netgp_bundle.reliability_dim if netgp_bundle is not None else 0),
    )
    device = resolve_device(args.device)
    model = model.to(device)
    checkpoint = torch.load(
        run / f"fold{args.fold}" / "best.pt", map_location=device, weights_only=False
    )
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=False)

    loader = make_loader(
        base,
        tensors,
        drugs,
        split="val",
        batch_size=args.batch_size,
        shuffle=False,
        ppi_features=ppi_bundle,
        netgp_features=netgp_bundle,
    )
    metrics, frame = evaluate(model, loader, device=device, base=base)
    frame = frame.loc[frame["label_mask"]].copy()

    per_drug = per_drug_metrics(frame)
    per_drug_frame = pd.DataFrame(per_drug.get("drugs", []))
    per_drug_frame = per_drug_frame.rename(columns={"scc": "final_scc"})
    per_drug_frame["compound_uid"] = per_drug_frame["compound_uid"].astype(str)

    p1_per_drug = pd.DataFrame(
        per_drug_metrics(frame.assign(y_pred=frame["y_p1_pred"])).get("drugs", [])
    ).rename(columns={"scc": "p1_scc"})
    t2_per_drug = pd.DataFrame(
        per_drug_metrics(frame.assign(y_pred=frame["y_t2_pred"])).get("drugs", [])
    ).rename(columns={"scc": "t2_scc"})
    for extra in (p1_per_drug, t2_per_drug):
        if not extra.empty:
            extra["compound_uid"] = extra["compound_uid"].astype(str)
            per_drug_frame = per_drug_frame.merge(
                extra[["compound_uid", *[c for c in extra.columns if c.endswith("_scc")]]],
                on="compound_uid",
                how="left",
            )

    tier_map: dict[str, str] = {}
    if tier_bundle is not None and not tier_bundle.audit.empty:
        tier_map = dict(
            zip(
                tier_bundle.audit["compound_uid"].astype(str),
                tier_bundle.audit["tier"].astype(str),
                strict=False,
            )
        )
    per_drug_frame["tier"] = (
        per_drug_frame["compound_uid"].map(tier_map).astype(object).fillna("none")
    )

    occlusion = _occlusion_frames(model, loader, device=device)
    occlusion = occlusion.merge(
        frame[["row_index", "compound_uid", "y_true"]], on="row_index", how="inner"
    )
    view_columns = {"final": "y_pred", "t2": "y_pred_t2", "ctx": "y_pred_ctx"}
    occ_by_view: dict[str, dict[str, pd.Series]] = {}
    for view, column in view_columns.items():
        if column not in occlusion.columns:
            continue
        view_frame = occlusion.dropna(subset=[column])
        per_block = {
            block: _per_drug_scc(group, column=column)
            for block, group in view_frame.groupby("block", sort=True)
        }
        if "baseline" not in per_block:
            continue
        occ_by_view[view] = per_block
        baseline_scc = per_block["baseline"]
        suffix = "" if view == "final" else f"_{view}"
        for block in BLOCKS:
            if block not in per_block:
                continue
            delta = per_block[block] - baseline_scc
            common = per_drug_frame["compound_uid"].isin(delta.index)
            per_drug_frame.loc[common, f"delta_{block}{suffix}"] = per_drug_frame.loc[
                common, "compound_uid"
            ].map(delta)
    tier_frame = _tier_table(per_drug_frame)

    occlusion_rows = []
    for view, per_block in occ_by_view.items():
        baseline_scc = per_block["baseline"]
        for block in ["baseline", *BLOCKS]:
            if block not in per_block:
                continue
            series = per_block[block]
            delta = (
                pd.Series(0.0, index=series.index) if block == "baseline" else series - baseline_scc
            )
            for tier, group in per_drug_frame.groupby("tier", sort=True):
                ids = group["compound_uid"]
                values = delta.reindex(ids).dropna()
                occlusion_rows.append(
                    {
                        "view": view,
                        "block": block,
                        "tier": tier,
                        "n_drugs": int(values.size),
                        "mean_delta_scc": (float(values.mean()) if values.size else float("nan")),
                    }
                )
    occlusion_frame = pd.DataFrame(occlusion_rows)

    # pathway 遮挡影响最大的药物 -> top 通路与 RWR top 基因
    pathway_attr = pd.DataFrame()
    gene_attr = pd.DataFrame()
    if ppi_bundle is not None and "delta_zero_pathway" in per_drug_frame.columns:
        pathway_arrays = np.load(
            processed / "student_inputs" / "pathway_weights.npz", allow_pickle=False
        )
        pathway_ids = [str(value) for value in pathway_arrays["pathway_ids"]]
        top = per_drug_frame.dropna(subset=["delta_zero_pathway"]).nsmallest(
            10, "delta_zero_pathway"
        )
        rows = []
        for row in top.itertuples(index=False):
            compound = str(row.compound_uid)
            if compound not in set(ppi_bundle.audit["compound_uid"].astype(str)):
                continue
            drug_index = ppi_bundle.audit.index[
                ppi_bundle.audit["compound_uid"].astype(str) == compound
            ]
            if len(drug_index) == 0:
                continue
            index = int(drug_index[0])
            target_top = np.argsort(-ppi_bundle.target_pathway[index])[:5]
            neighbor_top = np.argsort(-ppi_bundle.neighbor_pathway[index])[:5]
            rows.append(
                {
                    "compound_uid": compound,
                    "tier": row.tier,
                    "delta_zero_pathway": float(row.delta_zero_pathway),
                    "top_target_pathways": ";".join(pathway_ids[int(item)] for item in target_top),
                    "top_neighbor_pathways": ";".join(
                        pathway_ids[int(item)] for item in neighbor_top
                    ),
                }
            )
        pathway_attr = pd.DataFrame(rows)
        if not pathway_attr.empty:
            gene_attr = ppi_bundle.audit[
                ppi_bundle.audit["compound_uid"].astype(str).isin(pathway_attr["compound_uid"])
            ][["compound_uid", "tier", "n_roots", "n_neighbors", "rwr_top_genes"]]

    per_drug_frame.to_csv(out / "per_drug.csv", index=False)
    tier_frame.to_csv(out / "per_tier.csv", index=False)
    occlusion_frame.to_csv(out / "occlusion_by_tier.csv", index=False)
    if not pathway_attr.empty:
        pathway_attr.to_csv(out / "pathway_attribution.csv", index=False)
    if not gene_attr.empty:
        gene_attr.to_csv(out / "gene_attribution.csv", index=False)

    summary = {
        "run": str(run),
        "fold": args.fold,
        "split": split_id,
        "alpha_mean": float(frame["t2_alpha"].mean()) if "t2_alpha" in frame else None,
        "coverage_fraction": (
            float(frame["ppi_coverage"].mean())
            if ppi_bundle is not None and "ppi_coverage" in frame
            else None
        ),
        "metrics": {
            "final": metrics.get("regression", {}),
            "p1": metrics.get("p1_regression", {}),
            "t2": metrics.get("t2_regression", {}),
            "final_per_drug_scc": per_drug.get("scc_mean"),
            "p1_per_drug_scc": per_drug_metrics(frame.assign(y_pred=frame["y_p1_pred"])).get(
                "scc_mean"
            ),
            "t2_per_drug_scc": per_drug_metrics(frame.assign(y_pred=frame["y_t2_pred"])).get(
                "scc_mean"
            ),
        },
        "per_tier": tier_frame.to_dict(orient="records"),
        "occlusion_by_tier": occlusion_frame.to_dict(orient="records"),
        "occlusion_mean": {
            view: {
                block: float((per_block[block] - per_block["baseline"]).mean())
                for block in BLOCKS
                if block in per_block
            }
            for view, per_block in occ_by_view.items()
        },
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()

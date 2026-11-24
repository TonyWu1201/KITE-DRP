#!/usr/bin/env python
"""PPI/DTI 特征残差探针：先验块里到底有没有“模型没用的信息”。

对给定 T2 run 的 val 预测：

1. 取 ``resid = y - y_p1``；
2. 用每个 PPI 块（meta / summary / rwr / pathway）单独拟合 Ridge；
3. 按 **药物分组** K 折交叉验证，得到 out-of-fold ``r_hat``；
4. 报告 ``resid`` 的 OOF R²，以及 ``y_p1 + r_hat`` 的逐药 SCC 增量；
5. 用列 shuffle / 药物置换作为零分布。

这是诊断脚本，不是模型：它在 val（已用于选模）上做药物分组 CV，
因此只能回答“这个特征空间里是否还有可用信息”，不能作为 test 结论。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from kite_drp.common.paths import REPO_ROOT
from kite_drp.data.dataset import TeacherPairDataset
from kite_drp.train.features import build_cell_features, build_drug_features
from kite_drp.train.ppi_features import build_ppi_features

BLOCKS = ("meta", "summary", "rwr", "pathway_ge")


def _per_drug_scc(frame: pd.DataFrame, column: str, *, min_samples: int) -> pd.Series:
    values: dict[str, float] = {}
    for drug, group in frame.groupby("compound_uid"):
        if len(group) < int(min_samples):
            continue
        y_true = group["y_true"].to_numpy(dtype=float)
        y_pred = group[column].to_numpy(dtype=float)
        if np.std(y_true) <= 0 or np.std(y_pred) <= 0:
            continue
        stat = spearmanr(y_true, y_pred).statistic
        if np.isfinite(stat):
            values[str(drug)] = float(stat)
    return pd.Series(values, dtype=float)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residual = float(((y_true - y_pred) ** 2).sum())
    total = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - residual / total if total > 0 else float("nan")


def _load_config(run: Path) -> dict[str, Any]:
    for name in ("resolved_config.yaml", "config.yaml"):
        path = run / name
        if path.is_file():
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
    raise FileNotFoundError(f"{run} 缺少 resolved_config.yaml")


def _block_matrix(
    block: str,
    *,
    ppi,
    drug_idx: np.ndarray,
    cell_idx: np.ndarray,
    cell_ge: np.ndarray,
    cell_cnv: np.ndarray,
    include_cnv: bool,
) -> np.ndarray:
    if block == "meta":
        return np.asarray(ppi.meta)[drug_idx].astype(np.float32)
    if block == "summary":
        return np.asarray(ppi.summary)[drug_idx, cell_idx].astype(np.float32)
    if block == "rwr":
        return np.asarray(ppi.rwr)[drug_idx, cell_idx].astype(np.float32)
    if block == "pathway_ge":
        target = np.asarray(ppi.target_pathway)[drug_idx].astype(np.float32)
        neighbor = np.asarray(ppi.neighbor_pathway)[drug_idx].astype(np.float32)
        pieces = [target * cell_ge, neighbor * cell_ge]
        if include_cnv:
            pieces.extend([target * cell_cnv, neighbor * cell_cnv])
        return np.concatenate(pieces, axis=1).astype(np.float32)
    raise KeyError(block)


def _oof_residual(
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    *,
    alpha: float,
    n_splits: int,
) -> np.ndarray:
    oof = np.zeros_like(target, dtype=np.float64)
    splitter = GroupKFold(n_splits=min(int(n_splits), len(np.unique(groups))))
    for train, test in splitter.split(features, target, groups):
        model = make_pipeline(StandardScaler(), Ridge(alpha=float(alpha), solver="lsqr"))
        model.fit(features[train], target[train])
        oof[test] = model.predict(features[test])
    return oof


def main() -> None:
    parser = argparse.ArgumentParser(description="PPI residual ceiling probe")
    parser.add_argument("--run", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--blocks", default="meta,summary,rwr,pathway_ge")
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--include-cnv", action="store_true")
    parser.add_argument("--seed", type=int, default=20261123)
    args = parser.parse_args()

    run = Path(args.run)
    if not run.is_absolute():
        run = REPO_ROOT / run
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    config = _load_config(run)
    paths = config["paths"]
    processed = REPO_ROOT / paths["processed_dir"]
    splits = REPO_ROOT / paths["splits_dir"]
    split_id = f"{config['split']['protocol']}_s{int(config['split']['seed'])}"
    fold = int(args.fold)
    val_predictions = run / f"fold{fold}" / "val_predictions.parquet"
    if not val_predictions.is_file():
        raise SystemExit(f"缺少 {val_predictions}")
    frame = pd.read_parquet(val_predictions)
    if "y_p1_pred" not in frame.columns:
        raise SystemExit("该 run 没有 y_p1_pred；需要 teacher=t2")

    base = TeacherPairDataset(
        processed,
        split_id=split_id,
        fold=fold,
        pool=str(config.get("pool", "M-716")),
        split_dir=splits,
    )
    features = build_cell_features(processed, split_id=split_id, fold=fold)
    drugs = build_drug_features(processed)
    teacher_cfg = config["teacher"]
    ppi_cfg = teacher_cfg.get("ppi", {}) or {}
    ppi = build_ppi_features(
        processed,
        features,
        drugs,
        config=ppi_cfg,
        cache_dir=REPO_ROOT / paths["interim_dir"] / "teacher" / "ppi_features",
        fold_key=f"{split_id}_fold{fold}",
    )

    rows = frame["row_index"].to_numpy(dtype=np.int64)
    drug_idx = base.drug_idx[rows]
    cell_idx = base.cell_idx[rows]
    y_true = frame["y_true"].to_numpy(dtype=float)
    y_p1 = frame["y_p1_pred"].to_numpy(dtype=float)
    residual = y_true - y_p1
    groups = frame["compound_uid"].to_numpy()
    cell_ge = features.ge[cell_idx].astype(np.float32)
    cell_cnv = features.cnv[cell_idx].astype(np.float32)

    rng = np.random.default_rng(args.seed)
    tier = np.asarray(ppi.tier)[drug_idx]
    report: dict[str, Any] = {
        "run": str(run),
        "fold": fold,
        "n_rows": int(len(rows)),
        "n_drugs": int(frame["compound_uid"].nunique()),
        "residual_std": float(residual.std()),
        "blocks": {},
    }
    per_drug_rows: list[pd.DataFrame] = []
    for block in [item.strip() for item in args.blocks.split(",") if item.strip()]:
        if block not in BLOCKS:
            raise SystemExit(f"未知 block：{block}")
        X = _block_matrix(
            block,
            ppi=ppi,
            drug_idx=drug_idx,
            cell_idx=cell_idx,
            cell_ge=cell_ge,
            cell_cnv=cell_cnv,
            include_cnv=bool(args.include_cnv),
        )
        oof = _oof_residual(X, residual, groups, alpha=args.alpha, n_splits=args.splits)
        blended = frame.assign(y_probe=y_p1 + oof)
        per_drug = _per_drug_scc(blended, "y_probe", min_samples=args.min_samples)
        per_drug_p1 = _per_drug_scc(frame, "y_p1_pred", min_samples=args.min_samples)
        common = per_drug.index.intersection(per_drug_p1.index)
        delta = (per_drug.reindex(common) - per_drug_p1.reindex(common)).dropna()

        # 零分布：逐列 shuffle（破坏特征-样本对应）。
        shuffled = X.copy()
        for column in range(shuffled.shape[1]):
            shuffled[:, column] = shuffled[rng.permutation(shuffled.shape[0]), column]
        oof_shuffle = _oof_residual(
            shuffled, residual, groups, alpha=args.alpha, n_splits=args.splits
        )
        blended_shuffle = frame.assign(y_probe=y_p1 + oof_shuffle)
        per_drug_shuffle = _per_drug_scc(blended_shuffle, "y_probe", min_samples=args.min_samples)
        delta_shuffle = (per_drug_shuffle.reindex(common) - per_drug_p1.reindex(common)).dropna()

        # 零分布 2：药物置换（同 tier 内换药的特征）。
        permuted_idx = drug_idx.copy()
        for tier_code in np.unique(tier):
            group = np.flatnonzero(tier == tier_code)
            if group.size < 2:
                continue
            permuted_idx[group] = drug_idx[rng.permutation(group)]
        X_permuted = _block_matrix(
            block,
            ppi=ppi,
            drug_idx=permuted_idx,
            cell_idx=cell_idx,
            cell_ge=cell_ge,
            cell_cnv=cell_cnv,
            include_cnv=bool(args.include_cnv),
        )
        oof_permuted = _oof_residual(
            X_permuted, residual, groups, alpha=args.alpha, n_splits=args.splits
        )
        blended_permuted = frame.assign(y_probe=y_p1 + oof_permuted)
        per_drug_permuted = _per_drug_scc(blended_permuted, "y_probe", min_samples=args.min_samples)
        delta_permuted = (per_drug_permuted.reindex(common) - per_drug_p1.reindex(common)).dropna()

        report["blocks"][block] = {
            "n_features": int(X.shape[1]),
            "residual_r2": _r2(residual, oof),
            "residual_r2_shuffle": _r2(residual, oof_shuffle),
            "residual_r2_permute": _r2(residual, oof_permuted),
            "delta_scc": float(delta.mean()) if len(delta) else float("nan"),
            "delta_scc_shuffle": (
                float(delta_shuffle.mean()) if len(delta_shuffle) else float("nan")
            ),
            "delta_scc_permute": (
                float(delta_permuted.mean()) if len(delta_permuted) else float("nan")
            ),
            "n_drugs": int(len(delta)),
        }
        per_drug_rows.append(
            pd.DataFrame(
                {
                    "block": block,
                    "compound_uid": common,
                    "scc_p1": per_drug_p1.reindex(common).values,
                    "scc_probe": per_drug.reindex(common).values,
                    "scc_probe_shuffle": per_drug_shuffle.reindex(common).values,
                    "scc_probe_permute": per_drug_permuted.reindex(common).values,
                }
            )
        )

    if per_drug_rows:
        pd.concat(per_drug_rows, ignore_index=True).to_parquet(
            out / "probe_per_drug.parquet", index=False
        )
    (out / "probe.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()

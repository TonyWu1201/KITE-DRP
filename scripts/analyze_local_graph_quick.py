#!/usr/bin/env python
"""对 T1-local quick-screen 的 fold0 预测做逐药配对与 coverage 分层。

只读 ``val_predictions.parquet`` 和 ``data/processed/dti.parquet``；不训练。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def _per_drug(frame: pd.DataFrame, pred_col: str = "y_pred") -> pd.DataFrame:
    rows = []
    for compound, group in frame.groupby("compound_uid", sort=False):
        group = group.dropna(subset=["y_true", pred_col])
        if len(group) < 10 or group["y_true"].nunique() < 2:
            continue
        value = group[[pred_col, "y_true"]].corr(method="spearman").iloc[0, 1]
        rows.append({"compound_uid": compound, "n": len(group), "scc": float(value)})
    return pd.DataFrame(rows)


def _coverage() -> pd.DataFrame:
    dti = pd.read_parquet(REPO_ROOT / "data/processed/dti.parquet")
    any_cov = set(dti["compound_uid"])
    ot_cov = set(dti.loc[dti["source"].str.contains("opentargets"), "compound_uid"])
    rows = [
        {
            "compound_uid": uid,
            "coverage": "none" if uid not in any_cov else ("ot" if uid in ot_cov else "gdsc_only"),
        }
        for uid in sorted(
            set(pd.read_parquet(REPO_ROOT / "data/processed/compounds.parquet")["compound_uid"])
        )
    ]
    return pd.DataFrame(rows)


def _load(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    keep = ["cell_line_uid", "compound_uid", "y_true", "y_pred"]
    if "y_graph_pred" in frame.columns:
        keep.append("y_graph_pred")
    return frame[keep].dropna(subset=["y_true", "y_pred"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        default="experiments/2026-11-20_dti_p1_ldo_quick_s0/fold0/val_predictions.parquet",
    )
    parser.add_argument(
        "--runs",
        default="experiments/local_graph_quick_s0/runs",
        help="包含 {variant}_s0/fold0/val_predictions.parquet 的目录",
    )
    parser.add_argument("--variants", default="t1,mlp,root,neighbors,pseudo")
    args = parser.parse_args()

    baseline = _load(REPO_ROOT / args.baseline)
    baseline_per = _per_drug(baseline).rename(columns={"scc": "p1_scc", "n": "p1_n"})
    cov = _coverage()
    results: dict[str, object] = {}
    for name in [item.strip() for item in args.variants.split(",") if item.strip()]:
        path = REPO_ROOT / args.runs / f"{name}_s0" / "fold0" / "val_predictions.parquet"
        if not path.is_file():
            continue
        frame = _load(path)
        merged = baseline.merge(
            frame,
            on=["cell_line_uid", "compound_uid"],
            suffixes=("_p1", "_var"),
            how="inner",
        )
        per = _per_drug(merged.rename(columns={"y_pred_var": "y_pred", "y_true_var": "y_true"}))
        if per.empty:
            continue
        per = per.rename(columns={"scc": "var_scc", "n": "var_n"}).merge(
            baseline_per, on="compound_uid", how="inner"
        )
        per = per.merge(cov, on="compound_uid", how="left")
        per["delta"] = per["var_scc"] - per["p1_scc"]
        graph_summary: dict[str, float] = {}
        if "y_graph_pred" in merged.columns:
            graph_per = _per_drug(
                merged.rename(columns={"y_true_var": "y_true"}),
                pred_col="y_graph_pred",
            ).rename(columns={"scc": "graph_scc"})
            if not graph_per.empty:
                per = per.merge(
                    graph_per[["compound_uid", "graph_scc"]], on="compound_uid", how="left"
                )
                per["graph_delta"] = per["graph_scc"] - per["p1_scc"]
                graph_summary = {
                    "mean_graph_scc": float(per["graph_scc"].mean()),
                    "mean_graph_delta": float(per["graph_delta"].mean()),
                    "graph_fraction_improved": float((per["graph_delta"] > 0).mean()),
                    "graph_by_coverage": {
                        key: {
                            "n": int(len(group)),
                            "mean_graph_delta": float(group["graph_delta"].mean()),
                        }
                        for key, group in per.groupby("coverage")
                    },
                }
        summary = {
            "n_drugs": int(len(per)),
            "mean_var_scc": float(per["var_scc"].mean()),
            "mean_p1_scc": float(per["p1_scc"].mean()),
            "mean_delta": float(per["delta"].mean()),
            "fraction_improved": float((per["delta"] > 0).mean()),
            "fraction_worse_002": float((per["delta"] < -0.02).mean()),
            **graph_summary,
            "by_coverage": {
                key: {
                    "n": int(len(group)),
                    "mean_delta": float(group["delta"].mean()),
                }
                for key, group in per.groupby("coverage")
            },
        }
        results[name] = summary
        print(name, json.dumps(summary, ensure_ascii=False, default=float))
    (REPO_ROOT / args.runs / "paired_summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

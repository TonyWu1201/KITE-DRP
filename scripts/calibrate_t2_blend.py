#!/usr/bin/env python
"""T2 预测级融合校准（val 选权重，test 报数）。

用法：

    uv run python scripts/calibrate_t2_blend.py \
        --run t2_base=experiments/t2_quick_f0/runs/t2_base_s0 \
        --run t2_all=experiments/t2_quick_f0/runs/t2_all_s0 \
        --out experiments/t2_calibration

对每个 run / fold：

1. 读取 ``val_predictions.parquet``（以及可选的 ``test_predictions.parquet``）；
2. 在 val 上网格搜索

   ``y = y_p1 + w_t * (y_t2 - y_p1) + w_c * y_ctx``

   最大化逐药 Spearman 均值；
3. 冻结 ``(w_t*, w_c*)``，在 test 上只评估一次；
4. 与 P1-only、固定 0.5、tower-only 对照，并做按药配对 bootstrap。

``y_ctx`` 只有在 run 使用 ``context_mode=head/both`` 时存在；否则退化为 1D
（只搜索 ``w_t``）。该脚本不改模型、不重训，只做预测级校准，因此可以直接
复用已有的 quick-screen 产物。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise SystemExit(f"--run 需要 name=path，收到 {value}")
    name, path = value.split("=", 1)
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return name.strip(), resolved


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


def _blend(frame: pd.DataFrame, w_t: float, w_c: float) -> np.ndarray:
    y = frame["y_p1_pred"].to_numpy(dtype=float)
    if "y_t2_pred" in frame:
        y = y + float(w_t) * (frame["y_t2_pred"].to_numpy(dtype=float) - y)
    if w_c and "y_ctx_pred" in frame:
        y = y + float(w_c) * frame["y_ctx_pred"].to_numpy(dtype=float)
    return y


def _grid(has_ctx: bool) -> tuple[np.ndarray, np.ndarray]:
    w_t = np.linspace(0.0, 0.8, 17)
    w_c = np.linspace(-0.2, 0.4, 13) if has_ctx else np.zeros(1)
    return w_t, w_c


def _select_weights(
    frame: pd.DataFrame,
    *,
    min_samples: int,
) -> tuple[float, float, float]:
    has_ctx = "y_ctx_pred" in frame.columns
    w_t_grid, w_c_grid = _grid(has_ctx)
    best_score = -np.inf
    best_w_t = 0.0
    best_w_c = 0.0
    for w_t in w_t_grid:
        for w_c in w_c_grid:
            blended = frame.assign(_blend=_blend(frame, float(w_t), float(w_c)))
            score = _per_drug_scc(blended, "_blend", min_samples=min_samples).mean()
            if np.isfinite(score) and score > best_score:
                best_score = float(score)
                best_w_t = float(w_t)
                best_w_c = float(w_c)
    return best_w_t, best_w_c, best_score


def _bootstrap_mean(
    values: pd.Series,
    *,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    series = values.dropna()
    if series.empty:
        return {"n": 0.0, "mean": float("nan"), "low": float("nan"), "high": float("nan")}
    rng = np.random.default_rng(seed)
    array = series.to_numpy(dtype=float)
    draws = np.asarray(
        [
            rng.choice(array, size=array.size, replace=True).mean()
            for _ in range(max(int(iterations), 1))
        ],
        dtype=float,
    )
    return {
        "n": float(array.size),
        "mean": float(array.mean()),
        "low": float(np.quantile(draws, 0.025)),
        "high": float(np.quantile(draws, 0.975)),
    }


def _load_frame(path: Path, split: str) -> pd.DataFrame | None:
    candidate = path / f"{split}_predictions.parquet"
    if not candidate.is_file():
        return None
    frame = pd.read_parquet(candidate)
    required = {"y_true", "y_p1_pred"}
    if not required.issubset(frame.columns):
        return None
    if "compound_uid" not in frame.columns:
        raise ValueError(f"{candidate} 缺少 compound_uid；请用 eval_test=true 重跑")
    return frame


def _summarize(
    frame: pd.DataFrame,
    *,
    w_t: float,
    w_c: float,
    min_samples: int,
) -> tuple[float, pd.Series]:
    work = frame.assign(_blend=_blend(frame, w_t, w_c))
    per_drug = _per_drug_scc(work, "_blend", min_samples=min_samples)
    return (float(per_drug.mean()) if len(per_drug) else float("nan")), per_drug


def main() -> None:
    parser = argparse.ArgumentParser(description="T2 blend calibration")
    parser.add_argument("--run", action="append", required=True, help="name=path")
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261123)
    args = parser.parse_args()

    runs = dict(_parse_run(value) for value in args.run)
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {"runs": {}, "paired": {}}
    test_scc: dict[tuple[str, int], pd.Series] = {}
    per_drug_rows: list[pd.DataFrame] = []

    for name, run_dir in runs.items():
        folds = sorted(
            int(fold.name.removeprefix("fold"))
            for fold in run_dir.glob("fold*")
            if fold.is_dir() and fold.name.removeprefix("fold").isdigit()
        )
        run_report: dict[str, Any] = {"folds": {}, "dir": str(run_dir)}
        for fold in folds:
            fold_dir = run_dir / f"fold{fold}"
            val = _load_frame(fold_dir, "val")
            if val is None:
                continue
            w_t, w_c, best_val = _select_weights(val, min_samples=args.min_samples)
            val_selected, _ = _summarize(val, w_t=w_t, w_c=w_c, min_samples=args.min_samples)
            entry: dict[str, Any] = {
                "selected_val_scc": best_val,
                "selected": {"w_t": w_t, "w_c": w_c},
                "val": {
                    "selected": val_selected,
                    "p1_only": _summarize(val, w_t=0.0, w_c=0.0, min_samples=args.min_samples)[0],
                    "fixed_0.5": _summarize(val, w_t=0.5, w_c=0.0, min_samples=args.min_samples)[0],
                    "tower_only": _summarize(val, w_t=1.0, w_c=0.0, min_samples=args.min_samples)[
                        0
                    ],
                },
            }
            test = _load_frame(fold_dir, "test")
            if test is not None:
                test_selected, per_drug_selected = _summarize(
                    test, w_t=w_t, w_c=w_c, min_samples=args.min_samples
                )
                test_fixed, per_drug_fixed = _summarize(
                    test, w_t=0.5, w_c=0.0, min_samples=args.min_samples
                )
                test_p1, per_drug_p1 = _summarize(
                    test, w_t=0.0, w_c=0.0, min_samples=args.min_samples
                )
                entry["test"] = {
                    "selected": test_selected,
                    "fixed_0.5": test_fixed,
                    "p1_only": test_p1,
                }
                test_scc[(name, fold)] = per_drug_selected
                common = per_drug_p1.index.intersection(per_drug_selected.index)
                per_drug_rows.append(
                    pd.DataFrame(
                        {
                            "run": name,
                            "fold": fold,
                            "compound_uid": common,
                            "scc_p1": per_drug_p1.reindex(common).values,
                            "scc_selected": per_drug_selected.reindex(common).values,
                            "scc_fixed_0.5": per_drug_fixed.reindex(common).values,
                        }
                    )
                )
            run_report["folds"][str(fold)] = entry
        report["runs"][name] = run_report

    names = list(runs)
    for left in names:
        for right in names:
            if left >= right:
                continue
            deltas: list[pd.Series] = []
            for fold in sorted({key[1] for key in test_scc if key[0] == left}):
                if (left, fold) not in test_scc or (right, fold) not in test_scc:
                    continue
                common = test_scc[(left, fold)].index.intersection(test_scc[(right, fold)].index)
                deltas.append(
                    test_scc[(left, fold)].reindex(common) - test_scc[(right, fold)].reindex(common)
                )
            if deltas:
                pooled = pd.concat(deltas).rename("delta")
                report["paired"][f"{left} - {right}"] = _bootstrap_mean(
                    pooled,
                    iterations=args.bootstrap,
                    seed=args.seed,
                )

    if per_drug_rows:
        pd.concat(per_drug_rows, ignore_index=True).to_parquet(
            out / "per_drug_scc.parquet", index=False
        )
    (out / "calibration.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()

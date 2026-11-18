"""训练前置检查：确认 clone（仅 LFS 的 processed/splits）也能跑教师训练。

`kite-drp data preflight --target teacher` 会逐项检查模型侧依赖，不读取 canonical，
并把「缺什么 / 怎么补」写进输出。任何 required 项失败时退出码为 1。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kite_drp.data import io
from kite_drp.data.graph_inputs import validate_graph_inputs
from kite_drp.data.pairs import PAIR_COLUMNS


def _check(name: str, ok: bool, detail: str, *, required: bool = True) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "required": required, "detail": detail}


def run_preflight(
    *,
    processed_dir: str | Path,
    splits_dir: str | Path,
    target: str = "teacher",
    split_id: str = "p1_lco_s0",
    n_folds: int = 5,
) -> dict[str, Any]:
    processed = Path(processed_dir)
    splits = Path(splits_dir)
    checks: list[dict[str, Any]] = []

    # 1) teacher_inputs + index
    teacher_path = processed / "teacher_inputs.npz"
    index_path = processed / "teacher_index.json"
    if teacher_path.is_file() and index_path.is_file():
        arrays = io.read_npz(teacher_path)
        index = io.read_json(index_path)
        shape_ok = all(
            tuple(arrays[key].shape) == tuple(index["shapes"][key])
            for key in ("ge", "cnv", "mu_A", "mu_B")
        )
        checks.append(
            _check(
                "teacher_inputs",
                shape_ok,
                f"ge={arrays['ge'].shape}, cnv={arrays['cnv'].shape}, "
                f"mu_A={arrays['mu_A'].shape}, cnv_observed={int(arrays['cnv_observed_mask'].sum())}",
            )
        )
    else:
        checks.append(_check("teacher_inputs", False, f"缺少 {teacher_path} 或 {index_path}"))

    # 2) teacher_pairs（训练侧唯一标签来源）
    pairs_path = processed / "teacher_pairs.parquet"
    if pairs_path.is_file():
        pairs = io.read_parquet(pairs_path)
        missing = [column for column in PAIR_COLUMNS if column not in pairs.columns]
        rows_teacher = int(pairs["has_teacher_omics"].sum()) if "has_teacher_omics" in pairs else 0
        ok = not missing and len(pairs) > 0 and rows_teacher > 0
        checks.append(
            _check(
                "teacher_pairs",
                ok,
                f"rows={len(pairs)}, rows_teacher_omics={rows_teacher}, missing={missing}",
            )
        )
    else:
        checks.append(_check("teacher_pairs", False, f"缺少 {pairs_path}"))

    # 3) 药物特征 + 分子图（GIN 可选）
    drug_path = processed / "drug_features.npz"
    if drug_path.is_file():
        drugs = io.read_npz(drug_path)
        has_graphs = "graph_has_graph" in drugs and bool(drugs["graph_has_graph"].any())
        checks.append(
            _check(
                "drug_features",
                "morgan" in drugs and "targets" in drugs,
                f"n={len(drugs['compound_uid'])}, morgan={drugs['morgan'].shape}, "
                f"targets={drugs['targets'].shape}",
            )
        )
        checks.append(
            _check(
                "drug_graphs",
                has_graphs,
                "GIN 分子图可用" if has_graphs else "缺分子图：B0/T0(Morgan+targets) 可跑，GIN 不可用",
                required=False,
            )
        )
    else:
        checks.append(_check("drug_features", False, f"缺少 {drug_path}"))

    # 4) 图输入（T1 用；B0/T0 可缺）
    graph_path = processed / "graph_inputs.npz"
    if graph_path.is_file():
        problems = validate_graph_inputs(io.read_npz(graph_path))
        arrays = io.read_npz(graph_path)
        checks.append(
            _check(
                "graph_inputs",
                not problems,
                f"genes={len(arrays['genes'])}, ppi_edges={arrays['ppi_edges'].shape[1]}, "
                f"dti_edges={arrays['dti_edges'].shape[1]}; problems={problems}",
                required=target in {"teacher_t1", "all"},
            )
        )
    else:
        checks.append(
            _check(
                "graph_inputs",
                False,
                f"缺少 {graph_path}：T1 不可用",
                required=target in {"teacher_t1", "all"},
            )
        )

    # 5) split + fit_stats
    split_path = splits / f"{split_id}.json"
    if split_path.is_file():
        split = io.read_json(split_path)
        folds = sorted(int(item["fold"]) for item in split.get("folds", []))
        checks.append(
            _check(
                "split",
                len(folds) == n_folds,
                f"{split_id}: protocol={split.get('protocol')}, folds={folds}",
            )
        )
        missing_folds = []
        for fold in folds:
            fold_dir = processed / "fit_stats" / split_id / f"fold{fold}"
            required_files = [
                "label_scaler.parquet",
                "expr_scaler.parquet",
                "cnv_scaler.parquet",
                "hvg.json",
                "fit_scope.json",
            ]
            if not all((fold_dir / name).is_file() for name in required_files):
                missing_folds.append(fold)
        checks.append(
            _check(
                "fit_stats",
                not missing_folds,
                f"{split_id}: 缺失 fold {missing_folds}" if missing_folds else f"{split_id}: {len(folds)} 折齐全",
            )
        )
    else:
        checks.append(_check("split", False, f"缺少 {split_path}"))

    # 6) clone 可训练性：只用 processed + splits + fit_stats 构造数据集
    try:
        from kite_drp.data.dataset import TeacherPairDataset

        dataset = TeacherPairDataset(
            processed,
            split_id=split_id,
            fold=0,
            pool="M-716",
            split_dir=splits,
        )
        counts = {
            name: int((dataset.split_label == name).sum())
            for name in ("train", "val", "test")
        }
        checks.append(
            _check(
                "dataset_offline",
                counts["train"] > 0,
                f"TeacherPairDataset(fold0) 可用；pairs={len(dataset)}, split={counts}",
            )
        )
    except Exception as exc:  # pragma: no cover - 环境依赖
        checks.append(_check("dataset_offline", False, f"构造失败：{exc}"))

    # 7) 可选依赖（不影响数据缺口判定）
    for module in ("rdkit", "torch", "torch_geometric"):
        try:
            __import__(module)
            checks.append(_check(f"import:{module}", True, "ok", required=False))
        except Exception:
            checks.append(_check(f"import:{module}", False, "未安装（按需 uv sync --extra）", required=False))

    required_failures = [item for item in checks if item["required"] and not item["ok"]]
    optional_failures = [item for item in checks if not item["required"] and not item["ok"]]
    return {
        "status": "ok" if not required_failures else "error",
        "target": target,
        "split": split_id,
        "checks": checks,
        "warnings": [item["name"] for item in optional_failures],
        "missing": [item["name"] for item in required_failures],
    }


__all__ = ["run_preflight"]

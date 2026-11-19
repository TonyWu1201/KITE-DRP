"""教师训练入口（Hydra）。

用法：
    uv run python -m kite_drp.train.teacher_train teacher=b0 split=lco
    uv run python -m kite_drp.train.teacher_train teacher=t0 split=lco fold=all seed=0
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from kite_drp.common.paths import CONFIGS_DIR, REPO_ROOT
from kite_drp.data.dataset import TeacherPairDataset
from kite_drp.models.teacher.b0 import BilinearTeacher
from kite_drp.models.teacher.common import count_parameters
from kite_drp.models.teacher.t0 import T0Teacher
from kite_drp.train.data import build_pair_tensors, make_loader
from kite_drp.train.engine import evaluate, fit_teacher, resolve_device, set_seed
from kite_drp.train.features import (
    build_cell_features,
    build_drug_features,
    build_fold_similarity,
    resolve_neighbor_source,
    resolve_similarity_config,
)


def _build_model(
    config: dict[str, Any],
    *,
    pathway_dim: int,
    morgan_dim: int,
    n_target_genes: int,
) -> torch.nn.Module:
    teacher = config["teacher"]
    name = str(teacher.get("name", "t0"))
    common = {
        "pathway_dim": pathway_dim,
        "morgan_dim": morgan_dim,
        "n_target_genes": n_target_genes,
        "d_model": int(teacher.get("d_model", 256)),
        "hidden": int(teacher.get("hidden", 256)),
        "dropout": float(teacher.get("dropout", 0.1)),
        "use_gin": bool(teacher.get("use_gin", False)),
        "use_morgan": bool(teacher.get("use_morgan", True)),
        "use_targets": bool(teacher.get("use_targets", True)),
        "target_embed_dim": int(teacher.get("target_embed_dim", 16)),
        "modalities": list(teacher.get("modalities", ["GE", "MU", "CNV"])),
    }
    if name == "b0":
        return BilinearTeacher(**common)
    if name in {"t0", "t0_p0", "t0_p1"}:
        return T0Teacher(
            **common,
            n_heads=int(teacher.get("n_heads", 4)),
            n_layers=int(teacher.get("n_layers", 2)),
            top_k=int(teacher.get("top_k", 64)),
            use_neighbor=bool(teacher.get("use_neighbor", True))
            and bool(config.get("similarity", {}).get("enabled", True)),
            modality_dropout=float(teacher.get("modality_dropout", 0.1)),
            readout=str(teacher.get("readout", "sum")),
            use_dense_shortcut=bool(teacher.get("use_dense_shortcut", False)),
            token_types=bool(teacher.get("token_types", False)),
            rank_embedding=bool(teacher.get("rank_embedding", False)),
            cell_readout=str(teacher.get("cell_readout", "mean")),
            masked_modalities=bool(teacher.get("masked_modalities", False)),
            cross_gate=bool(teacher.get("cross_gate", False)),
            shared_pathway=bool(teacher.get("shared_pathway", False)),
            shared_pathway_proj=bool(teacher.get("shared_pathway_proj", False)),
            use_cls_proj=bool(teacher.get("use_cls_proj", True)),
            cls_proj_mode=str(teacher.get("cls_proj_mode", "full")),
            cls_proj_rank=int(teacher.get("cls_proj_rank", 64)),
            target_branch=bool(teacher.get("target_branch", False)),
            target_gene_layers=int(teacher.get("target_gene_layers", 1)),
        )
    raise KeyError(f"本轮只实现 b0/t0/t0_p0/t0_p1，收到 teacher={name}")


def _split_id(config: dict[str, Any]) -> str:
    split = config["split"]
    return f"{split['protocol']}_s{int(split['seed'])}"


def _fold_list(config: dict[str, Any], total: int) -> list[int]:
    value = config.get("fold", 0)
    if isinstance(value, str):
        text = value.strip()
        if text == "all":
            return list(range(total))
        if "," in text:
            folds = sorted({int(item) for item in text.split(",") if item.strip()})
            if any(fold < 0 or fold >= total for fold in folds):
                raise ValueError(f"fold 越界：{folds}（共 {total} 折）")
            return folds
    elif isinstance(value, (list, tuple)):
        folds = sorted({int(item) for item in value})
        if any(fold < 0 or fold >= total for fold in folds):
            raise ValueError(f"fold 越界：{folds}（共 {total} 折）")
        return folds
    return [int(value)]


def _run_fold(
    config: dict[str, Any],
    *,
    fold: int,
    split_id: str,
    run_dir: Path,
) -> dict[str, Any]:
    paths = config["paths"]
    processed = REPO_ROOT / paths["processed_dir"]
    splits = REPO_ROOT / paths["splits_dir"]
    seed = int(config.get("seed", 0))
    fold_seed = seed * 100 + fold
    set_seed(fold_seed)

    base = TeacherPairDataset(
        processed,
        split_id=split_id,
        fold=fold,
        pool=str(config.get("pool", "M-716")),
        split_dir=splits,
    )
    features = build_cell_features(processed, split_id=split_id, fold=fold)
    drugs = build_drug_features(processed)

    active_modalities = list(config["teacher"].get("modalities", ["GE", "MU", "CNV"]))
    similarity_cfg = resolve_similarity_config(config, active_modalities)
    neighbor_source = resolve_neighbor_source(config["teacher"], active_modalities)
    similarity_enabled = (
        bool(config.get("similarity", {}).get("enabled", True))
        and str(config["teacher"].get("name", "t0")) in {"t0", "t0_p0", "t0_p1"}
        and bool(config["teacher"].get("use_neighbor", True))
    )
    target_max_tokens = (
        int(config["teacher"].get("target_max_tokens", 64))
        if bool(config["teacher"].get("target_branch", False))
        else 0
    )
    if similarity_enabled:
        fold_def = next(item for item in base.split["folds"] if int(item["fold"]) == fold)
        neighbors, weights, is_train, similarity_meta = build_fold_similarity(
            features,
            train_cells=list(fold_def["train"]),
            config=similarity_cfg,
            processed_dir=processed,
            split_id=split_id,
            fold=fold,
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
        similarity_meta = {"enabled": False, "neighbor_source": neighbor_source}
        tensors = build_pair_tensors(
            features,
            drugs,
            target_max_tokens=target_max_tokens,
            target_modalities=active_modalities,
        )

    train_cfg = dict(config["train"])
    train_cfg.update(config["teacher"].get("train", {}))
    use_gin = bool(config["teacher"].get("use_gin", False))
    batch_size = int(train_cfg.get("batch_size", 512))
    rank_weight = float(train_cfg.get("rank_weight", 0.0))
    group_by_drug = bool(train_cfg.get("group_by_drug", False)) or rank_weight > 0.0
    train_loader = make_loader(
        base,
        tensors,
        drugs,
        split="train",
        batch_size=batch_size,
        shuffle=True,
        use_gin=use_gin,
        seed=fold_seed,
        group_by_drug=group_by_drug,
        drugs_per_batch=int(train_cfg.get("drugs_per_batch", 8)),
        cells_per_drug=int(train_cfg.get("cells_per_drug", 64)),
    )
    val_loader = make_loader(
        base,
        tensors,
        drugs,
        split="val",
        batch_size=batch_size,
        shuffle=False,
        use_gin=use_gin,
    )
    test_loader = (
        make_loader(
            base,
            tensors,
            drugs,
            split="test",
            batch_size=batch_size,
            shuffle=False,
            use_gin=use_gin,
        )
        if bool(train_cfg.get("eval_test", False))
        else None
    )

    model = _build_model(
        config,
        pathway_dim=int(features.ge.shape[1]),
        morgan_dim=int(drugs.morgan.shape[1]),
        n_target_genes=drugs.target_dim,
    )
    params = count_parameters(model)
    target = int(config["teacher"].get("params_target", 5_000_000))
    variant = str(config["teacher"].get("variant", config["teacher"]["name"]))
    print(
        json.dumps(
            {
                "variant": variant,
                "fold": fold,
                "split": split_id,
                "params": params,
                "params_target": target,
                "train_pairs": int(base.indices("train").size),
                "val_pairs": int(base.indices("val").size),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if params > target:
        raise ValueError(f"教师参数 {params:,} 超过目标 {target:,}")

    device = resolve_device(str(train_cfg.get("device", "auto")))
    model = model.to(device)
    fold_dir = run_dir / f"fold{fold}"
    result = fit_teacher(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_cfg,
        device=device,
        run_dir=fold_dir,
        base=base,
        fold=fold,
    )
    result["params"] = params
    result["similarity"] = similarity_meta
    result["train_pairs"] = int(base.indices("train").size)
    result["val_pairs"] = int(base.indices("val").size)
    if test_loader is not None:
        test_metrics, test_frame = evaluate(model, test_loader, device=device, base=base)
        result["test"] = test_metrics
        test_frame.to_parquet(fold_dir / "test_predictions.parquet", index=False)
        (fold_dir / "test_metrics.json").write_text(
            json.dumps(test_metrics, ensure_ascii=False, indent=2, default=float),
            encoding="utf-8",
        )
    model_spec = {
        "model_class": str(config["teacher"]["name"]),
        "pathway_dim": int(features.ge.shape[1]),
        "morgan_dim": int(drugs.morgan.shape[1]),
        "n_target_genes": int(drugs.target_dim),
        "n_genes": int(features.gene_ge.shape[1]),
        "target_max_tokens": target_max_tokens,
        "teacher": config["teacher"],
    }
    (fold_dir / "model_spec.json").write_text(
        json.dumps(model_spec, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return result


@hydra.main(version_base=None, config_path=str(CONFIGS_DIR), config_name="config")
def main(config: DictConfig) -> None:
    resolved = OmegaConf.to_container(config, resolve=True)
    assert isinstance(resolved, dict)
    seed = int(resolved.get("seed", 0))
    set_seed(seed)
    torch.set_float32_matmul_precision("high")

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output_dir / "resolved_config.yaml")

    split_id = _split_id(resolved)
    summary_path = REPO_ROOT / resolved["paths"]["splits_dir"] / f"{split_id}.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"split 文件不存在：{summary_path}")
    split_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    folds = _fold_list(resolved, len(split_payload["folds"]))

    results = []
    for fold in folds:
        print(json.dumps({"start_fold": fold, "split": split_id}), flush=True)
        results.append(_run_fold(resolved, fold=fold, split_id=split_id, run_dir=output_dir))

    variant = str(resolved["teacher"].get("variant", resolved["teacher"]["name"]))
    summary = {
        "variant": variant,
        "split": split_id,
        "seed": seed,
        "pool": resolved.get("pool", "M-716"),
        "folds": results,
        "mean_val": {
            key: float(np.nanmean([result["val"]["regression"][key] for result in results]))
            for key in ("r2", "pcc", "scc", "rmse")
        }
        | {
            key: float(np.nanmean([result["val"]["classification"][key] for result in results]))
            for key in ("auroc", "auprc", "ece")
        }
        | {
            "per_drug_scc": float(
                np.nanmean(
                    [
                        result["val"].get("per_drug", {}).get("scc_mean", float("nan"))
                        for result in results
                    ]
                )
            )
        },
    }
    if all("test" in result for result in results):
        summary["mean_test"] = {
            key: float(
                np.nanmean([result["test"]["regression"][key] for result in results])
            )
            for key in ("r2", "pcc", "scc", "rmse")
        } | {
            key: float(
                np.nanmean([result["test"]["classification"][key] for result in results])
            )
            for key in ("auroc", "auprc", "ece")
        }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    print(json.dumps({"status": "ok", "summary": str(output_dir / "summary.json")}))


if __name__ == "__main__":
    main()

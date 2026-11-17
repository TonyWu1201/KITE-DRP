"""``kite-drp data`` 命令树。"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from kite_drp.common.paths import REPO_ROOT
from kite_drp.data import io, registry
from kite_drp.data.config import DataPaths, load_data_config
from kite_drp.data.manifest import code_hash, config_hash
from kite_drp.data.splits import (
    PROTOCOLS,
    load_split,
    make_split,
    split_path,
    validate_split,
    write_split,
    write_split_manifest,
)


def _paths() -> DataPaths:
    return DataPaths().ensure()


def _fail(result: dict[str, Any]) -> int:
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 1


def _ok(result: dict[str, Any]) -> int:
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


def cmd_inventory(args: argparse.Namespace) -> int:
    from kite_drp.data.inventory import build_inventory

    paths = _paths()
    out = Path(args.out) if args.out else paths.audit / "inventory.json"
    payload = build_inventory(paths.raw)
    io.write_json(out, payload)
    return _ok({"status": "ok", "artifact": str(out), "n_files": payload["n_files"], "bytes": payload["total_bytes"]})


def cmd_canonical(args: argparse.Namespace) -> int:
    from kite_drp.data.canonical import build_canonical

    try:
        result = build_canonical(
            config_path=args.config,
            only=args.only,
            force=args.force,
            fetch_file_index=args.fetch_file_index,
        )
    except Exception as exc:  # pragma: no cover - CLI 边界
        traceback.print_exc()
        return _fail({"status": "error", "error": str(exc)})
    return _ok(result)


def _parse_seeds(value: str | None, default: list[int]) -> list[int]:
    if not value:
        return default
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _canonical_input_hashes(paths: DataPaths) -> dict[str, str]:
    result = {}
    for name in ("cell_lines", "compounds", "labels", "genes"):
        spec = registry.ARTIFACTS[name]
        if spec.path.is_file():
            result[spec.path.name] = "sha256:" + io.hash_file(spec.path)
    return result


def cmd_splits(args: argparse.Namespace) -> int:
    paths = _paths()
    config = load_data_config(args.config)
    labels = io.read_parquet(paths.canonical / "labels.parquet")
    cell_lines = io.read_parquet(paths.canonical / "cell_lines.parquet")
    compounds = io.read_parquet(paths.canonical / "compounds.parquet")
    seeds = _parse_seeds(args.seeds, list(config["splits"].get("seeds", [0, 1, 2])))
    if args.seed is not None:
        seeds = [int(args.seed)]
    if args.protocol:
        protocols = [args.protocol]
    elif args.all or not args.protocol:
        protocols = list(config["splits"].get("protocols", PROTOCOLS))
    else:
        protocols = [args.protocol]

    written: list[Path] = []
    validations: dict[str, Any] = {}
    inputs = _canonical_input_hashes(paths)
    for protocol in protocols:
        for seed in seeds:
            split = make_split(
                protocol,
                labels=labels,
                cell_lines=cell_lines,
                compounds=compounds,
                seed=seed,
                n_folds=int(config["splits"].get("n_folds", 5)),
            )
            path = write_split(
                split,
                directory=paths.splits,
                inputs=inputs,
                config_hash=config_hash(config),
                code_hash=code_hash(),
            )
            for fold in split["folds"]:
                fold["_seed"] = seed
            report = validate_split(split, labels=labels, cell_lines=cell_lines, compounds=compounds)
            validations[path.name] = report
            written.append(path)
    manifest = write_split_manifest(
        paths.splits,
        paths=written,
        inputs=inputs,
        code_hash=code_hash(),
        config_hash=config_hash(config),
        validations=validations,
    )
    errors = [
        item
        for name, report in validations.items()
        for item in report.get("warnings", [])
    ]
    return _ok(
        {
            "status": "ok",
            "splits": [str(path) for path in written],
            "manifest": str(manifest),
            "warnings": errors,
        }
    )


def cmd_fit_stats(args: argparse.Namespace) -> int:
    from kite_drp.data.fit import fit_stats_for_split

    paths = _paths()
    config = load_data_config(args.config)
    labels = io.read_parquet(paths.canonical / "labels.parquet")
    expression = io.read_parquet(paths.canonical / "expression.parquet")
    cnv = io.read_parquet(paths.canonical / "cnv.parquet")
    mutation_matrix, mutation_models, mutation_genes = io.load_packed_matrix(
        paths.canonical / "mutations_A.npz", name="matrix"
    )
    external_panel = None
    panel_config = config.get("raw_files", {}).get("mutation_panel")
    if panel_config:
        universe = io.read_json(paths.canonical / "gene_universes.json")
        external_panel = universe.get("panel")
    if args.split:
        candidates = [Path(args.split)] if Path(args.split).is_file() else sorted(paths.splits.glob(f"{args.split}.json"))
    else:
        candidates = sorted(paths.splits.glob("*_s*.json"))
    if not candidates:
        return _fail({"status": "error", "error": f"没有匹配的 split：{args.split or '--all'}"})
    summaries = {}
    for path in candidates:
        split = load_split(path)
        summary = fit_stats_for_split(
            split=split,
            labels=labels,
            expression=expression,
            cnv=cnv,
            root=args.root or (paths.processed / "fit_stats"),
            seed=split["seed"],
            min_obs_per_compound=int(config["label"].get("min_obs_per_compound", 30)),
            n_hvg=int(args.n_hvg),
            force=args.force,
            mutation_matrix=mutation_matrix,
            mutation_model_ids=mutation_models,
            mutation_genes=mutation_genes,
            external_panel=external_panel,
        )
        summaries[path.name] = summary
    return _ok({"status": "ok", "fit_stats": summaries})


def cmd_build(args: argparse.Namespace) -> int:
    from kite_drp.data.assembly import build_processed

    paths = _paths()
    config = load_data_config(args.config)
    result = build_processed(
        canonical_dir=paths.canonical,
        processed_dir=paths.processed,
        config=config,
        stage=args.stage,
    )
    return _ok({"status": "ok", **result})


def cmd_validate(args: argparse.Namespace) -> int:
    from kite_drp.data.validation import validate_all

    paths = _paths()
    result = validate_all(paths=paths, strict=args.strict, run_leakage=not args.no_leakage)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result["status"] == "ok" else 1


def cmd_report(args: argparse.Namespace) -> int:
    from kite_drp.data.report import build_report

    target = build_report(_paths())
    return _ok({"status": "ok", "report": str(target)})


def cmd_pipeline(args: argparse.Namespace) -> int:
    paths = _paths()
    config = load_data_config(args.config)
    steps: list[tuple[str, argparse.Namespace]] = []
    ns = argparse.Namespace(
        config=args.config,
        only=None,
        force=args.force,
        fetch_file_index=args.fetch_file_index,
    )
    steps.append(("inventory", argparse.Namespace(out=None)))
    steps.append(("canonical", ns))
    steps.append(
        (
            "splits",
            argparse.Namespace(
                config=args.config,
                protocol=None,
                all=True,
                seed=None,
                seeds=None,
            ),
        )
    )
    steps.append(
        (
            "fit-stats",
            argparse.Namespace(
                config=args.config,
                split=None,
                root=None,
                n_hvg=2000,
                force=args.force,
            ),
        )
    )
    steps.append(
        (
            "build",
            argparse.Namespace(config=args.config, stage="all"),
        )
    )
    steps.append(("validate", argparse.Namespace(strict=False, no_leakage=False)))
    steps.append(("report", argparse.Namespace()))
    results: dict[str, Any] = {}
    for name, step_args in steps:
        handler = {
            "inventory": cmd_inventory,
            "canonical": cmd_canonical,
            "splits": cmd_splits,
            "fit-stats": cmd_fit_stats,
            "build": cmd_build,
            "validate": cmd_validate,
            "report": cmd_report,
        }[name]
        try:
            code = handler(step_args)
        except Exception as exc:
            traceback.print_exc()
            return _fail({"status": "error", "step": name, "error": str(exc), "results": results})
        results[name] = {"exit_code": code}
        if code != 0:
            return _fail({"status": "error", "step": name, "results": results})
    return _ok({"status": "ok", "results": results})


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    data = subparsers.add_parser("data", help="数据管线：清单/预处理/划分/装配/校验")
    data_sub = data.add_subparsers(dest="data_command", metavar="<command>")

    p = data_sub.add_parser("inventory", help="L0 原始数据清单")
    p.add_argument("--out", type=str, default=None)
    p.set_defaults(func=cmd_inventory)

    p = data_sub.add_parser("canonical", help="L1 canonical 产物")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--only", choices=["tcga", "priors", "drug_features", "all"], default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--fetch-file-index", action="store_true")
    p.set_defaults(func=cmd_canonical)

    p = data_sub.add_parser("splits", help="生成 P0–P4/LPO split")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--protocol", type=str, default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--seeds", type=str, default=None)
    p.set_defaults(func=cmd_splits)

    p = data_sub.add_parser("fit-stats", help="逐 split×fold 拟合统计量")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--split", type=str, default=None)
    p.add_argument("--root", type=str, default=None)
    p.add_argument("--n-hvg", dest="n_hvg", type=int, default=2000)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_fit_stats)

    p = data_sub.add_parser("build", help="装配 processed model-input")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--stage", choices=["all", "teacher", "student", "vae", "tcga"], default="all")
    p.set_defaults(func=cmd_build)

    p = data_sub.add_parser("validate", help="产物/划分/泄漏校验")
    p.add_argument("--all", action="store_true")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--no-leakage", action="store_true")
    p.set_defaults(func=cmd_validate)

    p = data_sub.add_parser("report", help="生成质量报告")
    p.set_defaults(func=cmd_report)

    p = data_sub.add_parser("pipeline", help="一键串联（core/full）")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--profile", choices=["core", "full"], default="core")
    p.add_argument("--force", action="store_true")
    p.add_argument("--fetch-file-index", action="store_true")
    p.add_argument("--skip-online", action="store_true")
    p.set_defaults(func=cmd_pipeline)


__all__ = ["register"]

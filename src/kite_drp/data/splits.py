"""划分协议 P0–P4/LPO 的生成、校验与加载。

铁律：split 只依赖身份信息（细胞系/化合物/组织/scaffold）与标签**数量**，
不读取标签值；所有折内统计量在 split 之后由 :mod:`kite_drp.data.fit` 拟合。
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from kite_drp.data import io, registry
from kite_drp.data.drugs import compound_split_groups

PROTOCOLS = ("p0_random", "p1_lco", "p2_ldo", "p3_scaffold", "p4_lto", "lpo")
CELL_PROTOCOLS = {"p1_lco"}
COMPOUND_PROTOCOLS = {"p2_ldo", "p3_scaffold"}
PAIR_PROTOCOLS = {"p0_random", "lpo"}
LINEAGE_PROTOCOL = "p4_lto"


def _greedy_groups(counts: dict[str, int], *, seed: int, n_folds: int) -> list[list[str]]:
    """按含标签数降序做确定性贪心装箱（同 load 取序号小的折）。"""
    order = sorted(counts, key=lambda group: (-counts[group], io.stable_seed(seed, group, modulo=2**32)))
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    loads = [0] * n_folds
    for group in order:
        target = min(range(n_folds), key=lambda index: (loads[index], index))
        folds[target].append(group)
        loads[target] += counts[group]
    return [sorted(fold) for fold in folds]


def _fold_doc(
    fold: int,
    *,
    train_groups: list[str],
    val_groups: list[str],
    test_groups: list[str],
    train: list[str],
    val: list[str],
    test: list[str],
    stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "fold": fold,
        "train_groups": sorted(train_groups),
        "val_groups": sorted(val_groups),
        "test_groups": sorted(test_groups),
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(test),
        "stats": stats or {},
    }


def _group_counts(keys: Iterable[str], labels: pd.DataFrame, key_column: str) -> dict[str, int]:
    counts = labels.groupby(key_column).size().to_dict()
    return {str(key): int(counts.get(key, 0)) for key in keys}


def make_split(
    protocol: str,
    *,
    labels: pd.DataFrame,
    cell_lines: pd.DataFrame,
    compounds: pd.DataFrame,
    seed: int = 0,
    n_folds: int = 5,
    min_lineage_cells: int = 20,
) -> dict[str, Any]:
    """生成单个协议 × seed 的 split JSON（不写盘）。"""
    protocol = _canonical_protocol(protocol)
    folds: list[dict[str, Any]] = []
    group_key = ""
    level = ""
    degraded = False
    notes: list[str] = []

    pools = labels[labels["eligible_cell_line"] & labels["mapped"]].copy()
    if protocol in CELL_PROTOCOLS:
        cell_to_group = dict(zip(cell_lines["cell_line_uid"], cell_lines["source_group"], strict=False))
        pools["_group"] = pools["cell_line_uid"].map(cell_to_group).fillna(
            pools["cell_line_uid"].map(lambda value: f"name:{value}")
        )
        counts = _group_counts(pools["_group"].unique(), pools, "_group")
        groups = _greedy_groups(counts, seed=seed, n_folds=n_folds)
        group_to_cells: dict[str, list[str]] = defaultdict(list)
        pool_cells = set(pools["cell_line_uid"])
        for cell, group in zip(cell_lines["cell_line_uid"], cell_lines["source_group"], strict=False):
            if cell in pool_cells:
                group_to_cells[group].append(cell)
        for k in range(n_folds):
            test_groups = groups[k]
            val_groups = groups[(k + 1) % n_folds]
            train_groups = [group for j, fold in enumerate(groups) if j not in {k, (k + 1) % n_folds} for group in fold]
            train = sorted(cell for group in train_groups for cell in group_to_cells.get(group, []))
            val = sorted(cell for group in val_groups for cell in group_to_cells.get(group, []))
            test = sorted(cell for group in test_groups for cell in group_to_cells.get(group, []))
            folds.append(
                _fold_doc(
                    k,
                    train_groups=train_groups,
                    val_groups=val_groups,
                    test_groups=test_groups,
                    train=train,
                    val=val,
                    test=test,
                    stats=_fold_stats(pools, train, val, test, level="cell_line"),
                )
            )
        group_key, level = "source_group", "cell_line"

    elif protocol in COMPOUND_PROTOCOLS:
        if protocol == "p3_scaffold":
            groups_series, degraded = compound_split_groups(compounds)
            groups_series.index = compounds["compound_uid"].to_numpy()
            group_key = "target_family" if degraded else "scaffold"
            if degraded:
                notes.append("SMILES/scaffold unavailable or incomplete: P3 degraded to target-family split")
            else:
                missing = int(compounds["scaffold"].isna().sum()) if "scaffold" in compounds else 0
                if missing:
                    notes.append(
                        f"{missing} compounds have no Bemis-Murcko scaffold; "
                        "assigned singleton target/compound groups"
                    )
        else:
            groups_series = pd.Series(compounds["compound_uid"].to_numpy(), index=compounds["compound_uid"])
            group_key = "compound_uid"
        compound_to_group = {uid: str(group) for uid, group in groups_series.items()}
        pools["_group"] = pools["compound_uid"].map(compound_to_group).fillna(
            pools["compound_uid"].map(lambda value: f"compound:{value}")
        )
        counts = _group_counts(pools["_group"].unique(), pools, "_group")
        groups = _greedy_groups(counts, seed=seed, n_folds=n_folds)
        group_to_compounds: dict[str, list[str]] = defaultdict(list)
        for compound, group in compound_to_group.items():
            group_to_compounds[group].append(compound)
        for k in range(n_folds):
            test_groups = groups[k]
            val_groups = groups[(k + 1) % n_folds]
            train_groups = [group for j, fold in enumerate(groups) if j not in {k, (k + 1) % n_folds} for group in fold]
            train_compounds = sorted(
                compound for group in train_groups for compound in group_to_compounds.get(group, [])
            )
            val_compounds = sorted(
                compound for group in val_groups for compound in group_to_compounds.get(group, [])
            )
            test_compounds = sorted(
                compound for group in test_groups for compound in group_to_compounds.get(group, [])
            )
            train = sorted(pools[pools["_group"].isin(train_groups)]["cell_line_uid"].unique().tolist())
            val = sorted(pools[pools["_group"].isin(val_groups)]["cell_line_uid"].unique().tolist())
            test = sorted(pools[pools["_group"].isin(test_groups)]["cell_line_uid"].unique().tolist())
            fold_doc = _fold_doc(
                k,
                train_groups=train_groups,
                val_groups=val_groups,
                test_groups=test_groups,
                train=train,
                val=val,
                test=test,
                stats=_fold_stats(pools, train, val, test, level="compound", groups=(train_groups, val_groups, test_groups)),
            )
            fold_doc["train_compounds"] = train_compounds
            fold_doc["val_compounds"] = val_compounds
            fold_doc["test_compounds"] = test_compounds
            folds.append(fold_doc)
        level = "compound"

    elif protocol == LINEAGE_PROTOCOL:
        pool_cells = cell_lines[cell_lines["in_main_pool"]].copy()
        lineage_counts = pool_cells["oncotree_lineage"].value_counts()
        test_lineages = sorted(lineage_counts[lineage_counts >= min_lineage_cells].index.tolist())
        all_lineages = sorted(pool_cells["oncotree_lineage"].dropna().unique().tolist())
        for k, test_lineage in enumerate(test_lineages):
            remaining = [lineage for lineage in all_lineages if lineage != test_lineage]
            eligible_val = [
                lineage
                for lineage in remaining
                if int((pool_cells["oncotree_lineage"] == lineage).sum()) >= 10
            ]
            val_pool = eligible_val or remaining
            val_lineage = val_pool[k % len(val_pool)] if val_pool else None
            train_lineages = [lineage for lineage in remaining if lineage != val_lineage]
            train = sorted(pool_cells[pool_cells["oncotree_lineage"].isin(train_lineages)]["cell_line_uid"].tolist())
            val = sorted(pool_cells[pool_cells["oncotree_lineage"] == val_lineage]["cell_line_uid"].tolist())
            test = sorted(pool_cells[pool_cells["oncotree_lineage"] == test_lineage]["cell_line_uid"].tolist())
            folds.append(
                _fold_doc(
                    k,
                    train_groups=train_lineages,
                    val_groups=[val_lineage] if val_lineage else [],
                    test_groups=[test_lineage],
                    train=train,
                    val=val,
                    test=test,
                    stats=_fold_stats(pools, train, val, test, level="cell_line"),
                )
            )
        group_key, level = "oncotree_lineage", "cell_line"
        n_folds = len(folds)

    elif protocol in PAIR_PROTOCOLS:
        group_key, level = "observed_pair", "pair"
        n_buckets = 10 if protocol == "p0_random" else 5
        if protocol == "p0_random":
            rule = "bucket=stable_seed(seed|cell_line_uid|compound_uid)%10; test=(fold+9)%10; val=(fold+8)%10"
        else:
            rule = "bucket=stable_seed(seed|cell_line_uid|compound_uid)%5; test=fold; val=(fold+1)%5"
        buckets = [
            io.stable_seed(seed, cell, compound, modulo=n_buckets)
            for cell, compound in zip(pools["cell_line_uid"], pools["compound_uid"], strict=False)
        ]
        pools["_bucket"] = buckets
        for k in range(n_folds):
            if protocol == "p0_random":
                test_bucket, val_bucket = (k + 9) % n_buckets, (k + 8) % n_buckets
            else:
                test_bucket, val_bucket = k, (k + 1) % n_buckets
            if k >= n_buckets:
                continue
            test_mask = pools["_bucket"] == test_bucket
            val_mask = pools["_bucket"] == val_bucket
            train_mask = ~(test_mask | val_mask)
            train = sorted(pools.loc[train_mask, "cell_line_uid"].unique().tolist())
            val = sorted(pools.loc[val_mask, "cell_line_uid"].unique().tolist())
            test = sorted(pools.loc[test_mask, "cell_line_uid"].unique().tolist())
            folds.append(
                {
                    "fold": k,
                    "train_groups": [],
                    "val_groups": [],
                    "test_groups": [],
                    "train": train,
                    "val": val,
                    "test": test,
                    "assignment_rule": rule,
                    "stats": {
                        "n_train_pairs": int(train_mask.sum()),
                        "n_val_pairs": int(val_mask.sum()),
                        "n_test_pairs": int(test_mask.sum()),
                        "n_train_cells": len(train),
                        "n_val_cells": len(val),
                        "n_test_cells": len(test),
                    },
                }
            )
        n_folds = len(folds)

    else:
        raise ValueError(f"未知协议：{protocol}（可用：{', '.join(PROTOCOLS)}）")

    return {
        "schema_version": "1.0",
        "protocol": protocol,
        "seed": int(seed),
        "n_folds": int(n_folds),
        "group_key": group_key,
        "level": level,
        "degraded": degraded,
        "notes": notes,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code_git_commit": _git_commit(),
        "folds": folds,
    }


def _fold_stats(
    pools: pd.DataFrame,
    train: list[str],
    val: list[str],
    test: list[str],
    *,
    level: str,
    groups: tuple[list[str], list[str], list[str]] | None = None,
) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
    }
    if groups:
        train_groups, val_groups, test_groups = groups
        stats.update(
            {
                "n_labels_train": int(pools["_group"].isin(train_groups).sum()),
                "n_labels_val": int(pools["_group"].isin(val_groups).sum()),
                "n_labels_test": int(pools["_group"].isin(test_groups).sum()),
                "n_compounds_train": len(train_groups),
                "n_compounds_val": len(val_groups),
                "n_compounds_test": len(test_groups),
            }
        )
    else:
        stats.update(
            {
                "n_labels_train": int(pools["cell_line_uid"].isin(train).sum()),
                "n_labels_val": int(pools["cell_line_uid"].isin(val).sum()),
                "n_labels_test": int(pools["cell_line_uid"].isin(test).sum()),
            }
        )
        if level == "compound":
            stats["n_compounds_train"] = int(
                pools[pools["cell_line_uid"].isin(train)]["compound_uid"].nunique()
            )
    return stats


def _canonical_protocol(protocol: str) -> str:
    alias = {
        "p0": "p0_random",
        "random": "p0_random",
        "p1": "p1_lco",
        "lco": "p1_lco",
        "p2": "p2_ldo",
        "ldo": "p2_ldo",
        "p3": "p3_scaffold",
        "scaffold": "p3_scaffold",
        "p4": "p4_lto",
        "lto": "p4_lto",
        "lpo": "lpo",
        "leave_pairs_out": "lpo",
    }
    return alias.get(protocol, protocol)


def _git_commit() -> str:
    try:
        import subprocess

        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def split_path(protocol: str, seed: int, directory: str | Path | None = None) -> Path:
    directory = Path(directory) if directory else registry.resolve("split_manifest").parent
    return directory / f"{_canonical_protocol(protocol)}_s{seed}.json"


def write_split(
    split: dict[str, Any],
    *,
    directory: str | Path | None = None,
    inputs: dict[str, str] | None = None,
    config_hash: str | None = None,
    code_hash: str | None = None,
) -> Path:
    path = split_path(split["protocol"], split["seed"], directory)
    split = dict(split)
    split["input_artifacts"] = inputs or {}
    if config_hash:
        split["config_hash"] = config_hash
    if code_hash:
        split["code_hash"] = code_hash
    io.write_json(path, split)
    return path


def load_split(path: str | Path) -> dict[str, Any]:
    return io.read_json(path)


def fold_masks(fold: dict[str, Any], labels: pd.DataFrame, protocol: str) -> dict[str, pd.Series]:
    """把 fold 定义转换为标签行的布尔 mask（pair 协议按确定性规则重建）。"""
    protocol = _canonical_protocol(protocol)
    index = labels.index
    mapped = labels["mapped"].astype(bool) if "mapped" in labels.columns else pd.Series(True, index=index)
    if protocol in CELL_PROTOCOLS or protocol == LINEAGE_PROTOCOL:
        return {
            "train": labels["cell_line_uid"].isin(fold["train"]) & mapped,
            "val": labels["cell_line_uid"].isin(fold["val"]) & mapped,
            "test": labels["cell_line_uid"].isin(fold["test"]) & mapped,
        }
    if protocol in COMPOUND_PROTOCOLS:
        if "train_compounds" in fold:
            return {
                "train": labels["compound_uid"].isin(fold["train_compounds"]) & mapped,
                "val": labels["compound_uid"].isin(fold["val_compounds"]) & mapped,
                "test": labels["compound_uid"].isin(fold["test_compounds"]) & mapped,
            }
        return {
            "train": labels["compound_uid"].isin(fold["train_groups"]) & mapped,
            "val": labels["compound_uid"].isin(fold["val_groups"]) & mapped,
            "test": labels["compound_uid"].isin(fold["test_groups"]) & mapped,
        }
    if protocol in PAIR_PROTOCOLS:
        seed = int(fold.get("_seed", 0))
        n_buckets = 10 if protocol == "p0_random" else 5
        buckets = np.array(
            [
                io.stable_seed(seed, cell, compound, modulo=n_buckets)
                for cell, compound in zip(labels["cell_line_uid"], labels["compound_uid"], strict=False)
            ]
        )
        k = int(fold["fold"])
        if protocol == "p0_random":
            test_bucket, val_bucket = (k + 9) % n_buckets, (k + 8) % n_buckets
        else:
            test_bucket, val_bucket = k, (k + 1) % n_buckets
        test = pd.Series(buckets == test_bucket, index=index) & mapped
        val = pd.Series(buckets == val_bucket, index=index) & mapped
        train = ~(test | val) & mapped
        return {"train": train, "val": val, "test": test}
    raise ValueError(protocol)


def train_ids_for_fold(
    fold: dict[str, Any], labels: pd.DataFrame, protocol: str, seed: int = 0
) -> list[str]:
    fold = dict(fold)
    fold["_seed"] = seed
    masks = fold_masks(fold, labels, protocol)
    return sorted(labels.loc[masks["train"], "cell_line_uid"].unique().tolist())


def validate_split(
    split: dict[str, Any],
    *,
    labels: pd.DataFrame,
    cell_lines: pd.DataFrame,
    compounds: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """结构/泄漏/覆盖校验；返回 ``{checks, warnings, per_fold}``。"""
    protocol = _canonical_protocol(split["protocol"])
    warnings: list[str] = []
    checks: dict[str, bool] = {}
    per_fold: list[dict[str, Any]] = []
    if protocol in CELL_PROTOCOLS or protocol == LINEAGE_PROTOCOL:
        for fold in split["folds"]:
            train, val, test = set(fold["train"]), set(fold["val"]), set(fold["test"])
            checks[f"fold{fold['fold']}_disjoint"] = not (train & val or train & test or val & test)
            checks[f"fold{fold['fold']}_nonempty"] = bool(train and test)
            per_fold.append(
                {
                    "fold": fold["fold"],
                    "n_train": len(train),
                    "n_val": len(val),
                    "n_test": len(test),
                    "n_test_labels": int(labels["cell_line_uid"].isin(test).sum()),
                }
            )
        checks["no_group_overlap"] = all(
            not (set(fold["train_groups"]) & set(fold["test_groups"])) for fold in split["folds"]
        )
    elif protocol in COMPOUND_PROTOCOLS:
        for fold in split["folds"]:
            train, val, test = set(fold["train_groups"]), set(fold["val_groups"]), set(fold["test_groups"])
            checks[f"fold{fold['fold']}_disjoint"] = not (train & val or train & test or val & test)
            checks[f"fold{fold['fold']}_nonempty"] = bool(train and test)
            per_fold.append(
                {
                    "fold": fold["fold"],
                    "n_train": len(train),
                    "n_val": len(val),
                    "n_test": len(test),
                    "n_test_labels": int(
                        labels["compound_uid"].isin(
                            fold.get("test_compounds", fold.get("test_groups", []))
                        ).sum()
                    ),
                }
            )
        checks["no_group_overlap"] = all(
            not (set(fold["train_groups"]) & set(fold["test_groups"])) for fold in split["folds"]
        )
        if protocol == "p3_scaffold" and split.get("degraded"):
            warnings.append("P3 scaffold 已降级为 target-family split（缺 SMILES/RDKit）")
    else:
        fold = dict(split["folds"][0])
        fold["_seed"] = split["seed"]
        mapped = labels["mapped"].astype(bool) if "mapped" in labels.columns else pd.Series(True, index=labels.index)
        for fold_doc in split["folds"]:
            local = dict(fold_doc)
            local["_seed"] = split["seed"]
            masks = fold_masks(local, labels, protocol)
            checks[f"fold{fold_doc['fold']}_disjoint"] = not (
                (masks["train"] & masks["val"]).any()
                or (masks["train"] & masks["test"]).any()
                or (masks["val"] & masks["test"]).any()
            )
            checks[f"fold{fold_doc['fold']}_partition"] = bool(
                ((masks["train"] | masks["val"] | masks["test"]) | ~mapped).all()
            )
            per_fold.append(
                {
                    "fold": fold_doc["fold"],
                    "n_train_pairs": int(masks["train"].sum()),
                    "n_val_pairs": int(masks["val"].sum()),
                    "n_test_pairs": int(masks["test"].sum()),
                }
            )
        checks["no_group_overlap"] = all(checks.values())
    checks["all_folds_present"] = len(split["folds"]) == int(split["n_folds"])
    return {"checks": checks, "warnings": warnings, "per_fold": per_fold}


def write_split_manifest(
    directory: str | Path,
    *,
    paths: list[Path],
    inputs: dict[str, str] | None = None,
    code_hash: str | None = None,
    config_hash: str | None = None,
    validations: dict[str, Any] | None = None,
) -> Path:
    directory = Path(directory)
    manifest = {
        "schema_version": "1.0",
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code_hash": code_hash,
        "config_hash": config_hash,
        "input_artifacts": inputs or {},
        "splits": [
            {
                "file": path.name,
                "protocol": path.stem.split("_s")[0],
                "seed": int(path.stem.split("_s")[-1]),
                "sha256": io.hash_file(path),
                "size": path.stat().st_size,
            }
            for path in sorted(paths)
        ],
        "validations": validations or {},
    }
    target = directory / "split_manifest.json"
    io.write_json(target, manifest)
    return target


__all__ = [
    "CELL_PROTOCOLS",
    "COMPOUND_PROTOCOLS",
    "LINEAGE_PROTOCOL",
    "PAIR_PROTOCOLS",
    "PROTOCOLS",
    "fold_masks",
    "load_split",
    "make_split",
    "split_path",
    "train_ids_for_fold",
    "validate_split",
    "write_split",
    "write_split_manifest",
]

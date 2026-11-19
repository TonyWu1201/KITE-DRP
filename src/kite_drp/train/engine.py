"""教师训练循环：单折训练、验证、早停、指标落盘。"""

from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from kite_drp.eval.metrics import evaluate_predictions, regression_metrics
from kite_drp.models.teacher.common import TeacherBatch, TeacherOutput
from kite_drp.train.losses import pairwise_rank_loss


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def move_batch(batch: TeacherBatch, device: torch.device) -> TeacherBatch:
    for field, value in list(batch.__dict__.items()):
        if isinstance(value, torch.Tensor):
            setattr(batch, field, value.to(device, non_blocking=True))
    return batch


def _loss(
    model_output: TeacherOutput,
    batch: TeacherBatch,
    config: dict[str, Any],
    *,
    rank_weight: float = 0.0,
) -> torch.Tensor:
    delta = float(config.get("huber_delta", 1.0))
    regression = F.huber_loss(model_output.y_hat, batch.y, delta=delta, reduction="none")
    if batch.label_mask is not None:
        label_mask = batch.label_mask.to(regression.dtype)
        regression = (regression * label_mask).sum() / label_mask.sum().clamp_min(1.0)
    else:
        regression = regression.mean()
    mask = batch.cls_mask.to(model_output.logits.dtype)
    bce = F.binary_cross_entropy_with_logits(model_output.logits, batch.y_cls, reduction="none")
    classification = (bce * mask).sum() / mask.sum().clamp_min(1.0)
    weight = float(config.get("cls_weight", 0.5))
    loss = regression + weight * classification
    graph_weight = float(config.get("graph_weight", 0.0))
    if graph_weight > 0 and "y_graph" in model_output.aux:
        graph_reg = F.huber_loss(
            model_output.aux["y_graph"], batch.y, delta=delta, reduction="none"
        )
        if batch.label_mask is not None:
            mask_reg = batch.label_mask.to(graph_reg.dtype)
            graph_reg = (graph_reg * mask_reg).sum() / mask_reg.sum().clamp_min(1.0)
        else:
            graph_reg = graph_reg.mean()
        loss = loss + graph_weight * graph_reg
        graph_cls_weight = float(config.get("graph_cls_weight", 0.0))
        if graph_cls_weight > 0 and "logits_graph" in model_output.aux:
            graph_bce = F.binary_cross_entropy_with_logits(
                model_output.aux["logits_graph"], batch.y_cls, reduction="none"
            )
            graph_bce = (graph_bce * mask).sum() / mask.sum().clamp_min(1.0)
            loss = loss + graph_cls_weight * graph_bce
    if rank_weight > 0 and batch.drug_idx is not None:
        rank_mask = batch.label_mask if batch.label_mask is not None else None
        ranking = pairwise_rank_loss(
            model_output.y_hat.float(),
            batch.y.float(),
            batch.drug_idx,
            mask=rank_mask,
            temperature=float(config.get("rank_temperature", 1.0)),
            min_delta=float(config.get("rank_min_delta", 0.1)),
            max_pairs_per_group=int(config.get("rank_max_pairs", 1024)),
        )
        loss = loss + float(rank_weight) * ranking
    return loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    base: Any = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    model.eval()
    predictions = []
    graph_predictions: list[np.ndarray] = []
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(batch)
        graph_y = output.aux.get("y_graph")
        if graph_y is not None:
            graph_predictions.append(graph_y.float().cpu().numpy())
        predictions.append(
            pd.DataFrame(
                {
                    "row_index": batch.row_index.cpu().numpy(),
                    "y_true": batch.y.cpu().numpy(),
                    "y_pred": output.y_hat.float().cpu().numpy(),
                    "y_cls": batch.y_cls.cpu().numpy(),
                    "y_prob": torch.sigmoid(output.logits.float()).cpu().numpy(),
                    "cls_mask": batch.cls_mask.cpu().numpy().astype(bool),
                    "label_mask": (
                        batch.label_mask.cpu().numpy().astype(bool)
                        if batch.label_mask is not None
                        else np.ones(len(batch.y), dtype=bool)
                    ),
                }
            )
        )
    frame = pd.concat(predictions, ignore_index=True)
    if base is not None:
        rows = frame["row_index"].to_numpy()
        frame["cell_line_uid"] = base.pairs.iloc[rows]["cell_line_uid"].to_numpy()
        frame["compound_uid"] = base.pairs.iloc[rows]["compound_uid"].to_numpy()
    metrics = evaluate_predictions(frame)
    if graph_predictions:
        graph_pred = np.concatenate(graph_predictions)
        frame["y_graph_pred"] = graph_pred
        mask = frame["label_mask"].to_numpy(dtype=bool)
        metrics["graph_regression"] = regression_metrics(
            frame["y_true"].to_numpy()[mask],
            graph_pred[mask],
        )
    return metrics, frame


def _warmup_cosine(step: int, *, warmup: int, total: int) -> float:
    if warmup > 0 and step < warmup:
        return float(step + 1) / float(warmup)
    progress = float(step - warmup) / float(max(total - warmup, 1))
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def _update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters(), strict=False):
            ema_param.mul_(decay).add_(param.detach(), alpha=1.0 - decay)
        for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers(), strict=False):
            ema_buffer.copy_(buffer)


def fit_teacher(
    model: nn.Module,
    *,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    run_dir: str | Path,
    base: Any = None,
    fold: int = 0,
) -> dict[str, Any]:
    """训练一个 fold，返回 history 与 best 指标。"""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    epochs = int(config.get("epochs", 80))
    patience = int(config.get("patience", 10))
    monitor = str(config.get("monitor", "val_scc"))
    monitor_mode = str(config.get("monitor_mode", "max"))
    grad_clip = float(config.get("grad_clip", 1.0))
    amp_enabled = bool(config.get("amp", True)) and device.type == "cuda"

    lr = float(config.get("lr", 5e-4))
    weight_decay = float(config.get("weight_decay", 1e-4))
    backbone_params = [
        parameter for name, parameter in model.named_parameters() if name.startswith("backbone.")
    ]
    other_params = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone.")
    ]
    p1_lr_mult = float(config.get("p1_lr_mult", 1.0))
    if backbone_params and p1_lr_mult != 1.0:
        # T1：P1 预训练权重用小学习率，图分支用主学习率，避免把 P1 推离最优点。
        optimizer = torch.optim.AdamW(
            [
                {"params": other_params, "lr": lr, "weight_decay": weight_decay},
                {
                    "params": backbone_params,
                    "lr": lr * p1_lr_mult,
                    "weight_decay": weight_decay,
                },
            ]
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    freeze_epochs = max(int(config.get("freeze_p1_epochs", 0)), 0)
    if freeze_epochs > 0 and backbone_params:
        for parameter in backbone_params:
            parameter.requires_grad_(False)
    total_steps = max(int(config.get("warmup_steps", 200)), 0) + epochs * len(train_loader)
    warmup = int(config.get("warmup_steps", 200))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _warmup_cosine(step, warmup=warmup, total=total_steps),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    ema_decay = float(config.get("ema_decay", 0.0))
    ema_model = copy.deepcopy(model) if ema_decay > 0 else None
    if ema_model is not None:
        ema_model.eval()
    base_rank_weight = float(config.get("rank_weight", 0.0))
    rank_warmup_epochs = max(int(config.get("rank_warmup_epochs", 0)), 0)
    min_epochs = max(int(config.get("min_epochs", 0)), 0)

    history: list[dict[str, Any]] = []
    best_value = -math.inf if monitor_mode == "max" else math.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    step = 0
    started = time.time()

    for epoch in range(epochs):
        if freeze_epochs > 0 and epoch == freeze_epochs and backbone_params:
            for parameter in backbone_params:
                parameter.requires_grad_(True)
        model.train()
        if ema_model is not None:
            ema_model.eval()
        epoch_loss = 0.0
        rank_weight = base_rank_weight
        if rank_warmup_epochs > 0:
            rank_weight *= min(1.0, float(epoch + 1) / float(rank_warmup_epochs))
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                output = model(batch)
                loss = _loss(output, batch, config, rank_weight=rank_weight)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if ema_model is not None:
                _update_ema(ema_model, model, ema_decay)
            epoch_loss += float(loss.detach().cpu())
            step += 1

        eval_model = ema_model if ema_model is not None else model
        val_metrics, _ = evaluate(eval_model, val_loader, device=device, base=base)
        gates = {
            name: float(torch.tanh(getattr(eval_model, name)).detach())
            for name in ("gene_gate", "cell_gate", "drug_gate", "graph_pred_gate")
            if hasattr(eval_model, name)
        }
        record = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(len(train_loader), 1),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "rank_weight": rank_weight,
            "gates": gates,
            "val": val_metrics,
        }
        history.append(record)
        value = _monitor_value(val_metrics, monitor)
        if not math.isfinite(value):
            value = -math.inf if monitor_mode == "max" else math.inf
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": round(record["train_loss"], 5),
                    "rank_weight": round(rank_weight, 4),
                    "val_r2": round(val_metrics["regression"]["r2"], 4),
                    "val_scc": round(val_metrics["regression"]["scc"], 4),
                    "val_graph_scc": round(
                        val_metrics.get("graph_regression", {}).get("scc", float("nan")),
                        4,
                    ),
                    "val_auroc": round(val_metrics["classification"]["auroc"], 4),
                    "gates": {key: round(value, 4) for key, value in gates.items()},
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        improved = (
            (value > best_value + 1e-6) if monitor_mode == "max" else (value < best_value - 1e-6)
        )
        if improved:
            best_value = value
            best_epoch = epoch
            stale = 0
            best_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in eval_model.state_dict().items()
            }
        else:
            stale += 1
            if stale >= patience and (epoch + 1) >= min_epochs:
                print(json.dumps({"early_stop": epoch, "best_epoch": best_epoch}))
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    final_metrics, final_frame = evaluate(model, val_loader, device=device, base=base)
    result = {
        "fold": fold,
        "monitor": monitor,
        "best_value": best_value,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "elapsed_sec": time.time() - started,
        "val": final_metrics,
        "history": history,
        "ema_decay": ema_decay,
        "rank_weight": base_rank_weight,
        "p1_lr_mult": p1_lr_mult,
        "freeze_p1_epochs": freeze_epochs,
    }
    (run_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    final_frame.to_parquet(run_dir / "val_predictions.parquet", index=False)
    torch.save(
        {"model": model.state_dict(), "config": config, "fold": fold},
        run_dir / "best.pt",
    )
    return result


def _monitor_value(metrics: dict[str, Any], monitor: str) -> float:
    if monitor == "val_scc":
        return float(metrics["regression"]["scc"])
    if monitor == "val_r2":
        return float(metrics["regression"]["r2"])
    if monitor == "val_auroc":
        return float(metrics["classification"]["auroc"])
    if monitor == "val_pcc":
        return float(metrics["regression"]["pcc"])
    if monitor == "val_per_drug_scc":
        return float(metrics.get("per_drug", {}).get("scc_mean", float("nan")))
    raise KeyError(f"未知 monitor：{monitor}")


__all__ = ["evaluate", "fit_teacher", "move_batch", "resolve_device", "set_seed"]

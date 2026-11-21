"""局部图深监督/中心化损失的门控契约（纯合成张量）。"""

from __future__ import annotations

import torch

from kite_drp.models.teacher.common import TeacherBatch, TeacherOutput
from kite_drp.train.engine import _loss


def _batch(*, drug_idx: torch.Tensor | None = None) -> TeacherBatch:
    n = 3
    return TeacherBatch(
        ge=torch.zeros(n, 2),
        mu=torch.zeros(n, 2),
        cnv=torch.zeros(n, 2),
        cnv_mask=torch.ones(n, dtype=torch.bool),
        morgan=torch.zeros(n, 2),
        target_index=torch.zeros(0, dtype=torch.long),
        target_offset=torch.zeros(n, dtype=torch.long),
        y=torch.zeros(n),
        y_cls=torch.zeros(n),
        cls_mask=torch.ones(n, dtype=torch.bool),
        drug_idx=drug_idx if drug_idx is not None else torch.tensor([0, 0, 1]),
        label_mask=torch.ones(n, dtype=torch.bool),
    )


def _output(*, y_graph: torch.Tensor, local_delta: torch.Tensor) -> TeacherOutput:
    n = y_graph.shape[0]
    zeros = torch.zeros(n, 4)
    return TeacherOutput(
        y_hat=torch.zeros(n),
        logits=torch.zeros(n),
        z_pair=zeros,
        z_cell=zeros,
        z_drug=zeros,
        aux={
            "y_graph": y_graph,
            "local_delta": local_delta,
            "local_gate": torch.tensor([1.0, 1.0, 0.0]),
        },
    )


def test_graph_loss_ignores_uncovered_samples() -> None:
    batch = _batch()
    config = {"graph_weight": 1.0, "graph_cls_weight": 0.0, "cls_weight": 0.0}
    covered = _output(
        y_graph=torch.zeros(3),
        local_delta=torch.zeros(3),
    )
    extreme = _output(
        y_graph=torch.tensor([0.0, 0.0, 1e6]),
        local_delta=torch.zeros(3),
    )
    torch.testing.assert_close(_loss(covered, batch, config), _loss(extreme, batch, config))


def test_tower_loss_uses_y_t2_and_label_mask() -> None:
    batch = _batch()
    batch.label_mask = torch.tensor([True, True, False])
    batch.y = torch.tensor([0.0, 1.0, 100.0])
    config = {
        "graph_weight": 0.0,
        "graph_cls_weight": 0.0,
        "cls_weight": 0.0,
        "tower_weight": 1.0,
        "tower_cls_weight": 0.0,
    }
    zeros = torch.zeros(3, 4)
    output = TeacherOutput(
        y_hat=torch.zeros(3),
        logits=torch.zeros(3),
        z_pair=zeros,
        z_cell=zeros,
        z_drug=zeros,
        aux={"y_t2": torch.tensor([0.0, 0.0, 1e6])},
    )
    without_tower = TeacherOutput(
        y_hat=torch.zeros(3),
        logits=torch.zeros(3),
        z_pair=zeros,
        z_cell=zeros,
        z_drug=zeros,
    )
    base_config = {**config, "tower_weight": 0.0}
    # 第 2 行被 label_mask 排除；前两行的 tower Huber 损失 = (0 + 0.5) / 2。
    torch.testing.assert_close(
        _loss(output, batch, config) - _loss(without_tower, batch, base_config),
        torch.tensor(0.25),
    )


def test_center_loss_ignores_uncovered_samples() -> None:
    batch = _batch()
    config = {
        "graph_weight": 0.0,
        "graph_cls_weight": 0.0,
        "cls_weight": 0.0,
        "center_weight": 1.0,
    }
    covered = _output(
        y_graph=torch.zeros(3),
        local_delta=torch.tensor([1.0, 1.0, 0.0]),
    )
    extreme = _output(
        y_graph=torch.zeros(3),
        local_delta=torch.tensor([1.0, 1.0, 1e6]),
    )
    torch.testing.assert_close(_loss(covered, batch, config), _loss(extreme, batch, config))
    # 药物 0 的残差均值为 1；药物 1 被门控排除。
    torch.testing.assert_close(_loss(covered, batch, config), torch.tensor(1.0))

"""教师公共组件：batch 契约、MLP、输出结构。"""

from __future__ import annotations

from dataclasses import dataclass, field

from torch import Tensor, nn


@dataclass
class TeacherBatch:
    """教师前向所需的全部张量（由 collate 构建）。"""

    ge: Tensor
    mu: Tensor
    cnv: Tensor
    cnv_mask: Tensor
    morgan: Tensor
    target_index: Tensor
    target_offset: Tensor
    y: Tensor
    y_cls: Tensor
    cls_mask: Tensor
    row_index: Tensor | None = None
    label_mask: Tensor | None = None
    drug_idx: Tensor | None = None
    cell_idx: Tensor | None = None
    ge_mask: Tensor | None = None
    mu_mask: Tensor | None = None
    neighbor_ge: Tensor | None = None
    neighbor_mask: Tensor | None = None
    neighbor_confidence: Tensor | None = None
    # P1 target-aware gene branch：每对 (drug, cell) 的靶点基因表。
    target_gene_ids: Tensor | None = None
    target_gene_values: Tensor | None = None  # (B, T, 2×M)：每个模态值 + availability flag
    target_gene_mask: Tensor | None = None
    graph_node_feat: Tensor | None = None
    graph_edge_index: Tensor | None = None
    graph_edge_feat: Tensor | None = None
    graph_batch: Tensor | None = None
    graph_num_graphs: int = 0


@dataclass
class TeacherOutput:
    """教师输出契约（学生蒸馏/离线缓存按此字段读取）。"""

    y_hat: Tensor
    logits: Tensor
    z_pair: Tensor
    z_cell: Tensor
    z_drug: Tensor
    aux: dict[str, Tensor] = field(default_factory=dict)


class MLP(nn.Module):
    """Linear → GELU → Dropout 堆叠；最后一层可选 LayerNorm。"""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int] | tuple[int, ...],
        out_dim: int,
        *,
        dropout: float = 0.1,
        final_norm: bool = False,
    ) -> None:
        super().__init__()
        dims = [in_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for left, right in zip(dims[:-1], dims[1:], strict=False):
            layers.extend([nn.Linear(left, right), nn.GELU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(dims[-1], out_dim))
        if final_norm:
            layers.append(nn.LayerNorm(out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.net(inputs)


def masked_mean(values: Tensor, mask: Tensor, *, dim: int = 1) -> Tensor:
    """``mask`` 为 bool/0-1，形状需可广播到 ``values``。"""
    weights = mask.to(values.dtype)
    total = (values * weights).sum(dim=dim)
    count = weights.sum(dim=dim).clamp_min(1.0)
    return total / count


def count_parameters(module: nn.Module, *, trainable_only: bool = True) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad or not trainable_only
    )


__all__ = [
    "MLP",
    "TeacherBatch",
    "TeacherOutput",
    "count_parameters",
    "masked_mean",
]

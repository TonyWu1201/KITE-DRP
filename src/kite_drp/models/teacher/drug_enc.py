"""药物编码器：Morgan 指纹 + 靶点基因嵌入 + 可选 GIN 分子图。

对应计划书 D3：GIN（77 维原子特征）+ Morgan FP + 靶点 multi-hot 三路融合。
GIN 默认关闭（T0 v1 先用 Morgan+targets，MRI 消融时打开）。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from kite_drp.models.teacher.common import MLP, TeacherBatch


class GINLayer(nn.Module):
    def __init__(self, hidden: int, edge_dim: int, *, dropout: float = 0.1) -> None:
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden),
        )
        self.edge_proj = nn.Linear(edge_dim, hidden) if edge_dim > 0 else None
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_feat: Tensor, edge_index: Tensor, edge_feat: Tensor | None) -> Tensor:
        if edge_index.numel() == 0:
            return node_feat
        source, target = edge_index[0], edge_index[1]
        messages = node_feat[source]
        if self.edge_proj is not None and edge_feat is not None:
            messages = messages + self.edge_proj(edge_feat)
        aggregated = torch.zeros_like(node_feat).index_add_(0, target, messages)
        updated = self.mlp((1.0 + self.eps) * node_feat + aggregated)
        return self.norm(node_feat + self.dropout(updated))


class MolecularGIN(nn.Module):
    """纯 PyTorch 实现的 GIN（不依赖 PyG），输入为拼接后的 ragged 图 batch。"""

    def __init__(
        self,
        *,
        node_dim: int = 77,
        edge_dim: int = 4,
        hidden: int = 128,
        n_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden)
        self.layers = nn.ModuleList(
            [GINLayer(hidden, edge_dim, dropout=dropout) for _ in range(n_layers)]
        )
        self.output_dim = hidden

    def forward(
        self,
        node_feat: Tensor,
        edge_index: Tensor,
        edge_feat: Tensor | None,
        batch_index: Tensor,
        num_graphs: int,
    ) -> Tensor:
        if num_graphs == 0 or node_feat.numel() == 0:
            return torch.zeros(
                (num_graphs, self.output_dim),
                dtype=self.node_proj.weight.dtype,
                device=self.node_proj.weight.device,
            )
        hidden = self.node_proj(node_feat)
        for layer in self.layers:
            hidden = layer(hidden, edge_index, edge_feat)
        pooled = torch.zeros(
            (num_graphs, hidden.size(1)), dtype=hidden.dtype, device=hidden.device
        ).index_add_(0, batch_index, hidden)
        counts = torch.zeros(num_graphs, dtype=hidden.dtype, device=hidden.device).index_add_(
            0, batch_index, torch.ones_like(batch_index, dtype=hidden.dtype)
        )
        return pooled / counts.unsqueeze(1).clamp_min(1.0)


class DrugEncoder(nn.Module):
    def __init__(
        self,
        *,
        morgan_dim: int,
        n_target_genes: int,
        d_model: int = 256,
        hidden: int = 256,
        dropout: float = 0.1,
        use_gin: bool = False,
        gin_layers: int = 3,
        graph_node_dim: int = 77,
        graph_edge_dim: int = 4,
        target_embed_dim: int = 16,
    ) -> None:
        super().__init__()
        self.use_gin = use_gin
        self.morgan = MLP(morgan_dim, [hidden], hidden, dropout=dropout)
        self.targets = nn.EmbeddingBag(n_target_genes, target_embed_dim, mode="mean")
        self.target_proj = nn.Linear(target_embed_dim, hidden)
        self.gin = (
            MolecularGIN(
                node_dim=graph_node_dim,
                edge_dim=graph_edge_dim,
                hidden=hidden,
                n_layers=gin_layers,
                dropout=dropout,
            )
            if use_gin
            else None
        )
        fusion_dim = hidden * (3 if use_gin else 2)
        self.fuse = nn.Sequential(
            nn.Linear(fusion_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

    def forward(self, batch: TeacherBatch) -> Tensor:
        parts = [self.morgan(batch.morgan)]
        parts.append(self.target_proj(self.targets(batch.target_index, batch.target_offset)))
        if self.use_gin and self.gin is not None:
            parts.append(
                self.gin(
                    batch.graph_node_feat,
                    batch.graph_edge_index,
                    batch.graph_edge_feat,
                    batch.graph_batch,
                    batch.graph_num_graphs,
                )
            )
        return self.fuse(torch.cat(parts, dim=-1))


__all__ = ["DrugEncoder", "GINLayer", "MolecularGIN"]

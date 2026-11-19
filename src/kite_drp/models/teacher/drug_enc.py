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
        use_morgan: bool = True,
        use_targets: bool = True,
        gin_layers: int = 3,
        graph_node_dim: int = 77,
        graph_edge_dim: int = 4,
        target_embed_dim: int = 16,
        target_embedding: nn.Embedding | None = None,
        use_target_table: bool = True,
    ) -> None:
        super().__init__()
        if not (use_morgan or use_targets or use_gin):
            raise ValueError("药物编码至少需要 morgan / targets / gin 之一")
        self.use_gin = use_gin
        self.use_morgan = use_morgan
        self.use_targets = use_targets
        self.target_embed_dim = int(target_embed_dim)
        self.morgan = MLP(morgan_dim, [hidden], hidden, dropout=dropout) if use_morgan else None
        if use_targets:
            if use_target_table:
                self.targets = nn.EmbeddingBag(n_target_genes, target_embed_dim, mode="mean")
                if target_embedding is not None:
                    # P1：与细胞侧 target gene token 共享同一张基因 ID 嵌入表。
                    self.targets.weight = target_embedding.weight
            else:
                # T1：靶点嵌入由外部图基因嵌入提供（见 forward(target_vec=...)）。
                self.targets = None
            self.target_proj = nn.Linear(target_embed_dim, hidden)
        else:
            self.targets = None
            self.target_proj = None
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
        fusion_dim = hidden * (use_morgan + use_targets + use_gin)
        self.fuse = nn.Sequential(
            nn.Linear(fusion_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

    def encode(
        self,
        morgan: Tensor,
        *,
        target_vec: Tensor | None = None,
        target_index: Tensor | None = None,
        target_offset: Tensor | None = None,
        graph_node_feat: Tensor | None = None,
        graph_edge_index: Tensor | None = None,
        graph_edge_feat: Tensor | None = None,
        graph_batch: Tensor | None = None,
        graph_num_graphs: int = 0,
    ) -> Tensor:
        """从原始张量编码药物；T1 需要一次性编码全部 286 个药物节点。"""
        parts: list[Tensor] = []
        if self.morgan is not None:
            parts.append(self.morgan(morgan))
        if self.use_targets and self.target_proj is not None:
            if target_vec is None:
                if (
                    self.targets is not None
                    and target_index is not None
                    and target_offset is not None
                ):
                    target_vec = self.targets(target_index, target_offset)
                else:
                    target_vec = morgan.new_zeros(morgan.size(0), self.target_embed_dim)
            parts.append(self.target_proj(target_vec))
        if self.use_gin and self.gin is not None:
            num_graphs = graph_num_graphs or morgan.size(0)
            if graph_node_feat is None or graph_edge_index is None or graph_batch is None:
                parts.append(
                    torch.zeros(
                        (num_graphs, self.gin.output_dim),
                        dtype=self.gin.node_proj.weight.dtype,
                        device=morgan.device,
                    )
                )
            else:
                parts.append(
                    self.gin(
                        graph_node_feat,
                        graph_edge_index,
                        graph_edge_feat,
                        graph_batch,
                        num_graphs,
                    )
                )
        return self.fuse(torch.cat(parts, dim=-1))

    def forward(self, batch: TeacherBatch, target_vec: Tensor | None = None) -> Tensor:
        return self.encode(
            batch.morgan,
            target_vec=target_vec,
            target_index=batch.target_index,
            target_offset=batch.target_offset,
            graph_node_feat=batch.graph_node_feat,
            graph_edge_index=batch.graph_edge_index,
            graph_edge_feat=batch.graph_edge_feat,
            graph_batch=batch.graph_batch,
            graph_num_graphs=batch.graph_num_graphs,
        )


__all__ = ["DrugEncoder", "GINLayer", "MolecularGIN"]

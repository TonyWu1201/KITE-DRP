"""T1：图增强的 P1 教师（计划书 §4.4 的收敛版）。

设计要点：

1. ``G0`` 是 P1 基因嵌入表在 T1 基因空间上的子集；PPI 只更新一个独立的
   ``H``，再以 gate=0 的残差写回 ``G``，因此初始化时 ``T1 == P1``；
2. DTI 与 cell-gene 都是**有向边**（gene → drug / gene → cell），避免 gene
   节点聚合全部细胞导致表示被平均；
3. target 分支与药物靶点分支共用图更新后的基因嵌入 ``G``，直接补 P1
   “药物基因级、细胞通路级”的共享空间缺口；
4. 图分支有独立深监督头 ``y_graph``，防止融合 gate 学到 0 后图分支不学习。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from kite_drp.models.teacher.common import TeacherBatch, TeacherOutput
from kite_drp.models.teacher.heads import ResponseHeads
from kite_drp.models.teacher.t0 import T0Teacher

try:  # pragma: no cover - 可选依赖
    from torch_geometric.nn import GATv2Conv, HeteroConv
except Exception as exc:  # pragma: no cover
    raise ImportError("T1 需要 torch-geometric：uv sync --extra graph") from exc


class T1Teacher(nn.Module):
    def __init__(
        self,
        backbone: T0Teacher,
        *,
        n_full_genes: int,
        n_graph_genes: int,
        cell_in_dim: int,
        gene_emb_dim: int = 16,
        graph_hidden: int = 128,
        graph_layers: int = 2,
        graph_heads: int = 4,
        dropout: float = 0.1,
        use_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.use_backbone = bool(use_backbone)
        self.n_full_genes = int(n_full_genes)
        self.n_graph_genes = int(n_graph_genes)
        self.gene_emb_dim = int(gene_emb_dim)
        self.d_model = int(backbone.d_model)
        self.gene_embedding = nn.Embedding(self.n_graph_genes, self.gene_emb_dim)
        self.gene_stats_proj = nn.Linear(4, self.gene_emb_dim)
        self.gene_proj = nn.Linear(self.gene_emb_dim, graph_hidden)
        self.cell_in_proj = nn.Sequential(
            nn.Linear(cell_in_dim, graph_hidden),
            nn.GELU(),
            nn.LayerNorm(graph_hidden),
        )
        self.drug_in_proj = nn.Linear(self.d_model, graph_hidden)
        self.dropout = nn.Dropout(dropout)

        self.convs = nn.ModuleList(
            [
                HeteroConv(
                    {
                        ("gene", "ppi", "gene"): GATv2Conv(
                            graph_hidden,
                            graph_hidden,
                            heads=graph_heads,
                            concat=False,
                            edge_dim=1,
                            dropout=dropout,
                        ),
                        ("gene", "dti", "drug"): GATv2Conv(
                            graph_hidden,
                            graph_hidden,
                            heads=graph_heads,
                            concat=False,
                            edge_dim=1,
                            dropout=dropout,
                            add_self_loops=False,
                        ),
                        ("gene", "cg", "cell"): GATv2Conv(
                            graph_hidden,
                            graph_hidden,
                            heads=graph_heads,
                            concat=False,
                            edge_dim=4,
                            dropout=dropout,
                            add_self_loops=False,
                        ),
                    },
                    aggr="sum",
                )
                for _ in range(graph_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "gene": nn.LayerNorm(graph_hidden),
                        "drug": nn.LayerNorm(graph_hidden),
                        "cell": nn.LayerNorm(graph_hidden),
                    }
                )
                for _ in range(graph_layers)
            ]
        )
        self.gene_out_proj = nn.Linear(graph_hidden, self.gene_emb_dim)
        self.gene_gate = nn.Parameter(torch.zeros(1))
        self.cell_res_proj = nn.Linear(graph_hidden, self.d_model)
        self.drug_res_proj = nn.Linear(graph_hidden, self.d_model)
        self.cell_gate = nn.Parameter(torch.zeros(1))
        self.drug_gate = nn.Parameter(torch.zeros(1))
        self.graph_pair_proj = nn.Sequential(
            nn.Linear(graph_hidden * 3, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(self.d_model),
        )
        self.graph_heads = ResponseHeads(self.d_model, hidden=self.d_model // 2, dropout=dropout)

        # gate 初始 0：T1 从 P1 出发；图分支必须自己学出贡献。
        # gene_stats_proj 零初始化保证 G0 == P1 基因表；gene_out_proj 保持随机，
        # 否则 gene_gate 在 0 点的梯度恒为 0，PPI 知识永远进不了 target 分支。
        nn.init.zeros_(self.gene_stats_proj.weight)
        nn.init.zeros_(self.gene_stats_proj.bias)
        self._graph_ready = False

    # ------------------------------------------------------------------ graph
    def set_graph(self, graph: dict[str, Tensor | list[str]]) -> None:
        device = next(self.parameters()).device
        for key, value in graph.items():
            if isinstance(value, Tensor):
                setattr(self, key, value.to(device))
            else:
                setattr(self, key, value)
        self._graph_ready = True

    def _mean_target_vec(self, gene_embedding: Tensor) -> Tensor:
        # 药物编码器需要与 P1 的 EmbeddingBag 完全一致：使用全部靶点，
        # 不能用 target 分支的 64-token 截断表。
        ids = self.drug_target_ids_full
        mask = self.drug_target_mask_full
        embedded = gene_embedding[ids]
        weights = mask.unsqueeze(-1).to(embedded.dtype)
        return (embedded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _scatter_full(self, gene_embedding: Tensor) -> Tensor:
        full = gene_embedding.new_zeros(self.n_full_genes, gene_embedding.size(1))
        return full.index_copy(0, self.gene_full_idx, gene_embedding)

    def _encode_graph(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if not self._graph_ready:
            raise RuntimeError("T1Teacher 需要先 set_graph()")
        g0 = self.gene_embedding.weight + self.gene_stats_proj(self.gene_stats)
        hidden = self.gene_proj(g0)
        target0 = self._mean_target_vec(g0)
        drug_init = self.backbone.drug.encode(self.drug_morgan, target_vec=target0)
        h_drug = self.drug_in_proj(drug_init)
        h_cell = self.cell_in_proj(self.cell_features)

        for conv, norms in zip(self.convs, self.norms, strict=False):
            x_dict = {"gene": hidden, "drug": h_drug, "cell": h_cell}
            edge_index_dict = {}
            edge_attr_dict = {}
            if self.ppi_edge_index.numel():
                edge_index_dict[("gene", "ppi", "gene")] = self.ppi_edge_index
                edge_attr_dict[("gene", "ppi", "gene")] = self.ppi_edge_attr
            if self.dti_edge_index.numel():
                edge_index_dict[("gene", "dti", "drug")] = self.dti_edge_index
                edge_attr_dict[("gene", "dti", "drug")] = self.dti_edge_attr
            if self.cg_edge_index.numel():
                edge_index_dict[("gene", "cg", "cell")] = self.cg_edge_index
                edge_attr_dict[("gene", "cg", "cell")] = self.cg_edge_attr
            if not edge_index_dict:
                break
            out = conv(x_dict, edge_index_dict, edge_attr_dict)
            if "gene" in out:
                hidden = norms["gene"](hidden + self.dropout(out["gene"]))
            if "drug" in out:
                h_drug = norms["drug"](h_drug + self.dropout(out["drug"]))
            if "cell" in out:
                h_cell = norms["cell"](h_cell + self.dropout(out["cell"]))

        gene_embedding = g0 + torch.tanh(self.gene_gate) * self.gene_out_proj(hidden)
        target_vec = self._mean_target_vec(gene_embedding)
        return self._scatter_full(gene_embedding), target_vec, h_drug, h_cell

    # ----------------------------------------------------------------- forward
    def forward(self, batch: TeacherBatch) -> TeacherOutput:
        if batch.drug_idx is None or batch.cell_idx is None:
            raise ValueError("T1 需要 batch.drug_idx 与 batch.cell_idx")
        gene_full, target_vec, h_drug_nodes, h_cell_nodes = self._encode_graph()
        h_drug = h_drug_nodes[batch.drug_idx]
        h_cell = h_cell_nodes[batch.cell_idx]
        z_graph = self.graph_pair_proj(torch.cat([h_drug, h_cell, h_drug * h_cell], dim=-1))
        if self.use_backbone:
            p1 = self.backbone(
                batch,
                gene_embedding=gene_full,
                drug_target_vec=target_vec[batch.drug_idx],
            )
            z_cell = p1.z_cell + torch.tanh(self.cell_gate) * self.cell_res_proj(h_cell)
            z_drug = p1.z_drug + torch.tanh(self.drug_gate) * self.drug_res_proj(h_drug)
            z_pair = z_drug + z_cell
            y_hat, logits = self.backbone.heads(z_pair)
            aux = dict(p1.aux)
            aux.update(
                {
                    "y_p1": p1.y_hat,
                    "logits_p1": p1.logits,
                    "gene_gate": torch.tanh(self.gene_gate).detach(),
                    "cell_gate": torch.tanh(self.cell_gate).detach(),
                    "drug_gate": torch.tanh(self.drug_gate).detach(),
                }
            )
        else:
            z_cell = self.cell_res_proj(h_cell)
            z_drug = self.drug_res_proj(h_drug)
            z_pair = z_drug + z_cell
            y_hat, logits = self.graph_heads(z_graph)
            aux = {}
        y_graph, logits_graph = self.graph_heads(z_graph)
        aux.update(
            {
                "y_graph": y_graph,
                "logits_graph": logits_graph,
            }
        )
        return TeacherOutput(
            y_hat=y_hat,
            logits=logits,
            z_pair=z_pair,
            z_cell=z_cell,
            z_drug=z_drug,
            aux=aux,
        )

    # -------------------------------------------------------------- checkpoint
    @torch.no_grad()
    def load_p1_checkpoint(self, state: dict[str, Tensor]) -> None:
        """从 P1 的 state_dict 初始化：backbone 参数 + 图基因嵌入子集。"""
        backbone_state: dict[str, Tensor] = {}
        for key, value in state.items():
            if key.startswith("target.gene_embedding.") or key == "drug.targets.weight":
                continue
            backbone_state[key] = value
        self.backbone.load_state_dict(backbone_state, strict=False)
        table = state.get("target.gene_embedding.weight")
        if table is not None:
            if table.shape[0] < self.n_full_genes:
                raise ValueError(
                    f"P1 基因表行数 {table.shape[0]} < n_full_genes {self.n_full_genes}"
                )
            self.gene_embedding.weight.data.copy_(
                table[self.gene_full_idx].to(self.gene_embedding.weight.device)
            )


def count_t1_parameters(model: T1Teacher) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    backbone = sum(parameter.numel() for parameter in model.backbone.parameters())
    return {"total": total, "backbone": backbone, "graph": total - backbone}


__all__ = ["T1Teacher", "count_t1_parameters"]

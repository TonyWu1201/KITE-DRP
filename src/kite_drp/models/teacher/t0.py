"""T0：多模态交叉注意力教师（计划书 §4.3）。

结构：
- 三个组学模态（GE/MU/CNV）各自把 1,000 维通路分数压成
  ``[CLS] + top-k 通路 token``（默认 k=64，控制端到端序列长度）；
- 药物 token 作为 query 交叉注意组学 token（MKDR 式 cross-modality attention）；
- 可选相似性邻居 token（H3CDR 式 top-N 邻居的 GE 通路加权均值，gate 初始 0）；
- 2 层 TransformerEncoder 建模 token 间关系，回归 + 分类双头。
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from kite_drp.models.teacher.common import TeacherBatch, TeacherOutput
from kite_drp.models.teacher.drug_enc import DrugEncoder
from kite_drp.models.teacher.heads import ResponseHeads


class AttentionPool(nn.Module):
    """带 mask 的 attention pooling；全 mask 时返回 0。"""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(d_model))
        self.scale = math.sqrt(d_model)

    def forward(self, tokens: Tensor, mask: Tensor) -> Tensor:
        scores = torch.einsum("bnd,d->bn", tokens, self.query) / self.scale
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1) * mask.to(tokens.dtype)
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return (weights.unsqueeze(-1) * tokens).sum(dim=1) / denom


class TargetGeneEncoder(nn.Module):
    """把每个药物的靶点基因细胞特征编码为一个 target token。

    输入是逐 pair 的靶点基因表：``ids`` (B, T)、``values`` (B, T, 2×M)
    （每个启用模态的值 + availability flag）、``mask`` (B, T)。
    基因 ID 嵌入与药物编码器的靶点嵌入共享，构成药物-细胞共享基因空间。
    """

    def __init__(
        self,
        *,
        n_genes: int,
        d_model: int,
        gene_emb_dim: int = 16,
        n_values: int = 6,
        n_layers: int = 1,
        n_heads: int = 4,
        dropout: float = 0.1,
        gene_embedding: nn.Embedding | None = None,
        use_gene_table: bool = True,
    ) -> None:
        super().__init__()
        if gene_embedding is not None:
            self.gene_embedding = gene_embedding
        elif use_gene_table:
            self.gene_embedding = nn.Embedding(n_genes, gene_emb_dim)
        else:
            # T1：由外部图基因嵌入提供（forward(gene_embedding=...)）。
            self.gene_embedding = None
        self.gene_proj = nn.Linear(gene_emb_dim, d_model)
        self.value_proj = nn.Linear(n_values, d_model)
        self.type_embedding = nn.Parameter(torch.zeros(d_model))
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.pool = AttentionPool(d_model)
        self.n_layers = int(n_layers)
        if self.n_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                layer, num_layers=self.n_layers, enable_nested_tensor=False
            )
        else:
            self.encoder = None

    def forward(
        self,
        ids: Tensor,
        values: Tensor,
        mask: Tensor,
        gene_embedding: nn.Embedding | Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        table = gene_embedding if gene_embedding is not None else self.gene_embedding
        if table is None:
            raise ValueError("TargetGeneEncoder 未提供 gene_embedding")
        embedded = table(ids) if callable(table) else table[ids]
        tokens = self.gene_proj(embedded) + self.value_proj(values) + self.type_embedding
        tokens = self.dropout(self.norm(tokens))
        if self.encoder is not None:
            # 无靶点药物（LDO 中约 17%）会得到全 False 的 mask；直接喂给
            # TransformerEncoder 会让全 -inf 的 attention 产生 NaN。
            valid_rows = mask.any(dim=1)
            if bool(valid_rows.all()):
                tokens = self.encoder(tokens, src_key_padding_mask=~mask)
            else:
                encoded = tokens.new_zeros(tokens.shape)
                if bool(valid_rows.any()):
                    encoded[valid_rows] = self.encoder(
                        tokens[valid_rows],
                        src_key_padding_mask=~mask[valid_rows],
                    )
                tokens = encoded
        pooled = self.pool(tokens, mask)
        return pooled, mask.any(dim=1)


class PathwayTokenEncoder(nn.Module):
    """把 (B, P) 通路分数编码为 (B, 1+K, d)。"""

    def __init__(
        self,
        pathway_dim: int,
        d_model: int,
        *,
        top_k: int = 64,
        dropout: float = 0.1,
        token_types: bool = False,
        rank_embedding: bool = False,
        pathway_embedding: nn.Embedding | None = None,
        use_cls_proj: bool = True,
        cls_proj: nn.Module | None = None,
        modality_proj: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.top_k = int(top_k)
        self.pathway_embedding = (
            pathway_embedding
            if pathway_embedding is not None
            else nn.Embedding(pathway_dim, d_model)
        )
        self.score_proj = nn.Linear(1, d_model)
        self.modality_proj = modality_proj
        if not use_cls_proj:
            self.cls_proj = None
        elif cls_proj is not None:
            self.cls_proj = cls_proj
        else:
            self.cls_proj = nn.Linear(pathway_dim, d_model)
        self.cls_token = None if use_cls_proj else nn.Parameter(torch.zeros(1, d_model))
        self.type_embedding = nn.Parameter(torch.zeros(d_model))
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.token_types = bool(token_types)
        self.rank_embedding = bool(rank_embedding)
        if self.token_types:
            self.token_type_embedding = nn.Parameter(torch.zeros(2, d_model))
        if self.rank_embedding:
            self.rank_embedding_table = nn.Parameter(torch.zeros(self.top_k, d_model))

    def forward(self, scores: Tensor) -> Tensor:
        k = min(self.top_k, scores.size(1))
        indices = scores.abs().topk(k, dim=1).indices
        values = scores.gather(1, indices).unsqueeze(-1)
        embedding = self.pathway_embedding(indices)
        if self.modality_proj is not None:
            embedding = embedding + self.modality_proj(embedding)
        tokens = embedding + self.score_proj(values)
        if self.token_types:
            tokens = tokens + self.token_type_embedding[1]
        if self.rank_embedding:
            tokens = tokens + self.rank_embedding_table[:k].unsqueeze(0)
        if self.cls_proj is not None:
            cls = self.cls_proj(scores).unsqueeze(1)
        else:
            assert self.cls_token is not None
            cls = self.cls_token.expand(scores.size(0), -1).unsqueeze(1)
        if self.token_types:
            cls = cls + self.token_type_embedding[0]
        stacked = torch.cat([cls, tokens], dim=1) + self.type_embedding
        return self.dropout(self.norm(stacked))


class T0Teacher(nn.Module):
    def __init__(
        self,
        *,
        pathway_dim: int,
        morgan_dim: int,
        n_target_genes: int,
        d_model: int = 256,
        hidden: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        top_k: int = 64,
        use_gin: bool = False,
        use_morgan: bool = True,
        use_targets: bool = True,
        use_neighbor: bool = True,
        modality_dropout: float = 0.1,
        target_embed_dim: int = 16,
        readout: str = "sum",
        use_dense_shortcut: bool = False,
        token_types: bool = False,
        rank_embedding: bool = False,
        cell_readout: str = "mean",
        masked_modalities: bool = False,
        cross_gate: bool = False,
        shared_pathway: bool = False,
        shared_pathway_proj: bool = False,
        use_cls_proj: bool = True,
        cls_proj_mode: str = "full",
        cls_proj_rank: int = 64,
        target_branch: bool = False,
        target_gene_layers: int = 1,
        target_n_values: int = 0,
        use_gene_table: bool = True,
        use_target_table: bool = True,
        modalities: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        super().__init__()
        if isinstance(modalities, str):
            modalities = [modalities]
        self.modalities = tuple(modalities or ("GE", "MU", "CNV"))
        if not self.modalities:
            raise ValueError("T0 至少需要一个组学模态")
        unknown = set(self.modalities) - {"GE", "MU", "CNV"}
        if unknown:
            raise KeyError(f"未知模态：{sorted(unknown)}")
        if cell_readout not in {"mean", "attention"}:
            raise KeyError(f"未知 cell_readout：{cell_readout}")
        if readout not in {"sum", "bilinear", "hadamard", "hybrid"}:
            raise KeyError(f"未知 readout：{readout}")
        self.token_types = bool(token_types)
        self.cell_readout = str(cell_readout)
        # target token 在无靶点药物上必须可屏蔽，因此 target branch 隐含 mask 模式。
        self.masked_modalities = bool(masked_modalities or target_branch)
        self.cross_gate = bool(cross_gate)
        self.target_branch = bool(target_branch)
        self.target_n_values = int(target_n_values) or 2 * len(self.modalities)
        if self.target_branch and self.target_n_values != 2 * len(self.modalities):
            raise ValueError(
                "target_n_values 必须等于 2 × 启用模态数（每个模态一个值 + 一个 flag）"
            )
        self.shared_pathway = bool(shared_pathway)
        self.shared_pathway_proj = bool(shared_pathway_proj and shared_pathway)
        if cls_proj_mode not in {"full", "shared", "lowrank", "none"}:
            raise KeyError(f"未知 cls_proj_mode：{cls_proj_mode}")
        self.cls_proj_mode = "none" if not use_cls_proj else str(cls_proj_mode)
        shared_embedding = nn.Embedding(pathway_dim, d_model) if self.shared_pathway else None
        shared_cls_proj: nn.Module | None = None
        if self.cls_proj_mode == "shared":
            shared_cls_proj = nn.Linear(pathway_dim, d_model)
        elif self.cls_proj_mode == "lowrank":
            shared_cls_proj = nn.Sequential(
                nn.Linear(pathway_dim, cls_proj_rank),
                nn.GELU(),
                nn.Linear(cls_proj_rank, d_model),
            )
        encoders: list[nn.Module] = []
        for _ in self.modalities:
            modality_proj = (
                nn.Linear(d_model, d_model, bias=False) if self.shared_pathway_proj else None
            )
            encoders.append(
                PathwayTokenEncoder(
                    pathway_dim,
                    d_model,
                    top_k=top_k,
                    dropout=dropout,
                    token_types=token_types,
                    rank_embedding=rank_embedding,
                    pathway_embedding=shared_embedding,
                    use_cls_proj=self.cls_proj_mode != "none",
                    cls_proj=shared_cls_proj,
                    modality_proj=modality_proj,
                )
            )
        self.encoders = nn.ModuleList(encoders)
        self.missing_tokens = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, 1, d_model)) for _ in self.modalities]
        )
        self.missing_cnv = nn.Parameter(torch.zeros(1, 1, d_model))
        gene_embedding = (
            nn.Embedding(n_target_genes, target_embed_dim)
            if (self.target_branch and use_gene_table)
            else None
        )
        self.drug = DrugEncoder(
            morgan_dim=morgan_dim,
            n_target_genes=n_target_genes,
            d_model=d_model,
            hidden=hidden,
            dropout=dropout,
            use_gin=use_gin,
            use_morgan=use_morgan,
            use_targets=use_targets,
            target_embed_dim=target_embed_dim,
            target_embedding=gene_embedding,
            use_target_table=use_target_table,
        )
        self.target = (
            TargetGeneEncoder(
                n_genes=n_target_genes,
                d_model=d_model,
                gene_emb_dim=target_embed_dim,
                n_values=self.target_n_values,
                n_layers=target_gene_layers,
                n_heads=n_heads,
                dropout=dropout,
                gene_embedding=gene_embedding,
                use_gene_table=use_gene_table,
            )
            if self.target_branch
            else None
        )
        # 初始为 0，让 target branch 从“无贡献”开始学，避免随机基因嵌入早期扰动。
        self.target_gate = nn.Parameter(torch.zeros(1)) if self.target_branch else None
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.cell_pool = AttentionPool(d_model) if cell_readout == "attention" else None
        self.drug_type = nn.Parameter(torch.zeros(d_model)) if token_types else None
        self.cross_gate_param = nn.Parameter(torch.zeros(1)) if cross_gate else None
        self.presence_proj = (
            nn.Linear(
                len(self.modalities) + int(use_neighbor) + int(self.target_branch),
                d_model,
            )
            if self.masked_modalities
            else None
        )
        self.use_dense_shortcut = bool(use_dense_shortcut)
        if self.use_dense_shortcut:
            self.dense_cell = nn.Sequential(
                nn.Linear(d_model * len(self.modalities), d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(d_model),
            )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.readout = readout
        self.d_model = d_model
        head_dim = d_model * 3 if readout == "bilinear" else d_model
        self.heads = ResponseHeads(head_dim, hidden=d_model // 2, dropout=dropout)
        self.pair_proj = nn.Linear(d_model * 3, d_model) if readout == "hybrid" else None
        self.bilinear_scale = nn.Parameter(torch.ones(1)) if readout == "hybrid" else None
        self.bias = nn.Parameter(torch.zeros(1))
        self.use_neighbor = use_neighbor
        self.modality_dropout = float(modality_dropout)
        if use_neighbor:
            self.neighbor_proj = nn.Linear(pathway_dim, d_model)
            self.neighbor_type = nn.Parameter(torch.zeros(d_model))
            self.neighbor_gate = nn.Parameter(torch.zeros(1))

    def _maybe_drop(self, tokens: Tensor, index: int) -> Tensor:
        if not self.training or self.modality_dropout <= 0:
            return tokens
        drop = (torch.rand(tokens.size(0), 1, 1, device=tokens.device) < self.modality_dropout).to(
            tokens.dtype
        )
        missing = self.missing_tokens[index].expand_as(tokens)
        return tokens * (1.0 - drop) + missing * drop

    def _maybe_drop_mask(self, valid: Tensor) -> Tensor:
        if not self.training or self.modality_dropout <= 0:
            return valid
        drop = torch.rand(valid.size(0), device=valid.device) < self.modality_dropout
        return valid & ~drop

    @staticmethod
    def _availability(mask: Tensor | None, batch_size: int, device: torch.device) -> Tensor:
        if mask is None:
            return torch.ones(batch_size, dtype=torch.bool, device=device)
        return mask.view(-1).to(device=device, dtype=torch.bool)

    def forward(
        self,
        batch: TeacherBatch,
        *,
        gene_embedding: nn.Embedding | Tensor | None = None,
        drug_target_vec: Tensor | None = None,
        pair_context: Tensor | None = None,
    ) -> TeacherOutput:
        """``pair_context`` 是可选的 (B, d_model) 残差，用于 T2 的 DTI/PPI 上下文。

        只在 ``readout=sum/hadamard/hybrid`` 下支持；``bilinear`` 的维度契约不同，
        传入会直接报错，避免静默改变读出语义。
        """
        inputs = {"GE": batch.ge, "MU": batch.mu, "CNV": batch.cnv}
        omics: list[Tensor] = []
        omics_masks: list[Tensor] = []
        modality_presence: list[Tensor] = []
        cls_parts: list[Tensor] = []
        batch_size = batch.ge.size(0)
        for index, name in enumerate(self.modalities):
            tokens = self.encoders[index](inputs[name])
            if name == "CNV":
                valid = self._availability(batch.cnv_mask, batch_size, tokens.device)
            elif name == "GE":
                valid = self._availability(batch.ge_mask, batch_size, tokens.device)
            else:
                valid = self._availability(batch.mu_mask, batch_size, tokens.device)

            if self.masked_modalities:
                valid = self._maybe_drop_mask(valid)
                token_mask = valid.unsqueeze(1).expand(-1, tokens.size(1))
            else:
                if name == "CNV":
                    mask = batch.cnv_mask.view(-1, 1, 1).to(tokens.dtype)
                    tokens = tokens * mask + self.missing_cnv.expand_as(tokens) * (1.0 - mask)
                tokens = self._maybe_drop(tokens, index)
                token_mask = torch.ones(
                    tokens.size(0), tokens.size(1), dtype=torch.bool, device=tokens.device
                )
            omics.append(tokens)
            omics_masks.append(token_mask)
            modality_presence.append(valid.to(tokens.dtype))
            cls_parts.append(tokens[:, 0])

        if self.masked_modalities:
            presence = torch.stack(modality_presence, dim=1)
            no_modality = ~presence.any(dim=1)
            if bool(no_modality.any()):
                presence[no_modality, 0] = 1.0
                omics_masks[0] = omics_masks[0] | no_modality.unsqueeze(1).expand_as(omics_masks[0])
                modality_presence[0] = presence[:, 0]
            presence = torch.stack(modality_presence, dim=1)
        else:
            presence = None

        if self.use_neighbor:
            if batch.neighbor_ge is not None:
                gate = torch.sigmoid(self.neighbor_gate)
                neighbor = self.neighbor_proj(batch.neighbor_ge).unsqueeze(1)
                neighbor = gate * (neighbor + self.neighbor_type)
                if self.masked_modalities:
                    neighbor_valid = self._availability(
                        batch.neighbor_mask, batch_size, neighbor.device
                    )
                    neighbor_mask = neighbor_valid.unsqueeze(1)
                else:
                    neighbor_valid = torch.ones(
                        batch_size, dtype=neighbor.dtype, device=neighbor.device
                    )
                    neighbor_mask = torch.ones(
                        batch_size, 1, dtype=torch.bool, device=neighbor.device
                    )
                omics.append(neighbor)
                omics_masks.append(neighbor_mask)
                cls_parts.append(neighbor[:, 0])
                if self.masked_modalities:
                    presence = torch.cat(
                        [presence, neighbor_valid.to(presence.dtype).unsqueeze(1)], dim=1
                    )
            elif self.masked_modalities:
                presence = torch.cat(
                    [presence, torch.zeros(batch_size, 1, device=presence.device)], dim=1
                )

        if self.target is not None:
            if (
                batch.target_gene_ids is not None
                and batch.target_gene_values is not None
                and batch.target_gene_mask is not None
            ):
                target_token, target_valid = self.target(
                    batch.target_gene_ids,
                    batch.target_gene_values,
                    batch.target_gene_mask,
                    gene_embedding=gene_embedding,
                )
                assert self.target_gate is not None
                target_token = target_token * torch.tanh(self.target_gate)
            else:
                target_token = omics[0].new_zeros(batch_size, self.d_model)
                target_valid = torch.zeros(batch_size, dtype=torch.bool, device=omics[0].device)
            omics.append(target_token.unsqueeze(1))
            omics_masks.append(target_valid.unsqueeze(1))
            cls_parts.append(target_token)
            if self.masked_modalities:
                presence = torch.cat(
                    [presence, target_valid.to(presence.dtype).unsqueeze(1)], dim=1
                )

        tokens = torch.cat(omics, dim=1)
        token_mask = torch.cat(omics_masks, dim=1)
        if self.masked_modalities and not bool(token_mask.any(dim=1).all()):
            token_mask = token_mask.clone()
            token_mask[~token_mask.any(dim=1), 0] = True

        drug0 = self.drug(batch, target_vec=drug_target_vec)
        if self.masked_modalities:
            cross, attention = self.cross_attn(
                query=drug0.unsqueeze(1),
                key=tokens,
                value=tokens,
                key_padding_mask=~token_mask,
                need_weights=True,
            )
        else:
            cross, attention = self.cross_attn(
                query=drug0.unsqueeze(1),
                key=tokens,
                value=tokens,
                need_weights=True,
            )
        if self.cross_gate_param is not None:
            cross = cross * torch.sigmoid(self.cross_gate_param)
        drug_fused = self.cross_norm(drug0 + self.dropout(cross.squeeze(1)))
        if self.token_types and self.drug_type is not None:
            drug_fused = drug_fused + self.drug_type

        sequence = torch.cat([drug_fused.unsqueeze(1), tokens], dim=1)
        if self.masked_modalities:
            sequence_mask = torch.cat(
                [
                    torch.ones(batch_size, 1, dtype=torch.bool, device=tokens.device),
                    token_mask,
                ],
                dim=1,
            )
            encoded = self.encoder(sequence, src_key_padding_mask=~sequence_mask)
        else:
            encoded = self.encoder(sequence)

        z_drug = encoded[:, 0] + drug0
        if self.cell_pool is not None:
            z_cell = self.cell_pool(encoded[:, 1:], token_mask)
        elif self.masked_modalities:
            weights = token_mask.unsqueeze(-1).to(encoded.dtype)
            z_cell = (encoded[:, 1:] * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        else:
            z_cell = encoded[:, 1:].mean(dim=1)
        if self.presence_proj is not None and presence is not None:
            z_cell = z_cell + self.presence_proj(presence)
        if self.use_dense_shortcut:
            if self.masked_modalities:
                cls_tokens = torch.cat(
                    [
                        cls_parts[index] * modality_presence[index].unsqueeze(-1)
                        for index in range(len(self.modalities))
                    ],
                    dim=-1,
                )
            else:
                cls_tokens = torch.cat(
                    [cls_parts[index] for index in range(len(self.modalities))], dim=-1
                )
            z_cell = z_cell + self.dense_cell(cls_tokens)

        if self.readout == "bilinear":
            if pair_context is not None:
                raise ValueError("readout=bilinear 不支持 pair_context；请用 sum/hadamard/hybrid")
            interaction = z_cell * z_drug
            y_hat = interaction.sum(dim=-1) / math.sqrt(self.d_model) + self.bias
            z_pair = torch.cat([interaction, z_cell, z_drug], dim=-1)
            _, logits = self.heads(z_pair)
        elif self.readout == "hadamard":
            z_pair = z_cell * z_drug
            if pair_context is not None:
                z_pair = z_pair + pair_context
            y_hat, logits = self.heads(z_pair)
        elif self.readout == "hybrid":
            interaction = z_cell * z_drug
            z_pair = self.pair_proj(torch.cat([z_cell, z_drug, interaction], dim=-1))
            if pair_context is not None:
                z_pair = z_pair + pair_context
            y_head, logits = self.heads(z_pair)
            y_hat = (
                self.bilinear_scale * interaction.sum(dim=-1) / math.sqrt(self.d_model)
                + y_head
                + self.bias
            )
        else:
            z_pair = z_drug + z_cell
            if pair_context is not None:
                z_pair = z_pair + pair_context
            y_hat, logits = self.heads(z_pair)
        aux = {"attention_tokens": tokens, "omics_mask": token_mask}
        if attention is not None:
            aux["drug_pathway_attention"] = attention.squeeze(1)
        return TeacherOutput(
            y_hat=y_hat,
            logits=logits,
            z_pair=z_pair,
            z_cell=z_cell,
            z_drug=z_drug,
            aux=aux,
        )


__all__ = ["PathwayTokenEncoder", "T0Teacher", "TargetGeneEncoder"]

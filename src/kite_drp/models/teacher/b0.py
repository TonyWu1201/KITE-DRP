"""B0：双线性潜交互基线（无图、无注意力，CPU 可跑）。

计划书 §4.2 B0 / NeurixAI 结构的缩放版：肿瘤塔与药物塔各输出 d 维潜向量，
响应分数为点积。输入为通路分数（1,000 维）而非全转录组，参数控制在 ~2M。
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from kite_drp.models.teacher.common import MLP, TeacherBatch, TeacherOutput
from kite_drp.models.teacher.drug_enc import DrugEncoder
from kite_drp.models.teacher.heads import ResponseHeads


class ModalityProjector(nn.Module):
    """单模态投影；缺失时使用可学习 missing 向量。"""

    def __init__(self, in_dim: int, hidden: int, *, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden),
        )
        self.missing = nn.Parameter(torch.zeros(hidden))

    def forward(self, scores: Tensor, mask: Tensor | None = None) -> Tensor:
        projected = self.net(scores)
        if mask is None:
            return projected
        mask = mask.unsqueeze(-1).to(projected.dtype)
        return projected * mask + self.missing * (1.0 - mask)


class BilinearTeacher(nn.Module):
    def __init__(
        self,
        *,
        pathway_dim: int,
        morgan_dim: int,
        n_target_genes: int,
        d_model: int = 256,
        hidden: int = 256,
        dropout: float = 0.1,
        use_gin: bool = False,
        use_morgan: bool = True,
        use_targets: bool = True,
        target_embed_dim: int = 16,
        modalities: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        super().__init__()
        if isinstance(modalities, str):
            modalities = [modalities]
        self.modalities = tuple(modalities or ("GE", "MU", "CNV"))
        if not self.modalities:
            raise ValueError("B0 至少需要一个组学模态")
        unknown = set(self.modalities) - {"GE", "MU", "CNV"}
        if unknown:
            raise KeyError(f"未知模态：{sorted(unknown)}")
        self.ge = (
            ModalityProjector(pathway_dim, hidden, dropout=dropout)
            if "GE" in self.modalities
            else None
        )
        self.mu = (
            ModalityProjector(pathway_dim, hidden, dropout=dropout)
            if "MU" in self.modalities
            else None
        )
        self.cnv = (
            ModalityProjector(pathway_dim, hidden, dropout=dropout)
            if "CNV" in self.modalities
            else None
        )
        self.cell = MLP(
            hidden * len(self.modalities),
            [d_model * 2],
            d_model,
            dropout=dropout,
            final_norm=True,
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
        )
        self.heads = ResponseHeads(d_model, hidden=d_model // 2, dropout=dropout)
        self.bias = nn.Parameter(torch.zeros(1))
        self.d_model = d_model

    def forward(self, batch: TeacherBatch) -> TeacherOutput:
        parts: list[Tensor] = []
        if self.ge is not None:
            parts.append(self.ge(batch.ge))
        if self.mu is not None:
            parts.append(self.mu(batch.mu))
        if self.cnv is not None:
            parts.append(self.cnv(batch.cnv, batch.cnv_mask))
        z_cell = self.cell(torch.cat(parts, dim=-1))
        z_drug = self.drug(batch)
        z_pair = z_cell * z_drug
        y_hat = (z_cell * z_drug).sum(dim=-1) / math.sqrt(self.d_model) + self.bias
        _, logits = self.heads(z_pair)
        return TeacherOutput(
            y_hat=y_hat,
            logits=logits,
            z_pair=z_pair,
            z_cell=z_cell,
            z_drug=z_drug,
        )


__all__ = ["BilinearTeacher", "ModalityProjector"]

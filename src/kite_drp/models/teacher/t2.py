"""T2：冻结 P1 + 独立第二塔 + 无图 DTI/PPI 上下文。

设计依据：

- 旧 T1 的随机 PPI/DTI/去 cell-gene 边都能复现大部分增益，图拓扑不是来源；
- 1.5M MLP 头只有 +0.0009，单纯容量也不是来源；
- 剩余可复现的成分是“第二套独立编码 + 与 P1 的预测级融合”。

因此 T2 不再建图：

1. ``backbone`` 是冻结的 P1，只提供 ``y_p1`` 与稳定的基线；
2. ``tower`` 是独立初始化的 T0/P1 骨架（建议更窄/更浅、关闭相似性邻居），
   从原始 Morgan + targets + GE/MU/CNV 重新编码；
3. ``DTI/PPI`` 只作为固定特征进入 ``tower`` 的 ``pair_context``：
   根/邻居集合统计、通路 profile、RWR 加权组学；没有 GNN/消息传递；
4. 融合为预测级 gate：``y = y_p1 + tanh(alpha) * (y_t2 - y_p1)``，
   ``alpha`` 初始 0，第二塔通过深监督 ``tower_weight`` 先学起来。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from kite_drp.models.teacher.common import TeacherBatch, TeacherOutput
from kite_drp.models.teacher.t0 import T0Teacher


class T2DualTeacher(nn.Module):
    def __init__(
        self,
        backbone: T0Teacher,
        tower: T0Teacher,
        *,
        summary_dim: int = 0,
        rwr_dim: int = 0,
        n_pathways: int = 0,
        context_hidden: int = 64,
        dropout: float = 0.1,
        fusion: str = "global",  # global | sample
        alpha_init: float = 0.0,
        tower_weight: float = 0.3,
        context_scale_init: float = 1.0,
    ) -> None:
        super().__init__()
        if fusion not in {"global", "sample", "mean"}:
            raise ValueError(f"fusion 必须是 global/sample/mean，收到 {fusion}")
        self.backbone = backbone
        self.tower = tower
        self.d_model = int(tower.d_model)
        self.fusion = str(fusion)
        self.tower_weight = float(tower_weight)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.context_scale = nn.Parameter(torch.tensor(float(context_scale_init)))
        self.use_context = summary_dim > 0 or rwr_dim > 0 or n_pathways > 0

        self.summary_proj = (
            nn.Sequential(
                nn.Linear(int(summary_dim), int(context_hidden)),
                nn.GELU(),
                nn.LayerNorm(int(context_hidden)),
            )
            if summary_dim > 0
            else None
        )
        self.rwr_proj = (
            nn.Sequential(
                nn.Linear(int(rwr_dim), int(context_hidden)),
                nn.GELU(),
                nn.LayerNorm(int(context_hidden)),
            )
            if rwr_dim > 0
            else None
        )
        self.pathway_proj = (
            nn.Linear(4 * int(n_pathways), int(context_hidden)) if n_pathways > 0 else None
        )
        context_in = int(context_hidden) * sum(
            item is not None for item in (self.summary_proj, self.rwr_proj, self.pathway_proj)
        )
        self.context_mlp = (
            nn.Sequential(
                nn.Linear(context_in, self.d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(self.d_model),
            )
            if context_in > 0
            else None
        )
        if self.fusion == "sample":
            gate_in = 3 + 1 + 3  # y_p1, y_t2, |diff|, coverage, tier one-hot
            self.fusion_gate = nn.Sequential(
                nn.Linear(gate_in, 32),
                nn.GELU(),
                nn.Linear(32, 1),
            )
        else:
            self.fusion_gate = None
        self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True) -> T2DualTeacher:
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def _p1_forward(self, batch: TeacherBatch) -> TeacherOutput:
        self.backbone.eval()
        return self.backbone(batch)

    def _context(self, batch: TeacherBatch) -> Tensor | None:
        if not self.use_context:
            return None
        pieces: list[Tensor] = []
        if self.summary_proj is not None:
            if batch.ppi_summary is None:
                raise ValueError("T2 需要 batch.ppi_summary")
            pieces.append(self.summary_proj(batch.ppi_summary.float()))
        if self.rwr_proj is not None:
            if batch.ppi_rwr is None:
                raise ValueError("T2 需要 batch.ppi_rwr")
            pieces.append(self.rwr_proj(batch.ppi_rwr.float()))
        if self.pathway_proj is not None:
            if (
                batch.ppi_target_pathway is None
                or batch.ppi_neighbor_pathway is None
            ):
                raise ValueError("T2 需要 batch.ppi_target_pathway/ppi_neighbor_pathway")
            ge = batch.ge.float()
            cnv = batch.cnv.float()
            target = batch.ppi_target_pathway.float()
            neighbor = batch.ppi_neighbor_pathway.float()
            pathway = torch.cat(
                [target * ge, neighbor * ge, target * cnv, neighbor * cnv],
                dim=-1,
            )
            pieces.append(self.pathway_proj(pathway))
        if self.context_mlp is None or not pieces:
            return None
        context = self.context_mlp(torch.cat(pieces, dim=-1))
        if batch.ppi_coverage is not None:
            context = context * batch.ppi_coverage.to(context.dtype).unsqueeze(-1)
        return torch.tanh(self.context_scale) * context

    def forward(self, batch: TeacherBatch) -> TeacherOutput:
        if batch.drug_idx is None or batch.cell_idx is None:
            raise ValueError("T2 需要 batch.drug_idx 与 batch.cell_idx")
        p1 = self._p1_forward(batch)
        context = self._context(batch)
        t2 = self.tower(batch, pair_context=context)
        y_t2 = t2.y_hat
        if self.fusion == "sample":
            assert self.fusion_gate is not None
            coverage = (
                batch.ppi_coverage.to(y_t2.dtype)
                if batch.ppi_coverage is not None
                else torch.ones_like(y_t2)
            )
            tier = (
                batch.ppi_tier
                if batch.ppi_tier is not None
                else torch.zeros_like(batch.drug_idx)
            )
            tier_onehot = torch.nn.functional.one_hot(
                tier.clamp(0, 2).long(), num_classes=3
            ).to(y_t2.dtype)
            gate_in = torch.cat(
                [
                    p1.y_hat.detach().float().unsqueeze(-1),
                    y_t2.detach().float().unsqueeze(-1),
                    (y_t2 - p1.y_hat).abs().float().unsqueeze(-1),
                    coverage.float().unsqueeze(-1),
                    tier_onehot,
                ],
                dim=-1,
            )
            gate = torch.sigmoid(self.fusion_gate(gate_in).squeeze(-1)).to(y_t2.dtype)
            y_hat = p1.y_hat + gate * (y_t2 - p1.y_hat)
            logits = p1.logits + gate.unsqueeze(-1) * (t2.logits - p1.logits)
            aux_gate: Tensor | float = gate.detach()
        elif self.fusion == "mean":
            blend = y_t2.new_tensor(0.5)
            y_hat = p1.y_hat + blend * (y_t2 - p1.y_hat)
            logits = p1.logits + blend * (t2.logits - p1.logits)
            aux_gate = 0.5
        else:
            blend = torch.tanh(self.alpha.to(y_t2.dtype))
            y_hat = p1.y_hat + blend * (y_t2 - p1.y_hat)
            logits = p1.logits + blend * (t2.logits - p1.logits)
            aux_gate = float(blend.detach().cpu())
        aux = dict(p1.aux)
        aux.update(
            {
                "y_p1": p1.y_hat,
                "y_t2": y_t2,
                "logits_t2": t2.logits,
                "t2_alpha": aux_gate,
                "context_scale": torch.tanh(self.context_scale).detach(),
                "ppi_coverage": (
                    batch.ppi_coverage
                    if batch.ppi_coverage is not None
                    else torch.ones_like(y_t2, dtype=torch.bool)
                ),
            }
        )
        return TeacherOutput(
            y_hat=y_hat,
            logits=logits,
            z_pair=t2.z_pair,
            z_cell=t2.z_cell,
            z_drug=t2.z_drug,
            aux=aux,
        )

    def load_p1_checkpoint(self, state: dict[str, Tensor]) -> None:
        self.backbone.load_state_dict(state, strict=False)


def count_t2_parameters(model: T2DualTeacher) -> dict[str, int]:
    backbone = sum(parameter.numel() for parameter in model.backbone.parameters())
    tower = sum(parameter.numel() for parameter in model.tower.parameters())
    context = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone.") and not name.startswith("tower.")
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    return {"total": total, "backbone": backbone, "tower": tower, "context": context}


__all__ = ["T2DualTeacher", "count_t2_parameters"]

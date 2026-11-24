"""T2：冻结 P1 + 独立第二塔 + 无图 DTI/PPI 上下文。

设计依据（2026-11-21 实施记录 + 2026-11-23 调研）：

- 旧 T1 的随机 PPI/DTI/去 cell-gene 边都能复现大部分增益，图拓扑不是来源；
- T2 的基础增益来自“第二套独立编码 + 预测级融合”，不是 PPI 特征；
- 当前 RWR 与靶点集合统计高度相关（GE 0.963 / MU 0.991 / CNV 0.984），
  因为 tower 的 target branch 已经读取同一组靶点基因。

因此本版本在**不改变 P1** 的前提下，把先验分支改造成可辨识的三段：

1. ``context``：summary / rwr / pathway / meta / NetGP 间接靶点；
2. ``context_mode``：``add``（旧行为）| ``head``（先验独立输出）| ``both``；
3. ``fusion``：``none``（只训练塔，事后校准）| ``global`` | ``mean`` | ``sample``。

``head`` 模式下先验分支有自己的 ``y_ctx``，因此可以：

- 单独评估“先验自己有多强”（``y_ctx`` 的逐药 SCC）；
- 在遮挡实验里直接看 ``y_t2`` / ``y_ctx``，而不是被融合系数 α 缩放的最终输出；
- 事后用 ``scripts/calibrate_t2_blend.py`` 在 val 上校准
  ``y_p1 + w_t (y_t2 - y_p1) + w_c y_ctx``，再一次性评估 test。

NetGP 间接靶点通过一个独立的 gene-token encoder 进入 context：它与 tower 的
target branch 共享基因嵌入表，但输入是**非直接靶点**的 top-k 基因，因此
提供了 target branch 看不到的 gene-level 基。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from kite_drp.models.teacher.common import TeacherBatch, TeacherOutput
from kite_drp.models.teacher.t0 import T0Teacher, TargetGeneEncoder


class T2DualTeacher(nn.Module):
    def __init__(
        self,
        backbone: T0Teacher,
        tower: T0Teacher,
        *,
        summary_dim: int = 0,
        rwr_dim: int = 0,
        n_pathways: int = 0,
        meta_dim: int = 0,
        netgp_top_k: int = 0,
        netgp_pathway_dim: int = 0,
        netgp_reliability_dim: int = 0,
        n_target_genes: int = 0,
        modalities: tuple[str, ...] | list[str] = ("GE", "MU", "CNV"),
        context_hidden: int = 64,
        dropout: float = 0.1,
        fusion: str = "global",  # global | sample | mean | none
        alpha_init: float = 0.0,
        tower_weight: float = 0.3,
        context_scale_init: float = 1.0,
        context_mode: str = "add",  # add | head | both
        ctx_gate_init: float = 0.0,
        netgp_gate_init: float = 0.0,
        zero_uncovered: bool = True,
    ) -> None:
        super().__init__()
        if fusion not in {"global", "sample", "mean", "none"}:
            raise ValueError(f"fusion 必须是 global/sample/mean/none，收到 {fusion}")
        if context_mode not in {"add", "head", "both"}:
            raise ValueError(f"context_mode 必须是 add/head/both，收到 {context_mode}")
        self.backbone = backbone
        self.tower = tower
        self.d_model = int(tower.d_model)
        self.fusion = str(fusion)
        self.context_mode = str(context_mode)
        self.tower_weight = float(tower_weight)
        self.zero_uncovered = bool(zero_uncovered)
        self.meta_dim = int(meta_dim)
        self.netgp_reliability_dim = int(netgp_reliability_dim)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.context_scale = nn.Parameter(torch.tensor(float(context_scale_init)))
        self.ctx_gate = nn.Parameter(torch.tensor(float(ctx_gate_init)))
        self.netgp_gate = nn.Parameter(torch.tensor(float(netgp_gate_init)))

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
        self.meta_proj = (
            nn.Sequential(
                nn.Linear(int(meta_dim), int(context_hidden)),
                nn.GELU(),
                nn.LayerNorm(int(context_hidden)),
            )
            if meta_dim > 0
            else None
        )
        self.netgp_path_proj = (
            nn.Sequential(
                nn.Linear(int(netgp_pathway_dim), int(context_hidden)),
                nn.GELU(),
                nn.LayerNorm(int(context_hidden)),
            )
            if netgp_pathway_dim > 0
            else None
        )
        context_pieces = (
            self.summary_proj,
            self.rwr_proj,
            self.pathway_proj,
            self.meta_proj,
            self.netgp_path_proj,
        )
        context_in = int(context_hidden) * sum(item is not None for item in context_pieces)
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

        self.netgp_encoder: TargetGeneEncoder | None = None
        if netgp_top_k > 0 and n_target_genes > 0:
            shared_gene_table = (
                tower.target.gene_embedding if getattr(tower, "target", None) is not None else None
            )
            gene_emb_dim = (
                int(getattr(shared_gene_table, "embedding_dim", 16))
                if shared_gene_table is not None
                else 16
            )
            netgp_heads = 4 if self.d_model % 4 == 0 else (2 if self.d_model % 2 == 0 else 1)
            self.netgp_encoder = TargetGeneEncoder(
                n_genes=int(n_target_genes),
                d_model=self.d_model,
                gene_emb_dim=gene_emb_dim,
                n_values=2 * len(modalities) + 1,
                n_layers=1,
                n_heads=netgp_heads,
                dropout=dropout,
                gene_embedding=shared_gene_table,
                use_gene_table=True,
            )
            self.netgp_norm = nn.LayerNorm(self.d_model)
            self.netgp_type = nn.Parameter(torch.zeros(self.d_model))

        reliability_in = int(meta_dim) + int(netgp_reliability_dim)
        if reliability_in > 0:
            self.reliability_gate = nn.Sequential(
                nn.Linear(reliability_in, 32),
                nn.GELU(),
                nn.Linear(32, 1),
            )
            # 初始等价于旧的全局 context_scale，避免随机 gate 扰动早期训练。
            nn.init.zeros_(self.reliability_gate[-1].weight)
            nn.init.zeros_(self.reliability_gate[-1].bias)
        else:
            self.reliability_gate = None

        if self.context_mode in {"head", "both"}:
            self.ctx_norm = nn.LayerNorm(self.d_model)
            self.ctx_head = nn.Sequential(
                nn.Linear(self.d_model, self.d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.d_model // 2, 2),
            )
            nn.init.zeros_(self.ctx_head[-1].weight)
            nn.init.zeros_(self.ctx_head[-1].bias)
        else:
            self.ctx_head = None

        if self.fusion == "sample":
            gate_in = 3 + 1 + 3  # y_p1, y_t2, |diff|, coverage, tier one-hot
            self.fusion_gate = nn.Sequential(
                nn.Linear(gate_in, 32),
                nn.GELU(),
                nn.Linear(32, 1),
            )
        else:
            self.fusion_gate = None

        self.use_context = any(item is not None for item in context_pieces) or (
            self.netgp_encoder is not None
        )
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

    def _context(self, batch: TeacherBatch) -> tuple[Tensor | None, Tensor | None]:
        if not self.use_context:
            return None, None
        pieces: list[Tensor] = []
        coverage = (
            batch.ppi_coverage.to(torch.float32).unsqueeze(-1)
            if batch.ppi_coverage is not None and self.zero_uncovered
            else None
        )

        def _cover(values: Tensor) -> Tensor:
            return values * coverage if coverage is not None else values

        if self.summary_proj is not None:
            if batch.ppi_summary is None:
                raise ValueError("T2 需要 batch.ppi_summary")
            pieces.append(self.summary_proj(_cover(batch.ppi_summary.float())))
        if self.rwr_proj is not None:
            if batch.ppi_rwr is None:
                raise ValueError("T2 需要 batch.ppi_rwr")
            pieces.append(self.rwr_proj(_cover(batch.ppi_rwr.float())))
        if self.pathway_proj is not None:
            if batch.ppi_target_pathway is None or batch.ppi_neighbor_pathway is None:
                raise ValueError("T2 需要 batch.ppi_target_pathway/ppi_neighbor_pathway")
            ge = batch.ge.float()
            cnv = batch.cnv.float()
            target = _cover(batch.ppi_target_pathway.float())
            neighbor = _cover(batch.ppi_neighbor_pathway.float())
            pathway = torch.cat(
                [target * ge, neighbor * ge, target * cnv, neighbor * cnv],
                dim=-1,
            )
            pieces.append(self.pathway_proj(pathway))
        if self.meta_proj is not None:
            if batch.ppi_meta is None:
                raise ValueError("T2 需要 batch.ppi_meta")
            pieces.append(self.meta_proj(batch.ppi_meta.float()))
        if self.netgp_path_proj is not None:
            if batch.netgp_pathway is None:
                raise ValueError("T2 需要 batch.netgp_pathway")
            pieces.append(self.netgp_path_proj(batch.netgp_pathway.float()))

        context = (
            self.context_mlp(torch.cat(pieces, dim=-1))
            if self.context_mlp is not None and pieces
            else None
        )
        if self.netgp_encoder is not None and batch.netgp_gene_ids is not None:
            if (
                batch.netgp_gene_values is None
                or batch.netgp_gene_weight is None
                or batch.netgp_gene_mask is None
            ):
                raise ValueError("T2 需要 netgp_gene_values/weight/mask")
            netgp_values = torch.cat(
                [
                    batch.netgp_gene_values.float(),
                    batch.netgp_gene_weight.float().unsqueeze(-1),
                ],
                dim=-1,
            )
            token, _ = self.netgp_encoder(
                batch.netgp_gene_ids,
                netgp_values,
                batch.netgp_gene_mask,
            )
            token = self.netgp_norm(token + self.netgp_type)
            token = torch.tanh(self.netgp_gate).to(token.dtype) * token
            context = token if context is None else context + token
        if context is None:
            return None, None

        gate_inputs: list[Tensor] = []
        if self.meta_dim > 0 and batch.ppi_meta is not None:
            gate_inputs.append(batch.ppi_meta.float())
        if self.netgp_reliability_dim > 0 and batch.netgp_reliability is not None:
            gate_inputs.append(batch.netgp_reliability.float())
        if self.reliability_gate is not None and gate_inputs:
            gate_logit = self.reliability_gate(torch.cat(gate_inputs, dim=-1))
            gate = torch.sigmoid(self.context_scale + gate_logit.squeeze(-1))
        else:
            gate = torch.tanh(self.context_scale).expand(context.size(0))
        if batch.ppi_coverage is not None:
            gate = gate * batch.ppi_coverage.to(gate.dtype)
        gate = gate.to(context.dtype)
        return context * gate.unsqueeze(-1), gate

    def forward(self, batch: TeacherBatch) -> TeacherOutput:
        if batch.drug_idx is None or batch.cell_idx is None:
            raise ValueError("T2 需要 batch.drug_idx 与 batch.cell_idx")
        p1 = self._p1_forward(batch)
        context, gate = self._context(batch)
        pair_context = context if self.context_mode in {"add", "both"} else None
        t2 = self.tower(batch, pair_context=pair_context)
        y_t2 = t2.y_hat
        logits_t2 = t2.logits

        y_ctx: Tensor | None = None
        logits_ctx: Tensor | None = None
        if self.ctx_head is not None and context is not None:
            raw = self.ctx_head(self.ctx_norm(context))
            y_ctx = raw[..., 0]
            logits_ctx = raw[..., 1]
            ctx_scale = torch.tanh(self.ctx_gate).to(y_t2.dtype)
            y_t2_ctx = y_t2 + ctx_scale * y_ctx
            logits_t2_ctx = logits_t2 + ctx_scale * logits_ctx
        else:
            y_t2_ctx = y_t2
            logits_t2_ctx = logits_t2

        if self.fusion == "sample":
            assert self.fusion_gate is not None
            coverage = (
                batch.ppi_coverage.to(y_t2.dtype)
                if batch.ppi_coverage is not None
                else torch.ones_like(y_t2)
            )
            tier = (
                batch.ppi_tier if batch.ppi_tier is not None else torch.zeros_like(batch.drug_idx)
            )
            tier_onehot = torch.nn.functional.one_hot(tier.clamp(0, 2).long(), num_classes=3).to(
                y_t2.dtype
            )
            gate_in = torch.cat(
                [
                    p1.y_hat.detach().float().unsqueeze(-1),
                    y_t2_ctx.detach().float().unsqueeze(-1),
                    (y_t2_ctx - p1.y_hat).abs().float().unsqueeze(-1),
                    coverage.float().unsqueeze(-1),
                    tier_onehot,
                ],
                dim=-1,
            )
            blend = torch.sigmoid(self.fusion_gate(gate_in).squeeze(-1)).to(y_t2.dtype)
            y_hat = p1.y_hat + blend * (y_t2_ctx - p1.y_hat)
            logits = p1.logits + blend.unsqueeze(-1) * (logits_t2_ctx - p1.logits)
            aux_gate: Tensor | float = blend.detach()
        elif self.fusion == "mean":
            blend = y_t2.new_tensor(0.5)
            y_hat = p1.y_hat + blend * (y_t2_ctx - p1.y_hat)
            logits = p1.logits + blend * (logits_t2_ctx - p1.logits)
            aux_gate = 0.5
        elif self.fusion == "none":
            y_hat = p1.y_hat
            logits = p1.logits
            aux_gate = 0.0
        else:
            blend = torch.tanh(self.alpha.to(y_t2.dtype))
            y_hat = p1.y_hat + blend * (y_t2_ctx - p1.y_hat)
            logits = p1.logits + blend * (logits_t2_ctx - p1.logits)
            aux_gate = float(blend.detach().cpu())

        aux = dict(p1.aux)
        aux.update(
            {
                "y_p1": p1.y_hat,
                "y_t2": y_t2,
                "y_t2_ctx": y_t2_ctx,
                "logits_t2": logits_t2,
                "logits_t2_ctx": logits_t2_ctx,
                "t2_alpha": aux_gate,
                "context_scale": torch.tanh(self.context_scale).detach(),
                "context_gate": (gate.detach() if gate is not None else torch.ones_like(y_t2)),
                "ctx_gate": torch.tanh(self.ctx_gate).detach(),
                "netgp_gate": torch.tanh(self.netgp_gate).detach(),
                "ppi_coverage": (
                    batch.ppi_coverage
                    if batch.ppi_coverage is not None
                    else torch.ones_like(y_t2, dtype=torch.bool)
                ),
            }
        )
        if y_ctx is not None and logits_ctx is not None:
            aux["y_ctx"] = y_ctx
            aux["logits_ctx"] = logits_ctx
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

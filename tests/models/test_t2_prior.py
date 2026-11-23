"""T2 先验读出：独立头、NetGP token、reliability gate、fusion=none。"""

from __future__ import annotations

import torch

from kite_drp.models.teacher.common import TeacherBatch
from kite_drp.models.teacher.t0 import T0Teacher
from kite_drp.models.teacher.t2 import T2DualTeacher


def _tower(*, d_model: int = 8, top_k: int = 2) -> T0Teacher:
    return T0Teacher(
        pathway_dim=6,
        morgan_dim=8,
        n_target_genes=10,
        d_model=d_model,
        hidden=d_model,
        dropout=0.0,
        target_embed_dim=4,
        n_heads=2,
        n_layers=1,
        top_k=top_k,
        token_types=True,
        rank_embedding=True,
        cell_readout="attention",
        masked_modalities=True,
        cross_gate=True,
        use_cls_proj=True,
        cls_proj_mode="lowrank",
        cls_proj_rank=2,
        target_branch=True,
        target_gene_layers=1,
        readout="sum",
        use_neighbor=False,
        use_targets=True,
        use_gene_table=True,
        use_target_table=True,
    )


def _batch(*, n_pairs: int = 3, coverage: torch.Tensor | None = None) -> TeacherBatch:
    generator = torch.Generator().manual_seed(0)
    coverage = coverage if coverage is not None else torch.tensor([True, True, False])
    netgp_mask = torch.tensor([[True, True, False]] * n_pairs)
    netgp_weight = torch.tensor([[0.6, 0.4, 0.0]] * n_pairs)
    return TeacherBatch(
        ge=torch.randn(n_pairs, 6, generator=generator),
        mu=torch.rand(n_pairs, 6, generator=generator),
        cnv=torch.randn(n_pairs, 6, generator=generator),
        cnv_mask=torch.ones(n_pairs, dtype=torch.bool),
        morgan=torch.rand(n_pairs, 8, generator=generator),
        target_index=torch.arange(n_pairs * 2),
        target_offset=torch.arange(n_pairs) * 2,
        y=torch.randn(n_pairs, generator=generator),
        y_cls=torch.zeros(n_pairs),
        cls_mask=torch.ones(n_pairs, dtype=torch.bool),
        row_index=torch.arange(n_pairs),
        drug_idx=torch.arange(n_pairs),
        cell_idx=torch.arange(n_pairs),
        ge_mask=torch.ones(n_pairs, dtype=torch.bool),
        mu_mask=torch.ones(n_pairs, dtype=torch.bool),
        target_gene_ids=torch.arange(n_pairs * 2).reshape(n_pairs, 2),
        target_gene_values=torch.randn(n_pairs, 2, 6, generator=generator),
        target_gene_mask=torch.ones(n_pairs, 2, dtype=torch.bool),
        ppi_coverage=coverage,
        ppi_tier=torch.tensor([2, 1, 0])[:n_pairs],
        ppi_meta=torch.randn(n_pairs, 8, generator=generator),
        netgp_gene_ids=torch.tensor([[1, 2, 0]] * n_pairs),
        netgp_gene_values=torch.randn(n_pairs, 3, 6, generator=generator),
        netgp_gene_weight=netgp_weight,
        netgp_gene_mask=netgp_mask,
        netgp_pathway=torch.rand(n_pairs, 6, generator=generator),
        netgp_reliability=torch.rand(n_pairs, 8, generator=generator),
    )


def _model(**kwargs) -> T2DualTeacher:
    return T2DualTeacher(
        _tower(),
        _tower(d_model=6),
        summary_dim=0,
        rwr_dim=0,
        n_pathways=0,
        meta_dim=0,
        netgp_top_k=3,
        netgp_pathway_dim=0,
        netgp_reliability_dim=8,
        n_target_genes=10,
        context_hidden=4,
        dropout=0.0,
        **kwargs,
    )


def test_t2_decomposed_identity_and_aux() -> None:
    model = _model(fusion="none", context_mode="head")
    batch = _batch()
    with torch.no_grad():
        p1 = model.backbone(batch)
        output = model(batch)
    torch.testing.assert_close(output.y_hat, p1.y_hat)
    torch.testing.assert_close(output.logits, p1.logits)
    assert "y_ctx" in output.aux
    assert "y_t2" in output.aux
    assert "y_t2_ctx" in output.aux
    torch.testing.assert_close(output.aux["y_t2_ctx"], output.aux["y_t2"])


def test_netgp_token_starts_at_zero() -> None:
    model = _model(fusion="none", context_mode="head")
    batch = _batch()
    with torch.no_grad():
        context, gate = model._context(batch)
    assert context is not None and gate is not None
    assert float(context.abs().max()) == 0.0
    # gate 仍由 reliability 决定，覆盖行应为正。
    assert float(gate[0]) > 0.0
    assert float(gate[2]) == 0.0


def test_netgp_token_learns_when_gate_open() -> None:
    model = _model(fusion="none", context_mode="head", netgp_gate_init=1.0)
    model.ctx_head[-1].weight.data.normal_(0.0, 0.1)
    batch = _batch()
    output = model(batch)
    loss = output.aux["y_ctx"].pow(2).mean() + output.aux["y_t2_ctx"].pow(2).mean()
    loss.backward()
    encoder_grad = model.netgp_encoder.gene_proj.weight.grad
    assert encoder_grad is not None
    assert float(encoder_grad.abs().sum()) > 0.0


def test_reliability_gate_is_bounded() -> None:
    model = _model(fusion="none", context_mode="head", netgp_gate_init=1.0)
    batch = _batch()
    with torch.no_grad():
        _, gate = model._context(batch)
    assert gate is not None
    assert torch.all(gate >= 0) and torch.all(gate <= 1)
    assert gate.shape == (3,)


def test_t2_both_mode_keeps_independent_head() -> None:
    model = _model(fusion="none", context_mode="both", netgp_gate_init=1.0)
    model.ctx_head[-1].weight.data.normal_(0.0, 0.1)
    batch = _batch()
    with torch.no_grad():
        output = model(batch)
        expected = output.aux["y_t2"] + torch.tanh(model.ctx_gate) * output.aux["y_ctx"]
    torch.testing.assert_close(output.aux["y_t2_ctx"], expected)
    assert not torch.allclose(output.aux["y_ctx"], torch.zeros_like(output.aux["y_ctx"]))

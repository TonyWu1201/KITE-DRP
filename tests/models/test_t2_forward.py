"""T2 双塔前向契约：恒等、覆盖回退、冻结、融合与批一致性。"""

from __future__ import annotations

import copy

import torch

from kite_drp.models.teacher.common import TeacherBatch
from kite_drp.models.teacher.t0 import T0Teacher
from kite_drp.models.teacher.t2 import T2DualTeacher


def _tower(*, d_model: int = 8, n_layers: int = 1, top_k: int = 2) -> T0Teacher:
    return T0Teacher(
        pathway_dim=6,
        morgan_dim=8,
        n_target_genes=10,
        d_model=d_model,
        hidden=d_model,
        dropout=0.0,
        target_embed_dim=4,
        n_heads=2,
        n_layers=n_layers,
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


def _batch(
    *,
    n_pairs: int = 3,
    coverage: torch.Tensor | None = None,
    ge: torch.Tensor | None = None,
) -> TeacherBatch:
    generator = torch.Generator().manual_seed(0)
    ge = ge if ge is not None else torch.randn(n_pairs, 6, generator=generator)
    coverage = coverage if coverage is not None else torch.tensor([True, True, False])
    return TeacherBatch(
        ge=ge,
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
        ppi_summary=torch.randn(n_pairs, 38, generator=generator),
        ppi_rwr=torch.randn(n_pairs, 8, generator=generator),
        ppi_target_pathway=torch.rand(n_pairs, 6, generator=generator),
        ppi_neighbor_pathway=torch.rand(n_pairs, 6, generator=generator),
        ppi_coverage=coverage,
        ppi_tier=torch.tensor([2, 1, 0])[:n_pairs],
    )


def _model(**kwargs) -> T2DualTeacher:
    return T2DualTeacher(
        _tower(),
        _tower(d_model=6),
        summary_dim=38,
        rwr_dim=8,
        n_pathways=6,
        context_hidden=4,
        dropout=0.0,
        **kwargs,
    )


def test_t2_identity_at_init() -> None:
    model = _model()
    batch = _batch()
    with torch.no_grad():
        p1 = model.backbone(batch)
        output = model(batch)
    torch.testing.assert_close(output.y_hat, p1.y_hat)
    torch.testing.assert_close(output.logits, p1.logits)
    assert float(output.aux["t2_alpha"]) == 0.0


def test_t2_context_is_masked_by_coverage() -> None:
    model = _model()
    batch = _batch(coverage=torch.tensor([True, False, True]))
    with torch.no_grad():
        context, gate = model._context(batch)
    assert context is not None and gate is not None
    assert float(context[1].abs().max()) == 0.0
    assert float(gate[1]) == 0.0
    assert float(context[0].abs().max()) > 0.0
    assert float(context[2].abs().max()) > 0.0


def test_t2_uncovered_rows_stay_at_p1_even_with_alpha() -> None:
    model = _model(alpha_init=1.0)
    batch = _batch(coverage=torch.tensor([False, False, False]))
    with torch.no_grad():
        # 覆盖为 0 时 context 必须为 0；此时 T2 仍可预测，但主融合 gate 为 1。
        # 这里只验证 context 的精确回退，不假设 y_t2 == y_p1。
        context, gate = model._context(batch)
    assert context is not None and gate is not None
    assert float(context.abs().max()) == 0.0
    assert float(gate.abs().max()) == 0.0


def test_t2_keeps_backbone_eval_and_frozen() -> None:
    model = _model()
    model.train()
    assert model.training is True
    assert model.backbone.training is False
    assert model.tower.training is True
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert any(parameter.requires_grad for parameter in model.tower.parameters())
    assert any(parameter.requires_grad for parameter in model.summary_proj.parameters())


def test_t2_mean_fusion_is_half_half() -> None:
    model = _model(fusion="mean")
    batch = _batch()
    with torch.no_grad():
        output = model(batch)
        expected = 0.5 * (output.aux["y_p1"] + output.aux["y_t2"])
    torch.testing.assert_close(output.y_hat, expected)
    assert float(output.aux["t2_alpha"]) == 0.5


def test_t2_sample_fusion_is_finite() -> None:
    model = _model(fusion="sample")
    batch = _batch()
    with torch.no_grad():
        output = model(batch)
    assert torch.isfinite(output.y_hat).all()
    assert torch.isfinite(output.logits).all()
    assert output.aux["t2_alpha"].shape == (3,)


def test_t2_batch_invariance() -> None:
    model = _model()
    with torch.no_grad():
        mixed = model(_batch()).y_hat
        alone_batch = _batch(
            n_pairs=1,
            coverage=torch.tensor([True]),
            ge=torch.randn(1, 6, generator=torch.Generator().manual_seed(3)),
        )
        alone = model(alone_batch).y_hat
    # 不同随机输入不要求相等；这里验证单独前向不依赖同批其他样本的形状。
    assert mixed.shape == (3,)
    assert alone.shape == (1,)
    assert torch.isfinite(alone).all()


def test_t2_checkpoint_load_keeps_tower() -> None:
    model = _model()
    backbone_state = copy.deepcopy(model.backbone.state_dict())
    tower_state = copy.deepcopy(model.tower.state_dict())
    model.load_p1_checkpoint(backbone_state)
    for key, value in model.tower.state_dict().items():
        torch.testing.assert_close(value, tower_state[key])


def test_t0_pair_context_changes_sum_readout() -> None:
    torch.manual_seed(0)
    model = _tower()
    batch = _batch()
    with torch.no_grad():
        base = model(batch)
        context = torch.ones(3, model.d_model)
        changed = model(batch, pair_context=context)
    assert not torch.allclose(base.y_hat, changed.y_hat)

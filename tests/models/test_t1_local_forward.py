"""T1-local 前向契约：恒等、回退、冻结、批量一致性。"""

from __future__ import annotations

import copy

import pytest
import torch

from kite_drp.models.teacher.common import TeacherBatch
from kite_drp.models.teacher.t0 import T0Teacher
from kite_drp.models.teacher.t1_local import T1LocalTeacher, T1MlpHeadTeacher


def _backbone() -> T0Teacher:
    return T0Teacher(
        pathway_dim=6,
        morgan_dim=8,
        n_target_genes=10,
        d_model=16,
        hidden=16,
        dropout=0.0,
        target_embed_dim=4,
        n_heads=2,
        n_layers=1,
        top_k=2,
        token_types=True,
        rank_embedding=True,
        cell_readout="attention",
        masked_modalities=True,
        cross_gate=True,
        use_cls_proj=True,
        cls_proj_mode="lowrank",
        cls_proj_rank=2,
        target_branch=False,
        target_gene_layers=1,
        readout="sum",
        use_neighbor=False,
        use_targets=False,
        use_gene_table=True,
        use_target_table=True,
    )


def _batch(
    *,
    n_pairs: int = 3,
    node_values: torch.Tensor | None = None,
    node_root: torch.Tensor | None = None,
    node_pair: torch.Tensor | None = None,
    edge_index: torch.Tensor | None = None,
    coverage: torch.Tensor | None = None,
    observed: torch.Tensor | None = None,
    has_neighbor: torch.Tensor | None = None,
    ge: torch.Tensor | None = None,
    mu: torch.Tensor | None = None,
    cnv: torch.Tensor | None = None,
    morgan: torch.Tensor | None = None,
    y: torch.Tensor | None = None,
) -> TeacherBatch:
    generator = torch.Generator().manual_seed(0)
    ge = ge if ge is not None else torch.randn(n_pairs, 6, generator=generator)
    mu = mu if mu is not None else torch.rand(n_pairs, 6, generator=generator)
    cnv = cnv if cnv is not None else torch.randn(n_pairs, 6, generator=generator)
    morgan = morgan if morgan is not None else torch.rand(n_pairs, 8, generator=generator)
    y = y if y is not None else torch.randn(n_pairs, generator=generator)
    if node_values is None:
        node_values = torch.tensor(
            [
                [1.0, 1.0, 0.0, 1.0, 0.5, 1.0],
                [-1.0, 1.0, 0.0, 1.0, -0.5, 1.0],
                [0.5, 1.0, 0.0, 1.0, 0.0, 1.0],
                [2.0, 1.0, 0.0, 1.0, 1.0, 1.0],
                [0.2, 1.0, 0.0, 1.0, 0.1, 1.0],
                [0.3, 1.0, 0.0, 1.0, 0.2, 1.0],
            ]
        )
    if node_root is None:
        node_root = torch.tensor([True, True, False, True, True, False])
    if node_pair is None:
        node_pair = torch.tensor([0, 0, 0, 1, 1, 1])
    if edge_index is None:
        edge_index = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]])
    if coverage is None:
        coverage = torch.tensor([True, True, False])
    if observed is None:
        observed = torch.tensor([True, True, False])
    if has_neighbor is None:
        has_neighbor = torch.tensor([True, True, False])
    return TeacherBatch(
        ge=ge,
        mu=mu,
        cnv=cnv,
        cnv_mask=torch.ones(n_pairs, dtype=torch.bool),
        morgan=morgan,
        target_index=torch.zeros(0, dtype=torch.long),
        target_offset=torch.zeros(n_pairs, dtype=torch.long),
        y=y,
        y_cls=torch.zeros(n_pairs),
        cls_mask=torch.ones(n_pairs, dtype=torch.bool),
        row_index=torch.arange(n_pairs),
        drug_idx=torch.arange(n_pairs),
        cell_idx=torch.arange(n_pairs),
        ge_mask=torch.ones(n_pairs, dtype=torch.bool),
        mu_mask=torch.ones(n_pairs, dtype=torch.bool),
        local_node_values=node_values,
        local_node_root=node_root,
        local_node_pair=node_pair,
        local_edge_index=edge_index,
        local_graph_ptr=torch.tensor([0, 3, 6]) if n_pairs == 3 else torch.tensor([0, 3]),
        local_graph_coverage=coverage,
        local_graph_observed=observed,
        local_graph_has_neighbor=has_neighbor,
    )


def _model(**kwargs) -> T1LocalTeacher:
    return T1LocalTeacher(
        _backbone(),
        hidden=8,
        readout_hidden=16,
        dropout=0.0,
        **kwargs,
    )


def test_t1_local_identity_at_init() -> None:
    model = _model()
    batch = _batch()
    with torch.no_grad():
        p1 = model.backbone(batch)
        output = model(batch)
    torch.testing.assert_close(output.y_hat, p1.y_hat)
    torch.testing.assert_close(output.logits, p1.logits)
    assert float(output.aux["local_delta"].abs().max()) == 0.0


def test_t1_local_exact_fallback_when_uncovered() -> None:
    model = _model()
    # 打破零初始化，确保 residual 非零；无覆盖时必须仍精确回退。
    with torch.no_grad():
        model.readout[-1].weight.normal_(0.0, 0.1)
        model.readout[-1].bias.normal_(0.0, 0.1)
    batch = _batch(
        coverage=torch.tensor([False, False, False]),
        observed=torch.tensor([False, False, False]),
    )
    with torch.no_grad():
        p1 = model.backbone(batch)
        output = model(batch)
    torch.testing.assert_close(output.y_hat, p1.y_hat)
    assert float(output.aux["local_delta"].abs().max()) == 0.0


def test_t1_local_observed_gate_blocks_missing_roots() -> None:
    model = _model()
    with torch.no_grad():
        model.readout[-1].weight.normal_(0.0, 0.1)
    batch = _batch(
        coverage=torch.tensor([True, True, True]),
        observed=torch.tensor([True, False, True]),
    )
    with torch.no_grad():
        p1 = model.backbone(batch)
        output = model(batch)
    delta = output.y_hat - p1.y_hat
    assert abs(float(delta[1])) < 1e-8
    assert float(delta[0].abs() + delta[2].abs()) > 0.0


def test_t1_local_keeps_backbone_eval_and_frozen() -> None:
    model = _model()
    model.train()
    assert model.training is True
    assert model.backbone.training is False
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert any(parameter.requires_grad for parameter in model.readout.parameters())


def test_t1_local_batch_invariance() -> None:
    model = _model()
    with torch.no_grad():
        model.readout[-1].weight.normal_(0.0, 0.1)
    batch = _batch()
    with torch.no_grad():
        mixed = model(batch).y_hat
        # 单独取第 0 个 pair 的节点（前 3 个节点）与边。
        single = _batch(
            n_pairs=1,
            node_values=batch.local_node_values[:3],
            node_root=batch.local_node_root[:3],
            node_pair=torch.zeros(3, dtype=torch.long),
            edge_index=torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]]),
            coverage=batch.local_graph_coverage[:1],
            observed=batch.local_graph_observed[:1],
            has_neighbor=batch.local_graph_has_neighbor[:1],
            ge=batch.ge[:1],
            mu=batch.mu[:1],
            cnv=batch.cnv[:1],
            morgan=batch.morgan[:1],
            y=batch.y[:1],
        )
        alone = model(single).y_hat
    torch.testing.assert_close(mixed[0], alone[0], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("propagation", ["set", "none", "fixed", "trainable"])
def test_t1_local_propagation_modes(propagation: str) -> None:
    model = _model(propagation=propagation)
    with torch.no_grad():
        output = model(_batch())
    assert torch.isfinite(output.y_hat).all()


def test_t1_mlp_identity_and_fallback() -> None:
    model = T1MlpHeadTeacher(_backbone(), hidden=16, dropout=0.0)
    batch = _batch()
    with torch.no_grad():
        p1 = model.backbone(batch)
        output = model(batch)
    torch.testing.assert_close(output.y_hat, p1.y_hat)
    with torch.no_grad():
        model.head[-1].weight.normal_(0.0, 0.1)
    uncovered = _batch(
        coverage=torch.tensor([False, False, False]),
        observed=torch.tensor([False, False, False]),
    )
    with torch.no_grad():
        p1 = model.backbone(uncovered)
        output = model(uncovered)
    torch.testing.assert_close(output.y_hat, p1.y_hat)


def test_t1_local_checkpoint_load_keeps_local_head() -> None:
    model = _model()
    backbone_state = copy.deepcopy(model.backbone.state_dict())
    local_state = copy.deepcopy(model.readout.state_dict())
    model.load_p1_checkpoint(backbone_state)
    for key, value in model.readout.state_dict().items():
        torch.testing.assert_close(value, local_state[key])

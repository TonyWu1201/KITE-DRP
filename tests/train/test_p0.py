"""P0 组件测试：排序损失、drug-balanced sampler、T0 P0 前向。"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kite_drp.models.teacher.common import TeacherBatch, count_parameters  # noqa: E402
from kite_drp.models.teacher.t0 import T0Teacher  # noqa: E402
from kite_drp.train.data import DrugGroupedBatchSampler, PairDataset  # noqa: E402
from kite_drp.train.losses import pairwise_rank_loss  # noqa: E402


def test_pairwise_rank_loss_prefers_correct_order() -> None:
    labels = torch.tensor([0.0, 2.0, 1.0])
    groups = torch.zeros(3, dtype=torch.long)
    good = pairwise_rank_loss(torch.tensor([0.0, 2.0, 1.0]), labels, groups)
    bad = pairwise_rank_loss(torch.tensor([2.0, 0.0, 1.0]), labels, groups)
    assert torch.isfinite(good)
    assert good < bad


def test_pairwise_rank_loss_handles_small_groups_and_mask() -> None:
    scores = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)
    labels = torch.tensor([0.0, 1.0, 2.0])
    groups = torch.tensor([0, 1, 1])
    mask = torch.tensor([True, False, True])
    loss = pairwise_rank_loss(scores, labels, groups, mask=mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert scores.grad is not None


class _DummyBase:
    drug_idx = np.asarray([0, 0, 0, 1, 1, 2], dtype=np.int64)

    def indices(self, split: str) -> np.ndarray:
        assert split == "train"
        return np.arange(6, dtype=np.int64)


def test_drug_grouped_batch_sampler() -> None:
    dataset = PairDataset(_DummyBase(), "train")  # type: ignore[arg-type]
    sampler = DrugGroupedBatchSampler(
        dataset, drugs_per_batch=2, cells_per_drug=2, seed=0
    )
    seen: set[int] = set()
    for _ in range(5):
        batches = list(sampler)
        assert len(batches) == 2
        for batch in batches:
            assert 0 < len(batch) <= 4
            drugs = {int(_DummyBase.drug_idx[position]) for position in batch}
            assert len(drugs) <= 2
            seen.update(batch)
    assert seen == set(range(6))


def _p0_batch(batch_size: int = 4, pathway_dim: int = 64) -> TeacherBatch:
    return TeacherBatch(
        ge=torch.randn(batch_size, pathway_dim),
        mu=torch.rand(batch_size, pathway_dim),
        cnv=torch.randn(batch_size, pathway_dim),
        cnv_mask=torch.tensor([True, False, True, True][:batch_size]),
        morgan=torch.rand(batch_size, 128).round(),
        target_index=torch.tensor([1, 2, 3, 4][:batch_size], dtype=torch.long),
        target_offset=torch.arange(batch_size, dtype=torch.long),
        y=torch.randn(batch_size),
        y_cls=torch.tensor([0.0, 1.0, 0.0, 1.0][:batch_size]),
        cls_mask=torch.ones(batch_size, dtype=torch.bool),
        ge_mask=torch.ones(batch_size, dtype=torch.bool),
        mu_mask=torch.ones(batch_size, dtype=torch.bool),
        neighbor_ge=torch.randn(batch_size, pathway_dim),
        neighbor_mask=torch.tensor([True, False, True, True][:batch_size]),
    )


def test_t0_p0_forward_and_parameter_budget() -> None:
    model = T0Teacher(
        pathway_dim=64,
        morgan_dim=128,
        n_target_genes=50,
        d_model=32,
        hidden=32,
        dropout=0.0,
        target_embed_dim=8,
        n_heads=4,
        n_layers=1,
        top_k=8,
        token_types=True,
        rank_embedding=True,
        cell_readout="attention",
        masked_modalities=True,
        cross_gate=True,
        readout="hybrid",
    )
    output = model(_p0_batch())
    assert output.y_hat.shape == (4,)
    assert output.z_pair.shape == (4, 32)
    assert output.aux["omics_mask"].shape[0] == 4
    assert output.aux["drug_pathway_attention"].shape[0] == 4
    assert torch.isfinite(output.y_hat).all()
    assert count_parameters(model) < 5_000_000


def test_t0_p0_all_modalities_missing_is_finite() -> None:
    model = T0Teacher(
        pathway_dim=8,
        morgan_dim=128,
        n_target_genes=50,
        d_model=16,
        hidden=16,
        dropout=0.0,
        target_embed_dim=4,
        n_heads=2,
        n_layers=1,
        top_k=4,
        cell_readout="attention",
        masked_modalities=True,
        readout="hybrid",
    )
    batch = _p0_batch(batch_size=2, pathway_dim=8)
    batch.ge_mask = torch.zeros(2, dtype=torch.bool)
    batch.mu_mask = torch.zeros(2, dtype=torch.bool)
    batch.cnv_mask = torch.zeros(2, dtype=torch.bool)
    output = model(batch)
    assert torch.isfinite(output.y_hat).all()

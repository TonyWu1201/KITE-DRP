"""P1 结构测试：target-aware gene branch、共享 pathway embedding、参数预算。"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kite_drp.models.teacher.common import TeacherBatch, count_parameters  # noqa: E402
from kite_drp.models.teacher.t0 import T0Teacher  # noqa: E402
from kite_drp.train.data import build_pair_tensors  # noqa: E402
from kite_drp.train.features import CellFeatureSet, DrugFeatureSet  # noqa: E402


def _p1_batch(batch_size: int = 4, pathway_dim: int = 64, n_targets: int = 6) -> TeacherBatch:
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
        target_gene_ids=torch.randint(0, 50, (batch_size, n_targets)),
        target_gene_values=torch.randn(batch_size, n_targets, 6),
        target_gene_mask=torch.tensor(
            [[True] * n_targets, [True] * (n_targets // 2) + [False] * (n_targets - n_targets // 2)]
            + [[True] * n_targets] * (batch_size - 2)
        ),
    )


def test_p1_forward_uses_target_branch_and_shared_gene_embedding() -> None:
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
        shared_pathway=True,
        shared_pathway_proj=True,
        use_cls_proj=False,
        target_branch=True,
        target_gene_layers=1,
        readout="sum",
    )
    assert model.drug.targets is not None
    assert model.target is not None
    assert model.drug.targets.weight is model.target.gene_embedding.weight
    assert model.target_gate is not None
    assert float(model.target_gate.detach()) == 0.0

    output = model(_p1_batch())
    assert output.y_hat.shape == (4,)
    assert output.z_pair.shape == (4, 32)
    assert output.aux["omics_mask"].shape[0] == 4
    assert torch.isfinite(output.y_hat).all()
    assert count_parameters(model) < 5_000_000


def test_p1_all_targets_missing_is_finite() -> None:
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
        shared_pathway=True,
        use_cls_proj=False,
        target_branch=True,
        target_gene_layers=0,
        readout="sum",
    )
    batch = _p1_batch(batch_size=2, pathway_dim=8, n_targets=4)
    batch.target_gene_mask = torch.zeros(2, 4, dtype=torch.bool)
    output = model(batch)
    assert torch.isfinite(output.y_hat).all()


def test_p1_full_size_parameter_budget() -> None:
    model = T0Teacher(
        pathway_dim=1000,
        morgan_dim=2048,
        n_target_genes=18_396,
        d_model=256,
        hidden=256,
        target_embed_dim=16,
        n_heads=4,
        n_layers=2,
        top_k=64,
        token_types=True,
        rank_embedding=True,
        cell_readout="attention",
        masked_modalities=True,
        cross_gate=True,
        shared_pathway=False,
        use_cls_proj=True,
        cls_proj_mode="lowrank",
        target_branch=True,
        target_gene_layers=1,
        readout="sum",
    )
    params = count_parameters(model)
    assert params < 5_000_000
    # target branch + 低秩 cls_proj 后仍应落在 T0 v1 的 4.73M 附近。
    assert params < 4_950_000


def test_build_pair_tensors_target_gene_table() -> None:
    n_cells, n_genes, n_drugs = 3, 4, 2
    features = CellFeatureSet(
        cell_ids=["c0", "c1", "c2"],
        ge=np.zeros((n_cells, 2), dtype=np.float32),
        mu=np.zeros((n_cells, 2), dtype=np.float32),
        cnv=np.zeros((n_cells, 2), dtype=np.float32),
        cnv_mask=np.ones(n_cells, dtype=bool),
        ge_mask=np.ones(n_cells, dtype=bool),
        mu_mask=np.ones(n_cells, dtype=bool),
        gene_ids=["G0", "G1", "G2", "G3"],
        gene_ge=np.arange(n_cells * n_genes, dtype=np.float32).reshape(n_cells, n_genes),
        gene_mu=np.ones((n_cells, n_genes), dtype=np.float32),
        gene_cnv=np.zeros((n_cells, n_genes), dtype=np.float32),
        gene_ge_valid=np.ones(n_genes, dtype=bool),
        gene_cnv_valid=np.ones(n_genes, dtype=bool),
        gene_mu_available=np.asarray([True, False, True, True]),
        index={"c0": 0, "c1": 1, "c2": 2},
    )
    drugs = DrugFeatureSet(
        compound_uid=["d0", "d1"],
        morgan=np.zeros((n_drugs, 8), dtype=np.float32),
        target_indices=[
            np.asarray([0, 1], dtype=np.int64),
            np.asarray([2, 3], dtype=np.int64),
        ],
        target_dim=n_genes,
        target_genes=["G0", "G1", "G2", "G3"],
        graph_atom_ptr=None,
        graph_edge_ptr=None,
        graph_node_feat=None,
        graph_edge_index=None,
        graph_edge_feat=None,
        graph_has_graph=None,
        index={"d0": 0, "d1": 1},
    )
    tensors = build_pair_tensors(features, drugs, target_max_tokens=2)
    assert tensors.target_ids is not None
    assert tensors.target_mask is not None
    assert tensors.target_ids.shape == (2, 2)
    assert tensors.target_mask.tolist() == [[True, True], [True, True]]
    assert tensors.target_ids.tolist() == [[0, 1], [2, 3]]

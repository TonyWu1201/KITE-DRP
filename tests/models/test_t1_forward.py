"""T1 模型前向与“初始化等价 P1”单元测试（纯合成张量，不读数据）。"""

from __future__ import annotations

import copy

import pytest
import torch

pytest.importorskip("torch_geometric")

from kite_drp.models.teacher.common import TeacherBatch
from kite_drp.models.teacher.t0 import T0Teacher
from kite_drp.models.teacher.t1 import T1Teacher


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
        target_branch=True,
        target_gene_layers=1,
        readout="sum",
        use_neighbor=False,
        use_gene_table=True,
        use_target_table=True,
    )


def _batch() -> TeacherBatch:
    generator = torch.Generator().manual_seed(0)
    return TeacherBatch(
        ge=torch.randn(3, 6, generator=generator),
        mu=torch.rand(3, 6, generator=generator),
        cnv=torch.randn(3, 6, generator=generator),
        cnv_mask=torch.ones(3, dtype=torch.bool),
        morgan=torch.rand(3, 8, generator=generator),
        # 全基因空间；只含图内基因 (0/2/4/6/8)，与 G0 的 5 行一一对应。
        target_index=torch.tensor([0, 2, 4, 6, 8, 0], dtype=torch.long),
        target_offset=torch.tensor([0, 2, 4], dtype=torch.long),
        y=torch.randn(3, generator=generator),
        y_cls=torch.tensor([0.0, 1.0, 0.0]),
        cls_mask=torch.ones(3, dtype=torch.bool),
        row_index=torch.arange(3),
        label_mask=torch.ones(3, dtype=torch.bool),
        drug_idx=torch.tensor([0, 1, 2]),
        cell_idx=torch.tensor([0, 1, 2]),
        # 只取图内基因（gene_full_idx=0/2/4/6/8），否则 G_full 的零行与
        # P1 随机表不一致，等价性测试会误报。
        target_gene_ids=torch.tensor([[0, 2, 4, 6], [8, 0, 2, 4], [6, 8, 0, 2]], dtype=torch.long),
        target_gene_values=torch.randn(3, 4, 6, generator=generator),
        target_gene_mask=torch.ones(3, 4, dtype=torch.bool),
    )


def _graph() -> dict[str, torch.Tensor | list[str]]:
    return {
        "gene_full_idx": torch.tensor([0, 2, 4, 6, 8]),
        "gene_stats": torch.randn(5, 4),
        "cell_features": torch.randn(3, 7),
        "drug_morgan": torch.rand(4, 8),
        # 局部图基因空间（0..4），与 batch.target_index 的全基因空间对应。
        "drug_target_ids": torch.tensor([[0, 1], [2, 3], [4, 0], [0, 0]]),
        "drug_target_mask": torch.tensor([[True, True], [True, True], [True, True], [True, True]]),
        "drug_target_ids_full": torch.tensor([[0, 1], [2, 3], [4, 0], [0, 0]]),
        "drug_target_mask_full": torch.tensor(
            [[True, True], [True, True], [True, True], [True, True]]
        ),
        "ppi_edge_index": torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),
        "ppi_edge_attr": torch.ones(4, 1),
        "dti_edge_index": torch.tensor([[0, 1, 2], [0, 1, 2]]),
        "dti_edge_attr": torch.ones(3, 1),
        "cg_edge_index": torch.tensor([[0, 1, 2, 3], [0, 1, 2, 0]]),
        "cg_edge_attr": torch.ones(4, 4),
    }


def _model(backbone: T0Teacher) -> T1Teacher:
    return T1Teacher(
        backbone,
        n_full_genes=10,
        n_graph_genes=5,
        cell_in_dim=7,
        gene_emb_dim=4,
        graph_hidden=8,
        graph_layers=1,
        graph_heads=2,
        dropout=0.0,
    )


def test_t1_forward_is_finite() -> None:
    model = _model(_backbone())
    model.set_graph(_graph())
    output = model(_batch())
    assert output.y_hat.shape == (3,)
    assert output.z_pair.shape == (3, 16)
    assert output.aux["y_graph"].shape == (3,)
    assert torch.isfinite(output.y_hat).all()


def test_t1_requires_graph() -> None:
    model = _model(_backbone())
    with pytest.raises(RuntimeError, match="set_graph"):
        model(_batch())


def test_t1_init_equals_p1() -> None:
    torch.manual_seed(0)
    p1 = _backbone().eval()
    t1 = _model(copy.deepcopy(p1)).eval()
    graph = _graph()
    t1.set_graph(graph)
    batch = _batch()
    with torch.no_grad():
        t1.gene_embedding.weight.copy_(p1.target.gene_embedding.weight[graph["gene_full_idx"]])
        reference = p1(batch)
        output = t1(batch)
    assert torch.allclose(output.y_hat, reference.y_hat, atol=1e-5)
    assert float(torch.tanh(t1.cell_gate).detach()) == 0.0
    assert float(torch.tanh(t1.drug_gate).detach()) == 0.0


def test_t1_gates_receive_gradients() -> None:
    model = _model(_backbone())
    model.set_graph(_graph())
    output = model(_batch())
    loss = output.y_hat.sum() + output.aux["y_graph"].sum()
    loss.backward()
    for name in ("gene_gate", "cell_gate", "drug_gate", "graph_pred_gate"):
        grad = getattr(model, name).grad
        assert grad is not None
        assert float(grad.abs()) > 0.0


def test_t1_graph_only_forward() -> None:
    backbone = _backbone()
    model = T1Teacher(
        backbone,
        n_full_genes=10,
        n_graph_genes=5,
        cell_in_dim=7,
        gene_emb_dim=4,
        graph_hidden=8,
        graph_layers=1,
        graph_heads=2,
        dropout=0.0,
        use_backbone=False,
    )
    model.set_graph(_graph())
    output = model(_batch())
    assert output.y_hat.shape == (3,)
    assert torch.isfinite(output.y_hat).all()

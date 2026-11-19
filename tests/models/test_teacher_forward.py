"""教师模型前向/参数预算测试（无 torch 环境自动跳过）。"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from kite_drp.models.teacher.b0 import BilinearTeacher  # noqa: E402
from kite_drp.models.teacher.common import TeacherBatch, count_parameters  # noqa: E402
from kite_drp.models.teacher.t0 import T0Teacher  # noqa: E402


def _batch(batch_size: int = 4, pathway_dim: int = 64, n_targets: int = 50) -> TeacherBatch:
    return TeacherBatch(
        ge=torch.randn(batch_size, pathway_dim),
        mu=torch.randn(batch_size, pathway_dim),
        cnv=torch.randn(batch_size, pathway_dim),
        cnv_mask=torch.tensor([True, False, True, True][:batch_size]),
        morgan=torch.rand(batch_size, 128).round(),
        target_index=torch.tensor([1, 2, 3, 4], dtype=torch.long),
        target_offset=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        y=torch.randn(batch_size),
        y_cls=torch.tensor([0.0, 1.0, 0.0, 1.0][:batch_size]),
        cls_mask=torch.ones(batch_size, dtype=torch.bool),
        neighbor_ge=torch.randn(batch_size, pathway_dim),
    )


@pytest.mark.parametrize("model_cls", [BilinearTeacher, T0Teacher])
def test_forward_shapes(model_cls) -> None:
    model = model_cls(
        pathway_dim=64,
        morgan_dim=128,
        n_target_genes=50,
        d_model=32,
        hidden=32,
        dropout=0.0,
        target_embed_dim=8,
        **({"n_heads": 4, "n_layers": 1, "top_k": 8} if model_cls is T0Teacher else {}),
    )
    output = model(_batch())
    assert output.y_hat.shape == (4,)
    assert output.logits.shape == (4,)
    assert output.z_pair.shape == (4, 32)
    assert torch.isfinite(output.y_hat).all()


def test_missing_cnv_is_handled() -> None:
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
    )
    batch = _batch()
    batch.cnv = torch.zeros_like(batch.cnv)
    output = model(batch)
    assert torch.isfinite(output.y_hat).all()


def test_parameter_budget() -> None:
    b0 = BilinearTeacher(pathway_dim=1000, morgan_dim=2048, n_target_genes=18396)
    t0 = T0Teacher(pathway_dim=1000, morgan_dim=2048, n_target_genes=18396)
    assert count_parameters(b0) < 3_000_000
    assert count_parameters(t0) < 5_000_000


@pytest.mark.parametrize("model_cls", [BilinearTeacher, T0Teacher])
def test_modality_subset_reduces_parameters(model_cls) -> None:
    """模态消融要真正去掉分支参数，而不只是把输入置零。"""
    kwargs = {
        "pathway_dim": 64,
        "morgan_dim": 128,
        "n_target_genes": 50,
        "d_model": 32,
        "hidden": 32,
        "dropout": 0.0,
        "target_embed_dim": 8,
    }
    if model_cls is T0Teacher:
        kwargs.update(use_neighbor=False, n_heads=4, n_layers=1, top_k=8)
    extra: dict = {}
    full = model_cls(**kwargs, modalities=["GE", "MU", "CNV"], **extra)
    ge_only = model_cls(**kwargs, modalities=["GE"], **extra)
    assert count_parameters(ge_only) < count_parameters(full)
    output = ge_only(_batch())
    assert output.y_hat.shape == (4,)
    assert torch.isfinite(output.y_hat).all()


def test_drug_input_ablation() -> None:
    common = {
        "pathway_dim": 64,
        "morgan_dim": 128,
        "n_target_genes": 50,
        "d_model": 32,
        "hidden": 32,
        "dropout": 0.0,
        "target_embed_dim": 8,
        "n_heads": 4,
        "n_layers": 1,
        "top_k": 8,
        "use_neighbor": False,
    }
    batch = _batch()
    for use_morgan, use_targets in ((True, False), (False, True)):
        model = T0Teacher(**common, use_morgan=use_morgan, use_targets=use_targets)
        output = model(batch)
        assert output.y_hat.shape == (4,)
        assert torch.isfinite(output.y_hat).all()

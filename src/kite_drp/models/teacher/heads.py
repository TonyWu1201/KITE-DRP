"""教师输出头：回归（-z LN_IC50）+ 分类（敏感/耐药）。"""

from __future__ import annotations

from torch import Tensor, nn

from kite_drp.models.teacher.common import MLP


class ResponseHeads(nn.Module):
    def __init__(self, d_model: int, *, hidden: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.regression = MLP(d_model, [hidden], 1, dropout=dropout)
        self.classification = MLP(d_model, [hidden], 1, dropout=dropout)

    def forward(self, z_pair: Tensor) -> tuple[Tensor, Tensor]:
        return self.regression(z_pair).squeeze(-1), self.classification(z_pair).squeeze(-1)


__all__ = ["ResponseHeads"]

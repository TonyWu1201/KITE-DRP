"""T1-local：冻结 P1 + 每药一跳局部图残差。

与旧 T1 的区别：

- 不再建一张跨样本共享的 drug-cell-gene 异构图；每个药物有自己的
  靶点/邻居节点集。
- 当前细胞的 GE/MU/CNV 在消息传递**之前**写入节点状态，因此同一邻接
  在不同细胞上产生不同消息。
- P1 完全冻结；局部图只输出标量残差，初值为 0，无覆盖时精确回退 P1。

首轮只实现一跳、等权均值、共享投影；两跳/边权/联合微调留到一跳对照
有信号后再评估。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from kite_drp.models.teacher.common import TeacherBatch, TeacherOutput
from kite_drp.models.teacher.t0 import T0Teacher


def _masked_pool(
    values: Tensor,
    pair_index: Tensor,
    mask: Tensor,
    *,
    n_pairs: int,
) -> tuple[Tensor, Tensor]:
    """按 pair 对节点做 masked mean；返回池化向量和每 pair 的计数。"""

    weights = mask.to(values.dtype).unsqueeze(-1)
    total = torch.zeros((n_pairs, values.shape[-1]), dtype=values.dtype, device=values.device)
    total = total.index_add_(0, pair_index, values * weights)
    count = torch.zeros(n_pairs, dtype=values.dtype, device=values.device)
    count = count.index_add_(0, pair_index, mask.to(values.dtype))
    return total / count.clamp_min(1.0).unsqueeze(-1), count


def _neighbor_mean(values: Tensor, edge_index: Tensor) -> Tensor:
    """无向边上的等权邻居均值；孤立节点以自身状态作为消息。"""

    if edge_index.numel() == 0:
        return values
    source, target = edge_index[0], edge_index[1]
    total = torch.zeros_like(values)
    total = total.index_add_(0, target, values[source])
    degree = torch.zeros(values.shape[0], dtype=values.dtype, device=values.device)
    degree = degree.index_add_(0, target, torch.ones_like(target, dtype=values.dtype))
    mean = total / degree.clamp_min(1.0).unsqueeze(-1)
    return torch.where(degree.unsqueeze(-1) > 0, mean, values)


class T1LocalTeacher(nn.Module):
    """P1 backbone（冻结）+ 局部图残差分支。"""

    def __init__(
        self,
        backbone: T0Teacher,
        *,
        hidden: int = 64,
        readout_hidden: int = 128,
        condition_on: str = "pair",  # pair | cell | none
        propagation: str = "trainable",  # trainable | fixed | none
        dropout: float = 0.1,
        residual_shrink: float = 1.0,
    ) -> None:
        super().__init__()
        if condition_on not in {"pair", "cell", "none"}:
            raise ValueError(f"condition_on 必须是 pair/cell/none，收到 {condition_on}")
        if propagation not in {"trainable", "fixed", "none", "set"}:
            raise ValueError(f"propagation 必须是 trainable/fixed/none/set，收到 {propagation}")
        self.backbone = backbone
        self.d_model = int(backbone.d_model)
        self.hidden = int(hidden)
        self.condition_on = str(condition_on)
        self.propagation = str(propagation)
        self.residual_shrink = float(residual_shrink)
        self.dropout = nn.Dropout(dropout)
        # 6 维：GE/MU/CNV 值 + 三个 observed flag；根标记单独传入。
        self.node_in = nn.Sequential(
            nn.Linear(6, self.hidden),
            nn.GELU(),
            nn.LayerNorm(self.hidden),
        )
        self.message = nn.Linear(self.hidden, self.hidden)
        self.norm = nn.LayerNorm(self.hidden)
        cond_dim = {
            "pair": self.d_model,
            "cell": self.d_model,
            "none": 0,
        }[self.condition_on]
        readout_in = self.hidden * 2 + 6 + cond_dim
        self.readout = nn.Sequential(
            nn.Linear(readout_in, int(readout_hidden)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(readout_hidden), 1),
        )
        # 首次参数更新前严格等于 P1；图层必须自己“挣出”残差。
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)
        self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True) -> T1LocalTeacher:
        """外层切 train 时仍保持 P1 eval/no-grad；否则 dropout 会改变基线。"""

        super().train(mode)
        self.backbone.eval()
        return self

    def set_local_graph(self, tensors: dict[str, Tensor]) -> None:
        for key, value in tensors.items():
            setattr(self, key, value)

    @torch.no_grad()
    def _p1_forward(self, batch: TeacherBatch) -> TeacherOutput:
        self.backbone.eval()
        return self.backbone(batch)

    def _readout_input(
        self,
        pooled_root: Tensor,
        pooled_nonroot: Tensor,
        markers: Tensor,
        p1: TeacherOutput,
    ) -> Tensor:
        if self.condition_on == "pair":
            condition = p1.z_pair
        elif self.condition_on == "cell":
            condition = p1.z_cell
        else:
            condition = pooled_root.new_zeros((pooled_root.shape[0], 0))
        return torch.cat(
            [pooled_root, pooled_nonroot, markers, condition.to(pooled_root.dtype)], dim=-1
        )

    def forward(self, batch: TeacherBatch) -> TeacherOutput:
        if batch.local_node_values is None or batch.local_node_pair is None:
            raise ValueError("T1LocalTeacher 需要 batch.local_* 字段；检查 make_loader")
        p1 = self._p1_forward(batch)
        n_pairs = int(batch.y.shape[0])
        node_values = batch.local_node_values
        node_root = batch.local_node_root
        pair_index = batch.local_node_pair
        edge_index = batch.local_edge_index
        assert node_root is not None and edge_index is not None

        with torch.autocast(device_type=node_values.device.type, enabled=False):
            h0 = self.node_in(node_values.float())
            if self.propagation == "set":
                h1 = h0
            elif self.propagation == "none":
                message = h0
                h1 = self.norm(h0 + F.gelu(self.message(message)))
            elif self.propagation == "fixed":
                message = _neighbor_mean(h0, edge_index)
                h1 = self.norm(h0 + message)
            else:
                message = _neighbor_mean(h0, edge_index)
                h1 = self.norm(h0 + F.gelu(self.message(message)))

            pooled_root, n_roots = _masked_pool(h1, pair_index, node_root, n_pairs=n_pairs)
            pooled_nonroot, n_nonroot = _masked_pool(h1, pair_index, ~node_root, n_pairs=n_pairs)
            n_nodes = (n_roots + n_nonroot).clamp_min(1.0)
            coverage = batch.local_graph_coverage
            observed = batch.local_graph_observed
            has_neighbor = batch.local_graph_has_neighbor
            assert coverage is not None and observed is not None and has_neighbor is not None
            markers = torch.stack(
                [
                    (n_roots > 0).to(h1.dtype),
                    (n_nonroot > 0).to(h1.dtype),
                    n_roots / n_nodes,
                    n_nonroot / n_nodes,
                    has_neighbor.to(h1.dtype),
                    observed.to(h1.dtype),
                ],
                dim=-1,
            )
            readout_in = self._readout_input(pooled_root, pooled_nonroot, markers, p1)
            raw_residual = self.readout(self.dropout(readout_in)).squeeze(-1)
            gate = coverage.to(raw_residual.dtype) * observed.to(raw_residual.dtype)
            delta = self.residual_shrink * gate * raw_residual
        y_hat = p1.y_hat + delta.to(p1.y_hat.dtype)
        aux = dict(p1.aux)
        aux.update(
            {
                "y_p1": p1.y_hat,
                "local_residual": raw_residual,
                "local_delta": delta,
                "local_gate": gate,
                "local_nodes": n_nodes,
                "local_roots": n_roots,
                "local_has_neighbor": has_neighbor,
                "local_coverage": coverage,
                "local_observed": observed,
            }
        )
        return TeacherOutput(
            y_hat=y_hat,
            logits=p1.logits,
            z_pair=p1.z_pair,
            z_cell=p1.z_cell,
            z_drug=p1.z_drug,
            aux=aux,
        )

    def load_p1_checkpoint(self, state: dict[str, Tensor]) -> None:
        self.backbone.load_state_dict(state, strict=False)


class T1MlpHeadTeacher(nn.Module):
    """容量对照：冻结 P1 + 只读 ``z_pair`` 的残差头。

    用于回答“局部图分支的增益是否只是多了一个预测头/更多参数”。实际
    参数量由配置决定并在 ``model_spec.json`` 里记录。
    """

    def __init__(
        self,
        backbone: T0Teacher,
        *,
        hidden: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.d_model = int(backbone.d_model)
        self.head = nn.Sequential(
            nn.Linear(self.d_model, int(hidden)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden), 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()

    def set_local_graph(self, tensors: dict[str, Tensor]) -> None:
        """容量对照不需要图节点；保留接口以复用训练入口。"""

        self._local_graph_tensors = tensors

    def train(self, mode: bool = True) -> T1MlpHeadTeacher:
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def _p1_forward(self, batch: TeacherBatch) -> TeacherOutput:
        self.backbone.eval()
        return self.backbone(batch)

    def forward(self, batch: TeacherBatch) -> TeacherOutput:
        p1 = self._p1_forward(batch)
        coverage = batch.local_graph_coverage
        observed = batch.local_graph_observed
        if coverage is None or observed is None:
            raise ValueError("T1MlpHeadTeacher 需要 local_graph_coverage/observed")
        with torch.autocast(device_type=p1.z_pair.device.type, enabled=False):
            raw = self.head(p1.z_pair.float()).squeeze(-1)
            gate = coverage.to(raw.dtype) * observed.to(raw.dtype)
            delta = gate * raw
        y_hat = p1.y_hat + delta.to(p1.y_hat.dtype)
        aux = dict(p1.aux)
        aux.update(
            {
                "y_p1": p1.y_hat,
                "local_residual": raw,
                "local_delta": delta,
                "local_gate": gate,
                "local_coverage": coverage,
                "local_observed": observed,
            }
        )
        return TeacherOutput(
            y_hat=y_hat,
            logits=p1.logits,
            z_pair=p1.z_pair,
            z_cell=p1.z_cell,
            z_drug=p1.z_drug,
            aux=aux,
        )

    def load_p1_checkpoint(self, state: dict[str, Tensor]) -> None:
        self.backbone.load_state_dict(state, strict=False)


__all__ = ["T1LocalTeacher", "T1MlpHeadTeacher"]

"""训练损失：Huber/BCE 之外的排序目标。

主指标是 Spearman/SCC，而训练默认只优化点误差。这里实现同药物内的
pairwise RankNet 代理损失：同一药物下，若 ``y_i > y_j``，则要求
``s_i > s_j``。实际训练时配合 drug-balanced batch sampler 使用，
保证每个 batch 内有足够的同药物细胞对。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def pairwise_rank_loss(
    scores: Tensor,
    labels: Tensor,
    groups: Tensor,
    *,
    mask: Tensor | None = None,
    temperature: float = 1.0,
    min_delta: float = 0.1,
    max_pairs_per_group: int = 1024,
) -> Tensor:
    """同组内 pairwise RankNet。

    Args:
        scores: 模型回归输出，形状 ``(N,)``。
        labels: 连续标签，形状 ``(N,)``；标签越大表示越敏感。
        groups: 分组索引（这里是药物索引），形状 ``(N,)``。
        mask: 可选有效样本掩码，形状 ``(N,)``；``False`` 的样本不参与。
        temperature: 排序温度，越小越强调难 pair。
        min_delta: 标签差小于该值的 pair 视为并列，跳过。
        max_pairs_per_group: 每组最多采样的 pair 数，控制 O(M²) 开销。

    Returns:
        标量损失；若没有有效 pair，返回与 ``scores`` 相连的 0。
    """
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)
    groups = groups.reshape(-1)
    if mask is None:
        mask = torch.ones_like(labels, dtype=torch.bool)
    else:
        mask = mask.reshape(-1).to(torch.bool)
    if scores.numel() == 0:
        return scores.sum() * 0.0

    total: Tensor = scores.new_zeros(())
    count: Tensor = scores.new_zeros(())
    for group in torch.unique(groups):
        members = torch.nonzero((groups == group) & mask, as_tuple=False).flatten()
        if members.numel() < 2:
            continue
        group_scores = scores[members]
        group_labels = labels[members]
        left, right = torch.triu_indices(
            members.numel(), members.numel(), offset=1, device=scores.device
        )
        label_diff = group_labels[left] - group_labels[right]
        valid = label_diff.abs() > float(min_delta)
        if not bool(valid.any()):
            continue
        left = left[valid]
        right = right[valid]
        label_diff = label_diff[valid]
        if left.numel() > int(max_pairs_per_group):
            keep = torch.randperm(left.numel(), device=scores.device)[: int(max_pairs_per_group)]
            left = left[keep]
            right = right[keep]
            label_diff = label_diff[keep]

        direction = torch.sign(label_diff)
        ordered = (group_scores[left] - group_scores[right]) * direction
        pair_loss = F.softplus(-ordered / float(temperature))
        weight = label_diff.abs()
        total = total + (pair_loss * weight).sum()
        count = count + weight.sum()

    if float(count) == 0.0:
        return scores.sum() * 0.0
    return total / count.clamp_min(1e-8)


__all__ = ["pairwise_rank_loss"]

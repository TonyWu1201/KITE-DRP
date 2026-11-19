"""教师训练 DataLoader 与 batch 拼装（纯张量，不含 PyG）。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from kite_drp.data.dataset import TeacherPairDataset
from kite_drp.models.teacher.common import TeacherBatch
from kite_drp.train.features import (
    CellFeatureSet,
    DrugFeatureSet,
    neighbor_scores,
)


@dataclass
class PairTensors:
    cell_ge: np.ndarray
    cell_mu: np.ndarray
    cell_cnv: np.ndarray
    cell_cnv_mask: np.ndarray
    cell_ge_mask: np.ndarray
    cell_mu_mask: np.ndarray
    neighbor_ge: np.ndarray | None
    neighbor_mask: np.ndarray | None
    drug_morgan: np.ndarray
    # P1 target-aware gene branch：逐基因细胞特征 + 每药物的靶点基因表。
    gene_ge: np.ndarray | None = None
    gene_mu: np.ndarray | None = None
    gene_cnv: np.ndarray | None = None
    gene_ge_valid: np.ndarray | None = None
    gene_cnv_valid: np.ndarray | None = None
    gene_mu_available: np.ndarray | None = None
    target_ids: np.ndarray | None = None  # (n_drugs, T) gene index，右侧 padding 0
    target_mask: np.ndarray | None = None  # (n_drugs, T) bool
    target_modalities: tuple[str, ...] = ("GE", "MU", "CNV")


def _build_target_gene_table(
    features: CellFeatureSet,
    drugs: DrugFeatureSet,
    *,
    max_tokens: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """把每个药物的靶点基因映射到细胞基因顺序，并截断到 ``max_tokens``。"""
    if max_tokens <= 0:
        return None, None
    if drugs.target_genes != features.gene_ids:
        raise ValueError(
            "P1 target branch 要求 drug target_genes 与 cell gene_ids 顺序一致；"
            "当前 processed 产物不满足，需要先重映射药物靶点索引。"
        )
    lookup = {gene: position for position, gene in enumerate(features.gene_ids)}
    target_to_gene = np.asarray(
        [lookup.get(gene, -1) for gene in drugs.target_genes], dtype=np.int64
    )
    positions: list[np.ndarray] = []
    for indices in drugs.target_indices:
        mapped = target_to_gene[np.asarray(indices, dtype=np.int64)]
        positions.append(mapped[mapped >= 0])
    if not any(item.size for item in positions):
        return None, None
    width = min(max(int(item.size) for item in positions), int(max_tokens))
    if width <= 0:
        return None, None
    ids = np.zeros((len(positions), width), dtype=np.int64)
    mask = np.zeros((len(positions), width), dtype=bool)
    # 固定种子：仅对靶点过多的药物做一次可复现的随机截断；
    # 99% 药物靶点数 <= 106，默认 64 只会影响极少数多靶点药物。
    rng = np.random.default_rng(20261119)
    for row, item in enumerate(positions):
        if item.size > width:
            item = np.sort(rng.choice(item, size=width, replace=False))
        ids[row, : item.size] = item
        mask[row, : item.size] = True
    return ids, mask


def build_pair_tensors(
    features: CellFeatureSet,
    drugs: DrugFeatureSet,
    *,
    neighbors: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    neighbor_source: str = "GE",
    target_max_tokens: int = 64,
    target_modalities: list[str] | tuple[str, ...] | None = None,
) -> PairTensors:
    if neighbors is not None and weights is not None:
        source = {
            "GE": features.ge,
            "MU": features.mu,
            "CNV": features.cnv,
        }[neighbor_source]
        neighbor_ge = neighbor_scores(source, neighbors, weights)
        # 归一化权重本身不含绝对距离；max 权重可作为“最近邻有多近”的廉价置信度。
        neighbor_mask = np.isfinite(neighbor_ge).all(axis=1) & (weights.max(axis=1) > 0)
    else:
        neighbor_ge = None
        neighbor_mask = None
    target_ids, target_mask = _build_target_gene_table(
        features, drugs, max_tokens=target_max_tokens
    )
    target_names = tuple(target_modalities or ("GE", "MU", "CNV"))
    unknown = set(target_names) - {"GE", "MU", "CNV"}
    if unknown:
        raise KeyError(f"未知 target_modalities：{sorted(unknown)}")
    return PairTensors(
        cell_ge=features.ge,
        cell_mu=features.mu,
        cell_cnv=features.cnv,
        cell_cnv_mask=features.cnv_mask.astype(bool),
        cell_ge_mask=features.ge_mask.astype(bool),
        cell_mu_mask=features.mu_mask.astype(bool),
        neighbor_ge=neighbor_ge,
        neighbor_mask=neighbor_mask,
        drug_morgan=drugs.morgan,
        gene_ge=features.gene_ge,
        gene_mu=features.gene_mu,
        gene_cnv=features.gene_cnv,
        gene_ge_valid=features.gene_ge_valid,
        gene_cnv_valid=features.gene_cnv_valid,
        gene_mu_available=features.gene_mu_available,
        target_ids=target_ids,
        target_mask=target_mask,
        target_modalities=target_names,
    )


class PairDataset(Dataset):
    """把 ``TeacherPairDataset`` 的 index/label 包装成 torch Dataset（返回索引）。"""

    def __init__(self, base: TeacherPairDataset, split: str) -> None:
        self.base = base
        self.indices = base.indices(split)
        if self.indices.size == 0:
            raise ValueError(f"split={split} 没有样本")

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, position: int) -> int:
        return int(self.indices[position])


class DrugGroupedBatchSampler(Sampler[list[int]]):
    """按药物分组的 batch sampler，保证 batch 内有同药物细胞对。

    排序损失只对同药物内样本对生效；随机 pair batch 中同药物碰撞概率约为
    1/n_drugs，rank 信号会被稀释。该 sampler 每个 batch 取 ``drugs_per_batch``
    个药物，每个药物再取至多 ``cells_per_drug`` 个训练 pair。
    """

    def __init__(
        self,
        dataset: PairDataset,
        *,
        drugs_per_batch: int = 8,
        cells_per_drug: int = 64,
        seed: int = 0,
    ) -> None:
        if drugs_per_batch <= 0 or cells_per_drug <= 0:
            raise ValueError("drugs_per_batch / cells_per_drug 必须为正")
        self.dataset = dataset
        self.drugs_per_batch = int(drugs_per_batch)
        self.cells_per_drug = int(cells_per_drug)
        self.seed = int(seed)
        self.epoch = 0
        drug_idx = dataset.base.drug_idx[dataset.indices]
        self.groups: list[np.ndarray] = []
        for drug in np.unique(drug_idx):
            positions = np.flatnonzero(drug_idx == drug).astype(np.int64)
            if positions.size:
                self.groups.append(positions)
        if not self.groups:
            raise ValueError("DrugGroupedBatchSampler 没有可用药物")
        self._chunks_per_epoch = sum(
            math.ceil(positions.size / self.cells_per_drug) for positions in self.groups
        )

    def __len__(self) -> int:
        return math.ceil(self._chunks_per_epoch / self.drugs_per_batch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        # 每个药物拆成多个 chunk；一个 epoch 覆盖全部训练 pair，避免
        # “每药只取一个 chunk”导致排序实验实际看到的样本数只有 v1 的 1/3。
        chunks: list[np.ndarray] = []
        for positions in self.groups:
            shuffled = rng.permutation(positions)
            for start in range(0, shuffled.size, self.cells_per_drug):
                chunks.append(shuffled[start : start + self.cells_per_drug])
        order = rng.permutation(len(chunks))
        for start in range(0, len(order), self.drugs_per_batch):
            batch: list[int] = []
            for chunk_index in order[start : start + self.drugs_per_batch]:
                batch.extend(int(position) for position in chunks[int(chunk_index)])
            if batch:
                yield batch


def _graph_batch(drug_indices: np.ndarray, drugs: DrugFeatureSet) -> dict[str, torch.Tensor | int]:
    if (
        drugs.graph_atom_ptr is None
        or drugs.graph_edge_ptr is None
        or drugs.graph_node_feat is None
        or drugs.graph_edge_index is None
    ):
        return {"num_graphs": 0}
    node_parts: list[np.ndarray] = []
    edge_parts: list[np.ndarray] = []
    edge_feat_parts: list[np.ndarray] = []
    batch_parts: list[np.ndarray] = []
    atom_offset = 0
    for graph_id, drug in enumerate(drug_indices):
        atom_start, atom_end = int(drugs.graph_atom_ptr[drug]), int(drugs.graph_atom_ptr[drug + 1])
        edge_start, edge_end = int(drugs.graph_edge_ptr[drug]), int(drugs.graph_edge_ptr[drug + 1])
        if atom_end <= atom_start:
            continue
        node_parts.append(drugs.graph_node_feat[atom_start:atom_end])
        if edge_end > edge_start:
            edges = drugs.graph_edge_index[:, edge_start:edge_end].astype(np.int64)
            edges = edges - atom_start + atom_offset
            edge_parts.append(edges)
            if drugs.graph_edge_feat is not None:
                edge_feat_parts.append(drugs.graph_edge_feat[edge_start:edge_end])
        batch_parts.append(np.full(atom_end - atom_start, graph_id, dtype=np.int64))
        atom_offset += atom_end - atom_start
    if not node_parts:
        return {"num_graphs": 0}
    return {
        "node_feat": torch.from_numpy(np.concatenate(node_parts, axis=0).astype(np.float32)),
        "edge_index": torch.from_numpy(
            np.concatenate(edge_parts, axis=1) if edge_parts else np.zeros((2, 0), dtype=np.int64)
        ),
        "edge_feat": torch.from_numpy(
            np.concatenate(edge_feat_parts, axis=0).astype(np.float32)
            if edge_feat_parts
            else np.zeros((0, 4), dtype=np.float32)
        ),
        "batch": torch.from_numpy(np.concatenate(batch_parts, axis=0)),
        "num_graphs": len(drug_indices),
    }


def collate_pairs(
    indices: list[int],
    *,
    base: TeacherPairDataset,
    tensors: PairTensors,
    drugs: DrugFeatureSet,
    use_gin: bool = False,
) -> TeacherBatch:
    rows = np.asarray(indices, dtype=np.int64)
    cell_idx = base.cell_idx[rows]
    drug_idx = base.drug_idx[rows]

    target_index: list[np.ndarray] = []
    offsets = [0]
    for drug in drug_idx:
        values = drugs.target_indices[int(drug)]
        target_index.append(values)
        offsets.append(offsets[-1] + int(values.size))
    flattened = (
        np.concatenate(target_index).astype(np.int64)
        if target_index
        else np.zeros(0, dtype=np.int64)
    )

    batch = TeacherBatch(
        ge=torch.from_numpy(tensors.cell_ge[cell_idx]),
        mu=torch.from_numpy(tensors.cell_mu[cell_idx]),
        cnv=torch.from_numpy(tensors.cell_cnv[cell_idx]),
        cnv_mask=torch.from_numpy(tensors.cell_cnv_mask[cell_idx]),
        morgan=torch.from_numpy(tensors.drug_morgan[drug_idx]),
        target_index=torch.from_numpy(flattened),
        target_offset=torch.as_tensor(offsets[:-1], dtype=torch.long),
        y=torch.from_numpy(base.y[rows]),
        y_cls=torch.from_numpy(base.y_cls[rows]),
        cls_mask=torch.from_numpy(base.cls_mask[rows]),
        label_mask=torch.from_numpy(base.label_mask[rows]),
        row_index=torch.from_numpy(rows),
        drug_idx=torch.from_numpy(drug_idx),
        ge_mask=torch.from_numpy(tensors.cell_ge_mask[cell_idx]),
        mu_mask=torch.from_numpy(tensors.cell_mu_mask[cell_idx]),
        neighbor_ge=(
            torch.from_numpy(tensors.neighbor_ge[cell_idx])
            if tensors.neighbor_ge is not None
            else None
        ),
        neighbor_mask=(
            torch.from_numpy(tensors.neighbor_mask[cell_idx])
            if tensors.neighbor_mask is not None
            else None
        ),
    )
    if tensors.target_ids is not None and tensors.target_mask is not None:
        target_ids = tensors.target_ids[drug_idx]
        target_mask = tensors.target_mask[drug_idx]
        names = tensors.target_modalities
        values = np.zeros((*target_ids.shape, 2 * len(names)), dtype=np.float32)
        if target_ids.shape[1] > 0:
            grid = cell_idx[:, None]
            sources = {
                "GE": (
                    tensors.gene_ge,
                    tensors.gene_ge_valid,
                    tensors.cell_ge_mask,
                ),
                "MU": (
                    tensors.gene_mu,
                    tensors.gene_mu_available,
                    tensors.cell_mu_mask,
                ),
                "CNV": (
                    tensors.gene_cnv,
                    tensors.gene_cnv_valid,
                    tensors.cell_cnv_mask,
                ),
            }
            for position, name in enumerate(names):
                matrix, valid_genes, cell_mask = sources[name]
                values[..., 2 * position] = matrix[grid, target_ids]
                values[..., 2 * position + 1] = (
                    cell_mask[cell_idx, None] & valid_genes[target_ids]
                ).astype(np.float32)
            values[~target_mask] = 0.0
        batch.target_gene_ids = torch.from_numpy(target_ids)
        batch.target_gene_values = torch.from_numpy(values)
        batch.target_gene_mask = torch.from_numpy(target_mask)
    if use_gin:
        graph = _graph_batch(drug_idx, drugs)
        batch.graph_node_feat = graph.get("node_feat")  # type: ignore[assignment]
        batch.graph_edge_index = graph.get("edge_index")  # type: ignore[assignment]
        batch.graph_edge_feat = graph.get("edge_feat")  # type: ignore[assignment]
        batch.graph_batch = graph.get("batch")  # type: ignore[assignment]
        batch.graph_num_graphs = int(graph.get("num_graphs", 0))
    return batch


def make_loader(
    base: TeacherPairDataset,
    tensors: PairTensors,
    drugs: DrugFeatureSet,
    *,
    split: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    use_gin: bool = False,
    seed: int = 0,
    group_by_drug: bool = False,
    drugs_per_batch: int = 8,
    cells_per_drug: int = 64,
) -> DataLoader:
    dataset = PairDataset(base, split)

    def _collate(indices: list[int]) -> TeacherBatch:
        return collate_pairs(indices, base=base, tensors=tensors, drugs=drugs, use_gin=use_gin)

    if group_by_drug:
        sampler = DrugGroupedBatchSampler(
            dataset,
            drugs_per_batch=drugs_per_batch,
            cells_per_drug=cells_per_drug,
            seed=seed,
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            collate_fn=_collate,
            pin_memory=torch.cuda.is_available(),
        )

    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


__all__ = [
    "DrugGroupedBatchSampler",
    "PairDataset",
    "PairTensors",
    "build_pair_tensors",
    "collate_pairs",
    "make_loader",
]

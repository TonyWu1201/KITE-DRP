"""教师训练 DataLoader 与 batch 拼装（纯张量，不含 PyG）。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

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
    neighbor_ge: np.ndarray | None
    drug_morgan: np.ndarray


def build_pair_tensors(
    features: CellFeatureSet,
    drugs: DrugFeatureSet,
    *,
    neighbors: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> PairTensors:
    neighbor_ge = (
        neighbor_scores(features.ge, neighbors, weights)
        if neighbors is not None and weights is not None
        else None
    )
    return PairTensors(
        cell_ge=features.ge,
        cell_mu=features.mu,
        cell_cnv=features.cnv,
        cell_cnv_mask=features.cnv_mask.astype(bool),
        neighbor_ge=neighbor_ge,
        drug_morgan=drugs.morgan,
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


def _graph_batch(
    drug_indices: np.ndarray, drugs: DrugFeatureSet
) -> dict[str, torch.Tensor | int]:
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
        neighbor_ge=(
            torch.from_numpy(tensors.neighbor_ge[cell_idx])
            if tensors.neighbor_ge is not None
            else None
        ),
    )
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
) -> DataLoader:
    dataset = PairDataset(base, split)
    generator = torch.Generator().manual_seed(seed) if shuffle else None

    def _collate(indices: list[int]) -> TeacherBatch:
        return collate_pairs(
            indices, base=base, tensors=tensors, drugs=drugs, use_gin=use_gin
        )

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
    "PairDataset",
    "PairTensors",
    "build_pair_tensors",
    "collate_pairs",
    "make_loader",
]

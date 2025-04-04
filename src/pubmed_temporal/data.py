import os.path as osp
from os import makedirs
from shutil import copy
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
# from torch_geometric.datasets import Planetoid  # pytorch_geometric#9966
from torch_geometric.utils.convert import from_networkx

from .graph import build_graph
from .planetoid import Planetoid


def build_data(root: str) -> Dataset:
    """
    Returns PyTorch object from NetworkX graph object,
    split into train, validation, and test sets based on
    time intervals going from 1967 (`t=0`) to 2010 (`t=42`).

    To build the dataset, the following steps are taken:
        1. Download original PubMed graph dataset.
        2. Build NetworkX object from dataset.
        3. Obtain PyTorch Geometric (Planetoid) node index map.
        4. Relabel nodes to match Planetoid's index map.
        5. Add weight vectors `x`.
        6. Add classes `y`.
        7. Add time steps `t`.
        8. Verify if dataset matches Planetoid's.
        9. Save data with edge time steps starting from zero.

    Train, validation and test sets consider a node-level split:
        - Trainining nodes: `t <= train_time`
        - Validation nodes: `train_time < t < test_time`
        - Test nodes: `t >= test_time`

    To load the dataset, use the following code:
        >>> from pubmed_temporal import Planetoid
        >>> # from torch_geometric.datasets import Planetoid  # pytorch_geometric#9982
        >>> dataset = Planetoid(root=".", name="pubmed", split="temporal")

    :param root: Root folder to save data.
    """
    makedirs(osp.join(root, "pubmed", "temporal", "raw"), exist_ok=True)

    G = build_graph(root=root, planetoid_index=True, factorize_time=True)
    data = from_networkx(G.to_undirected())

    # Map edges from converted graph dataset to Planetoid.
    data_ = Planetoid(root=root, name="pubmed", split="public")[0]
    edge_map = {(u, v): i for i, (u, v) in enumerate(data.edge_index.t().numpy())}
    edge_map = np.array([edge_map[(u, v)] for u, v in data_.edge_index.t().numpy()])
    data.edge_index = data.edge_index[:, edge_map]
    data.time = data.time[edge_map]

    assert np.array_equal(data.edge_attr, data_.edge_attr)
    assert np.array_equal(data.x, data_.x)
    assert np.array_equal(data.y, data_.y)

    # Copy original dataset files to temporal folder.
    names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
    for n in names:
        copy(osp.join(root, "pubmed", "raw", f"ind.pubmed.{n}"),
             osp.join(root, "pubmed", "temporal", "raw"))

    # Save edge time steps starting from zero.
    np.save(osp.join(root, "pubmed", "temporal", "raw", "edge_time.npy"),
            (data.time - data.time.min()).numpy())

    # Create temporal split for 60/20/20% training, validation, and test sets.
    # temporal_node_split(data, 0.6, 0.2) == temporal_edge_split(data, 0.6, 0.2)
    train_mask, val_mask, test_mask = temporal_node_split(data, 0.6, 0.2)
    np.savez(osp.join(root, "pubmed", "temporal", "raw", "temporal_split_0.6_0.2.npz"),
             train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    # Map directed edges presented in the original graph dataset.
    edge_directed = [G.has_edge(u, v) for u, v in zip(*data.edge_index.numpy())]
    np.save(osp.join(root, "pubmed", "temporal", "raw", "edge_directed.npy"),
            np.array(edge_directed, dtype=bool))

    return Planetoid(root=root, name="pubmed", split="temporal")[0]


def temporal_node_split(data, train_split: float, val_split: Optional[float] = None):
    """
    Create node-level temporal split for training, validation, and test sets
    based on disjoint time intervals (i.e., unique time steps for each split).

    :param data: PyTorch Geometric object.
    :param train_split: Proportion of training nodes.
    :param val_split: Proportion of validation nodes. Optional.
    """
    for train_time in data.time.unique().numpy()[::-1]:
        train_mask = data.time <= train_time
        subgraph = data.edge_subgraph(train_mask)
        if subgraph.edge_index.unique().shape[0]/data.num_nodes <= train_split:
            break

    for val_time in range(train_time, data.time.max()+1)[::-1]:
        val_mask = (data.time > train_time) & (data.time <= val_time)
        subgraph = data.edge_subgraph(val_mask)
        if subgraph.edge_index.unique().shape[0]/data.num_nodes <= (val_split or 0):
            break

    test_mask = ~(train_mask|val_mask)
    return train_mask, val_mask, test_mask


def temporal_edge_split(data, train_split: float, val_split: Optional[float] = None):
    """
    Create edge-level temporal split for training, validation, and test sets
    based on disjoint time intervals (i.e., unique time steps for each split).

    :param data: PyTorch Geometric object.
    :param train_split: Proportion of training edges.
    :param val_split: Proportion of validation edges. Optional.
    """
    for train_time in data.time.unique().numpy()[::-1]:
        train_mask = data.time <= train_time
        subgraph = data.edge_subgraph(train_mask)
        if subgraph.num_edges/data.num_edges <= train_split:
            break

    for val_time in range(train_time, data.time.max()+1)[::-1]:
        val_mask = (data.time > train_time) & (data.time <= val_time)
        subgraph = data.edge_subgraph(val_mask)
        if subgraph.num_edges/data.num_edges < (val_split or 0):
            break

    test_mask = ~(train_mask|val_mask)
    return train_mask, val_mask, test_mask

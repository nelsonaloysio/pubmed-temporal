#!/usr/bin/env python3

import os.path as osp
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

# Add library to Python path (required if not installed).
PATH = Path(__file__).absolute().parent.parent.__str__()
if PATH not in sys.path:
    sys.path.append(PATH)

from pubmed_temporal.graph import read_times
from pubmed_temporal.planetoid import Planetoid


def build_extra(root: str = PATH) -> list:
    """
    Build table, plot figures and save them to disk.

    Requires the 'matplotlib' and 'tabulate' packages installed.

    :param root: Root path to read data from.
    """
    assert osp.isdir(osp.join(root, "pubmed", "temporal")),\
        "Please run `build_data.py` first to obtain the temporal split."

    dataset = Planetoid(root=root, name="pubmed", split="temporal")
    data = dataset[0]

    # Fill missing node time with value inferred from connected paper.
    node_time = [t for t in read_times(root=root).values()]
    node_time = pd.Series(node_time).fillna(2009).astype(int)
    year = dict(enumerate(sorted(node_time.unique())))

    # Align edge times starting from zero with corresponding node times (years).
    data.time += (len(node_time.unique()) - len(data.time.unique())) - 1
    edge_time = pd.Series(data.time)
    mask = pd.Series([
        "train" if data.train_mask[i] else
        "val" if data.val_mask[i] else
        "test" for i in range(data.num_nodes)
    ])

    # Transductive split.
    train = data.edge_subgraph(data.train_mask)
    train = train.subgraph(train.edge_index.unique())
    val = data.edge_subgraph(data.val_mask)
    val = val.subgraph(val.edge_index.unique())
    test = data.edge_subgraph(data.test_mask)
    test = test.subgraph(test.edge_index.unique())

    # Inductive split.
    train_nodes = data.edge_subgraph(data.train_mask).edge_index.unique()
    val_nodes = data.edge_subgraph(data.val_mask).edge_index.unique()
    test_nodes = data.edge_subgraph(data.test_mask).edge_index.unique()

    test_nodes = test_nodes[~torch.isin(
        test_nodes, torch.cat([train_nodes, val_nodes]).unique())]
    val_nodes = val_nodes[~torch.isin(
        val_nodes, torch.cat([train_nodes, test_nodes]).unique())]
    train_nodes = train_nodes[~torch.isin(
        train_nodes, torch.cat([val_nodes, test_nodes]).unique())]

    train_mask_ = torch.zeros(data.num_nodes, dtype=bool)
    val_mask_ = torch.zeros(data.num_nodes, dtype=bool)
    test_mask_ = torch.zeros(data.num_nodes, dtype=bool)

    train_mask_[train_nodes] = True
    val_mask_[val_nodes] = True
    test_mask_[test_nodes] = True

    train_ = data.subgraph(train_mask_)
    val_ = data.subgraph(val_mask_)
    test_ = data.subgraph(test_mask_)

    # Plot temporal nodes per class.
    y = pd.Series(data.y).apply(lambda x: f"Class {x}")
    y_node_time_count = y.groupby(node_time).value_counts().unstack().fillna(0)
    node_plot = y_node_time_count.plot.bar(
        figsize=(9, 4),
        rot=45,
        stacked=True,
        title="Node time distribution by class",
    )

    # Plot temporal edges per mask.
    edge_time_mask_count = edge_time.groupby(mask).value_counts().unstack().fillna(0).T.sort_index()
    edge_time_mask_count.index = [year[x] for x in edge_time_mask_count.index]
    edge_plot = edge_time_mask_count.iloc[:, [1,2,0]].plot.bar(
        figsize=(9, 4),
        log=True,
        rot=45,
        stacked=True,
        title="Edge time distribution by mask (log-scale)",
    )

    # Save figures.
    for name, fig in zip(("nodes", "edges"), (node_plot, edge_plot)):
        fig.grid(axis="y", color="#cccccc50", zorder=0)
        fig.set_axisbelow(True)
        fig.get_figure().set_tight_layout(True)
        fig.get_figure().savefig(f"fig-{name}.png")

    # Build and save table.
    df = pd.DataFrame({
        ('Full', 'None'): {
            'Nodes': data.num_nodes,
            'Edges': data.num_edges//2,
            'Class 0': data.y.eq(0).sum().item(),
            'Class 1': data.y.eq(1).sum().item(),
            'Class 2': data.y.eq(2).sum().item(),
            'Time steps': f'{data.time.unique().shape[0]}',
            'Interval (Years)': f'{year[data.time.min().item()]} - {year[data.time.max().item()]}',
        },
        ('Transductive', 'Train'): {
            'Nodes': train.num_nodes,
            'Edges': train.num_edges//2,
            'Class 0': train.y.eq(0).sum().item(),
            'Class 1': train.y.eq(1).sum().item(),
            'Class 2': train.y.eq(2).sum().item(),
            'Time steps': f'{train.time.unique().shape[0]}',
            'Interval (Years)': f'{year[train.time.min().item()]} - {year[train.time.max().item()]}',
        },
        ('Transductive', 'Validation'): {
            'Nodes': val.num_nodes,
            'Edges': val.num_edges//2,
            'Class 0': val.y.eq(0).sum().item(),
            'Class 1': val.y.eq(1).sum().item(),
            'Class 2': val.y.eq(2).sum().item(),
            'Time steps': f'{val.time.unique().shape[0]}',
            'Interval (Years)': f'{year[val.time.min().item()]} - {year[val.time.max().item()]}',
        },
        ('Transductive', 'Test'): {
            'Nodes': test.num_nodes,
            'Edges': test.num_edges//2,
            'Class 0': test.y.eq(0).sum().item(),
            'Class 1': test.y.eq(1).sum().item(),
            'Class 2': test.y.eq(2).sum().item(),
            'Time steps': f'{test.time.unique().shape[0]}',
            'Interval (Years)': f'{year[test.time.min().item()]} - {year[test.time.max().item()]}',
        },
        ('Inductive', 'Train'): {
            'Nodes': train_.num_nodes,
            'Edges': train_.num_edges//2,
            'Class 0': train_.y.eq(0).sum().item(),
            'Class 1': train_.y.eq(1).sum().item(),
            'Class 2': train_.y.eq(2).sum().item(),
            'Time steps': f'{train_.time.unique().shape[0]}',
            'Interval (Years)': f'{year[train_.time.min().item()]} - {year[train_.time.max().item()]}',
            },
        ('Inductive', 'Validation'): {
            'Nodes': val_.num_nodes,
            'Edges': val_.num_edges//2,
            'Class 0': val_.y.eq(0).sum().item(),
            'Class 1': val_.y.eq(1).sum().item(),
            'Class 2': val_.y.eq(2).sum().item(),
            'Time steps': f'{val_.time.unique().shape[0]}',
            'Interval (Years)': f'{year[val_.time.min().item()]} - {year[val_.time.max().item()]}',
        },
        ('Inductive', 'Test'): {
            'Nodes': test_.num_nodes,
            'Edges': test_.num_edges//2,
            'Class 0': test_.y.eq(0).sum().item(),
            'Class 1': test_.y.eq(1).sum().item(),
            'Class 2': test_.y.eq(2).sum().item(),
            'Time steps': f'{test_.time.unique().shape[0]}',
            'Interval (Years)': f'{year[test_.time.min().item()]} - {year[test_.time.max().item()]}',
        },
    }).T

    df = df.reset_index(names=["Graph", "Split"])
    df.to_markdown("table.md", colalign=["center"]*df.shape[1], index=False)


if __name__ == "__main__":
    build_extra()

#!/usr/bin/env python3

import os.path as osp
import sys
from pathlib import Path

import pandas as pd

# Add library to Python path (required if not installed).
PATH = osp.join(Path(__file__).parent.parent, "src")
if PATH not in sys.path:
    sys.path.append(PATH)

from pubmed_temporal.build import read_nodes_time
from pubmed_temporal.planetoid import Planetoid, ROOT
from pubmed_temporal.split import split_train_val_test


def build_extra(root: str = ROOT) -> list:
    """
    Build table, plot figures and save them to disk.

    Requires the 'matplotlib' and 'tabulate' packages installed.

    :param root: Root path to read data from.
    """
    assert osp.isdir(osp.join(root, "pubmed", "temporal")),\
        "Please run `build_data.py` first to obtain the temporal split."

    dataset = Planetoid(root=root, name="pubmed", split="temporal")
    data = dataset[0]

    train, val, test = split_train_val_test(data)
    train_, val_, test_ = split_train_val_test(data, inductive_split=True)

    y = pd.Series(data.y).apply(lambda x: f"Class {x}")
    edge_time = pd.Series(data.time)
    mask = pd.Series([
        "train" if data.train_mask[i] else
        "val" if data.val_mask[i] else
        "test" for i in range(data.num_nodes)
    ])

    # Fill missing node time with value inferred from connected paper.
    node_time = [t for t in read_nodes_time().values()]
    node_time = pd.Series(node_time).fillna(2009).astype(int)
    time = dict(enumerate(sorted(node_time.unique())))

    y_node_time_count = y.groupby(node_time).value_counts().unstack().fillna(0)
    edge_time_mask_count = edge_time.groupby(mask).value_counts().unstack().fillna(0).T.sort_index()
    edge_time_mask_count.index = [time[x] for x in edge_time_mask_count.index]

    node_plot = y_node_time_count.plot.bar(
        stacked=True, figsize=(9, 4), rot=45,
        title="Node time distribution by class")

    edge_plot = edge_time_mask_count.iloc[:, [1,2,0]].plot.bar(
        stacked=True, figsize=(9, 4), rot=45,
        title="Edge time distribution by mask")

    for i, fig in enumerate((node_plot, edge_plot)):
        fig.grid(axis="y", color="#cccccc50", zorder=0)
        fig.set_axisbelow(True)
        fig.get_figure().set_tight_layout(True)
        fig.get_figure().savefig(f"fig-{i}.png")

    df = pd.DataFrame({
        ('Full', 'None'): {
            'Nodes': data.num_nodes,
            'Edges': data.num_edges//2,
            'Class 0': data.y.eq(0).sum().item(),
            'Class 1': data.y.eq(1).sum().item(),
            'Class 2': data.y.eq(2).sum().item(),
            'Time steps': f'{data.time.unique().shape[0]}',
            'Interval (Years)': f'{time[data.time.min().item()]} - {time[data.time.max().item()]}',
        },
        ('Transductive', 'Train'): {
            'Nodes': train.num_nodes,
            'Edges': train.num_edges//2,
            'Class 0': train.y.eq(0).sum().item(),
            'Class 1': train.y.eq(1).sum().item(),
            'Class 2': train.y.eq(2).sum().item(),
            'Time steps': f'{train.time.unique().shape[0]}',
            'Interval (Years)': f'{time[train.time.min().item()]} - {time[train.time.max().item()]}',
        },
        ('Transductive', 'Validation'): {
            'Nodes': val.num_nodes,
            'Edges': val.num_edges//2,
            'Class 0': val.y.eq(0).sum().item(),
            'Class 1': val.y.eq(1).sum().item(),
            'Class 2': val.y.eq(2).sum().item(),
            'Time steps': f'{val.time.unique().shape[0]}',
            'Interval (Years)': f'{time[val.time.min().item()]} - {time[val.time.max().item()]}',
        },
        ('Transductive', 'Test'): {
            'Nodes': test.num_nodes,
            'Edges': test.num_edges//2,
            'Class 0': test.y.eq(0).sum().item(),
            'Class 1': test.y.eq(1).sum().item(),
            'Class 2': test.y.eq(2).sum().item(),
            'Time steps': f'{test.time.unique().shape[0]}',
            'Interval (Years)': f'{time[test.time.min().item()]} - {time[test.time.max().item()]}',
        },
        ('Inductive', 'Train'): {
            'Nodes': train_.num_nodes,
            'Edges': train_.num_edges//2,
            'Class 0': train_.y.eq(0).sum().item(),
            'Class 1': train_.y.eq(1).sum().item(),
            'Class 2': train_.y.eq(2).sum().item(),
            'Time steps': f'{train_.time.unique().shape[0]}',
            'Interval (Years)': f'{time[train_.time.min().item()]} - {time[train_.time.max().item()]}',
            },
        ('Inductive', 'Validation'): {
            'Nodes': val_.num_nodes,
            'Edges': val_.num_edges//2,
            'Class 0': val_.y.eq(0).sum().item(),
            'Class 1': val_.y.eq(1).sum().item(),
            'Class 2': val_.y.eq(2).sum().item(),
            'Time steps': f'{val_.time.unique().shape[0]}',
            'Interval (Years)': f'{time[val_.time.min().item()]} - {time[val_.time.max().item()]}',
        },
        ('Inductive', 'Test'): {
            'Nodes': test_.num_nodes,
            'Edges': test_.num_edges//2,
            'Class 0': test_.y.eq(0).sum().item(),
            'Class 1': test_.y.eq(1).sum().item(),
            'Class 2': test_.y.eq(2).sum().item(),
            'Time steps': f'{test_.time.unique().shape[0]}',
            'Interval (Years)': f'{time[test_.time.min().item()]} - {time[test_.time.max().item()]}',
        },
    }).T

    df = df.reset_index(names=["Graph", "Split"])
    df.to_markdown("table.md", colalign=["center"]*df.shape[1], index=False)


if __name__ == "__main__":
    build_extra()

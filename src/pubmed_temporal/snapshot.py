from typing import Optional, Union
from typing_extensions import Self

import torch
from torch import Tensor


# https://github.com/pyg-team/pytorch_geometric/issues/9967
def snapshot(
    data,
    start_time: Union[float, int],
    end_time: Union[float, int],
    attr: Union[str, Tensor] = 'time',
    filter_all: bool = False,
) -> Self:
    r"""Returns a snapshot of :obj:`data` to only hold events that occurred
    in period :obj:`[start_time, end_time]`.

    See `pytorch_geometric#9967
    <https://github.com/pyg-team/pytorch_geometric/pull/9966>`__ for details.

    Args:
        data (Data): The data object.
        start_time (float or int): The start time of the snapshot.
        end_time (float or int): The end time of the snapshot.
        attr (str, optional): The attribute to use. (default: :obj:`time`)
        filter_all (bool, optional): If set to :obj:`True`, filters both
            node- and edge-level data. (default: :obj:`False`)
    """
    if attr in data:
        time = data[attr]
        mask = (time >= start_time) & (time <= end_time)

        if data.is_node_attr(attr):
            keys = data.node_attrs()
        elif data.is_edge_attr(attr):
            keys = data.edge_attrs()

        data._select(keys, mask)

        if data.is_node_attr(attr) and 'num_nodes' in data:
            data.num_nodes = int(mask.sum())

        if not filter_all:
            return data

        if data.is_node_attr(attr):
            keys = data.edge_attrs()
            vals = torch.where(mask == True)[0]
            mask = data.edge_index
            mask = torch.isin(mask, vals).sum(axis=0).bool()
        elif data.is_edge_attr(attr):
            keys = data.node_attrs()
            vals = data.edge_index.reshape(-1).unique()
            mask = torch.zeros(data.num_nodes, dtype=bool)
            mask[vals] = True

        data._select(keys, mask)

        if data.is_edge_attr(attr) and 'num_nodes' in data:
            data.num_nodes = int(mask.sum())

    return data

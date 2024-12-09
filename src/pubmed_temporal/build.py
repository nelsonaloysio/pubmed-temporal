import gzip
import json
import logging as log
import os.path as osp
from functools import partial, wraps
from os import makedirs
from pathlib import Path
from shutil import copy
from typing import Callable, Optional
from zipfile import ZipFile

import networkx as nx
import numpy as np
import pandas as pd
import torch
from numpy import ndarray
from requests import request
from torch.utils.data import Dataset
from torch_geometric.utils.convert import from_networkx, to_networkx
from tqdm.contrib.concurrent import process_map

from .planetoid import Planetoid
from .snapshot import snapshot

try:
    from pubmed_id import PubMedAPI
except ImportError:
    PubMedAPI = None

DATASET_URL = "https://linqs-data.soe.ucsc.edu/public/datasets/pubmed-diabetes/pubmed-diabetes.zip"
ROOT = Path(__file__).parent.parent.parent


def build_dataset(root: str = ROOT) -> Dataset:
    """
    Returns PyTorch object from NetworkX graph object,
    split into train, validation, and test sets based on
    time intervals going from 1964 (`t=0`) to 2010 (`t=45`).

    To build the dataset, the following steps are taken:
        1. Download original PubMed graph dataset.
        2. Build NetworkX object from dataset.
        3. Obtain Planetoid node index map.
        4. Relabel nodes to match Planetoid's index map.
        5. Add weight vectors `x`.
        6. Add classes `y`.
        7. Add time steps `t`.
        8. Verify if dataset matches Planetoid's.
        9. Save temporal node index and split.

    The train, validation and test sets consider the following node split:
        - Trainining nodes (0.58):
            `t < 37`
        - Validation nodes (0.22):
            `37 <= t <= 40`
        - Test nodes (0.20):
            `t > 40`

    To load the dataset, use the following code:
        >>> from torch_geometric.datasets import Planetoid
        >>> dataset = Planetoid(root=..., name="pubmed", split="temporal")

    Or using the internal loader function:
        >>> from pubmed_temporal import Planetoid
        >>> dataset = Planetoid(root=..., name="pubmed", split="temporal")

    :param root: Root folder to save data.
    """
    G = build_graph(root=root)
    data = from_networkx(G.to_undirected())

    assert verify_data(root=root, data=data),\
        "Error: built dataset does not match Planetoid's. Aborting..."

    edge_directed = {(u, v): 0 for u, v in zip(*data.edge_index.numpy())}
    edge_directed.update({(u, v): 1 for u, v in G.edges()})

    train_mask = torch.from_numpy(np.array([True if t < 37 else False for t in data.time]))
    val_mask = torch.from_numpy(np.array([True if 37 <= t <= 40 else False for t in data.time]))
    test_mask = torch.from_numpy(np.array([True if t > 40 else False for t in data.time]))

    makedirs(osp.join(root, "pubmed", "temporal", "raw"), exist_ok=True)

    names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
    for n in names:
        copy(osp.join(root, "pubmed", "raw", f"ind.pubmed.{n}"),
             osp.join(root, "pubmed", "temporal", "raw"))

    np.save(osp.join(root, "pubmed", "temporal", "raw", "edge_time.npy"),
            data.time.numpy())

    # NOTE: Additional attribute.
    np.save(osp.join(root, "pubmed", "temporal", "raw", "node_time.npy"),
            data.node_time.numpy())

    # NOTE: Additional attribute.
    np.save(osp.join(root, "pubmed", "temporal", "raw", "edge_directed.npy"),
            np.array(list(edge_directed.values()), dtype=bool))

    np.savez(osp.join(root, "pubmed", "temporal", "raw", "temporal_split_0.6_0.2.npz"),
             train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    return Planetoid(root=root, name="pubmed", split="temporal")


def build_graph(
    root: str = ROOT,
    planetoid_index: bool = True,
    factorize: bool = False
) -> nx.DiGraph:
    """
    Builds and returns a NetworkX digraph, with attributed
    nodes containing weight vectors `x` and classes `y`,

    Passing `planetoid_index=True` will relabel nodes to
    integer-based, matching Planetoid's index map. Otherwise,
    nodes will be labeled with their original PubMed IDs.

    :param root: Root folder to save data.
    :param planetoid_index: Whether to relabel nodes to match
      Planetoid's index. Default is True.
    :param factorize: Whether to factorize time from strings to integers.
    """
    G = nx.DiGraph()
    download_dataset(root=root)

    # Load original PubMed dataset and times obtained from API.
    nodes = read_nodes(root=root)
    edges = read_edges(root=root)
    times = read_nodes_time(root=root, factorize=factorize)

    # Fill missing time information (n=1, PMID: '17874530').
    # Paper is not indexed anymore on PubMed: <https://pubmed.ncbi.nlm.nih.gov/17874530>.
    # We can infer its time ('2009' or 44) from the edge connecting it to another paper.
    pmid = next(iter(pmid for pmid, time in times.items() if time is None))
    times[pmid] = times[edges.query(f"target == '{pmid}'")["source"].values[0]]

    # Add nodes to graph.
    G.add_nodes_from(nodes.index)
    G.add_edges_from(edges[["source", "target"]].values.tolist())

    # Remove self-loops (n=3).
    G.remove_edges_from(nx.selfloop_edges(G))

    # [pyg-team/pytorch_geometric#6203]
    # Remove 11 bi-directional edges to match Planetoid edge index,
    # keeping those in which the source node has the lowest PMID.
    # G.remove_edges_from(set(tuple(sorted(edge, reverse=True))
    #                         for edge in G.edges()
    #                         if G.has_edge(*edge[::-1])))

    # Add features `x`, corresponding to the original tf-idf weighted vectors.
    nx.set_node_attributes(G, values=nodes.drop("y", axis=1).apply(list, axis=1), name="x")

    # Add classes `y`, mapping them to match Planetoid's.
    nx.set_node_attributes(G, values=nodes["y"].apply({3: 1, 1: 0, 2: 2}.get), name="y")

    # Add node time steps, using the year of publication obtained from the metadata.
    nx.set_node_attributes(G, values=times, name="node_time")

    # Add edge time steps, corresponding to the source node.
    nx.set_edge_attributes(G, values={(u, v): times[u] for u, v in G.edges(data=False)}, name="time")

    # Relabel nodes to match Planetoid's index.
    if planetoid_index:
        planetoid_index_map = get_planetoid_index_map(root=root)

        H = nx.DiGraph()
        H.add_nodes_from(range(len(planetoid_index_map)))

        G = nx.compose(
            H,
            nx.relabel_nodes(
                G,
                dict(zip(nodes.index, planetoid_index_map))
            )
        )

    # Write to disk in GEXF and GraphML formats.
    G._node = {node: {k: v for k, v in attr.items() if k != "x"} for node, attr in G.nodes(data=True)}
    makedirs(osp.join(root, "pubmed", "temporal", "graph"), exist_ok=True)
    nx.write_gexf(G, osp.join(root, "pubmed", "temporal", "graph", "pubmed-temporal.gexf"))
    nx.write_graphml(G, osp.join(root, "pubmed", "temporal", "graph", "pubmed-temporal.graphml"))

    return G


def check_dataset(func: Callable) -> Callable:
    """
    Decorator to check if dataset has been downloaded.
    """
    @wraps(func)
    def func_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                "Dataset not found in root folder. "
                "Please download it first with `download_dataset`."
            ) from e
    return func_wrapper


def download_dataset(root: str = ROOT, url: str = DATASET_URL) -> None:
    """
    Downloads original PubMed dataset.

    See [reference](https://paperswithcode.com/dataset/pubmed).

    :param root: Root folder to save data.
    :param url: URL to download dataset from.
    """
    makedirs(osp.join(root, "input"), exist_ok=True)
    name = osp.join(root, "input", "pubmed-dataset.zip")

    if osp.isfile(name):
        return

    r = request("GET", url, stream=True)
    log.info("Downloading dataset from '%s'...", url)

    with gzip.open(name, "wt") as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

    if osp.isfile(name):
        log.info("Dataset saved to '%s'.", name)


def get_planetoid_index_map(root: str = ROOT, max_workers: Optional[int] = None) -> list:
    """
    Builds and returns Pubmed to Planetoid node index map.

    Due to some nodes having the same weight vector, building the
    index map has a complexity of O(n^2), where `n` is the number
    of nodes in the dataset. Setting the `max_workers` parameter
    to a value greater than 1 allows to parallelize the process.

    :param root: Root folder to save data.
    :param max_workers: Maximum number of workers to use.
    """
    name = osp.join(root, "input", "planetoid-index-map.json.gz")

    if not osp.isfile(name):
        log.info("Building Planetoid index map...")

        # Original dataset.
        G = build_graph(root=root, planetoid_index=False).to_undirected()
        data = from_networkx(G)
        V = list(G.nodes())
        X = data.x.numpy()

        # Planetoid dataset.
        data_ = Planetoid(root=root, name="pubmed")[0]
        G_ = to_networkx(data_).to_undirected()
        V_ = list(G_.nodes())
        X_ = data_.x.numpy()

        # Finds matching weight vectors, with squared time complexity O(n^2).
        if not max_workers or max_workers == 1:
            index_map = []
            for i, x in enumerate(X):
                print(f"Comparing weight vectors: {i+1}/{len(X)}...", end="\r")
                index_map.append([j for j, x_ in enumerate(X_) if np.array_equal(x, x_)])

        else: # Parallelize process.
            chunksize = max(len(X) // (max_workers + 2), 1)
            index_map = list(
                process_map(
                    partial(arrays_equal, X_),
                    X,
                    ascii=True,
                    max_workers=max_workers,
                    chunksize=chunksize,
                    total=len(X),
                    desc=f"Comparing weight vectors "
                         f"(workers: {max_workers}, "
                         f"chunksize: {chunksize})"
                    )
                )

        # Disambiguate multiple matches considering number of node neighbors.
        for i in list(i for i, v in enumerate(index_map) if len(v) > 1):
            n_neighbors = len(list(G.neighbors(V[i])))
            index_map[i] = [j for j in index_map[i]
                            if n_neighbors == len(list(G_.neighbors(V_[j])))]

        assert list(set([len(m) for m in index_map])) == [1],\
               "Error: matches are not unique; number of matches differ."

        assert all(np.array_equal(data.x[i], data_.x[j[0]]) for i, j in enumerate(index_map)),\
               "Error: matches are not unique; weight vectors differ."

        # Flatten list, now containing one-to-one matches.
        index_map = [i for i in index_map for i in i]

        with gzip.open(name, "wt") as j:
            json.dump(index_map, j, indent=2)

        log.info("File saved to '%s'.", name)
        return index_map

    with gzip.open(name, "rb") as j:
        return json.load(j)


def get_pubmed_metadata(
    root: str = ROOT,
    max_workers: Optional[int] = None,
    chunksize: Optional[int] = None
) -> dict:
    """
    Obtains data from PubMed via scraping.

    Requires the 'pubmed-id' package. Setting `max_workers` to
    a value greater than `1` allows to parallelize the process.

    :param root: Root folder to save data.
    :param max_workers: Maximum number of workers to use. Optional.
    :param chunksize: Maximum number of IDs to request per worker. Optional.
    """
    name = osp.join(root, "input", "pubmed-metadata.json.gz")

    if PubMedAPI is None:
        raise ImportError(
            "Module 'pubmed_id' not found. "
            "Please install it first with `pip install pubmed-id`.")

    if not osp.isfile(name):
        log.info("Obtaining data from PubMed...")
        api = PubMedAPI()
        ids = read_ids(root)

        results = {}

        while True:
            results.update(
                api(ids, method="scrape", max_workers=max_workers, chunksize=chunksize)
            )
            # Keep requesting data until every expected ID (all except one) is obtained.
            # Paper is not indexed anymore on PubMed: <https://pubmed.ncbi.nlm.nih.gov/17874530>.
            ids = [pmid for pmid in ids if not results.get(pmid)]
            if len(ids) == 1:
                break

        with gzip.open(name, "wt") as j:
            json.dump(results, j, indent=2)

        log.info("File saved to '%s'.", name)
        return results

    with gzip.open(name, "rb") as j:
        return json.load(j)


@check_dataset
def read_edges(root: str = ROOT) -> pd.DataFrame:
    """
    Reads edges from zipped dataset.

    :param root: Root folder where dataset is found.
    """
    name = osp.join(root, "input", "pubmed-dataset.zip")
    path = osp.join("pubmed-diabetes", "data", "Pubmed-Diabetes.DIRECTED.cites.tab")
    applymap = "applymap" if int(pd.__version__.split(".", 1)[0]) == 1 else "map"

    with ZipFile(name) as z:
        with z.open(path, "r") as zf:
            df = pd.read_table(
                zf,
                dtype=str,
                header=None,
                sep="\t",
                skiprows=2,
                usecols=[0, 1, 3],
                names=["edge_id", "source", "target"],
                index_col="edge_id"
            )

    return getattr(df, applymap)(lambda x: x.replace("paper:", ""))


@check_dataset
def read_ids(root: str = ROOT) -> list:
    """
    Reads IDs from zipped dataset.

    :param root: Root folder where dataset is found.
    """
    name = osp.join(root, "input", "pubmed-dataset.zip")
    path = osp.join("pubmed-diabetes", "data", "Pubmed-Diabetes.NODE.paper.tab")

    with ZipFile(name) as z:
        with z.open(path, "r") as zf:
            return sorted(
                [line.decode().split("\t", 1)[0] for line in zf.readlines()[2:]],
                key=int
            )


@check_dataset
def read_nodes(root: str = ROOT) -> pd.DataFrame:
    """
    Reads nodes from zipped dataset.

    :param root: Root folder where dataset is found.
    """
    name = osp.join(root, "input", "pubmed-dataset.zip")
    path = osp.join("pubmed-diabetes", "data", "Pubmed-Diabetes.NODE.paper.tab")

    with ZipFile(name) as z:
        with z.open(path, "r") as zf:
            return pd.DataFrame(
                {
                line[0]: {
                    **({"y": line[1].split("=")[1]}),
                    **{item[0]: item[1] for item in [item.split("=") for item in line[2:-1]]}
                }
                for line in [line.decode().split("\t") for line in zf.readlines()[2:]]
                },
                dtype=str
            )\
            .transpose()\
            .fillna(0)\
            .astype(float)


def read_nodes_time(root: str = ROOT, factorize: bool = False) -> pd.Series:
    """
    Reads node times from compressed JSON file.

    If not found, reads PubMed data obtained via
    scraping, extracts times, and saves them to file.

    :param root: Root folder where dataset is found.
    :param factorize: Whether to factorize time from strings to integers.
    """
    name_metadata = f"{root}/input/pubmed-metadata.json.gz"
    name_times = f"{root}/input/pubmed-times.json.gz"

    if osp.isfile(name_times):
        with gzip.open(name_times, "rb") as j:
            times = json.load(j)

    elif osp.isfile(name_metadata):
        with gzip.open(name_metadata, "rb") as j:
            pubmed_metadata = json.load(j)

        # Extract year information from dates.
        times = {
            k: v["date"].split()[0].split(":")[0]
            if v.get("date", None) else None
            for k, v in pubmed_metadata.items()
        }

        with gzip.open(name_times, "wt") as j:
            json.dump(times, j, indent=2)

    else:
        raise FileNotFoundError(
            "PubMed metadata not found in root folder. "
            "Please obtain it first with `get_pubmed_metadata`."
        )

    # Factorize time from strings (years) to integers,
    # keeping None values as they are in the dictionary.
    if factorize:
        times = {
            pmid: None if time == -1 else time
            for pmid, time in zip(
                times.keys(),
                pd.factorize(pd.Series(times), sort=True)[0]
            )
        }

    return times


def arrays_equal(arrays: ndarray, array: ndarray) -> list:
    """
    Returns indices of matching weight vectors.

    :param arrays: n-dimensional weight vectors from Planetoid dataset.
    :param array: 1-dimensional weight vector from original dataset.
    """
    return [i for i in range(arrays.shape[0]) if np.array_equal(array, arrays[i])]


def verify_data(data: Dataset, root: str = ROOT) -> bool:
    """
    Verifies built dataset's features match Planetoid's.

    Features to be verified:
    - Node features `x`.
    - Node classes `y`.
    - Total edges in `edge_index`.

    :param root: Root folder to save Planetoid data.
    :param data: PyTorch object from NetworkX graph object.
    """
    data_ = Planetoid(root=root, name="pubmed", split="public")[0]

    return np.array_equal(data.x, data_.x) and\
           np.array_equal(data.y, data_.y) and\
           data.edge_index.shape[1] == data_.edge_index.shape[1]


def save_graphs(data: Dataset, frmt: str = "gexf") -> None:
    """
    Writes directed graphs from full, train, validation, and test sets.

    :param G: NetworkX graph object.
    :param frmt: Format to save graphs in. Default is `'gexf'`.
    """
    from torch_geometric.utils.convert import to_networkx

    digraph = data.clone().snapshot(1, 1, "directed")

    G = to_networkx(
        digraph,
        node_attrs=["y"],
        edge_attrs=["time", "train_mask", "val_mask", "test_mask"])

    G_train = to_networkx(
        snapshot(digraph.clone(), 1, 1, attr="train_mask", filter_all=True),
        node_attrs=["y"],
        edge_attrs=["time"])

    G_val = to_networkx(
        snapshot(digraph.clone(), 1, 1, attr="val_mask", filter_all=True),
        node_attrs=["y"],
        edge_attrs=["time"])

    G_test = to_networkx(
        snapshot(digraph.clone(), 1, 1, attr="test_mask", filter_all=True),
        node_attrs=["y"],
        edge_attrs=["time"])

    writer = getattr(nx, f"write_{frmt}")
    writer(G, "graph.gexf")
    writer(G_train, "graph_train.gexf")
    writer(G_val, "graph_val.gexf")
    writer(G_test, "graph_test.gexf")

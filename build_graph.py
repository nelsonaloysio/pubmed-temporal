#!/usr/bin/env python3

import os.path as osp
from argparse import ArgumentParser, Namespace
from pathlib import Path
from sys import argv

import networkx as nx

from src.pubmed_temporal.graph import (build_graph,
                                       download_pubmed_metadata,
                                       download_graph_dataset)

ROOT = Path(__file__).parent.absolute().__str__()


def argparser(args: list = argv[1:]) -> Namespace:
    """ Parse command line arguments. """
    parser = ArgumentParser()

    parser.add_argument("--root",
                        action="store",
                        default=ROOT,
                        metavar="PATH",
                        help="Set root directory to save data.")

    parser.add_argument("-w", "--max-workers",
                        action="store",
                        metavar="WORKERS",
                        help="Set number of workers to use.",
                        type=int)

    parser.add_argument("-c", "--chunksize",
                        action="store",
                        metavar="WORKERS",
                        help="Set number of IDs to send to each worker at a time.",
                        type=int)

    parser.add_argument("--output",
                        action="store",
                        default='pubmed-temporal.graphml',
                        metavar="PATH",
                        help="Output file name (default: pubmed-temporal.graphml).")

    return parser.parse_args(args)


if __name__ == "__main__":
    args = argparser()
    download_pubmed_metadata(root=args.root, max_workers=args.max_workers, chunksize=args.chunksize)
    download_graph_dataset(root=args.root)
    G = build_graph(root=args.root)
    G._node = {
        node: {k: v for k, v in attr.items() if k != "x"}
        for node, attr in G.nodes(data=True)
    }
    fmt = osp.splitext(args.output)[-1]
    getattr(nx, f"write_{fmt}")(G, args.output)

#!/usr/bin/env python3

from argparse import ArgumentParser, Namespace
from pathlib import Path
from sys import argv

from src.pubmed_temporal.data import build_data
from src.pubmed_temporal.graph import download_pubmed_metadata, download_graph_dataset

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

    return parser.parse_args(args)


if __name__ == "__main__":
    args = argparser()
    download_pubmed_metadata(root=args.root, max_workers=args.max_workers, chunksize=args.chunksize)
    download_graph_dataset(root=args.root)
    build_data(root=args.root)

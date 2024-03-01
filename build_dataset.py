#!/usr/bin/env python3

from argparse import ArgumentParser, Namespace
from sys import argv

from src.pubmed_temporal.build import (
    ROOT,
    build_dataset,
    download_dataset,
    get_planetoid_index_map,
    get_pubmed_metadata
)

def argparser(args: list = argv[1:]) -> Namespace:
    """ Parse command line arguments. """
    parser = ArgumentParser(description=str("Query or build temporal dataset from PubMed."))

    parser.add_argument("--root",
                        action="store",
                        default=ROOT,
                        metavar="PATH",
                        help=f"Root to save files to (default: '{ROOT}').")

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
    download_dataset(root=args.root)
    get_pubmed_metadata(root=args.root, max_workers=args.max_workers, chunksize=args.chunksize)
    get_planetoid_index_map(root=args.root, max_workers=args.max_workers)
    build_dataset(root=args.root)

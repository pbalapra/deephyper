import argparse
import sys

from deephyper.core.logs import parsing, topk, notebooks, balsam
from deephyper.core.plot import quick_plot



def create_parser():
    parser = argparse.ArgumentParser(description="Command line to analysis the outputs produced by DeepHyper.")

    subparsers = parser.add_subparsers(help="Kind of analytics.")

    mapping = dict()

    modules = [
        notebooks,
        parsing,  # parsing deephyper.log
        quick_plot,  # output quick plots
        topk,
        balsam
    ]

    for module in modules:
        name, func = module.add_subparser(subparsers)
        mapping[name] = func

    return parser, mapping


def main():
    parser, mapping = create_parser()

    args = parser.parse_args()

    mapping[sys.argv[1]](**vars(args))

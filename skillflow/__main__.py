"""CLI 入口：python -m skillflow"""

from __future__ import annotations

import argparse

from .create.cli import add_parser as add_create
from .eval.cli import add_parser as add_eval
from .evolve.cli import add_parser as add_evolve


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="skillflow",
        description="Skill creation, evaluation and evolution framework",
    )
    subparsers = parser.add_subparsers(dest="command")

    add_create(subparsers)
    add_eval(subparsers)
    add_evolve(subparsers)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()

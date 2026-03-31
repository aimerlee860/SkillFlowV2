"""create 子命令 CLI。"""

from __future__ import annotations

import argparse
from pathlib import Path

from .creator import create_skill


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("create", help="根据 SPEC 创建技能")
    parser.add_argument("name", help="技能名称（用作输出目录名）")
    parser.add_argument("--spec", required=True, help="SPEC YAML 文件路径")
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="技能输出目录（默认: ./skills/<name>）",
    )
    parser.add_argument(
        "--lang", "-L",
        choices=["auto", "zh", "en"],
        default="auto",
        help="Output language (default: auto-detect)",
    )
    parser.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> None:
    output_dir = args.output or str(Path.cwd() / "skills" / args.name)
    create_skill(args.spec, output_dir, name=args.name, lang=args.lang)

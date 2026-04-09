"""evolve 子命令 CLI。"""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from ..core.utils import validate_skill_dir
from .orchestrator import evolve_skill

console = Console()


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("evolve", help="演化技能")
    parser.add_argument("skill", help="技能目录路径")
    parser.add_argument("--spec", default=None, help="SPEC 文件路径（可选）")
    parser.add_argument("--threshold", type=float, default=0.01, help="通过率提升阈值（默认0.01）")
    parser.add_argument("--trials", type=int, default=5, help="每个用例运行次数（默认5）")
    parser.add_argument(
        "--parallel", "-j",
        type=int,
        default=1,
        help="Number of tasks to evaluate in parallel (default: 1, sequential)",
    )
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=100,
        dest="max_iterations",
        help="Maximum evolution iterations (default: 100)",
    )
    parser.add_argument(
        "--patience", "-p",
        type=int,
        default=10,
        help="Stop after N no-improvements (default: 10)",
    )
    parser.add_argument(
        "--mode", "-M",
        choices=["steady", "greedy"],
        default="steady",
        help="Evolution mode: steady=from baseline (default), greedy=from best",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="输出目录（默认: ./results/<skill目录名>/evolve）",
    )
    parser.add_argument(
        "--speed", "-s",
        type=float,
        default=0.3,
        help="演化速度 0.1~1.0，越小改动越保守，越大越激进（默认 0.3）",
    )
    parser.add_argument("--ignore-cache", action="store_true", help="忽略缓存")
    parser.add_argument("--debug", action="store_true", help="启用 debug 中间件")
    parser.add_argument("--save-trace", action="store_true", help="将执行轨迹落盘到各 iter-*/trace/ 目录")
    parser.add_argument(
        "--test-cases",
        default=None,
        help="测试用例 JSON 文件路径，提供则跳过生成直接加载",
    )
    parser.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> None:
    output = args.output or str(Path.cwd() / "results" / Path(args.skill).name / "evolve")

    try:
        validate_skill_dir(args.skill)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        return

    evolve_skill(
        skill_path=args.skill,
        spec_path=args.spec,
        output_dir=output,
        threshold=args.threshold,
        trials=args.trials,
        parallel=args.parallel,
        max_iterations=args.max_iterations,
        patience=args.patience,
        mode=args.mode,
        speed=args.speed,
        ignore_cache=args.ignore_cache,
        test_cases_file=args.test_cases,
        debug=args.debug,
        save_trace=args.save_trace,
    )

"""eval 子命令 CLI。"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from rich.console import Console

from ..core.utils import ensure_dir, load_json, save_json, validate_skill_dir
from .runner import run_eval
from .test_generator import generate_test_cases

console = Console()


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("eval", help="评估技能")
    parser.add_argument("skill", help="技能目录路径")
    parser.add_argument("--spec", default=None, help="SPEC 文件路径（可选）")
    parser.add_argument("--trials", type=int, default=5, help="每个用例运行次数（默认5）")
    parser.add_argument(
        "--parallel", "-j",
        type=int,
        default=1,
        help="Number of tasks to evaluate in parallel (default: 1, sequential)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="结果输出目录（默认: ./results/<skill目录名>）",
    )
    parser.add_argument("--ignore-cache", action="store_true", help="忽略缓存重新生成测试用例")
    parser.add_argument("--debug", action="store_true", help="启用 debug 中间件，输出 agent 执行详细日志")
    parser.add_argument(
        "--init",
        action="store_true",
        dest="init_only",
        help="只生成测试用例，不运行评估",
    )
    parser.add_argument(
        "--test-cases",
        default=None,
        help="测试用例 JSON 文件路径，提供则跳过生成直接加载",
    )
    parser.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> None:
    output = args.output or str(Path.cwd() / "results" / Path(args.skill).name)

    try:
        validate_skill_dir(args.skill)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        return

    if args.init_only and args.test_cases:
        console.print("[red]--init and --test-cases are mutually exclusive[/red]")
        return

    test_cases = []
    tc_filename = None

    if args.test_cases:
        console.print(f"[blue]加载测试用例文件:[/blue] {args.test_cases}")
        data = load_json(args.test_cases)
        test_cases = data["test_cases"] if isinstance(data, dict) else data
    else:
        test_cases, tc_filename = generate_test_cases(
            skill_path=args.skill,
            spec_path=args.spec,
            output_dir=output,
            ignore_cache=args.ignore_cache,
        )

    console.print(f"[green]共 {len(test_cases)} 个测试用例[/green]")

    # --init 模式：只生成不测试
    if args.init_only:
        return

    # 运行评估
    result = run_eval(
        skill_path=args.skill,
        test_cases=test_cases,
        trials=args.trials,
        parallel=args.parallel,
        debug=args.debug,
    )

    # 保存结果：使用与 test_cases 相同的时间戳，若 --skip-init 则用当前时间
    if tc_filename:
        ts = tc_filename.replace("test_cases_", "").replace(".json", "")
    else:
        ts = time.strftime("%Y%m%d%H%M")
    ensure_dir(output)
    result_file = f"{output}/eval_result_{ts}.json"
    save_json(result_file, result)
    console.print(f"[green]评估结果已保存:[/green] {result_file}")
    console.print(f"[green]总体通过率:[/green] {result['overall_pass_rate']}")
    console.print(f"[green]平均用例通过率:[/green] {result['avg_case_pass_rate']}")

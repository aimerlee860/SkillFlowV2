"""演化流程编排，支持多轮迭代和早停。"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

from ..core.utils import ensure_dir, load_json, load_text, save_json, save_text
from ..eval.runner import run_eval
from ..eval.test_generator import generate_test_cases
from ..eval.trace import aggregate_case_traces, format_trace_context
from .guards import check_regression, should_run_eval, snapshot_files
from .mutator import mutate_skill

console = Console()

# 连续判定"无改进空间"的最大轮次
MAX_NO_CHANGE = 3

STOP_REASON_MAP = {
    "converged": "技能已收敛：连续 {n} 轮判定无改进空间",
    "plateaued": "评估停滞：连续 {patience} 轮无提升",
    "max_iterations": "达到最大迭代次数",
    "trivial": "演化无效：连续产出无实质改动",
}


def evolve_skill(
    skill_path: str | Path,
    spec_path: str | Path | None = None,
    output_dir: str | Path = "./output/evolve",
    threshold: float = 0.01,
    trials: int = 5,
    parallel: int = 1,
    max_iterations: int = 100,
    patience: int = 10,
    mode: str = "steady",
    speed: float = 0.3,
    ignore_cache: bool = False,
    test_cases_file: str | Path | None = None,
    debug: bool = False,
    save_trace: bool = False,
) -> dict:
    """执行多轮技能演化流程。

    Args:
        skill_path: 技能目录路径
        spec_path: SPEC 文件路径
        output_dir: 输出目录
        threshold: 通过率提升阈值（单轮保留条件）
        trials: 每个用例运行次数
        parallel: 并行度（1=串行）
        max_iterations: 最大演化轮数
        patience: 连续无提升早停轮数
        mode: 演化模式 - steady=从 baseline 演化, greedy=从当前最优演化
        speed: 演化速度 0.1~1.0，越小改动越保守
        ignore_cache: 是否忽略缓存
        test_cases_file: 测试用例 JSON 文件路径，提供则跳过生成直接加载
        debug: 是否启用 debug 中间件
        save_trace: 是否将执行轨迹落盘到各 iter-*/trace/ 子目录

    Returns:
        演化记录 dict
    """
    skill_path = Path(skill_path).resolve()
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    # 创建 timestamp 运行目录，支持多次执行
    timestamp = time.strftime("%Y%m%d%H%M")
    run_dir = output_dir / timestamp
    if run_dir.exists():
        shutil.rmtree(run_dir)
    ensure_dir(run_dir)
    output_dir = run_dir

    # Step 1: 获取测试用例
    console.print("[bold blue]Step 1: 获取测试用例[/bold blue]")
    if test_cases_file:
        console.print(f"[blue]加载测试用例文件:[/blue] {test_cases_file}")
        data = load_json(test_cases_file)
        test_cases = data["test_cases"] if isinstance(data, dict) else data
    else:
        test_cases, _ = generate_test_cases(
            skill_path=skill_path,
            spec_path=spec_path,
            output_dir=output_dir,
            ignore_cache=ignore_cache,
        )
    console.print(f"[green]共 {len(test_cases)} 个测试用例[/green]\n")

    # Step 2: Baseline eval
    console.print("[bold blue]Step 2: 运行 Baseline 评估[/bold blue]")
    baseline_result = run_eval(
        skill_path=skill_path,
        test_cases=test_cases,
        trials=trials,
        parallel=parallel,
        debug=debug,
        collect_trace=True,
        output_dir=output_dir,
        save_trace=save_trace,
    )
    save_json(output_dir / "baseline.json",
              {k: v for k, v in baseline_result.items() if k != "_traces"})
    best_rate = baseline_result["overall_reward"]
    best_result = baseline_result  # 保存完整结果，用于 case 级回归检测
    best_iter = 0  # 最优版本对应的迭代号（0 = baseline）
    console.print(f"[green]Baseline Reward:[/green] {best_rate:.4f}")
    console.print(f"[blue]演化模式:[/blue] {mode}\n")

    # 生成 baseline 轨迹摘要
    trace_context = _build_trace_context(baseline_result)

    # 打印 baseline 详细评分表格
    _print_baseline_table(baseline_result)

    # 收集 baseline 失败用例，供第一轮反思使用
    failed_cases = _collect_failed_cases(baseline_result)

    # 备份原始技能（完整目录，保留技能目录名）
    skill_name = skill_path.name
    backup_dir = output_dir / "baseline_skill" / skill_name
    if backup_dir.exists():
        shutil.rmtree(backup_dir.parent)
    shutil.copytree(skill_path, backup_dir)
    best_dir = backup_dir  # 当前最优版本的技能目录

    # 迭代演化
    history: list[dict] = []
    no_improve_count = 0
    no_change_count = 0
    stop_reason = "max_iterations"
    log_file = output_dir / "evolve_log.json"

    for iteration in range(1, max_iterations + 1):
        console.rule(f"[bold cyan]演化轮次 {iteration}/{max_iterations} ({mode})[/bold cyan]")

        # 准备本轮工作目录：从基础版本完整复制
        iter_dir = output_dir / f"iter-{iteration}"
        evolved_skill_dir = iter_dir / skill_name
        if iter_dir.exists():
            shutil.rmtree(iter_dir)

        if mode == "steady":
            shutil.copytree(backup_dir, evolved_skill_dir)
        else:
            shutil.copytree(best_dir, evolved_skill_dir)

        # 快照修改前状态
        before_snapshot = snapshot_files(evolved_skill_dir)

        # 审视 + 可选反思 + 可选范例提取（就地修改 evolved_skill_dir 中的文件）
        if failed_cases:
            console.print(f"[yellow]{len(failed_cases)} 个失败用例，执行审视+反思[/yellow]")
        else:
            console.print(f"[blue]无失败用例，执行审视[/blue]")

        # 构建历史策略摘要
        past_strategies = _build_past_strategies(history) if history else ""

        analysis, strategy, files_changed = mutate_skill(
            evolved_skill_dir,
            failed_cases=failed_cases,
            eval_result=best_result,
            trace_context=trace_context,
            past_strategies=past_strategies,
            speed=speed,
        )
        save_text(iter_dir / "analysis.md", analysis)

        # 判断是否有文件修改（audit 判定"无改进空间"的情况）
        if not files_changed:
            no_change_count += 1
            console.print(
                f"[yellow]本轮未修改文件 ({no_change_count}/{MAX_NO_CHANGE})[/yellow]"
            )
            if no_change_count >= MAX_NO_CHANGE:
                stop_reason = "converged"
                history.append({
                    "iteration": iteration,
                    "mode": mode,
                    "strategy": strategy,
                    "base_rate": round(best_rate, 4),
                    "best_rate": round(best_rate, 4),
                    "evolved_rate": round(best_rate, 4),
                    "improvement": 0.0,
                    "accepted": False,
                    "skip_reason": "no_changes",
                    "summary": "判定无改进空间，未修改文件",
                })
                console.print(
                    f"[red]连续 {MAX_NO_CHANGE} 轮无文件修改，技能已收敛。[/red]\n"
                )
                _save_running_log(
                    log_file, skill_path, "running", stop_reason, threshold, mode,
                    baseline_result["overall_reward"], best_rate, history, best_iter,
                )
                break
            continue

        no_change_count = 0  # 有文件修改则重置计数

        # Diff 初筛：改动太小时调 LLM 语义判断
        after_snapshot = snapshot_files(evolved_skill_dir)
        should_run, eval_reason = should_run_eval(before_snapshot, after_snapshot)

        if not should_run:
            console.print(f"[yellow]改动过于细微，跳过评估: {eval_reason}[/yellow]")
            history.append({
                "iteration": iteration,
                "mode": mode,
                "strategy": strategy,
                "base_rate": round(best_rate, 4),
                "best_rate": round(best_rate, 4),
                "evolved_rate": round(best_rate, 4),
                "improvement": 0.0,
                "accepted": False,
                "skip_reason": "trivial_change",
                "summary": f"细微改动被过滤: {eval_reason}",
            })
            no_improve_count += 1
            if no_improve_count >= patience:
                stop_reason = "plateaued"
                console.print(
                    f"[red]连续 {patience} 轮无提升 (patience={patience})，早停。[/red]\n"
                )
                _save_running_log(
                    log_file, skill_path, "running", stop_reason, threshold, mode,
                    baseline_result["overall_reward"], best_rate, history, best_iter,
                )
                break
            _save_running_log(
                log_file, skill_path, "running", stop_reason, threshold, mode,
                baseline_result["overall_reward"], best_rate, history, best_iter,
            )
            console.print("")
            continue

        # 评估演化后的技能（greedy 模式需要采集轨迹）
        evolved_result = run_eval(
            skill_path=evolved_skill_dir,
            test_cases=test_cases,
            trials=trials,
            parallel=parallel,
            debug=debug,
            collect_trace=(mode == "greedy") or save_trace,
            output_dir=iter_dir,
            save_trace=save_trace,
        )
        evolved_rate = evolved_result["overall_reward"]
        save_json(iter_dir / "eval.json",
                  {k: v for k, v in evolved_result.items() if k != "_traces"})

        improvement = evolved_rate - best_rate
        adaptive_thr = _adaptive_threshold(best_rate, threshold)

        # 三重验收：① reward 提升 ② 无 case 回归
        accepted = improvement >= adaptive_thr

        if accepted:
            has_reg, reg_details = check_regression(
                best_result["test_cases"], evolved_result["test_cases"]
            )
            if has_reg:
                accepted = False
                console.print("[red]Case 回归，拒绝本轮：[/red]")
                for d in reg_details:
                    console.print(f"  [red]- {d}[/red]")

        summary = analysis.strip().replace("\n", " ")[:200]

        iter_record = {
            "iteration": iteration,
            "mode": mode,
            "strategy": strategy,
            "base_rate": round(
                baseline_result["overall_reward"] if mode == "steady" else best_rate, 4
            ),
            "best_rate": round(best_rate, 4),
            "evolved_rate": round(evolved_rate, 4),
            "improvement": round(improvement, 4),
            "threshold_used": round(adaptive_thr, 4),
            "accepted": accepted,
            "summary": summary,
        }
        history.append(iter_record)

        # 输出本轮对比
        _print_iteration_table(iteration, best_rate, evolved_rate, improvement, accepted)

        if accepted:
            best_dir = evolved_skill_dir
            best_iter = iteration
            best_rate = evolved_rate
            best_result = evolved_result
            # greedy 模式：更新轨迹摘要为当前最优版本
            if mode == "greedy":
                trace_context = _build_trace_context(evolved_result)
            no_improve_count = 0
        else:
            no_improve_count += 1

        # 更新失败用例，供下轮反思使用
        failed_cases = _collect_failed_cases(evolved_result)

        # 早停：评估停滞
        if no_improve_count >= patience:
            stop_reason = "plateaued"
            console.print(
                f"[red]连续 {patience} 轮无提升 (patience={patience})，早停。[/red]\n"
            )
            _save_running_log(
                log_file, skill_path, "running", stop_reason, threshold, mode,
                baseline_result["overall_reward"], best_rate, history, best_iter,
            )
            break

        _save_running_log(
            log_file, skill_path, "running", stop_reason, threshold, mode,
            baseline_result["overall_reward"], best_rate, history, best_iter,
        )
        console.print("")

    # 最终结果：记录最优版本（不修改原技能目录）
    final_improvement = best_rate - baseline_result["overall_reward"]

    # 格式化终止原因
    stop_detail = STOP_REASON_MAP.get(stop_reason, stop_reason)
    if stop_reason == "converged":
        stop_detail = stop_detail.format(n=no_change_count)
    elif stop_reason == "plateaued":
        stop_detail = stop_detail.format(patience=patience)

    console.rule("[bold green]演化结束[/bold green]")
    summary_table = Table(title="最终结果")
    summary_table.add_column("指标", style="cyan")
    summary_table.add_column("值", style="green")
    summary_table.add_row("终止原因", stop_detail)
    summary_table.add_row("Baseline Reward", f"{baseline_result['overall_reward']:.4f}")
    summary_table.add_row("最优 Reward", f"{best_rate:.4f}")
    summary_table.add_row("总提升", f"{final_improvement:+.4f}")
    summary_table.add_row("演化轮次", str(len(history)))
    summary_table.add_row("接受轮次", str(sum(1 for h in history if h["accepted"])))
    summary_table.add_row("最优版本", f"iter_{best_iter}" if best_iter > 0 else "baseline")
    console.print(summary_table)

    if best_iter == 0:
        status = "no_improvement"
    else:
        status = "improved"

    log = _build_log(
        skill_path, status, stop_reason, stop_detail, threshold, mode,
        baseline_rate=baseline_result["overall_reward"],
        evolved_rate=best_rate,
        iterations_done=len(history),
        history=history,
        best_iter=best_iter,
    )
    save_json(output_dir / "evolve_log.json", log)
    return log


def _adaptive_threshold(best_rate: float, min_threshold: float) -> float:
    """根据当前最优 reward 动态调整验收阈值。

    - best_rate < 0.5:  threshold = 0.02（初期容易提升）
    - 0.5 ~ 0.8:       threshold = 0.03
    - 0.8 ~ 0.9:       threshold = 0.05（高分段需要确定性）
    - >= 0.9:          threshold = 0.02（边际递减，放宽避免永远不接受）
    """
    if best_rate < 0.5:
        return max(0.02, min_threshold)
    elif best_rate < 0.8:
        return max(0.03, min_threshold)
    elif best_rate < 0.9:
        return max(0.05, min_threshold)
    else:
        return max(0.02, min_threshold)


def _print_baseline_table(baseline_result: dict) -> None:
    """打印 baseline 详细评分表格。"""
    table = Table(title="Baseline 评估详情")
    table.add_column("测试点", style="cyan", max_width=40)
    table.add_column("Reward", style="green")
    table.add_column("平均分", style="yellow")
    table.add_column("Trials", style="dim")
    table.add_column("耗时(s)", style="dim")

    for case in baseline_result["test_cases"]:
        results = case["results"]
        avg_score = sum(r["score"] for r in results) / len(results) if results else 0.0
        table.add_row(
            case["test_point"],
            f"{case['pass_rate']:.2f}",
            f"{avg_score:.2f}",
            f"{sum(1 for r in results if r['pass'])}/{len(results)}",
            f"{case.get('time_total', 0):.1f}",
        )

    table.add_section()
    table.add_row(
        "总计",
        f"{baseline_result['overall_reward']:.4f}",
        "-",
        f"{baseline_result['trials']}/case",
        f"{baseline_result.get('time_total', 0):.1f}",
    )
    console.print(table)


def _collect_failed_cases(eval_result: dict) -> list[dict]:
    """从评估结果中收集失败用例。"""
    failed = []
    for case in eval_result["test_cases"]:
        if case["pass_rate"] < 1.0:
            for r in case["results"]:
                if not r["pass"]:
                    failed.append({
                        "test_point": case["test_point"],
                        "question": case["question"],
                        "reason": r["reason"],
                    })
                    break
    return failed


def _build_past_strategies(history: list[dict]) -> str:
    """从演化历史中构建策略探索记录，供后续迭代参考。"""
    if not history:
        return ""

    lines = ["## 已尝试演化记录\n"]
    for h in history:
        tag = "✓" if h.get("accepted") else "✗"
        skip = h.get("skip_reason", "")
        summary = h.get("summary", "")
        rate = h.get("evolved_rate", "")
        improvement = h.get("improvement", 0)

        entry = f"- 轮次{h['iteration']} [{tag}] 策略={h.get('strategy', '?')}"
        if rate:
            entry += f" reward={rate:.4f} ({improvement:+.4f})"
        if skip:
            entry += f" 原因={skip}"
        if summary:
            entry += f"\n  摘要: {summary[:150]}"
        lines.append(entry)

    return "\n".join(lines) + "\n"


def _build_trace_context(eval_result: dict) -> str:
    """从 eval 结果中提取轨迹并生成摘要文本。"""
    traces_by_case = eval_result.get("_traces")
    if not traces_by_case:
        return ""

    aggregates = []
    for case_traces in traces_by_case:
        if case_traces:
            aggregates.append(aggregate_case_traces(case_traces))

    return format_trace_context(aggregates) if aggregates else ""


def _print_iteration_table(
    iteration: int,
    best_rate: float,
    evolved_rate: float,
    improvement: float,
    accepted: bool,
) -> None:
    """打印单轮对比表格。"""
    table = Table(title=f"轮次 {iteration}")
    table.add_column("版本", style="cyan")
    table.add_column("Reward", style="green")
    table.add_column("提升", style="yellow")
    table.add_column("结果", style="bold")
    table.add_row("当前最优", f"{best_rate:.4f}", "-", "-")
    status = "[green]ACCEPTED[/green]" if accepted else "[red]REJECTED[/red]"
    table.add_row("本轮演化", f"{evolved_rate:.4f}", f"{improvement:+.4f}", status)
    console.print(table)


def _save_running_log(
    log_file: Path,
    skill_path: Path,
    status: str,
    stop_reason: str,
    threshold: float,
    mode: str,
    baseline_rate: float,
    best_rate: float,
    history: list[dict],
    best_iter: int,
) -> None:
    """每轮结束后保存/更新 evolve_log.json，确保崩溃后可恢复。"""
    log = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "skill": skill_path.name if isinstance(skill_path, Path) else str(skill_path),
        "status": status,
        "stop_reason": stop_reason,
        "mode": mode,
        "threshold": threshold,
        "baseline_rate": round(baseline_rate, 4),
        "evolved_rate": round(best_rate, 4),
        "improvement": round(best_rate - baseline_rate, 4),
        "iterations_done": len(history),
        "accepted_iterations": sum(1 for h in history if h.get("accepted")),
        "best_iter": best_iter,
        "history": history,
    }
    save_json(log_file, log)


def _build_log(
    skill_path: Path,
    status: str,
    stop_reason: str,
    stop_detail: str,
    threshold: float,
    mode: str,
    baseline_rate: float,
    evolved_rate: float,
    iterations_done: int,
    history: list[dict],
    best_iter: int = 0,
) -> dict:
    """构建演化记录。"""
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "skill": skill_path.name if isinstance(skill_path, Path) else str(skill_path),
        "status": status,
        "stop_reason": stop_reason,
        "stop_detail": stop_detail,
        "mode": mode,
        "threshold": threshold,
        "baseline_rate": round(baseline_rate, 4),
        "evolved_rate": round(evolved_rate, 4),
        "improvement": round(evolved_rate - baseline_rate, 4),
        "iterations_done": iterations_done,
        "accepted_iterations": sum(1 for h in history if h.get("accepted")),
        "best_iter": best_iter,
        "history": history,
    }

"""测试执行器，运行测试用例并使用 LLM-as-judge 评估。"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Callable

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from ..core.agent import build_agent, run_agent, _setup_debug_logging
from ..core.llm import get_llm
from ..core.prompts import JUDGE_PROMPT, JUDGE_PROMPT_WITH_REFERENCE

console = Console()
debug_logger = logging.getLogger("skillflow.debug")

# JSONL 增量保存文件名
_PROGRESS_FILE = "eval_progress.jsonl"


def judge_response(test_point: str, question: str, response: str, checkpoints: list[str] | None = None, max_retries: int = 3, expected: str | None = None) -> tuple[dict, float]:
    """使用 LLM-as-judge 评估响应。

    Args:
        test_point: 测试点名称
        question: 测试问题
        response: agent 响应
        checkpoints: 评估要点列表
        max_retries: 最大重试次数（默认 3 次）
        expected: 人工参考答案（可选，提供后 Judge 使用带参考答案的 prompt）

    Returns:
        (result_dict, elapsed_seconds)
    """
    if checkpoints:
        checkpoints_text = "\n".join(f"- {cp}" for cp in checkpoints)
    else:
        checkpoints_text = "（无特定评估要点，按通用标准评估）"

    if expected and expected.strip():
        prompt = JUDGE_PROMPT_WITH_REFERENCE.format(
            test_point=test_point,
            question=question,
            response=response,
            checkpoints=checkpoints_text,
            expected=expected,
        )
    else:
        prompt = JUDGE_PROMPT.format(
            test_point=test_point,
            question=question,
            response=response,
            checkpoints=checkpoints_text,
        )
    llm = get_llm()

    total_elapsed = 0.0
    last_error = None

    for attempt in range(max_retries):
        t0 = time.perf_counter()
        try:
            resp = llm.invoke(prompt)
            elapsed = time.perf_counter() - t0
            total_elapsed += elapsed

            text = resp.content if hasattr(resp, "content") else str(resp)
            result = _extract_judge_result(text)

            # 检查是否为 JSON 解析失败
            if "JSON 解析失败" in result.get("reason", ""):
                last_error = result["reason"]
                console.print(f"[yellow]Judge JSON 解析失败 (attempt {attempt+1}/{max_retries}), 重试...[/yellow]")
                continue

            # 成功，返回结果
            return result, total_elapsed

        except Exception as e:
            elapsed = time.perf_counter() - t0
            total_elapsed += elapsed
            last_error = str(e)
            console.print(f"[yellow]Judge 调用异常 (attempt {attempt+1}/{max_retries}): {e}, 重试...[/yellow]")

    # 所有重试都失败，返回默认失败结果
    console.print(f"[red]Judge 重试 {max_retries} 次后仍失败[/red]")
    return {
        "pass": False,
        "score": 0.0,
        "reason": f"Judge 重试失败: {last_error[:100] if last_error else 'unknown'}",
    }, total_elapsed


def _run_single_trial(
    skill_path: str,
    test_point: str,
    question: str,
    trial_idx: int,
    debug: bool = False,
    collect_trace: bool = False,
    checkpoints: list[str] | None = None,
    expected: str | None = None,
) -> dict:
    """运行单次 trial：每次新建 agent 保持上下文干净。"""
    from .trace import ExecutionTrace, extract_trace

    # 阶段 1: 构建 agent
    t_build = time.perf_counter()
    agent = build_agent(
        skills=[skill_path],
        debug=debug,
        debug_trial_id=f"case={test_point[:20]} trial={trial_idx+1}",
    )
    build_elapsed = time.perf_counter() - t_build

    console.print(
        f"  [dim]  trial {trial_idx+1}: "
        f"build={build_elapsed:.2f}s[/dim]", end=""
    )

    # 阶段 2: 运行 agent
    t_agent = time.perf_counter()
    response, messages = run_agent(agent, question)
    agent_elapsed = time.perf_counter() - t_agent
    console.print(
        f"  [dim]  trial {trial_idx+1}: "
        f"agent={agent_elapsed:.2f}s, judge...[/dim]", end=""
    )

    # 阶段 3: judge 评估
    judge_result, judge_elapsed = judge_response(test_point, question, response, checkpoints=checkpoints, expected=expected)

    console.print(
        f"  [dim]  trial {trial_idx+1}: "
        f"judge={judge_elapsed:.2f}s, total={build_elapsed+agent_elapsed+judge_elapsed:.2f}s[/dim]", end=""
    )

    result = {
        "trial": trial_idx + 1,
        "pass": judge_result["pass"],
        "score": judge_result["score"],
        "reason": judge_result["reason"],
        "response": response,
        "time_build": round(build_elapsed, 2),
        "time_agent": round(agent_elapsed, 2),
        "time_judge": round(judge_elapsed, 2),
        "time_total": round(build_elapsed + agent_elapsed + judge_elapsed, 2),
    }

    # 阶段 4: 轨迹提取（可选）
    if collect_trace:
        trace = extract_trace(messages, trial_idx + 1, test_point, question, skill_path)
        trace.final_response = response
        trace.score = judge_result["score"]
        trace.passed = judge_result["pass"]
        trace.judge_reason = judge_result["reason"]
        result["_trace"] = trace

    return result


def _append_progress(progress_file: Path, record: dict) -> None:
    """追加一行 JSONL 记录到进度文件。"""
    with open(progress_file, "a", encoding="utf-8") as f:
        # 序列化时去掉 _trace（不可 JSON 化的数据类）
        clean = {k: v for k, v in record.items() if k != "_trace"}
        f.write(json.dumps(clean, ensure_ascii=False) + "\n")


def run_eval(
    skill_path: str | Path,
    test_cases: list[dict],
    trials: int = 5,
    parallel: int = 1,
    debug: bool = False,
    collect_trace: bool = False,
    output_dir: str | Path | None = None,
    save_trace: bool = False,
    emit_progress: Optional[Callable[[str, str, dict], None]] = None,
) -> dict:
    """运行技能评估。

    Args:
        skill_path: 技能目录路径
        test_cases: 测试用例列表
        trials: 每个用例运行次数
        parallel: 全局并发度（1=串行，>1 则所有用例的所有 trial 共享线程池）
        debug: 是否启用 debug 中间件
        collect_trace: 是否收集执行轨迹
        output_dir: 输出目录（提供则启用 JSONL 增量保存）
        save_trace: 是否将执行轨迹落盘（需要 output_dir，隐含 collect_trace）
        emit_progress: 进度回调函数，签名 (event_type, event_data)

    Returns:
        评估结果 dict
    """
    # save_trace 隐含 collect_trace
    if save_trace:
        collect_trace = True
    from ..core.utils import ensure_dir, save_json

    skill_path = str(Path(skill_path).resolve())
    console.print(f"[blue]技能路径:[/blue] {skill_path}")

    if debug:
        _setup_debug_logging()
        debug_logger.info("=" * 60)
        debug_logger.info("EVAL START | skill=%s | cases=%d | trials=%d", skill_path, len(test_cases), trials)
        debug_logger.info("=" * 60)

    # 增量保存
    progress_file = None
    if output_dir:
        output_dir = Path(output_dir)
        ensure_dir(output_dir)
        progress_file = output_dir / _PROGRESS_FILE
        # 清除旧进度
        if progress_file.exists():
            progress_file.unlink()

    results = [None] * len(test_cases)
    total = len(test_cases) * trials
    t_eval_start = time.perf_counter()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("运行评估...", total=total)

        if parallel <= 1:
            # 串行模式
            for i, tc in enumerate(test_cases):
                progress.update(task, description=f"用例 {i+1}/{len(test_cases)}")
                case_results = []
                for trial_idx in range(trials):
                    result = _run_single_trial(
                        skill_path, tc["test_point"], tc["question"], trial_idx,
                        debug=debug,
                        collect_trace=collect_trace,
                        checkpoints=tc.get("checkpoints"),
                        expected=tc.get("expected"),
                    )
                    case_results.append(result)
                    progress.advance(task)

                    # 每个 trial 完成后追加写入
                    if progress_file:
                        _append_progress(progress_file, {
                            "case_idx": i,
                            "test_point": tc["test_point"],
                            "question": tc["question"],
                            **result,
                        })

                    # 发送进度事件
                    if emit_progress:
                        emit_progress("progress", {
                            "phase": "eval",
                            "case_idx": i,
                            "total_cases": len(test_cases),
                            "trial": trial_idx + 1,
                            "total_trials": trials,
                            "test_point": tc["test_point"],
                            "score": result.get("score", 0),
                            "pass": result.get("pass", False),
                        })

                results[i] = {
                    "test_point": tc["test_point"],
                    "question": tc["question"],
                    "checkpoints": tc.get("checkpoints"),
                    "results": case_results,
                }
        else:
            # 并行模式：全局线程池，所有 case × trial 共享 parallel 个 worker
            with ThreadPoolExecutor(max_workers=parallel) as executor:
                future_map = {}

                for i, tc in enumerate(test_cases):
                    for trial_idx in range(trials):
                        future = executor.submit(
                            _run_single_trial,
                            skill_path, tc["test_point"], tc["question"], trial_idx,
                            debug,
                            collect_trace,
                            tc.get("checkpoints"),
                            tc.get("expected"),
                        )
                        future_map[future] = (i, trial_idx, tc["test_point"], tc["question"])

                # 收集结果
                case_results_map: dict[int, list[dict]] = defaultdict(list)
                for future in as_completed(future_map):
                    case_idx, trial_idx, tp, q = future_map[future]
                    result = future.result()
                    result["trial"] = trial_idx + 1
                    case_results_map[case_idx].append(result)
                    progress.advance(task)

                    # 每个 trial 完成后追加写入
                    if progress_file:
                        _append_progress(progress_file, {
                            "case_idx": case_idx,
                            "test_point": tp,
                            "question": q,
                            **{k: v for k, v in result.items() if k != "_trace"},
                        })

                    # 发送进度事件
                    if emit_progress:
                        emit_progress("progress", {
                            "phase": "eval",
                            "case_idx": case_idx,
                            "total_cases": len(test_cases),
                            "trial": trial_idx + 1,
                            "total_trials": trials,
                            "test_point": tp,
                            "score": result.get("score", 0),
                            "pass": result.get("pass", False),
                        })

                for i, tc in enumerate(test_cases):
                    case_results = sorted(case_results_map[i], key=lambda r: r["trial"])
                    results[i] = {
                        "test_point": tc["test_point"],
                        "question": tc["question"],
                        "checkpoints": tc.get("checkpoints"),
                        "results": case_results,
                    }

    eval_elapsed = time.perf_counter() - t_eval_start
    console.print(f"\n[bold]总评估耗时:[/bold] {eval_elapsed:.1f}s")

    # 正常结束：删除进度文件
    if progress_file and progress_file.exists():
        progress_file.unlink()

    result = _build_eval_result(skill_path, test_cases, results, trials, collect_trace)

    if output_dir:
        # 落盘评估结果
        clean = {k: v for k, v in result.items() if k != "_traces"}
        save_json(Path(output_dir) / "eval.json", clean)

        # 落盘轨迹
        if save_trace and "_traces" in result:
            from .trace import save_traces
            save_traces(result["_traces"], Path(output_dir) / "trace")

    return result


def load_progress(progress_file: str | Path) -> dict:
    """从 JSONL 进度文件恢复部分评估结果。

    Args:
        progress_file: eval_progress.jsonl 路径

    Returns:
        部分评估结果 dict，结构与完整结果一致，但只包含已完成的 case
    """
    progress_file = Path(progress_file)
    if not progress_file.exists():
        raise FileNotFoundError(f"进度文件不存在: {progress_file}")

    # 按 case_idx 分组
    case_map: dict[int, dict] = {}
    case_trials: dict[int, list[dict]] = defaultdict(list)

    with open(progress_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            idx = record["case_idx"]
            case_map[idx] = {
                "test_point": record["test_point"],
                "question": record["question"],
            }
            case_trials[idx].append({
                "trial": record["trial"],
                "pass": record["pass"],
                "score": record["score"],
                "reason": record["reason"],
                "response": record.get("response", ""),
                "time_total": record.get("time_total", 0),
            })

    # 构建 case results
    from .metrics import compute_case_metrics, compute_overall_reward

    case_results = []
    case_rewards = []

    for idx in sorted(case_map.keys()):
        trials_list = sorted(case_trials[idx], key=lambda r: r["trial"])
        case = {
            **case_map[idx],
            "results": trials_list,
        }
        passes = sum(1 for r in trials_list if r["pass"])
        case["pass_rate"] = passes / len(trials_list) if trials_list else 0.0
        case["time_total"] = round(sum(r.get("time_total", 0) for r in trials_list), 2)

        metrics = compute_case_metrics(trials_list)
        case.update(metrics)
        case_rewards.append(metrics["reward"])
        case_results.append(case)

    overall_reward = compute_overall_reward(case_rewards) if case_rewards else 0.0

    return {
        "skill": Path(progress_file).parent.name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "partial": True,
        "completed_cases": len(case_results),
        "total_cases_hint": max(case_map.keys()) + 1 if case_map else 0,
        "test_cases": case_results,
        "overall_reward": overall_reward,
    }


def _build_eval_result(
    skill_path: str,
    test_cases: list[dict],
    case_results: list[dict],
    trials: int,
    collect_trace: bool = False,
) -> dict:
    """构建完整的评估结果。"""
    from .metrics import compute_case_metrics, compute_overall_reward

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    skill_name = Path(skill_path).name

    # 提取轨迹（在 case_results 被 update 前收集）
    traces_by_case: list[list] = []
    if collect_trace:
        for case in case_results:
            case_traces = [
                r.pop("_trace") for r in case["results"] if "_trace" in r
            ]
            traces_by_case.append(case_traces)

    case_rewards = []
    for case in case_results:
        passes = sum(1 for r in case["results"] if r["pass"])
        case["pass_rate"] = passes / len(case["results"]) if case["results"] else 0.0
        case["time_total"] = round(sum(r.get("time_total", 0) for r in case["results"]), 2)

        metrics = compute_case_metrics(case["results"])
        case.update(metrics)
        case_rewards.append(metrics["reward"])

    overall_reward = compute_overall_reward(case_rewards)

    result = {
        "skill": skill_name,
        "timestamp": timestamp,
        "trials": trials,
        "total_cases": len(test_cases),
        "test_cases": case_results,
        "time_total": round(sum(c.get("time_total", 0) for c in case_results), 2),
        "overall_reward": overall_reward,
    }

    if collect_trace:
        result["_traces"] = traces_by_case

    return result


def _extract_judge_result(text: str) -> dict:
    """从 judge 响应中提取结果。"""
    stripped = text.strip()

    # 尝试提取 ```json ... ``` 代码块
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", stripped, re.DOTALL)
    if match:
        stripped = match.group(1).strip()

    # 尝试找到第一个 { 和最后一个 } 之间的内容
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        json_str = stripped[start : end + 1]
    else:
        json_str = stripped

    # 尝试解析 JSON
    try:
        result = json.loads(json_str)
    except json.JSONDecodeError as e:
        console.print(f"[red]Judge JSON 解析失败: {e}[/red]")
        console.print(f"[dim]响应内容: {stripped[:300]}[/dim]")
        # 返回默认失败结果
        return {
            "pass": False,
            "score": 0.0,
            "reason": f"JSON 解析失败: {str(e)[:100]}",
        }

    return {
        "pass": bool(result.get("pass", False)),
        "score": float(result.get("score", 0.0)),
        "reason": str(result.get("reason", "")),
    }

"""统一执行入口：封装 create/eval/evolve 调用，支持恢复。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

from .task_store import TaskRecord


class TaskExecutor:
    """统一执行入口。"""

    def __init__(self, emit_progress: Optional[Callable[[str, str, dict], None]] = None):
        """初始化执行器。

        Args:
            emit_progress: 进度回调函数，签名 (task_id, event_type, event_data)
        """
        self._emit_progress = emit_progress

    def execute(self, task: TaskRecord, recovery: bool = False) -> dict:
        """执行任务。

        Args:
            task: 任务记录
            recovery: 是否从进度文件恢复

        Returns:
            执行结果 dict
        """
        if task.task_type == "create":
            return self._execute_create(task)
        elif task.task_type == "eval":
            return self._execute_eval(task, recovery)
        elif task.task_type == "evolve":
            return self._execute_evolve(task, recovery)
        else:
            raise ValueError(f"Unknown task type: {task.task_type}")

    def _execute_create(self, task: TaskRecord) -> dict:
        """执行技能创建。"""
        from ..create.creator import create_skill

        params = task.params
        spec_path = Path(params["spec_path"])
        output_dir = Path(params["output_dir"])

        result_path = create_skill(
            spec_path=spec_path,
            output_dir=output_dir,
            name=params.get("name"),
            lang=params.get("lang", "auto"),
        )
        return {"path": str(result_path)}

    def _emit(self, task_id: str, event_type: str, data: dict) -> None:
        """发送进度事件。"""
        if self._emit_progress:
            self._emit_progress(task_id, event_type, data)

    def _execute_eval(self, task: TaskRecord, recovery: bool) -> dict:
        """执行技能评估。"""
        from ..eval.runner import run_eval, load_progress
        from ..eval.test_generator import generate_test_cases
        from ..core.utils import load_json
        from ..eval.metrics import compute_overall_reward

        params = task.params
        skill_path = Path(params["skill_path"])
        output_dir = Path(params["output_dir"])
        progress_file = output_dir / "eval_progress.jsonl"

        # 发送启动事件
        self._emit(task.id, "status", {"message": "准备评估...", "phase": "init"})

        # 确定测试用例
        test_cases: Optional[list[dict]] = None

        if params.get("test_cases"):
            test_cases = params["test_cases"]
            self._emit(task.id, "status", {"message": "使用已有测试用例", "count": len(test_cases)})
        elif params.get("test_cases_file"):
            data = load_json(params["test_cases_file"])
            test_cases = data["test_cases"] if isinstance(data, dict) else data
            self._emit(task.id, "status", {"message": "加载测试用例文件", "count": len(test_cases)})

        # 恢复模式：从进度文件继续
        partial_result: Optional[dict] = None
        if recovery and progress_file.exists():
            try:
                partial_result = load_progress(progress_file)
                completed_points = {
                    c["test_point"] for c in partial_result.get("test_cases", [])
                }
                if test_cases:
                    test_cases = [
                        tc for tc in test_cases
                        if tc["test_point"] not in completed_points
                    ]
                self._emit(task.id, "status", {"message": f"恢复评估，已完成 {len(completed_points)} 个测试点"})
            except Exception:
                partial_result = None

        # 生成测试用例（如果未提供）
        if not test_cases:
            spec_path = params.get("spec_path")
            self._emit(task.id, "status", {"message": "生成测试用例...", "phase": "generate_tc"})
            test_cases, _ = generate_test_cases(
                skill_path=skill_path,
                spec_path=spec_path,
                output_dir=output_dir,
                ignore_cache=params.get("ignore_cache", False),
            )
            self._emit(task.id, "test_cases", {"test_cases": test_cases})
            self._emit(task.id, "status", {"message": f"生成 {len(test_cases)} 个测试用例"})

        # 如果恢复模式下所有 case 都已完成，直接返回 partial
        if recovery and partial_result and not test_cases:
            self._emit(task.id, "status", {"message": "评估已完成（恢复）"})
            return partial_result

        # 运行评估
        self._emit(task.id, "status", {"message": f"开始评估 {len(test_cases)} 个测试点...", "phase": "eval"})

        # 创建进度回调
        def _eval_progress(event_type: str, data: dict) -> None:
            self._emit(task.id, event_type, data)

        result = run_eval(
            skill_path=skill_path,
            test_cases=test_cases,
            trials=params.get("trials", 5),
            parallel=params.get("parallel", 1),
            debug=params.get("debug", False),
            output_dir=output_dir,
            save_trace=params.get("save_trace", False),
            emit_progress=_eval_progress,
        )
        self._emit(task.id, "status", {"message": "评估完成", "overall_reward": result.get("overall_reward")})

        # 合并 partial 结果
        if partial_result:
            result = self._merge_eval_results(partial_result, result)

        return result

    def _execute_evolve(self, task: TaskRecord, recovery: bool) -> dict:
        """执行技能演化。"""
        from ..evolve.orchestrator import evolve_skill
        from ..core.utils import load_json, save_json

        params = task.params
        skill_path = Path(params["skill_path"])
        output_dir = Path(params["output_dir"])
        log_file = output_dir / "evolve_log.json"

        # 恢复模式：从 log 文件继续
        if recovery and log_file.exists():
            return self._resume_evolve(task, skill_path, output_dir, log_file, params)

        # 创建进度回调
        def _evolve_progress(event_type: str, data: dict) -> None:
            self._emit(task.id, event_type, data)

        # 新执行
        # 确定 output_dir 的 timestamp 子目录（与原逻辑一致）
        import time
        timestamp = time.strftime("%Y%m%d%H%M")
        run_dir = output_dir / timestamp

        result = evolve_skill(
            skill_path=skill_path,
            spec_path=params.get("spec_path"),
            output_dir=output_dir,
            threshold=params.get("threshold", 0.01),
            trials=params.get("trials", 5),
            parallel=params.get("parallel", 1),
            max_iterations=params.get("max_iterations", 100),
            patience=params.get("patience", 10),
            mode=params.get("mode", "steady"),
            speed=params.get("speed", "low"),
            ignore_cache=params.get("ignore_cache", False),
            test_cases_file=params.get("test_cases_file"),
            debug=params.get("debug", False),
            save_trace=params.get("save_trace", False),
            emit_progress=_evolve_progress,
        )
        # 添加 run_id 用于版本管理 UI
        result["run_id"] = timestamp

        # 发送终止原因事件
        self._emit(task.id, "done", {
            "status": result.get("status", "completed"),
            "stop_reason": result.get("stop_reason", ""),
            "stop_detail": result.get("stop_detail", ""),
            "baseline_rate": result.get("baseline_rate", 0),
            "evolved_rate": result.get("evolved_rate", 0),
            "improvement": result.get("improvement", 0),
            "iterations_done": result.get("iterations_done", 0),
            "best_iter": result.get("best_iter", 0),
        })

        return result

    def _resume_evolve(
        self,
        task: TaskRecord,
        skill_path: Path,
        output_dir: Path,
        log_file: Path,
        params: dict,
    ) -> dict:
        """从 evolve_log.json 恢复演化流程。"""
        import shutil
        import time
        from ..evolve.orchestrator import (
            evolve_skill,
            _adaptive_threshold,
            _build_trace_context,
            _build_past_strategies,
            _collect_failed_cases,
            _save_running_log,
            MAX_NO_CHANGE,
            STOP_REASON_MAP,
        )
        from ..evolve.mutator import mutate_skill
        from ..evolve.guards import check_regression, should_run_eval, snapshot_files
        from ..eval.runner import run_eval
        from ..core.utils import load_json, save_json, ensure_dir
        from rich.console import Console

        console = Console()
        log = load_json(log_file)

        # 找到实际的运行目录（output_dir 下的 timestamp 子目录）
        # log_file 已经是 run_dir/evolve_log.json
        run_dir = log_file.parent
        skill_name = skill_path.name

        # 恢复状态
        history = log.get("history", [])
        best_iter = log.get("best_iter", 0)
        baseline_rate = log.get("baseline_rate", 0)
        best_rate = log.get("evolved_rate", baseline_rate)
        mode = log.get("mode", params.get("mode", "steady"))
        threshold = log.get("threshold", params.get("threshold", 0.01))
        patience = params.get("patience", 10)
        max_iterations = params.get("max_iterations", 100)

        start_iteration = len(history) + 1

        if start_iteration > max_iterations:
            console.print("[yellow]已达到最大迭代次数，无需恢复[/yellow]")
            self._emit(task.id, "status", {"message": "已达到最大迭代次数，无需恢复"})
            return log

        console.print(f"[bold blue]恢复演化: 从轮次 {start_iteration} 继续[/bold blue]")
        self._emit(task.id, "status", {"phase": "resume", "message": f"恢复演化: 从轮次 {start_iteration} 继续"})

        # 定位 best_dir
        if best_iter == 0:
            best_dir = run_dir / "baseline_skill" / skill_name
        else:
            best_dir = run_dir / f"iter-{best_iter}" / skill_name

        # 加载测试用例
        test_cases: list[dict] = []
        if params.get("test_cases_file"):
            data = load_json(params["test_cases_file"])
            test_cases = data["test_cases"] if isinstance(data, dict) else data
        else:
            tc_file = run_dir / "test_cases.json"
            if tc_file.exists():
                test_cases = load_json(tc_file).get("test_cases", [])

        if not test_cases:
            from ..eval.test_generator import generate_test_cases
            self._emit(task.id, "status", {"phase": "test_cases", "message": "生成测试用例..."})
            test_cases, _ = generate_test_cases(
                skill_path=skill_path,
                spec_path=params.get("spec_path"),
                output_dir=run_dir,
                ignore_cache=params.get("ignore_cache", False),
            )

        # 加载 baseline 和 best result
        baseline_result = load_json(run_dir / "baseline.json")
        best_result = baseline_result
        if best_iter > 0:
            best_result = load_json(run_dir / f"iter-{best_iter}" / "eval.json")

        # 备份原始技能（baseline_skill）
        backup_dir = run_dir / "baseline_skill" / skill_name
        if not backup_dir.exists():
            shutil.copytree(skill_path, backup_dir)

        # 轨迹上下文
        trace_context = _build_trace_context(best_result) if mode == "greedy" else ""

        # 失败用例
        failed_cases = _collect_failed_cases(best_result)

        # 继续演化循环
        no_improve_count = 0
        no_change_count = 0
        stop_reason = "max_iterations"
        speed = params.get("speed", "low")

        for iteration in range(start_iteration, max_iterations + 1):
            console.rule(f"[bold cyan]演化轮次 {iteration}/{max_iterations} ({mode})[/bold cyan]")
            self._emit(task.id, "iteration_start", {
                "iteration": iteration,
                "max_iterations": max_iterations,
                "mode": mode,
                "best_rate": round(best_rate, 4),
            })

            iter_dir = run_dir / f"iter-{iteration}"
            evolved_skill_dir = iter_dir / skill_name
            if iter_dir.exists():
                shutil.rmtree(iter_dir)

            if mode == "steady":
                shutil.copytree(backup_dir, evolved_skill_dir)
            else:
                shutil.copytree(best_dir, evolved_skill_dir)

            before_snapshot = snapshot_files(evolved_skill_dir)

            past_strategies = _build_past_strategies(history) if history else ""

            self._emit(task.id, "status", {"phase": "mutate", "message": f"执行演化策略...", "iteration": iteration})
            analysis, strategy, files_changed = mutate_skill(
                evolved_skill_dir,
                failed_cases=failed_cases,
                eval_result=best_result,
                trace_context=trace_context,
                past_strategies=past_strategies,
                speed=speed,
            )

            from ..core.utils import save_text
            save_text(iter_dir / "analysis.md", analysis)

            if not files_changed:
                no_change_count += 1
                console.print(f"[yellow]本轮未修改文件 ({no_change_count}/{MAX_NO_CHANGE})[/yellow]")
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
                    _save_running_log(
                        log_file, skill_path, "running", stop_reason, threshold, mode,
                        baseline_rate, best_rate, history, best_iter,
                    )
                    break
                continue

            no_change_count = 0

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
                    _save_running_log(
                        log_file, skill_path, "running", stop_reason, threshold, mode,
                        baseline_rate, best_rate, history, best_iter,
                    )
                    break
                _save_running_log(
                    log_file, skill_path, "running", stop_reason, threshold, mode,
                    baseline_rate, best_rate, history, best_iter,
                )
                continue

            self._emit(task.id, "status", {"phase": "eval", "message": f"评估轮次 {iteration}...", "iteration": iteration})

            def _iter_progress(event_type: str, data: dict) -> None:
                self._emit(task.id, event_type, {**data, "phase": "iter_eval", "iteration": iteration})

            evolved_result = run_eval(
                skill_path=evolved_skill_dir,
                test_cases=test_cases,
                trials=params.get("trials", 5),
                parallel=params.get("parallel", 1),
                debug=params.get("debug", False),
                collect_trace=(mode == "greedy") or params.get("save_trace", False),
                output_dir=iter_dir,
                save_trace=params.get("save_trace", False),
                emit_progress=_iter_progress,
            )
            evolved_rate = evolved_result["overall_reward"]
            save_json(iter_dir / "eval.json", {k: v for k, v in evolved_result.items() if k != "_traces"})

            improvement = evolved_rate - best_rate
            adaptive_thr = _adaptive_threshold(best_rate, threshold)

            accepted = improvement >= adaptive_thr
            if accepted:
                has_reg, reg_details = check_regression(
                    best_result["test_cases"], evolved_result["test_cases"]
                )
                if has_reg:
                    accepted = False
                    console.print("[red]Case 回归，拒绝本轮[/red]")

            summary = analysis.strip().replace("\n", " ")[:200]

            history.append({
                "iteration": iteration,
                "mode": mode,
                "strategy": strategy,
                "base_rate": round(baseline_rate if mode == "steady" else best_rate, 4),
                "best_rate": round(best_rate, 4),
                "evolved_rate": round(evolved_rate, 4),
                "improvement": round(improvement, 4),
                "threshold_used": round(adaptive_thr, 4),
                "accepted": accepted,
                "summary": summary,
            })

            # 发送迭代结果事件
            self._emit(task.id, "iteration", {
                "iteration": iteration,
                "strategy": strategy,
                "evolved_rate": round(evolved_rate, 4),
                "improvement": round(improvement, 4),
                "accepted": accepted,
            })

            if accepted:
                best_dir = evolved_skill_dir
                best_iter = iteration
                best_rate = evolved_rate
                best_result = evolved_result
                if mode == "greedy":
                    trace_context = _build_trace_context(evolved_result)
                no_improve_count = 0
            else:
                no_improve_count += 1

            failed_cases = _collect_failed_cases(evolved_result)

            if no_improve_count >= patience:
                stop_reason = "plateaued"
                _save_running_log(
                    log_file, skill_path, "running", stop_reason, threshold, mode,
                    baseline_rate, best_rate, history, best_iter,
                )
                break

            _save_running_log(
                log_file, skill_path, "running", stop_reason, threshold, mode,
                baseline_rate, best_rate, history, best_iter,
            )

        # 最终结果
        final_improvement = best_rate - baseline_rate
        status = "no_improvement" if best_iter == 0 else "improved"

        stop_detail = STOP_REASON_MAP.get(stop_reason, stop_reason)
        if stop_reason == "converged":
            stop_detail = stop_detail.format(n=no_change_count)
        elif stop_reason == "plateaued":
            stop_detail = stop_detail.format(patience=patience)

        final_log = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "skill": skill_name,
            "run_id": log_file.parent.name,  # 用于版本管理 UI
            "status": status,
            "stop_reason": stop_reason,
            "stop_detail": stop_detail,
            "mode": mode,
            "threshold": threshold,
            "baseline_rate": round(baseline_rate, 4),
            "evolved_rate": round(best_rate, 4),
            "improvement": round(final_improvement, 4),
            "iterations_done": len(history),
            "accepted_iterations": sum(1 for h in history if h.get("accepted")),
            "best_iter": best_iter,
            "history": history,
        }
        save_json(log_file, final_log)

        console.rule("[bold green]演化结束[/bold green]")
        console.print(f"[green]最优 Reward: {best_rate:.4f}[/green]")
        console.print(f"[green]总提升: {final_improvement:+.4f}[/green]")

        # 发送终止原因事件
        self._emit(task.id, "done", {
            "status": final_log.get("status", "completed"),
            "stop_reason": final_log.get("stop_reason", ""),
            "stop_detail": final_log.get("stop_detail", ""),
            "baseline_rate": final_log.get("baseline_rate", 0),
            "evolved_rate": final_log.get("evolved_rate", 0),
            "improvement": final_log.get("improvement", 0),
            "iterations_done": final_log.get("iterations_done", 0),
            "best_iter": final_log.get("best_iter", 0),
        })

        return final_log

    def _merge_eval_results(self, partial: dict, new: dict) -> dict:
        """合并 partial 和新执行的 results。"""
        from ..eval.metrics import compute_overall_reward

        merged_cases = partial.get("test_cases", []) + new.get("test_cases", [])
        rewards = [c.get("reward", 0) for c in merged_cases]
        overall = compute_overall_reward(rewards)

        return {
            "skill": new.get("skill", partial.get("skill")),
            "timestamp": new.get("timestamp"),
            "trials": new.get("trials", partial.get("trials")),
            "total_cases": len(merged_cases),
            "test_cases": merged_cases,
            "overall_reward": overall,
            "partial": False,
        }
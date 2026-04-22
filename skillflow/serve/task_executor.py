"""统一执行入口：封装 create/eval/evolve 调用，支持恢复。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

from .task_store import TaskRecord, TaskStore


class TaskExecutor:
    """统一执行入口。"""

    def __init__(
        self,
        emit_progress: Optional[Callable[[str, str, dict], None]] = None,
        store: Optional[TaskStore] = None,
    ):
        """初始化执行器。

        Args:
            emit_progress: 进度回调函数，签名 (task_id, event_type, event_data)
            store: 任务存储实例，用于在执行过程中更新任务字段
        """
        self._emit_progress = emit_progress
        self._store = store

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
                enable_critic=params.get("enable_critic", True),
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

        # 过滤掉不可 JSON 序列化的字段（轨迹已单独保存）
        return {k: v for k, v in result.items() if k != "_traces"}

    def _execute_evolve(self, task: TaskRecord, recovery: bool) -> dict:
        """执行技能演化。"""
        import time
        from ..evolve.orchestrator import evolve_skill
        from ..core.utils import load_json

        params = task.params
        skill_path = Path(params["skill_path"])
        output_dir = Path(params["output_dir"])

        # 恢复模式：用 task.output_dir（已在首次运行时更新为 run_dir）
        if recovery and task.output_dir:
            run_dir = Path(task.output_dir)
            log_file = run_dir / "evolve_log.json"
            if log_file.exists():
                return self._resume_evolve(task, skill_path, log_file, params)

        # 使用预计算的 timestamp（由 API 层传入），确保 watcher 和 executor 一致
        timestamp = params.get("run_id") or time.strftime("%Y%m%d%H%M")
        run_dir = output_dir / timestamp

        # 在执行前持久化 run_dir，确保崩溃后可恢复
        if self._store:
            self._store.update_output_dir(task.id, str(run_dir))

        # 创建进度回调
        def _evolve_progress(event_type: str, data: dict) -> None:
            self._emit(task.id, event_type, data)

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
            run_id=timestamp,
        )
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
        log_file: Path,
        params: dict,
    ) -> dict:
        """从 evolve_log.json 恢复演化流程。"""
        from ..evolve.orchestrator import evolve_skill
        from ..core.utils import load_json

        log = load_json(log_file)
        run_dir = log_file.parent
        history = log.get("history", [])
        start_iteration = len(history) + 1
        max_iterations = params.get("max_iterations", 100)

        if start_iteration > max_iterations:
            self._emit(task.id, "status", {"message": "已达到最大迭代次数，无需恢复"})
            log["run_id"] = log_file.parent.name
            return log

        # 加载 baseline 和 best result
        baseline_result = load_json(run_dir / "baseline.json")
        best_iter = log.get("best_iter", 0)
        best_result = baseline_result
        if best_iter > 0:
            best_result = load_json(run_dir / f"iter-{best_iter}" / "eval.json")

        # 加载测试用例
        test_cases = self._load_resume_test_cases(params, run_dir, skill_path, task)

        # 进度回调
        def _evolve_progress(event_type: str, data: dict) -> None:
            self._emit(task.id, event_type, data)

        result = evolve_skill(
            skill_path=skill_path,
            spec_path=params.get("spec_path"),
            output_dir=run_dir,
            threshold=log.get("threshold", params.get("threshold", 0.01)),
            trials=params.get("trials", 5),
            parallel=params.get("parallel", 1),
            max_iterations=max_iterations,
            patience=params.get("patience", 10),
            mode=log.get("mode", params.get("mode", "steady")),
            speed=params.get("speed", "low"),
            test_cases_file=params.get("test_cases_file"),
            debug=params.get("debug", False),
            save_trace=params.get("save_trace", False),
            emit_progress=_evolve_progress,
            resume_from={
                "run_dir": run_dir,
                "history": history,
                "best_iter": best_iter,
                "best_rate": log.get("evolved_rate", log.get("baseline_rate", 0)),
                "baseline_result": baseline_result,
                "best_result": best_result,
                "test_cases": test_cases,
            },
        )

        # 添加 run_id + 发送 done 事件
        result["run_id"] = log_file.parent.name
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

    def _load_resume_test_cases(
        self,
        params: dict,
        run_dir: Path,
        skill_path: Path,
        task: TaskRecord,
    ) -> list[dict]:
        """加载恢复模式下的测试用例。"""
        from ..core.utils import load_json

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
        return test_cases

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
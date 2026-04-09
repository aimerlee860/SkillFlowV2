"""Eval API：启动评估任务 + SSE 进度流。"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from fastapi import APIRouter, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

router = APIRouter(tags=["eval"])


class EvalRequest(BaseModel):
    skill: str  # skill 目录名（在 skills/ 下）
    spec: str | None = None
    trials: int = 5
    parallel: int = 1
    debug: bool = False
    save_trace: bool = False
    ignore_cache: bool = False
    test_cases_file: str | None = None
    test_cases_content: str | None = None  # 前端传来的 JSON 字符串


@router.post("/eval")
def start_eval(req: EvalRequest, request: Request):
    tm = request.app.state.task_manager

    skill_path = Path.cwd() / "skills" / req.skill
    if not (skill_path / "SKILL.md").exists():
        return {"error": f"Skill not found: {req.skill}"}

    spec_path = None
    if req.spec:
        spec_path = str(Path.cwd() / "specs" / req.spec)

    output_dir = Path.cwd() / "results" / req.skill / "eval" / time.strftime("%Y%m%d%H%M")

    task = tm.create_task("eval")
    progress_file = output_dir / "eval_progress.jsonl"

    def _run():
        from ...eval.test_generator import generate_test_cases
        from ...eval.runner import run_eval

        try:
            # 获取测试用例
            task.progress_events.append(
                {"type": "status", "message": "Generating test cases..."}
            )

            if req.test_cases_file:
                from ...core.utils import load_json
                data = load_json(req.test_cases_file)
                test_cases = data["test_cases"] if isinstance(data, dict) else data
            elif req.test_cases_content:
                data = json.loads(req.test_cases_content)
                test_cases = data["test_cases"] if isinstance(data, dict) else data
            else:
                test_cases, _ = generate_test_cases(
                    skill_path=skill_path,
                    spec_path=spec_path,
                    output_dir=output_dir,
                    ignore_cache=req.ignore_cache,
                )
                # 将自动生成的测试用例推回前端，便于查看和后续复用
                task.progress_events.append(
                    {"type": "test_cases", "data": test_cases}
                )

            task.progress_events.append(
                {"type": "status", "message": f"{len(test_cases)} test cases ready, running eval..."}
            )

            # 运行评估
            result = run_eval(
                skill_path=skill_path,
                test_cases=test_cases,
                trials=req.trials,
                parallel=req.parallel,
                debug=req.debug,
                output_dir=output_dir,
                save_trace=req.save_trace,
            )

            # 清理内部不可序列化的字段
            clean_result = {k: v for k, v in result.items() if k != "_traces"}
            task.result = clean_result
            task.status = "completed"
            task.progress_events.append(
                {"type": "done", "status": "completed", "result": clean_result}
            )
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            task.progress_events.append(
                {"type": "done", "status": "failed", "error": str(e)}
            )

    task.progress_events.append({"type": "status", "message": "Starting evaluation..."})
    task.status = "running"
    tm.submit_with_watcher(task, _run, progress_file=progress_file)

    return {"task_id": task.id, "status": "running"}


@router.get("/eval/{task_id}/progress")
async def eval_progress(task_id: str, request: Request):
    tm = request.app.state.task_manager
    task = tm.get_task(task_id)
    if not task:
        return {"error": "task not found"}

    async def _stream():
        last_idx = 0
        while task.status in ("pending", "running"):
            while last_idx < len(task.progress_events):
                yield {"data": json.dumps(task.progress_events[last_idx])}
                last_idx += 1
            await asyncio.sleep(0.5)
        while last_idx < len(task.progress_events):
            yield {"data": json.dumps(task.progress_events[last_idx])}
            last_idx += 1

    return EventSourceResponse(_stream())

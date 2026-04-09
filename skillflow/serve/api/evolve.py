"""Evolve API：启动演化任务 + SSE 进度流。"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from fastapi import APIRouter, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

router = APIRouter(tags=["evolve"])


class EvolveRequest(BaseModel):
    skill: str  # skill 目录名（在 skills/ 下）
    spec: str | None = None
    threshold: float = 0.01
    trials: int = 5
    parallel: int = 1
    max_iterations: int = 100
    patience: int = 10
    mode: str = "steady"
    speed: float = 0.3
    ignore_cache: bool = False
    debug: bool = False
    save_trace: bool = False
    test_cases_file: str | None = None


@router.post("/evolve")
def start_evolve(req: EvolveRequest, request: Request):
    tm = request.app.state.task_manager

    skill_path = Path.cwd() / "skills" / req.skill
    if not (skill_path / "SKILL.md").exists():
        return {"error": f"Skill not found: {req.skill}"}

    spec_path = None
    if req.spec:
        spec_path = str(Path.cwd() / "specs" / req.spec)

    output_dir = Path.cwd() / "results" / req.skill / "evolve"

    task = tm.create_task("evolve")

    def _run():
        from ...evolve.orchestrator import evolve_skill

        try:
            result = evolve_skill(
                skill_path=skill_path,
                spec_path=spec_path,
                output_dir=output_dir,
                threshold=req.threshold,
                trials=req.trials,
                parallel=req.parallel,
                max_iterations=req.max_iterations,
                patience=req.patience,
                mode=req.mode,
                speed=req.speed,
                ignore_cache=req.ignore_cache,
                test_cases_file=req.test_cases_file,
                debug=req.debug,
                save_trace=req.save_trace,
            )
            task.result = result
            task.status = "completed"
            task.progress_events.append(
                {"type": "done", "status": "completed", "result": result}
            )
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            task.progress_events.append(
                {"type": "done", "status": "failed", "error": str(e)}
            )

    # evolve_log.json 路径在 _run 执行时才确定（内部创建 timestamp 子目录）
    # 但我们可以轮询 output_dir 下最新的 timestamp 目录
    task.progress_events.append(
        {"type": "status", "message": "Starting evolution..."}
    )
    task.status = "running"
    tm._executor.submit(_run)

    # 启动 evolve_log.json 轮询
    _start_evolve_watcher(tm, task, output_dir)

    return {"task_id": task.id, "status": "running"}


def _start_evolve_watcher(tm, task, output_dir: Path):
    """轮询 evolve_log.json 获取迭代进度。"""
    seen_iters = 0

    def _watch():
        nonlocal seen_iters
        while task.status in ("pending", "running"):
            try:
                # 找到最新的 timestamp 子目录
                if not output_dir.exists():
                    continue
                subdirs = sorted(
                    [d for d in output_dir.iterdir() if d.is_dir()],
                    key=lambda d: d.name,
                )
                if not subdirs:
                    continue
                latest = subdirs[-1]
                log_file = latest / "evolve_log.json"
                if not log_file.exists():
                    continue

                data = json.loads(log_file.read_text(encoding="utf-8"))
                history = data.get("history", [])
                if len(history) > seen_iters:
                    for record in history[seen_iters:]:
                        task.progress_events.append(
                            {"type": "iteration", "data": record}
                        )
                    seen_iters = len(history)
            except Exception:
                pass
            time.sleep(2)

    tm._executor.submit(_watch)


@router.get("/evolve/{task_id}/progress")
async def evolve_progress(task_id: str, request: Request):
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

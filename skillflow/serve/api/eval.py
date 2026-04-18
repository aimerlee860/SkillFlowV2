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
    no_critic: bool = False
    test_cases_file: str | None = None
    test_cases_content: str | None = None  # 前端传来的 JSON 字符串


@router.post("/eval")
def start_eval(req: EvalRequest, request: Request):
    """启动技能评估任务。"""
    tm = request.app.state.task_manager

    skill_path = Path.cwd() / "skills" / req.skill
    if not (skill_path / "SKILL.md").exists():
        return {"error": f"Skill not found: {req.skill}"}

    spec_path = None
    if req.spec:
        spec_path = str(Path.cwd() / "specs" / req.spec)

    output_dir = str(Path.cwd() / "results" / req.skill / "eval" / time.strftime("%Y%m%d%H%M"))

    # 构建参数
    params = {
        "skill_path": str(skill_path),
        "spec_path": spec_path,
        "output_dir": output_dir,
        "trials": req.trials,
        "parallel": req.parallel,
        "debug": req.debug,
        "save_trace": req.save_trace,
        "ignore_cache": req.ignore_cache,
        "enable_critic": not req.no_critic,
        "test_cases_file": req.test_cases_file,
        "test_cases": None,  # 运行时生成或从 test_cases_content 解析
    }

    # 处理 test_cases_content
    if req.test_cases_content:
        data = json.loads(req.test_cases_content)
        params["test_cases"] = data["test_cases"] if isinstance(data, dict) else data

    # 创建任务
    task = tm.create_task("eval", skill=req.skill, params=params, output_dir=output_dir)

    # 发送初始状态
    tm.emit_progress(task.id, "status", {"message": "Starting evaluation..."})

    # 提交执行
    status = tm.submit(task.id)

    return {"task_id": task.id, "status": status}


@router.get("/eval/{task_id}/progress")
async def eval_progress(task_id: str, request: Request):
    """SSE 进度流。"""
    tm = request.app.state.task_manager
    task = tm.get_task(task_id)
    if not task:
        return {"error": "task not found"}

    async def _stream():
        last_seq = -1
        current_task = task
        while current_task.status in ("pending", "queued", "running"):
            events = tm.get_progress_events(task_id, after_seq=last_seq)
            for ev in events:
                yield {"data": json.dumps(ev)}
                last_seq = ev["seq"]
            await asyncio.sleep(0.5)
            current_task = tm.get_task(task_id)
            if not current_task:
                break

        # 发送剩余事件
        events = tm.get_progress_events(task_id, after_seq=last_seq)
        for ev in events:
            yield {"data": json.dumps(ev)}

    return EventSourceResponse(_stream())
"""Create API：启动技能创建任务 + SSE 进度流。"""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

router = APIRouter(tags=["create"])


class CreateRequest(BaseModel):
    name: str
    spec: str = ""  # spec 文件名（在 specs/ 下），可选
    spec_content: str = ""  # spec YAML 内容文本，与 spec 二选一
    lang: str = "auto"


@router.post("/create")
def start_create(req: CreateRequest, request: Request):
    tm = request.app.state.task_manager

    # 确定_spec_path：优先用 spec_content，其次用 spec 文件名
    if req.spec_content.strip():
        # 将文本内容写入临时文件
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        )
        tmp.write(req.spec_content)
        tmp.close()
        spec_path = Path(tmp.name)
    elif req.spec:
        spec_path = Path.cwd() / "specs" / req.spec
        if not spec_path.exists():
            return {"error": f"Spec file not found: {req.spec}"}
    else:
        return {"error": "请提供 spec 文件名或直接输入 spec 内容"}

    output_dir = Path.cwd() / "skills" / req.name

    task = tm.create_task("create")

    def _run():
        from ...create.creator import create_skill

        try:
            result = create_skill(
                spec_path=spec_path,
                output_dir=output_dir,
                name=req.name,
                lang=req.lang,
            )
            task.result = {"path": str(result)}
            task.status = "completed"
            task.progress_events.append(
                {"type": "done", "status": "completed", "result": task.result}
            )
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            task.progress_events.append(
                {"type": "done", "status": "failed", "error": str(e)}
            )

    # 先发射一个 starting 事件
    task.progress_events.append({"type": "status", "message": "Starting skill creation..."})
    task.status = "running"
    tm._executor.submit(_run)

    return {"task_id": task.id, "status": "running"}


@router.get("/create/{task_id}/progress")
async def create_progress(task_id: str, request: Request):
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
        # 发送剩余事件
        while last_idx < len(task.progress_events):
            yield {"data": json.dumps(task.progress_events[last_idx])}
            last_idx += 1

    return EventSourceResponse(_stream())


@router.get("/tasks/{task_id}")
def get_task_status(task_id: str, request: Request):
    tm = request.app.state.task_manager
    task = tm.get_task(task_id)
    if not task:
        return {"error": "task not found"}
    return task.to_dict()

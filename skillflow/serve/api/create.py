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
    """启动技能创建任务。"""
    tm = request.app.state.task_manager

    # 确定 spec_path
    if req.spec_content.strip():
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        )
        tmp.write(req.spec_content)
        tmp.close()
        spec_path = str(tmp.name)
    elif req.spec:
        spec_path = str(Path.cwd() / "specs" / req.spec)
        if not Path(spec_path).exists():
            return {"error": f"Spec file not found: {req.spec}"}
    else:
        return {"error": "请提供 spec 文件名或直接输入 spec 内容"}

    output_dir = str(Path.cwd() / "skills" / req.name)

    # 构建参数
    params = {
        "spec_path": spec_path,
        "output_dir": output_dir,
        "name": req.name,
        "lang": req.lang,
    }

    # 创建任务
    task = tm.create_task("create", skill=req.name, params=params, output_dir=output_dir)

    # 提交执行
    status = tm.submit(task.id)

    return {"task_id": task.id, "status": status}


@router.get("/create/{task_id}/progress")
async def create_progress(task_id: str, request: Request):
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


@router.get("/tasks/{task_id}")
def get_task_status(task_id: str, request: Request):
    """获取任务状态。"""
    tm = request.app.state.task_manager
    task = tm.get_task(task_id)
    if not task:
        return {"error": "task not found"}
    return task.to_dict()
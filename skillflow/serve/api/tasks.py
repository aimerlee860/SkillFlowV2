"""任务管理 API：CRUD + 控制接口。"""

from __future__ import annotations

import asyncio
import io
import json
import zipfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Request, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

router = APIRouter(tags=["tasks"])


@router.get("/tasks")
def list_tasks(
    request: Request,
    status: Optional[str] = Query(None),
    skill: Optional[str] = Query(None),
    task_type: Optional[str] = Query(None),
    limit: int = Query(50),
):
    """查询任务列表。"""
    tm = request.app.state.task_manager
    tasks = tm.list_tasks(
        status=status,
        skill=skill,
        task_type=task_type,
        limit=limit,
    )
    return {"tasks": [t.to_dict() for t in tasks], "count": len(tasks)}


@router.get("/tasks/concurrent-status")
def get_concurrent_status(request: Request):
    """获取并发状态。"""
    tm = request.app.state.task_manager
    return tm.get_concurrent_status()


@router.get("/tasks/queue")
def get_queue(request: Request):
    """获取排队中的任务。"""
    tm = request.app.state.task_manager
    queued = tm.list_tasks(status="queued")
    return {"queue": [t.to_dict() for t in queued]}


@router.get("/tasks/{task_id}")
def get_task(task_id: str, request: Request):
    """获取任务详情。"""
    tm = request.app.state.task_manager
    task = tm.get_task(task_id)
    if not task:
        return {"error": "task not found"}
    events = tm.get_progress_events(task_id)
    return {"task": task.to_dict(), "events": events}


@router.get("/tasks/{task_id}/progress")
async def task_progress_stream(
    task_id: str,
    request: Request,
    last_seq: int = Query(-1, description="客户端已收到的最大序列号"),
):
    """SSE 进度流。"""
    tm = request.app.state.task_manager
    task = tm.get_task(task_id)
    if not task:
        return {"error": "task not found"}

    async def _stream():
        nonlocal task
        while task.status in ("pending", "queued", "running"):
            events = tm.get_progress_events(task_id, after_seq=last_seq)
            for ev in events:
                yield {"data": json.dumps(ev)}
                last_seq = ev["seq"]
            await asyncio.sleep(0.5)
            # 重新加载任务状态
            task = tm.get_task(task_id)
            if not task:
                break

        # 发送剩余事件
        events = tm.get_progress_events(task_id, after_seq=last_seq)
        for ev in events:
            yield {"data": json.dumps(ev)}

    return EventSourceResponse(_stream())


@router.post("/tasks/{task_id}/pause")
def pause_task(task_id: str, request: Request):
    """暂停任务。"""
    tm = request.app.state.task_manager
    success = tm.pause(task_id)
    return {"success": success}


@router.post("/tasks/{task_id}/resume")
def resume_task(task_id: str, request: Request):
    """恢复暂停的任务。"""
    tm = request.app.state.task_manager
    try:
        status = tm.resume(task_id)
        return {"success": True, "status": status}
    except ValueError as e:
        return {"error": str(e)}


@router.post("/tasks/{task_id}/retry")
def retry_task(task_id: str, request: Request):
    """重试失败或中断的任务。"""
    tm = request.app.state.task_manager
    try:
        status = tm.retry(task_id)
        return {"success": True, "status": status}
    except ValueError as e:
        return {"error": str(e)}


@router.delete("/tasks/{task_id}")
def cancel_task(task_id: str, request: Request):
    """取消任务。"""
    tm = request.app.state.task_manager
    success = tm.cancel(task_id)
    return {"success": success}


@router.delete("/tasks/{task_id}/delete")
def delete_task(task_id: str, request: Request):
    """删除任务（仅允许已结束的任务）。"""
    tm = request.app.state.task_manager
    success = tm.delete(task_id)
    if not success:
        return {"success": False, "error": "任务不存在或状态不允许删除"}
    return {"success": True}


@router.get("/tasks/{task_id}/log")
def get_task_log(task_id: str, request: Request):
    """获取任务执行日志。"""
    tm = request.app.state.task_manager
    task = tm.get_task(task_id)
    if not task:
        return {"error": "task not found"}

    if not task.output_dir:
        return {"error": "no output directory"}

    output_dir = Path(task.output_dir)
    from ...core.utils import load_json

    if task.task_type == "eval":
        # 尝试找 eval.json 或 progress.jsonl
        eval_file = output_dir / "eval.json"
        if eval_file.exists():
            return {"type": "json", "content": load_json(eval_file)}

        # 尝试找 timestamp 子目录（新版 eval 输出）
        subdirs = sorted(output_dir.iterdir(), key=lambda d: d.name, reverse=True)
        for subdir in subdirs:
            if subdir.is_dir():
                eval_file = subdir / "eval.json"
                if eval_file.exists():
                    return {"type": "json", "path": str(eval_file), "content": load_json(eval_file)}

        progress_file = output_dir / "eval_progress.jsonl"
        if progress_file.exists():
            lines = progress_file.read_text(encoding="utf-8").strip().split("\n")
            records = [json.loads(l) for l in lines if l]
            return {"type": "jsonl", "path": str(progress_file), "records": records}

        return {"error": "log not found"}

    elif task.task_type == "evolve":
        # 找 evolve_log.json
        log_file = output_dir / "evolve_log.json"
        if log_file.exists():
            return {"type": "json", "path": str(log_file), "content": load_json(log_file)}

        # 尝试找 timestamp 子目录
        subdirs = sorted(output_dir.iterdir(), key=lambda d: d.name, reverse=True)
        for subdir in subdirs:
            if subdir.is_dir():
                log_file = subdir / "evolve_log.json"
                if log_file.exists():
                    return {"type": "json", "path": str(log_file), "content": load_json(log_file)}

        return {"error": "log not found"}

    elif task.task_type == "create":
        # create 任务没有详细日志，只有结果路径
        if task.result:
            return {"type": "result", "content": task.result}
        return {"error": "no result yet"}

    return {"error": "unknown task type"}


@router.get("/tasks/{task_id}/download")
def download_task_results(task_id: str, request: Request):
    """下载任务结果目录为 zip 文件。"""
    tm = request.app.state.task_manager
    task = tm.get_task(task_id)
    if not task:
        return {"error": "task not found"}

    if task.task_type not in ("eval", "evolve"):
        return {"error": "仅支持下载 eval/evolve 任务结果"}

    if not task.output_dir:
        return {"error": "任务未设置输出目录"}

    output_dir = Path(task.output_dir)

    # 确定 actual_path（处理 evolve 的 timestamp 子目录）
    if task.task_type == "evolve" and task.result and task.result.get("run_id"):
        run_id = task.result["run_id"]
        # 新格式: output_dir 已含时间戳; 旧格式: output_dir 是父目录
        actual_path = output_dir if output_dir.name == run_id else output_dir / run_id
    else:
        actual_path = output_dir

    if not actual_path.exists():
        return {"error": "任务结果目录不存在，可能任务尚未开始或执行失败"}

    # 打包为 zip
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(actual_path.rglob("*")):
            if not f.is_file() or f.name.startswith("."):
                continue
            arcname = f.relative_to(actual_path)
            zf.write(f, arcname)
    buf.seek(0)

    # 生成文件名
    skill_name = task.skill or "unknown"
    timestamp = actual_path.name if task.task_type == "evolve" else output_dir.name
    filename = f"{skill_name}-{task.task_type}-{timestamp}.zip"

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
"""Evolve API：启动演化任务 + SSE 进度流 + 版本管理。"""

from __future__ import annotations

import asyncio
import difflib
import io
import json
import shutil
import time
import zipfile
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
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
    speed: str = "low"
    ignore_cache: bool = False
    debug: bool = False
    save_trace: bool = False
    test_cases_file: str | None = None


@router.post("/evolve")
def start_evolve(req: EvolveRequest, request: Request):
    """启动技能演化任务。"""
    tm = request.app.state.task_manager

    skill_path = Path.cwd() / "skills" / req.skill
    if not (skill_path / "SKILL.md").exists():
        return {"error": f"Skill not found: {req.skill}"}

    spec_path = None
    if req.spec:
        spec_path = str(Path.cwd() / "specs" / req.spec)

    output_dir = Path.cwd() / "results" / req.skill / "evolve"

    # 预计算 timestamp，确保 watcher 和 executor 使用同一个 run_dir
    import time as _time
    timestamp = _time.strftime("%Y%m%d%H%M")
    run_dir = output_dir / timestamp

    # 构建参数
    params = {
        "skill_path": str(skill_path),
        "spec_path": spec_path,
        "output_dir": str(output_dir),
        "threshold": req.threshold,
        "trials": req.trials,
        "parallel": req.parallel,
        "max_iterations": req.max_iterations,
        "patience": req.patience,
        "mode": req.mode,
        "speed": req.speed,
        "ignore_cache": req.ignore_cache,
        "test_cases_file": req.test_cases_file,
        "debug": req.debug,
        "save_trace": req.save_trace,
        "run_id": timestamp,
    }

    # 创建任务
    task = tm.create_task("evolve", skill=req.skill, params=params, output_dir=str(run_dir))

    # 发送初始状态
    tm.emit_progress(task.id, "status", {"message": "Starting evolution..."})

    # 提交执行
    status = tm.submit(task.id)

    # 启动 evolve_log 轮询（监听本次 run_dir，避免读取旧 run 的日志）
    _start_evolve_watcher(tm, task, run_dir)

    return {"task_id": task.id, "status": status}


def _start_evolve_watcher(tm, task, run_dir: Path):
    """轮询 run_dir/evolve_log.json 获取迭代进度。"""
    seen_iters = 0
    task_id = task.id

    def _watch():
        nonlocal seen_iters
        while True:
            current = tm.get_task(task_id)
            if not current or current.status not in ("pending", "queued", "running"):
                break
            try:
                log_file = run_dir / "evolve_log.json"
                if log_file.exists():
                    data = json.loads(log_file.read_text(encoding="utf-8"))
                    history = data.get("history", [])
                    if len(history) > seen_iters:
                        for record in history[seen_iters:]:
                            tm.emit_progress(task_id, "iteration", record)
                        seen_iters = len(history)
            except Exception:
                pass
            time.sleep(2)

    tm._executor.submit(_watch)


@router.get("/evolve/{task_id}/progress")
async def evolve_progress(task_id: str, request: Request):
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


# ===== 版本管理 API（保留原有功能）=====

_CWD = Path.cwd()


def _resolve_skill_dir(skill_name: str, run_id: str, version: str) -> Path | None:
    """解析演化版本对应的技能目录路径。"""
    run_dir = _CWD / "results" / skill_name / "evolve" / run_id
    if not run_dir.exists():
        return None
    if version == "baseline":
        candidates = list((run_dir / "baseline_skill").iterdir())
        return candidates[0] if candidates else None
    else:
        for d in run_dir.iterdir():
            if d.is_dir() and d.name == version:
                # 过滤掉 trace 目录，只返回技能目录
                skill_dirs = [
                    c for c in d.iterdir()
                    if c.is_dir() and c.name != "trace"
                ]
                return skill_dirs[0] if skill_dirs else None
    return None


@router.get("/evolve-versions/{skill_name}/{run_id}")
def list_evolve_versions(skill_name: str, run_id: str):
    """列出某次演化运行中所有可用版本。"""
    run_dir = _CWD / "results" / skill_name / "evolve" / run_id
    if not run_dir.exists():
        return {"error": f"Run not found: {run_id}"}

    log_file = run_dir / "evolve_log.json"
    log_data = {}
    if log_file.exists():
        log_data = json.loads(log_file.read_text(encoding="utf-8"))

    history = {h["iteration"]: h for h in log_data.get("history", [])}
    best_iter = log_data.get("best_iter", 0)
    baseline_rate = log_data.get("baseline_rate", 0)

    versions = []

    baseline_dir = _resolve_skill_dir(skill_name, run_id, "baseline")
    if baseline_dir and baseline_dir.exists():
        versions.append({
            "id": "baseline",
            "label": "Baseline",
            "rate": baseline_rate,
            "accepted": False,
            "best": best_iter == 0,
        })

    for d in sorted(run_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("iter-"):
            continue
        iter_num = int(d.name.split("-")[1])
        skill_dirs = [c for c in d.iterdir() if c.is_dir()]
        if not skill_dirs:
            continue
        h = history.get(iter_num, {})
        skip = h.get("skip_reason", "")
        versions.append({
            "id": d.name,
            "label": f"Iter {iter_num}",
            "rate": h.get("evolved_rate", 0),
            "accepted": h.get("accepted", False),
            "best": best_iter == iter_num,
            "strategy": h.get("strategy", ""),
            "summary": h.get("summary", ""),
            "skip_reason": skip,
        })

    return {"skill": skill_name, "run_id": run_id, "versions": versions}


@router.get("/evolve-diff/{skill_name}/{run_id}", response_class=PlainTextResponse)
def get_evolve_diff(skill_name: str, run_id: str, frm: str = "baseline", to: str = ""):
    """返回两个版本之间的 unified diff。"""
    from ...evolve.guards import build_diff_text, snapshot_files

    if not to:
        return PlainTextResponse("缺少 to 参数", status_code=400)

    before_dir = _resolve_skill_dir(skill_name, run_id, frm)
    after_dir = _resolve_skill_dir(skill_name, run_id, to)

    if not before_dir or not before_dir.exists():
        return PlainTextResponse(f"版本不存在: {frm}", status_code=404)
    if not after_dir or not after_dir.exists():
        return PlainTextResponse(f"版本不存在: {to}", status_code=404)

    before = snapshot_files(before_dir)
    after = snapshot_files(after_dir)
    diff = build_diff_text(before, after)
    return diff


@router.get("/evolve-diff-json/{skill_name}/{run_id}")
def get_evolve_diff_json(skill_name: str, run_id: str, frm: str = "baseline", to: str = ""):
    """返回两个版本之间的结构化 diff，用于左右对比展示。"""
    from ...evolve.guards import snapshot_files

    if not to:
        return {"error": "缺少 to 参数"}

    before_dir = _resolve_skill_dir(skill_name, run_id, frm)
    after_dir = _resolve_skill_dir(skill_name, run_id, to)

    if not before_dir or not before_dir.exists():
        return {"error": f"版本不存在: {frm}"}
    if not after_dir or not after_dir.exists():
        return {"error": f"版本不存在: {to}"}

    before = snapshot_files(before_dir)
    after = snapshot_files(after_dir)

    all_files = sorted(set(before) | set(after))
    files = []

    for fname in all_files:
        old_lines = (before.get(fname) or "").splitlines()
        new_lines = (after.get(fname) or "").splitlines()

        if old_lines == new_lines:
            continue

        if not old_lines:
            status = "added"
        elif not new_lines:
            status = "deleted"
        else:
            status = "modified"

        sm = difflib.SequenceMatcher(None, old_lines, new_lines)
        lines = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for k in range(i2 - i1):
                    lines.append({
                        "type": "equal",
                        "left_num": i1 + k + 1,
                        "right_num": j1 + k + 1,
                        "left": old_lines[i1 + k],
                        "right": new_lines[j1 + k],
                    })
            elif tag == "delete":
                for k in range(i1, i2):
                    lines.append({
                        "type": "delete",
                        "left_num": k + 1,
                        "right_num": None,
                        "left": old_lines[k],
                        "right": "",
                    })
            elif tag == "insert":
                for k in range(j1, j2):
                    lines.append({
                        "type": "insert",
                        "left_num": None,
                        "right_num": k + 1,
                        "left": "",
                        "right": new_lines[k],
                    })
            elif tag == "replace":
                # 交错显示删除和插入行
                left_range = list(range(i1, i2))
                right_range = list(range(j1, j2))
                max_len = max(len(left_range), len(right_range))
                for k in range(max_len):
                    li = left_range[k] if k < len(left_range) else None
                    ri = right_range[k] if k < len(right_range) else None
                    lines.append({
                        "type": "replace",
                        "left_num": li + 1 if li is not None else None,
                        "right_num": ri + 1 if ri is not None else None,
                        "left": old_lines[li] if li is not None else "",
                        "right": new_lines[ri] if ri is not None else "",
                    })

        files.append({"name": fname, "status": status, "lines": lines})

    return {"files": files}


@router.get("/evolve-download/{skill_name}/{run_id}/{version}")
def download_evolve_version(skill_name: str, run_id: str, version: str):
    """下载指定版本的技能目录为 zip。"""
    skill_dir = _resolve_skill_dir(skill_name, run_id, version)
    if not skill_dir or not skill_dir.exists():
        return {"error": f"版本不存在: {version}"}

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(skill_dir.rglob("*")):
            if not f.is_file() or f.name.startswith("."):
                continue
            arcname = f.relative_to(skill_dir)
            zf.write(f, arcname)
    buf.seek(0)

    filename = f"{skill_name}-{version}.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/evolve-apply/{skill_name}/{run_id}/{version}")
def apply_evolve_version(skill_name: str, run_id: str, version: str):
    """将指定版本应用到当前技能目录（覆盖 skills/{skill_name}/）。"""
    skill_dir = _resolve_skill_dir(skill_name, run_id, version)
    if not skill_dir or not skill_dir.exists():
        return {"error": f"版本不存在: {version}"}

    target = _CWD / "skills" / skill_name
    if not target.exists():
        return {"error": f"当前技能目录不存在: {skill_name}"}

    backup_dir = _CWD / "results" / skill_name / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d%H%M%S")
    backup_path = backup_dir / f"pre-apply-{version}-{ts}"
    shutil.copytree(target, backup_path)

    for item in target.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    for item in skill_dir.iterdir():
        if item.is_dir():
            shutil.copytree(item, target / item.name)
        else:
            shutil.copy2(item, target / item.name)

    return {"status": "ok", "applied": version, "backup": str(backup_path.relative_to(_CWD))}
"""Evolve API：启动演化任务 + SSE 进度流 + 版本管理。"""

from __future__ import annotations

import asyncio
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


# ===== 版本管理 =====

_CWD = Path.cwd()


def _resolve_skill_dir(skill_name: str, run_id: str, version: str) -> Path | None:
    """解析演化版本对应的技能目录路径。"""
    run_dir = _CWD / "results" / skill_name / "evolve" / run_id
    if not run_dir.exists():
        return None
    if version == "baseline":
        # baseline_skill/{skill_name}/
        candidates = list((run_dir / "baseline_skill").iterdir())
        return candidates[0] if candidates else None
    else:
        # iter-{n}/{skill_name}/
        for d in run_dir.iterdir():
            if d.is_dir() and d.name == version:
                skill_dirs = [c for c in d.iterdir() if c.is_dir()]
                return skill_dirs[0] if skill_dirs else None
    return None


@router.get("/evolve-versions/{skill_name}/{run_id}")
def list_evolve_versions(skill_name: str, run_id: str):
    """列出某次演化运行中所有可用版本。"""
    run_dir = _CWD / "results" / skill_name / "evolve" / run_id
    if not run_dir.exists():
        return {"error": f"Run not found: {run_id}"}

    # 读取 evolve_log 获取元数据
    log_file = run_dir / "evolve_log.json"
    log_data = {}
    if log_file.exists():
        log_data = json.loads(log_file.read_text(encoding="utf-8"))

    history = {h["iteration"]: h for h in log_data.get("history", [])}
    best_iter = log_data.get("best_iter", 0)
    baseline_rate = log_data.get("baseline_rate", 0)

    versions = []

    # baseline
    baseline_dir = _resolve_skill_dir(skill_name, run_id, "baseline")
    if baseline_dir and baseline_dir.exists():
        versions.append({
            "id": "baseline",
            "label": "Baseline",
            "rate": baseline_rate,
            "accepted": False,
            "best": best_iter == 0,
        })

    # iterations
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

    # 备份当前版本
    backup_dir = _CWD / "results" / skill_name / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d%H%M%S")
    backup_path = backup_dir / f"pre-apply-{version}-{ts}"
    shutil.copytree(target, backup_path)

    # 覆盖：先清空再复制
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

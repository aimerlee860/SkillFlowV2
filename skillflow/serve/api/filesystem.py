"""文件系统浏览 API。"""

from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path

from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

router = APIRouter(tags=["filesystem"])

_CWD = Path.cwd()


def _safe_path(base: str, *parts: str) -> Path:
    """解析路径并确保在 CWD 下。"""
    p = (Path(base) / Path(*parts)).resolve()
    return p


@router.get("/skills")
def list_skills():
    """列出所有技能目录。"""
    skills_dir = _CWD / "skills"
    if not skills_dir.exists():
        return []
    result = []
    for d in sorted(skills_dir.iterdir()):
        if d.is_dir() and (d / "SKILL.md").exists():
            result.append({"name": d.name, "path": str(d.relative_to(_CWD))})
    return result


@router.get("/specs")
def list_specs():
    """列出所有 SPEC YAML 文件。"""
    specs_dir = _CWD / "specs"
    if not specs_dir.exists():
        return []
    result = []
    for f in sorted(specs_dir.iterdir()):
        if f.is_file() and f.suffix in (".yaml", ".yml"):
            result.append({"name": f.name, "path": str(f.relative_to(_CWD))})
    return result


class SaveSpecRequest(BaseModel):
    name: str
    content: str


@router.post("/specs")
def save_spec(req: SaveSpecRequest):
    """保存 spec 内容到 specs/ 目录。"""
    specs_dir = _CWD / "specs"
    specs_dir.mkdir(parents=True, exist_ok=True)
    # 确保文件名以 .yaml 结尾
    filename = req.name
    if not filename.endswith((".yaml", ".yml")):
        filename += ".yaml"
    path = specs_dir / filename
    # 安全检查：确保路径在 specs/ 下
    path = path.resolve()
    if not str(path).startswith(str(specs_dir.resolve())):
        return {"error": "invalid path"}
    path.write_text(req.content, encoding="utf-8")
    return {"name": filename, "path": str(path.relative_to(_CWD))}


@router.get("/results")
def list_results():
    """列出所有评估/演化结果。"""
    results_dir = _CWD / "results"
    if not results_dir.exists():
        return []
    result = []
    for d in sorted(results_dir.iterdir()):
        if d.is_dir():
            eval_runs = []
            evolve_runs = []
            eval_dir = d / "eval"
            evolve_dir = d / "evolve"
            if eval_dir.exists():
                for run in sorted(eval_dir.iterdir()):
                    if run.is_dir() and (run / "eval.json").exists():
                        eval_runs.append({"id": run.name, "path": str(run.relative_to(_CWD))})
            if evolve_dir.exists():
                for run in sorted(evolve_dir.iterdir()):
                    if run.is_dir() and (run / "evolve_log.json").exists():
                        evolve_runs.append({"id": run.name, "path": str(run.relative_to(_CWD))})
            result.append({
                "skill": d.name,
                "eval_runs": eval_runs,
                "evolve_runs": evolve_runs,
            })
    return result


@router.get("/skill-content/{name}", response_class=PlainTextResponse)
def get_skill_content(name: str):
    """返回 SKILL.md 内容。"""
    path = _CWD / "skills" / name / "SKILL.md"
    if not path.exists():
        return PlainTextResponse(f"Skill '{name}' not found", status_code=404)
    return path.read_text(encoding="utf-8")


@router.get("/spec-content/{name}", response_class=PlainTextResponse)
def get_spec_content(name: str):
    """返回 spec YAML 内容。"""
    path = _CWD / "specs" / name
    if not path.exists():
        return PlainTextResponse(f"Spec '{name}' not found", status_code=404)
    return path.read_text(encoding="utf-8")


@router.get("/result/{skill_name}/eval/{run_id}")
def get_eval_result(skill_name: str, run_id: str):
    """返回某次评估结果。"""
    path = _CWD / "results" / skill_name / "eval" / run_id / "eval.json"
    if not path.exists():
        return {"error": "not found"}
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/result/{skill_name}/evolve/{run_id}")
def get_evolve_result(skill_name: str, run_id: str):
    """返回某次演化结果。"""
    path = _CWD / "results" / skill_name / "evolve" / run_id / "evolve_log.json"
    if not path.exists():
        return {"error": "not found"}
    return json.loads(path.read_text(encoding="utf-8"))


# ===== 测试用例管理 =====

_TC_DIR = "test_cases"  # results/{skill}/test_cases/{name}.json


@router.get("/test-cases/{skill_name}")
def list_test_cases(skill_name: str):
    """列出某技能的所有已保存测试用例名称。"""
    tc_dir = _CWD / "results" / skill_name / _TC_DIR
    if not tc_dir.exists():
        return []
    items = [
        {"name": f.stem, "path": str(f.relative_to(_CWD))}
        for f in tc_dir.iterdir()
        if f.is_file() and f.suffix == ".json"
    ]
    return sorted(items, key=lambda x: x["name"])


@router.get("/test-cases/{skill_name}/{tc_name}")
def get_test_cases(skill_name: str, tc_name: str):
    """获取某个具体的测试用例内容。"""
    path = _CWD / "results" / skill_name / _TC_DIR / f"{tc_name}.json"
    path = path.resolve()
    results_root = (_CWD / "results").resolve()
    if not str(path).startswith(str(results_root)):
        return {"error": "invalid path"}
    if not path.exists():
        return {"error": "not found"}
    return json.loads(path.read_text(encoding="utf-8"))


class SaveTestCasesRequest(BaseModel):
    name: str  # 测试用例名称
    test_cases: list[dict]


@router.post("/test-cases/{skill_name}")
def save_test_cases(skill_name: str, req: SaveTestCasesRequest):
    """保存测试用例到 results/{skill_name}/test_cases/{name}.json。"""
    tc_dir = _CWD / "results" / skill_name / _TC_DIR
    tc_dir.mkdir(parents=True, exist_ok=True)
    filename = req.name
    if not filename.endswith(".json"):
        filename += ".json"
    path = (tc_dir / filename).resolve()
    results_root = (_CWD / "results").resolve()
    if not str(path).startswith(str(results_root)):
        return {"error": "invalid path"}
    path.write_text(
        json.dumps({"test_cases": req.test_cases}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"name": filename, "saved": len(req.test_cases)}


class GenerateTestCasesRequest(BaseModel):
    spec: str | None = None
    ignore_cache: bool = False


@router.post("/test-cases/{skill_name}/generate")
def generate_test_cases_api(skill_name: str, req: GenerateTestCasesRequest, request: Request):
    """仅生成测试用例（不运行 eval，不保存），后台任务。"""
    tm = request.app.state.task_manager

    skill_path = _CWD / "skills" / skill_name
    if not (skill_path / "SKILL.md").exists():
        return {"error": f"Skill not found: {skill_name}"}

    spec_path = None
    if req.spec:
        spec_path = str(_CWD / "specs" / req.spec)

    task = tm.create_task("generate_test_cases")

    def _run():
        from ...eval.test_generator import generate_test_cases

        try:
            task.progress_events.append(
                {"type": "status", "message": "Generating test cases..."}
            )
            import tempfile
            test_cases, _ = generate_test_cases(
                skill_path=skill_path,
                spec_path=spec_path,
                output_dir=tempfile.mkdtemp(),
                ignore_cache=req.ignore_cache,
            )
            task.result = {"test_cases": test_cases, "count": len(test_cases)}
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

    task.progress_events.append(
        {"type": "status", "message": "Starting test case generation..."}
    )
    task.status = "running"
    tm._executor.submit(_run)

    return {"task_id": task.id, "status": "running"}


@router.get("/test-cases/{skill_name}/generate/{task_id}/progress")
async def generate_tc_progress(task_id: str, request: Request):
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
            import asyncio
            await asyncio.sleep(0.5)
        while last_idx < len(task.progress_events):
            yield {"data": json.dumps(task.progress_events[last_idx])}
            last_idx += 1

    from sse_starlette.sse import EventSourceResponse
    return EventSourceResponse(_stream())


# ===== 技能上传 =====


def _parse_frontmatter_name(text: str) -> str | None:
    """从 SKILL.md 的 YAML frontmatter 中提取 name 字段。"""
    m = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    if not m:
        return None
    for line in m.group(1).splitlines():
        if line.startswith("name:"):
            val = line.split(":", 1)[1].strip().strip("\"'")
            if val:
                return val
    return None


@router.post("/upload-skill")
async def upload_skill(
    file: UploadFile = File(...),
    name: str = Form(""),
):
    """上传 zip 格式的技能目录到 skills/。"""
    # 1. 校验文件后缀
    filename = file.filename or ""
    if not filename.lower().endswith(".zip"):
        return {"error": "仅支持 .zip 文件"}

    # 2. 读取 zip
    content = await file.read()
    try:
        zf = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile:
        return {"error": "无效的 zip 文件"}

    # 3. 查找 SKILL.md
    names = zf.namelist()
    skill_md_candidates = [n for n in names if n.rstrip("/").endswith("/SKILL.md") or n == "SKILL.md"]
    if not skill_md_candidates:
        return {"error": "zip 中未找到 SKILL.md 文件"}

    skill_md_path = skill_md_candidates[0]

    # 4. 确定技能名和内部前缀
    #    如果 SKILL.md 在子目录中（如 my-skill/SKILL.md），前缀为 "my-skill/"
    #    如果在根目录，前缀为空
    if "/" in skill_md_path:
        prefix = skill_md_path.rsplit("SKILL.md", 1)[0]  # e.g. "my-skill/"
        dir_name = prefix.rstrip("/").split("/")[-1]
    else:
        prefix = ""
        dir_name = ""

    # 提取名称优先级：用户指定 > frontmatter name > 目录名
    skill_md_text = zf.read(skill_md_path).decode("utf-8")
    frontmatter_name = _parse_frontmatter_name(skill_md_text)

    skill_name = name.strip() or frontmatter_name or dir_name
    if not skill_name:
        return {"error": "无法确定技能名称，请在 zip 中包含顶层目录或手动指定名称"}

    # 5. 名称安全校验
    if not re.match(r"^[a-zA-Z0-9_\-\u4e00-\u9fff]+$", skill_name):
        return {"error": f"技能名称不合法: {skill_name}（仅允许字母、数字、下划线、横线、中文）"}

    # 6. 检查冲突
    target_dir = _CWD / "skills" / skill_name
    if target_dir.exists():
        return {"error": f"技能 '{skill_name}' 已存在，请先删除或使用其他名称"}

    # 7. 解压
    target_dir.mkdir(parents=True, exist_ok=True)
    extracted_files = []
    for member in names:
        # 安全检查：防止路径穿越
        member_path = (target_dir / member).resolve()
        if not str(member_path).startswith(str(target_dir.resolve())):
            continue
        # 跳过前缀目录
        if prefix:
            if not member.startswith(prefix):
                continue
            relative = member[len(prefix):]
        else:
            relative = member
        if not relative:
            continue
        if member.endswith("/"):
            (target_dir / relative).mkdir(parents=True, exist_ok=True)
        else:
            dest = target_dir / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(zf.read(member))
            extracted_files.append(relative)

    zf.close()
    return {"name": skill_name, "files": extracted_files}

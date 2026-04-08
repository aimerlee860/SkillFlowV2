"""演化守护机制：case 回归检测、diff 检测、LLM 改动判断。"""

from __future__ import annotations

import difflib
import json
import re
from pathlib import Path

from rich.console import Console

from ..core.llm import get_llm
from ..core.prompts import CHANGE_JUDGE_PROMPT
from ..core.utils import load_text

console = Console()

# 跳过的目录（与 mutator 保持一致）
_SKIP_DIRS = {"assets", "__pycache__", ".git", "node_modules"}

# 判定为小改动的 diff 比例阈值
SMALL_DIFF_THRESHOLD = 0.02


def check_regression(
    prev_cases: list[dict],
    curr_cases: list[dict],
) -> tuple[bool, list[str]]:
    """检查是否存在 case 级回归。

    之前 pass_rate >= 1.0 的 case，本轮 pass_rate < 1.0 即为回归。

    Args:
        prev_cases: 上一轮（最优版本）eval 结果的 test_cases
        curr_cases: 本轮 eval 结果的 test_cases

    Returns:
        (has_regression, details)
    """
    prev_map = {c["test_point"]: c for c in prev_cases}
    curr_map = {c["test_point"]: c for c in curr_cases}

    regressions = []
    for tp, prev in prev_map.items():
        curr = curr_map.get(tp)
        if not curr:
            continue
        if prev["pass_rate"] >= 1.0 and curr["pass_rate"] < 1.0:
            regressions.append(
                f"{tp}: {prev['pass_rate']:.0%} -> {curr['pass_rate']:.0%}"
            )

    return len(regressions) > 0, regressions


def snapshot_files(skill_path: Path) -> dict[str, str]:
    """读取技能目录下所有文本文件，返回快照。

    Returns:
        {相对路径: 文件内容}
    """
    files: dict[str, str] = {}
    skill_md = skill_path / "SKILL.md"
    if skill_md.exists():
        files["SKILL.md"] = load_text(skill_md)

    for f in sorted(skill_path.rglob("*")):
        if not f.is_file() or f.name.startswith("."):
            continue
        rel = str(f.relative_to(skill_path))
        if any(part in _SKIP_DIRS for part in Path(rel).parts):
            continue
        if rel == "SKILL.md":
            continue
        try:
            files[rel] = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError):
            pass

    return files


def compute_diff_ratio(before: dict[str, str], after: dict[str, str]) -> float:
    """计算文件变更比例。

    Returns:
        有效变更比例 (0.0 ~ 1.0)
    """
    all_files = set(before) | set(after)
    if not all_files:
        return 0.0

    total_ratio = 0.0
    for f in all_files:
        old = before.get(f, "").splitlines(keepends=True)
        new = after.get(f, "").splitlines(keepends=True)
        diff_lines = list(difflib.unified_diff(old, new, n=0))
        changed = sum(
            1 for line in diff_lines if line.startswith("+") or line.startswith("-")
        )
        # 跳过 unified_diff 头部行（---, +++）
        header_lines = sum(
            1 for line in diff_lines if line.startswith("---") or line.startswith("+++")
        )
        changed -= header_lines
        total_lines = max(len(old), len(new), 1)
        total_ratio += max(changed, 0) / total_lines

    return total_ratio / len(all_files)


def build_diff_text(before: dict[str, str], after: dict[str, str]) -> str:
    """生成人类可读的 diff 文本，供 LLM 判断。"""
    all_files = sorted(set(before) | set(after))
    parts = []
    for f in all_files:
        old = before.get(f, "").splitlines(keepends=True)
        new = after.get(f, "").splitlines(keepends=True)
        diff = list(difflib.unified_diff(old, new, fromfile=f, tofile=f, n=2))
        if diff:
            parts.append("".join(diff))
    return "\n".join(parts) if parts else "(无变化)"


def llm_judge_change(before: dict[str, str], after: dict[str, str]) -> bool:
    """当 diff 较小时，调用 LLM 判断改动是否有实质意义。

    Returns:
        True 表示有意义的改动，False 表示表面改动
    """
    diff_text = build_diff_text(before, after)
    if diff_text == "(无变化)":
        return False

    prompt = CHANGE_JUDGE_PROMPT.format(diff_content=diff_text)
    llm = get_llm()
    resp = llm.invoke(prompt)
    text = resp.content if hasattr(resp, "content") else str(resp)

    # 提取 JSON
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        text = match.group(1)
    try:
        result = json.loads(text.strip())
        return bool(result.get("meaningful", False))
    except (json.JSONDecodeError, AttributeError):
        # 解析失败时保守放行
        console.print("[yellow]LLM 改动判断解析失败，保守放行[/yellow]")
        return True


def should_run_eval(
    before: dict[str, str],
    after: dict[str, str],
) -> tuple[bool, str]:
    """判断是否应该运行评估。

    Returns:
        (should_run, reason)
    """
    diff_ratio = compute_diff_ratio(before, after)

    if diff_ratio >= SMALL_DIFF_THRESHOLD:
        return True, f"diff_ratio={diff_ratio:.4f}"

    # diff 较小，调 LLM 语义判断
    is_meaningful = llm_judge_change(before, after)
    if is_meaningful:
        return True, f"diff_ratio={diff_ratio:.4f} but semantically meaningful"

    return False, f"trivial change (diff_ratio={diff_ratio:.4f})"

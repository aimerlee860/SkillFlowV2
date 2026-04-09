"""技能变异器：审视（全文件）+ 反思（失败用例）+ 范例提取。"""

from __future__ import annotations

import re
from pathlib import Path

from rich.console import Console

from ..core.llm import get_llm
from ..core.prompts import (
    EVOLVE_AUDIT_PROMPT,
    EVOLVE_REFLECT_PROMPT,
    EVOLVE_REWRITE_PROMPT,
    build_speed_constraint,
)
from ..core.utils import load_text, save_text
from .exemplar import (
    extract_exemplars,
    format_exemplars_instruction,
    format_exemplars_section,
)

console = Console()

# 审视时跳过的目录
_SKIP_DIRS = {"assets", "__pycache__", ".git", "node_modules"}


def _clean_markdown_wrapping(text: str) -> str:
    """去除 LLM 输出中可能的 markdown 代码块包裹。"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text


def _read_skill_files(skill_path: Path) -> str:
    """读取技能目录下所有文本文件，返回格式化内容。"""
    parts = []

    # SKILL.md 放在最前面
    skill_md = skill_path / "SKILL.md"
    if skill_md.exists():
        parts.append(f"### FILE: SKILL.md\n{load_text(skill_md)}")

    # 其余文件按路径排序，保证确定性
    for f in sorted(skill_path.rglob("*")):
        if not f.is_file() or f.name.startswith("."):
            continue
        rel = f.relative_to(skill_path)
        if any(part in _SKIP_DIRS for part in rel.parts):
            continue
        if f == skill_md:
            continue
        try:
            content = f.read_text(encoding="utf-8")
            parts.append(f"### FILE: {rel}\n{content}")
        except (UnicodeDecodeError, PermissionError):
            pass

    return "\n\n".join(parts)


def _parse_audit_output(text: str) -> tuple[str, dict[str, str]]:
    """解析审视输出，分离分析文本和修改后的文件。

    支持两种格式：
    1. SEARCH/REPLACE 块格式（推荐）：只输出被替换的片段
    2. 完整文件格式（兼容旧版）：输出完整文件内容

    Returns:
        (analysis_text, {relative_path: new_content})
    """
    pattern = r"<<<FILE:(.+?)>>>\n(.*?)<<<END>>>"
    matches = re.findall(pattern, text, re.DOTALL)

    if not matches:
        return text, {}

    first_marker_pos = text.index("<<<FILE:")
    analysis = text[:first_marker_pos].strip()

    modified_files = {}
    for file_path, block in matches:
        block = _clean_markdown_wrapping(block)
        rel_path = file_path.strip()

        # 检测是否为 SEARCH/REPLACE 格式
        sr_pattern = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE"
        sr_matches = re.findall(sr_pattern, block, re.DOTALL)

        if sr_matches:
            # SEARCH/REPLACE 模式：对原文做定点替换
            full_text = modified_files.get(rel_path, "")  # 支持同一文件多次出现
            if not full_text:
                # 第一次遇到该文件，从磁盘读取
                from . import _read_skill_files  # noqa: avoid circular
                # 尝试从已有的文件内容推断（这里直接用空串，后面处理）
                pass

            # 如果该文件还没有内容，跳过（后续由 apply_search_replace 处理）
            if rel_path in modified_files:
                modified_files[rel_path] = _apply_search_replace(modified_files[rel_path], sr_matches)
            else:
                modified_files[rel_path] = block  # 暂存，后续在 audit_skill 中处理
        else:
            # 完整文件模式（兼容旧版）
            modified_files[rel_path] = block

    return analysis, modified_files


def _apply_search_replace(original: str, sr_pairs: list[tuple[str, str]]) -> str:
    """对原始内容应用 SEARCH/REPLACE 替换。"""
    result = original
    for search_str, replace_str in sr_pairs:
        if search_str in result:
            result = result.replace(search_str, replace_str, 1)
    return result


# ---------- 审视分支 ----------

def audit_skill(skill_path: Path, trace_context: str = "", past_strategies: str = "", speed: float = 0.3) -> str:
    """审视技能目录所有文件，基于质量标准输出改进。

    读取所有文件 → LLM 审视 → 解析输出 → 就地写回修改的文件。

    Args:
        skill_path: 技能目录路径
        trace_context: 执行轨迹诊断文本（可选）
        past_strategies: 历史演化策略摘要（可选）
        speed: 演化速度，控制改动激进程度

    Returns:
        审视分析文本
    """
    files_content = _read_skill_files(skill_path)
    if not trace_context:
        trace_context = "（无执行轨迹数据）"
    if not past_strategies:
        past_strategies = ""
    speed_constraint = build_speed_constraint(speed)
    prompt = EVOLVE_AUDIT_PROMPT.format(
        skill_files=files_content,
        trace_context=trace_context,
        past_strategies=past_strategies,
        speed_constraint=speed_constraint,
    )

    console.print("[blue]审视技能中...[/blue]")
    llm = get_llm()
    response = llm.invoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)

    analysis, modified_files = _parse_audit_output(text)

    for rel_path, content in modified_files.items():
        file_path = skill_path / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # 检查是否为未处理的 SEARCH/REPLACE 块
        sr_pattern = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE"
        sr_matches = re.findall(sr_pattern, content, re.DOTALL)
        if sr_matches and file_path.exists():
            original = load_text(file_path)
            content = _apply_search_replace(original, sr_matches)

        save_text(file_path, content)

    n = len(modified_files)
    console.print(f"[green]审视完成，修改了 {n} 个文件[/green]")

    return analysis


# ---------- 反思分支 ----------

def reflect_on_failures(
    skill_content: str,
    failed_cases: list[dict],
    exemplars_section: str = "（无近阈值范例）",
    trace_context: str = "",
) -> str:
    """自我反思阶段：双面分析技能在失败用例上的表现。

    Args:
        skill_content: 当前 SKILL.md 内容
        failed_cases: 失败的测试用例及评估结果
        exemplars_section: 近阈值范例的格式化文本
        trace_context: 执行轨迹诊断文本（可选）

    Returns:
        反思分析文本
    """
    cases_text = "\n".join(
        f"- [{i+1}] 问题：{c['question']}\n  测试目标：{c['test_point']}"
        for i, c in enumerate(failed_cases)
    )
    reasons_text = "\n".join(
        f"- [{i+1}] {c.get('reason', '无原因')}"
        for i, c in enumerate(failed_cases)
    )

    if not trace_context:
        trace_context = "（无执行轨迹数据）"
    prompt = EVOLVE_REFLECT_PROMPT.format(
        skill_content=skill_content,
        failed_cases=cases_text,
        failure_reasons=reasons_text,
        exemplars_section=exemplars_section,
        trace_context=trace_context,
    )

    console.print("[blue]反思失败用例中...[/blue]")
    llm = get_llm()
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


def rewrite_skill(
    skill_content: str,
    reflection: str,
    exemplars_instruction: str = "",
) -> str:
    """基于反思改写技能。

    Args:
        skill_content: 当前 SKILL.md 内容
        reflection: 反思分析文本
        exemplars_instruction: 范例注入指令文本

    Returns:
        改写后的 SKILL.md 内容
    """
    prompt = EVOLVE_REWRITE_PROMPT.format(
        skill_content=skill_content,
        reflection=reflection,
        exemplars_instruction=exemplars_instruction,
    )

    console.print("[blue]改写技能中...[/blue]")
    llm = get_llm()
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


# ---------- 统一入口 ----------

def mutate_skill(
    skill_path: str | Path,
    failed_cases: list[dict] | None = None,
    eval_result: dict | None = None,
    trace_context: str = "",
    past_strategies: str = "",
    speed: float = 0.3,
) -> tuple[str, str, bool]:
    """在技能目录上就地执行审视 + 可选反思 + 可选范例提取。

    流程：
    1. 审视分支（必执行）：读取所有文件，按质量标准审视，就地写回修改
    2. 反思分支（有失败用例时执行）：基于上一轮失败用例反思，改写 SKILL.md
    3. 范例提取（有近阈值失败响应时执行）：提取优质模式注入 SKILL.md

    Args:
        skill_path: 技能目录路径（应为副本目录，会就地修改其中文件）
        failed_cases: 来自上一轮评估的失败用例（可选）
        eval_result: 上一轮完整 eval 结果，用于范例提取（可选）
        trace_context: 执行轨迹诊断文本（可选）

    Returns:
        (分析文本, 策略名称, files_changed)
        策略名称为 "audit"、"audit+reflect" 或 "audit+reflect+exploit"
        files_changed 表示是否有文件被修改
    """
    skill_path = Path(skill_path)

    # 记录修改前的 SKILL.md 用于判断 reflect/exploit 是否有实际改动
    skill_before_audit = load_text(skill_path / "SKILL.md") if (skill_path / "SKILL.md").exists() else ""

    # 分支 1：审视（必执行）
    audit_analysis = audit_skill(skill_path, trace_context=trace_context, past_strategies=past_strategies, speed=speed)
    combined = f"## 审视分析\n{audit_analysis}"
    strategy = "audit"

    # 范例提取（独立于 reflect，需要完整 eval_result）
    exemplars = []
    if eval_result and eval_result.get("test_cases"):
        exemplars = extract_exemplars(eval_result["test_cases"])
        if exemplars:
            console.print(f"[green]提取到 {len(exemplars)} 个优质范例[/green]")

    # 分支 2：反思（有失败用例时执行，在审视结果上继续改写 SKILL.md）
    if failed_cases:
        exemplars_section = format_exemplars_section(exemplars) if exemplars else "（无近阈值范例）"
        exemplars_instruction = format_exemplars_instruction(exemplars) if exemplars else ""

        skill_content = load_text(skill_path / "SKILL.md")
        reflection = reflect_on_failures(
            skill_content,
            failed_cases,
            exemplars_section=exemplars_section,
            trace_context=trace_context,
        )
        console.print("[green]反思完成[/green]")

        new_content = rewrite_skill(
            skill_content,
            reflection,
            exemplars_instruction=exemplars_instruction,
        )
        new_content = _clean_markdown_wrapping(new_content)
        save_text(skill_path / "SKILL.md", new_content)

        combined += f"\n\n## 失败用例反思\n{reflection}"
        strategy = "audit+reflect"

        if exemplars:
            strategy = "audit+reflect+exploit"

    # 判断是否有文件被修改
    skill_after = load_text(skill_path / "SKILL.md") if (skill_path / "SKILL.md").exists() else ""
    # audit 可能改了其他文件但没改 SKILL.md，通过 audit_skill 的返回判断
    files_changed = skill_before_audit != skill_after or _audit_modified_files(audit_analysis)

    return combined, strategy, files_changed


def _audit_modified_files(audit_text: str) -> bool:
    """从审视输出中判断是否修改了文件。"""
    return "<<<FILE:" in audit_text

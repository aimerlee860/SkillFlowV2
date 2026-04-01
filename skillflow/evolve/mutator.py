"""技能变异器：审视（全文件）+ 反思（失败用例）。"""

from __future__ import annotations

import re
from pathlib import Path

from rich.console import Console

from ..core.llm import get_llm
from ..core.prompts import EVOLVE_AUDIT_PROMPT, EVOLVE_REFLECT_PROMPT, EVOLVE_REWRITE_PROMPT
from ..core.utils import load_text, save_text

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
    for file_path, content in matches:
        content = _clean_markdown_wrapping(content)
        modified_files[file_path.strip()] = content

    return analysis, modified_files


# ---------- 审视分支 ----------

def audit_skill(skill_path: Path) -> str:
    """审视技能目录所有文件，基于质量标准输出改进。

    读取所有文件 → LLM 审视 → 解析输出 → 就地写回修改的文件。

    Returns:
        审视分析文本
    """
    files_content = _read_skill_files(skill_path)
    prompt = EVOLVE_AUDIT_PROMPT.format(skill_files=files_content)

    console.print("[blue]审视技能中...[/blue]")
    llm = get_llm()
    response = llm.invoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)

    analysis, modified_files = _parse_audit_output(text)

    for rel_path, content in modified_files.items():
        file_path = skill_path / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        save_text(file_path, content)

    n = len(modified_files)
    console.print(f"[green]审视完成，修改了 {n} 个文件[/green]")

    return analysis


# ---------- 反思分支 ----------

def reflect_on_failures(
    skill_content: str,
    failed_cases: list[dict],
) -> str:
    """自我反思阶段：分析技能在失败用例上的不足。

    Args:
        skill_content: 当前 SKILL.md 内容
        failed_cases: 失败的测试用例及评估结果

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

    prompt = EVOLVE_REFLECT_PROMPT.format(
        skill_content=skill_content,
        failed_cases=cases_text,
        failure_reasons=reasons_text,
    )

    console.print("[blue]反思失败用例中...[/blue]")
    llm = get_llm()
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


def rewrite_skill(
    skill_content: str,
    reflection: str,
) -> str:
    """基于反思改写技能。

    Args:
        skill_content: 当前 SKILL.md 内容
        reflection: 反思分析文本

    Returns:
        改写后的 SKILL.md 内容
    """
    prompt = EVOLVE_REWRITE_PROMPT.format(
        skill_content=skill_content,
        reflection=reflection,
    )

    console.print("[blue]改写技能中...[/blue]")
    llm = get_llm()
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


# ---------- 统一入口 ----------

def mutate_skill(
    skill_path: str | Path,
    failed_cases: list[dict] | None = None,
) -> tuple[str, str]:
    """在技能目录上就地执行审视 + 可选反思。

    流程：
    1. 审视分支（必执行）：读取所有文件，按质量标准审视，就地写回修改
    2. 反思分支（有失败用例时执行）：基于上一轮失败用例反思，改写 SKILL.md

    Args:
        skill_path: 技能目录路径（应为副本目录，会就地修改其中文件）
        failed_cases: 来自上一轮评估的失败用例（可选）

    Returns:
        (分析文本, 策略名称)
        策略名称为 "audit" 或 "audit+reflect"
    """
    skill_path = Path(skill_path)

    # 分支 1：审视（必执行）
    audit_analysis = audit_skill(skill_path)
    combined = f"## 审视分析\n{audit_analysis}"
    strategy = "audit"

    # 分支 2：反思（有失败用例时执行，在审视结果上继续改写 SKILL.md）
    if failed_cases:
        skill_content = load_text(skill_path / "SKILL.md")
        reflection = reflect_on_failures(skill_content, failed_cases)
        console.print("[green]反思完成[/green]")

        new_content = rewrite_skill(skill_content, reflection)
        new_content = _clean_markdown_wrapping(new_content)
        save_text(skill_path / "SKILL.md", new_content)

        combined += f"\n\n## 失败用例反思\n{reflection}"
        strategy = "audit+reflect"

    return combined, strategy

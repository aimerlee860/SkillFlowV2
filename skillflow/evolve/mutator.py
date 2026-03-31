"""技能变异器：自我反思 + 改写。"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from ..core.llm import get_llm
from ..core.prompts import EVOLVE_REFLECT_PROMPT, EVOLVE_REWRITE_PROMPT
from ..core.utils import load_text, save_text

console = Console()


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

    console.print("[blue]自我反思中...[/blue]")
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


def mutate_skill(
    skill_path: str | Path,
    failed_cases: list[dict],
) -> str:
    """在技能目录上就地执行自我反思 + 改写。

    Args:
        skill_path: 技能目录路径（应为副本目录，会就地修改其中文件）
        failed_cases: 失败的测试用例

    Returns:
        反思分析文本
    """
    skill_path = Path(skill_path)
    skill_content = load_text(skill_path / "SKILL.md")

    # 反思
    reflection = reflect_on_failures(skill_content, failed_cases)
    console.print("[green]反思完成[/green]")

    # 改写
    new_content = rewrite_skill(skill_content, reflection)

    # 清理可能的 markdown 代码块包裹
    if new_content.strip().startswith("```"):
        lines = new_content.strip().split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        new_content = "\n".join(lines)

    # 就地写回
    save_text(skill_path / "SKILL.md", new_content)

    return reflection

"""范例提取器：从近阈值失败响应中提取优质模式。"""

from __future__ import annotations

import re

from rich.console import Console

from ..core.llm import get_llm
from ..core.prompts import EXEMPLAR_EXTRACT_PROMPT

console = Console()

# 近阈值下限：score >= 此值但 pass=false 的 case 才值得提取
_EXEMPLAR_MIN_SCORE = 0.7


def extract_exemplars(
    case_results: list[dict],
    min_score: float = _EXEMPLAR_MIN_SCORE,
) -> list[dict]:
    """从失败用例中提取高质量范例。

    只关注 score >= min_score 但 pass=false 的 case，
    这些是"接近成功"的响应，最可能包含有价值模式。

    Args:
        case_results: eval 结果中的 test_cases 列表
        min_score: 近阈值下限

    Returns:
        [{"pattern": str, "snippet": str}] 或空列表
    """
    exemplars = []

    for case in case_results:
        if case["pass_rate"] >= 1.0:
            continue  # 已通过，不需要提取

        # 取最高分的失败 trial
        failed_trials = [
            r for r in case["results"] if not r["pass"] and r["score"] >= min_score
        ]
        if not failed_trials:
            continue

        best_trial = max(failed_trials, key=lambda r: r["score"])
        result = _extract_single(
            question=case["question"],
            test_point=case["test_point"],
            checkpoints=case.get("checkpoints"),
            score=best_trial["score"],
            response=best_trial["response"],
            reason=best_trial["reason"],
        )
        if result:
            exemplars.append(result)

    return exemplars


def _extract_single(
    question: str,
    test_point: str,
    score: float,
    response: str,
    reason: str,
    checkpoints: list[str] | None = None,
) -> dict | None:
    """从单个近阈值失败响应中提取优质模式。

    Returns:
        {"pattern": str, "snippet": str} 或 None
    """
    if checkpoints:
        checkpoints_text = "、".join(checkpoints)
    else:
        checkpoints_text = "（无特定评估要点）"

    prompt = EXEMPLAR_EXTRACT_PROMPT.format(
        question=question,
        test_point=test_point,
        checkpoints=checkpoints_text,
        score=score,
        response=response,
        reason=reason,
    )

    console.print(f"[dim]  提取范例: {test_point[:30]}...[/dim]")
    llm = get_llm()
    resp = llm.invoke(prompt)
    text = resp.content if hasattr(resp, "content") else str(resp)
    text = text.strip()

    if text.upper().startswith("EMPTY"):
        return None

    # 解析输出
    pattern = ""
    snippet = ""
    lines = text.split("\n")
    current_section = None

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- 模式描述") or stripped.startswith("模式描述"):
            # 取冒号后的内容
            parts = stripped.split("：", 1)
            if len(parts) < 2:
                parts = stripped.split(":", 1)
            pattern = parts[1].strip() if len(parts) >= 2 else ""
            current_section = "pattern"
        elif stripped.startswith("- 示例片段") or stripped.startswith("示例片段"):
            current_section = "snippet"
            parts = stripped.split("：", 1)
            if len(parts) < 2:
                parts = stripped.split(":", 1)
            if len(parts) >= 2 and parts[1].strip():
                snippet = parts[1].strip() + "\n"
        elif current_section == "snippet" and stripped:
            snippet += line + "\n"

    snippet = snippet.strip()
    if not pattern and not snippet:
        return None

    return {"pattern": pattern, "snippet": snippet}


def format_exemplars_section(exemplars: list[dict]) -> str:
    """将范例格式化为可注入 prompt 的文本。"""
    if not exemplars:
        return "（无近阈值范例）"

    parts = []
    for i, ex in enumerate(exemplars, 1):
        parts.append(f"### 范例 {i}：{ex['pattern']}")
        parts.append(ex["snippet"])
        parts.append("")

    return "\n".join(parts)


def format_exemplars_instruction(exemplars: list[dict]) -> str:
    """生成注入 EVOLVE_REWRITE_PROMPT 的范例指令。"""
    if not exemplars:
        return ""

    instruction = "\n6. 将以下高质量范例融入技能的示例部分：\n"
    for i, ex in enumerate(exemplars, 1):
        instruction += f"   范例 {i}（{ex['pattern']}）：\n"
        instruction += f"   {ex['snippet']}\n"

    return instruction

"""范例提取器：基于同一 case 的 trial 间分差，对比提取高分路径的成功模式。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console

from ..core.llm import get_llm
from ..core.prompts import EXEMPLAR_CONTRAST_PROMPT

if TYPE_CHECKING:
    from ..eval.trace import ExecutionTrace

console = Console()

# 分差阈值：max(score) - min(score) >= 此值才值得对比利用
_VARIANCE_THRESHOLD = 0.1
# 最高分下限：最高分至少要达到此值，才有可提取的成功模式
_MIN_BEST_SCORE = 0.7
# 最少 trial 数：至少需要 2 次 trial 才有对比意义
_MIN_TRIALS = 2


def _format_trace_summary(trace: ExecutionTrace) -> str:
    """将执行轨迹格式化为紧凑摘要，供对比 prompt 使用。"""
    lines = [
        f"Trial {trace.trial}: {trace.total_llm_calls} 次 LLM 调用, "
        f"{trace.total_tool_calls} 次工具调用, "
        f"技能触发={trace.skill_triggered}"
    ]

    # 工具调用序列
    tool_seq = []
    for step in trace.steps:
        if step.step_type == "tool_call" and step.tool_name:
            tool_seq.append(step.tool_name)
    if tool_seq:
        lines.append(f"工具序列: {' -> '.join(tool_seq)}")

    # 错误点
    errors = []
    for step in trace.steps:
        if step.tool_error:
            errors.append(f"{step.tool_name}: {step.tool_error}")
    if errors:
        lines.append(f"错误: {'; '.join(errors[:3])}")

    return "\n".join(lines)


def extract_exemplars(
    case_results: list[dict],
    traces_by_case: list[list[ExecutionTrace]] | None = None,
) -> list[dict]:
    """从高方差 case 中对比提取高分路径的成功模式。

    选取标准：同一 case 的多次 trial 中，score_spread >= 阈值 且
    最高分 >= 最低分数要求。对选中的 case，对比最高分和最低分 trial，
    提取差异启发。

    Args:
        case_results: eval 结果中的 test_cases 列表
        traces_by_case: 与 case_results 等长的轨迹列表（可选）

    Returns:
        [{"diagnosis": str, "pattern": str, "snippet": str}] 或空列表
    """
    exemplars = []

    for case_idx, case in enumerate(case_results):
        results = case.get("results", [])
        if len(results) < _MIN_TRIALS:
            continue

        scores = [r["score"] for r in results]
        score_spread = max(scores) - min(scores)
        if score_spread < _VARIANCE_THRESHOLD:
            continue

        best_trial = max(results, key=lambda r: r["score"])
        worst_trial = min(results, key=lambda r: r["score"])

        if best_trial["score"] < _MIN_BEST_SCORE:
            continue

        # 获取 trace（如果有）
        trace_best = None
        trace_worst = None
        if traces_by_case is not None and case_idx < len(traces_by_case):
            case_traces = traces_by_case[case_idx]
            if case_traces:
                trace_map = {t.trial: t for t in case_traces}
                trace_best = trace_map.get(best_trial["trial"])
                trace_worst = trace_map.get(worst_trial["trial"])

        result = _extract_contrast(
            question=case["question"],
            test_point=case["test_point"],
            checkpoints=case.get("checkpoints"),
            best_score=best_trial["score"],
            best_response=best_trial["response"],
            best_reason=best_trial["reason"],
            worst_score=worst_trial["score"],
            worst_response=worst_trial["response"],
            worst_reason=worst_trial["reason"],
            trace_best=trace_best,
            trace_worst=trace_worst,
        )
        if result:
            result["score_spread"] = score_spread
            exemplars.append(result)

    # 按分差降序排列，方差最大的 case 优先出现在 prompt 中
    exemplars.sort(key=lambda e: e.get("score_spread", 0), reverse=True)
    return exemplars


def _extract_contrast(
    question: str,
    test_point: str,
    best_score: float,
    best_response: str,
    best_reason: str,
    worst_score: float,
    worst_response: str,
    worst_reason: str,
    checkpoints: list[str] | None = None,
    trace_best: ExecutionTrace | None = None,
    trace_worst: ExecutionTrace | None = None,
) -> dict | None:
    """对比最高分和最低分 trial，提取成功模式。"""
    if checkpoints:
        checkpoints_text = "、".join(checkpoints)
    else:
        checkpoints_text = "（无特定评估要点）"

    # 构建轨迹对比段落
    trace_section = ""
    if trace_best and trace_worst:
        best_summary = _format_trace_summary(trace_best)
        worst_summary = _format_trace_summary(trace_worst)
        trace_section = (
            f"### 执行路径对比\n\n"
            f"**最高分路径**（得分 {best_score}）：\n{best_summary}\n\n"
            f"**最低分路径**（得分 {worst_score}）：\n{worst_summary}\n"
        )

    prompt = EXEMPLAR_CONTRAST_PROMPT.format(
        question=question,
        test_point=test_point,
        checkpoints=checkpoints_text,
        best_score=best_score,
        best_response=best_response,
        best_reason=best_reason,
        worst_score=worst_score,
        worst_response=worst_response,
        worst_reason=worst_reason,
        trace_section=trace_section,
    )

    console.print(f"[dim]  对比提取: {test_point[:30]}... "
                  f"(best={best_score:.2f} worst={worst_score:.2f})[/dim]")
    llm = get_llm()
    resp = llm.invoke(prompt)
    text = resp.content if hasattr(resp, "content") else str(resp)
    text = text.strip()

    if text.upper().startswith("EMPTY"):
        return None

    # 解析输出
    diagnosis = ""
    pattern = ""
    snippet = ""
    current_section = None

    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("- 诊断") or stripped.startswith("诊断"):
            parts = stripped.split("：", 1)
            if len(parts) < 2:
                parts = stripped.split(":", 1)
            diagnosis = parts[1].strip() if len(parts) >= 2 else ""
            current_section = None
        elif stripped.startswith("- 模式描述") or stripped.startswith("模式描述"):
            parts = stripped.split("：", 1)
            if len(parts) < 2:
                parts = stripped.split(":", 1)
            pattern = parts[1].strip() if len(parts) >= 2 else ""
            current_section = None
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
    if not pattern and not snippet and not diagnosis:
        return None

    return {"diagnosis": diagnosis, "pattern": pattern, "snippet": snippet}


def format_exemplars_section(exemplars: list[dict]) -> str:
    """将范例格式化为可注入 reflect prompt 的文本。"""
    if not exemplars:
        return "（无高方差范例）"

    parts = []
    for i, ex in enumerate(exemplars, 1):
        header = f"### 范例 {i}"
        if ex.get("diagnosis"):
            header += f"：{ex['diagnosis']}"
        elif ex.get("pattern"):
            header += f"：{ex['pattern']}"
        parts.append(header)
        if ex.get("pattern") and ex.get("diagnosis"):
            parts.append(f"模式：{ex['pattern']}")
        if ex.get("snippet"):
            parts.append(ex["snippet"])
        parts.append("")

    return "\n".join(parts)


def format_exemplars_instruction(exemplars: list[dict]) -> str:
    """生成注入 EVOLVE_REWRITE_PROMPT 的范例指令。"""
    if not exemplars:
        return ""

    instruction = "\n6. 将以下高分路径的成功模式融入技能的规则和示例部分：\n"
    for i, ex in enumerate(exemplars, 1):
        label = ex.get("diagnosis") or ex.get("pattern") or f"范例 {i}"
        instruction += f"   范例 {i}（{label}）：\n"
        if ex.get("snippet"):
            instruction += f"   {ex['snippet']}\n"

    return instruction

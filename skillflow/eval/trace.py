"""执行轨迹：从 agent messages 中提取、聚合、摘要。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage


@dataclass
class StepRecord:
    """单个执行步骤。"""

    step_type: str  # "llm_call" | "tool_call"
    # LLM 调用相关
    llm_output_preview: str = ""
    # 工具调用相关
    tool_name: str = ""
    tool_args_preview: str = ""
    tool_result_preview: str = ""
    tool_error: str | None = None
    # Token 统计
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass
class ExecutionTrace:
    """单次 trial 的执行轨迹。"""

    trial: int
    test_point: str
    question: str
    # 最终结果（由 runner 填充）
    final_response: str = ""
    score: float = 0.0
    passed: bool = False
    judge_reason: str = ""
    # 技能触发（由 extract_trace 填充）
    skill_triggered: bool = False
    skill_name: str = ""
    # 执行步骤
    steps: list[StepRecord] = field(default_factory=list)
    # 汇总
    total_llm_calls: int = 0
    total_tool_calls: int = 0


@dataclass
class PathSummary:
    """单次 trial 的路径摘要。"""

    steps: int
    tool_sequence: list[str]
    errors: list[str]
    score: float
    passed: bool


@dataclass
class CaseTraceAggregate:
    """单个 case 多次 trial 的聚合轨迹。"""

    test_point: str
    question: str
    total_trials: int = 0
    pass_count: int = 0
    fail_count: int = 0
    skill_triggered_count: int = 0
    # 层次1：共识问题（>50% trial 共有）
    consensus_errors: list[str] = field(default_factory=list)
    consensus_tool_misuse: list[str] = field(default_factory=list)
    # 层次2：分化问题
    path_stability: float = 1.0
    pass_fail_divergence: str | None = None
    # 层次3：效率信号
    best_path: PathSummary | None = None
    worst_path: PathSummary | None = None
    avg_path_length: float = 0.0


# ── 提取 ──────────────────────────────────────────────────────

_TRUNCATE_LEN = 300


def _truncate(text: str, max_len: int = _TRUNCATE_LEN) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def extract_trace(
    messages: list,
    trial: int,
    test_point: str,
    question: str,
    skill_path: str = "",
) -> ExecutionTrace:
    """从 agent.invoke 返回的 messages 中提取执行轨迹。

    Args:
        messages: agent.invoke 返回的 messages 列表
        trial: trial 序号
        test_point: 测试点名称
        question: 测试问题
        skill_path: 技能目录路径（用于检测技能是否被触发）
    """
    steps: list[StepRecord] = []
    total_llm_calls = 0
    total_tool_calls = 0
    skill_triggered = False
    skill_name = ""

    # 预计算技能路径关键词（用于匹配）
    skill_keywords: list[str] = []
    if skill_path:
        skill_name = Path(skill_path).name
        skill_keywords = [
            skill_path,
            # 相对路径形式
            skill_name,
        ]

    for msg in messages:
        if isinstance(msg, AIMessage):
            total_llm_calls += 1

            # LLM 文本输出
            content = msg.content if isinstance(msg.content, str) else ""
            if content:
                steps.append(StepRecord(
                    step_type="llm_call",
                    llm_output_preview=_truncate(content, 200),
                ))

            # 工具调用
            tool_calls = getattr(msg, "tool_calls", None) or []
            for tc in tool_calls:
                total_tool_calls += 1
                args_str = _truncate(str(tc.get("args", {})), 300)
                steps.append(StepRecord(
                    step_type="tool_call",
                    tool_name=tc.get("name", ""),
                    tool_args_preview=args_str,
                ))

                # 检测技能触发：工具参数中包含技能路径
                if skill_keywords and not skill_triggered:
                    args_raw = str(tc.get("args", {}))
                    for kw in skill_keywords:
                        if kw in args_raw:
                            skill_triggered = True
                            break

        elif isinstance(msg, ToolMessage):
            # 工具结果回填到最后一个 tool_call step
            for step in reversed(steps):
                if step.step_type == "tool_call" and not step.tool_result_preview:
                    content = msg.content
                    if isinstance(content, str):
                        step.tool_result_preview = _truncate(content, 300)
                    else:
                        step.tool_result_preview = _truncate(str(content), 300)
                    is_error = getattr(msg, "status", None) == "error"
                    if is_error:
                        step.tool_error = step.tool_result_preview
                    # Token 统计（usage_metadata 在 AIMessage 上）
                    break

    # 从 messages 中的 AIMessage 收集 token 统计
    total_tokens_in = 0
    total_tokens_out = 0
    for msg in messages:
        if isinstance(msg, AIMessage):
            usage = getattr(msg, "usage_metadata", None)
            if usage:
                total_tokens_in += usage.get("input_tokens", 0)
                total_tokens_out += usage.get("output_tokens", 0)

    # 回填 token 到 LLM step
    if steps:
        llm_steps = [s for s in steps if s.step_type == "llm_call"]
        if llm_steps:
            per_call_in = total_tokens_in // max(len(llm_steps), 1)
            per_call_out = total_tokens_out // max(len(llm_steps), 1)
            for s in llm_steps:
                s.tokens_in = per_call_in
                s.tokens_out = per_call_out

    return ExecutionTrace(
        trial=trial,
        test_point=test_point,
        question=question,
        skill_triggered=skill_triggered,
        skill_name=skill_name,
        steps=steps,
        total_llm_calls=total_llm_calls,
        total_tool_calls=total_tool_calls,
    )


# ── 聚合 ──────────────────────────────────────────────────────

def _to_path_summary(trace: ExecutionTrace) -> PathSummary:
    return PathSummary(
        steps=len(trace.steps),
        tool_sequence=[
            s.tool_name for s in trace.steps if s.step_type == "tool_call"
        ],
        errors=[
            s.tool_error for s in trace.steps
            if s.step_type == "tool_call" and s.tool_error
        ],
        score=trace.score,
        passed=trace.passed,
    )


def _find_divergence_point(
    pass_trace: ExecutionTrace,
    fail_trace: ExecutionTrace,
) -> str | None:
    """找到 pass 和 fail trial 路径的分化点。"""
    pass_tools = [s.tool_name for s in pass_trace.steps if s.step_type == "tool_call"]
    fail_tools = [s.tool_name for s in fail_trace.steps if s.step_type == "tool_call"]

    for i, (p, f) in enumerate(zip(pass_tools, fail_tools)):
        if p != f:
            return f"第{i+1}次工具调用分化：pass路径调用 {p}，fail路径调用 {f}"

    if len(fail_tools) > len(pass_tools):
        extra = fail_tools[len(pass_tools):]
        return f"fail路径多出 {len(extra)} 步额外调用: {', '.join(extra)}"

    if len(pass_tools) > len(fail_tools):
        return f"pass路径多出 {len(pass_tools) - len(fail_tools)} 步，fail路径提前结束"

    return None


def _lcs_length(a: tuple, b: tuple) -> int:
    """最长公共子序列长度。"""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def _compute_path_stability(traces: list[ExecutionTrace]) -> float:
    """计算多次 trial 路径的稳定度（基于工具序列 LCS 相似度）。"""
    if len(traces) <= 1:
        return 1.0

    sequences = []
    for t in traces:
        seq = tuple(s.tool_name for s in t.steps if s.step_type == "tool_call")
        sequences.append(seq)

    total_sim = 0.0
    pairs = 0
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            max_len = max(len(sequences[i]), len(sequences[j]), 1)
            sim = _lcs_length(sequences[i], sequences[j]) / max_len
            total_sim += sim
            pairs += 1

    return total_sim / pairs if pairs > 0 else 1.0


def _detect_tool_misuse(trace: ExecutionTrace) -> list[str]:
    """检测工具误用模式。"""
    misuses = []
    tool_calls = [s for s in trace.steps if s.step_type == "tool_call"]

    # 同一工具连续调用（可能表示重试/卡住）
    prev_name = None
    repeat_count = 0
    for s in tool_calls:
        if s.tool_name == prev_name and s.tool_name:
            repeat_count += 1
            if repeat_count >= 2:
                misuses.append(f"{s.tool_name} 连续调用 {repeat_count + 1} 次")
        else:
            repeat_count = 0
        prev_name = s.tool_name

    # 工具调用出错
    for s in tool_calls:
        if s.tool_error:
            misuses.append(f"{s.tool_name} 调用出错: {_truncate(s.tool_error, 80)}")

    return misuses


def aggregate_case_traces(
    traces: list[ExecutionTrace],
) -> CaseTraceAggregate:
    """聚合一个 case 的所有 trial 轨迹。"""
    if not traces:
        return CaseTraceAggregate(test_point="", question="")

    passed = [t for t in traces if t.passed]
    failed = [t for t in traces if not t.passed]
    skill_triggered_count = sum(1 for t in traces if t.skill_triggered)

    # 层次1：共识问题（>50% trial 共有）
    threshold = len(traces) * 0.5
    error_freq: dict[str, int] = {}
    misuse_freq: dict[str, int] = {}

    for trace in traces:
        seen_errors: set[str] = set()
        for step in trace.steps:
            if step.tool_error and step.tool_error not in seen_errors:
                error_freq[step.tool_error] = error_freq.get(step.tool_error, 0) + 1
                seen_errors.add(step.tool_error)

        for m in _detect_tool_misuse(trace):
            misuse_freq[m] = misuse_freq.get(m, 0) + 1

    consensus_errors = [
        e for e, c in error_freq.items() if c >= threshold
    ]
    consensus_tool_misuse = [
        e for e, c in misuse_freq.items() if c >= threshold
    ]

    # 层次2：分化点
    divergence = None
    if passed and failed:
        best_pass = min(passed, key=lambda t: len(t.steps))
        worst_fail = max(failed, key=lambda t: len(t.steps))
        divergence = _find_divergence_point(best_pass, worst_fail)

    path_stability = _compute_path_stability(traces)

    # 层次3：效率信号
    best_trace = min(traces, key=lambda t: len(t.steps))
    worst_trace = max(traces, key=lambda t: len(t.steps))

    return CaseTraceAggregate(
        test_point=traces[0].test_point,
        question=traces[0].question,
        total_trials=len(traces),
        pass_count=len(passed),
        fail_count=len(failed),
        skill_triggered_count=skill_triggered_count,
        consensus_errors=consensus_errors,
        consensus_tool_misuse=consensus_tool_misuse,
        path_stability=path_stability,
        pass_fail_divergence=divergence,
        best_path=_to_path_summary(best_trace),
        worst_path=_to_path_summary(worst_trace),
        avg_path_length=sum(len(t.steps) for t in traces) / len(traces),
    )


# ── 摘要格式化（注入 prompt）───────────────────────────────────────

TRACE_CONTEXT_TEMPLATE = """\
## 执行轨迹诊断

{case_summaries}
"""

CASE_TRACE_TEMPLATE = """\
### 用例：{test_point}
- 通过率：{pass_count}/{total_trials}
- 技能触发：{skill_triggered_count}/{total_trials} 次
- 路径稳定性：{path_stability:.0%}
- 效率对比：最优 {best_steps}步 vs 最差 {worst_steps}步，平均 {avg_steps:.1f}步
{consensus_section}
{divergence_section}
{misuse_section}
"""


def format_trace_context(aggregates: list[CaseTraceAggregate]) -> str:
    """将聚合轨迹格式化为可注入 prompt 的文本。"""
    if not aggregates:
        return "（无执行轨迹数据）"

    parts = []
    for agg in aggregates:
        consensus_section = ""
        if agg.consensus_errors:
            lines = ["**共识错误**（多数 trial 共有）：n"]
            for e in agg.consensus_errors:
                lines.append(f"  - {_truncate(e, 100)}")
            consensus_section = "\n".join(lines)

        divergence_section = ""
        if agg.pass_fail_divergence:
            divergence_section = f"**分化点**：{agg.pass_fail_divergence}"

        misuse_section = ""
        if agg.consensus_tool_misuse:
            lines = ["**工具误用模式**：n"]
            for m in agg.consensus_tool_misuse:
                lines.append(f"  - {_truncate(m, 100)}")
            misuse_section = "\n".join(lines)

        parts.append(CASE_TRACE_TEMPLATE.format(
            test_point=agg.test_point,
            pass_count=agg.pass_count,
            total_trials=agg.total_trials,
            skill_triggered_count=agg.skill_triggered_count,
            path_stability=agg.path_stability,
            best_steps=agg.best_path.steps if agg.best_path else 0,
            worst_steps=agg.worst_path.steps if agg.worst_path else 0,
            avg_steps=agg.avg_path_length,
            consensus_section=consensus_section,
            divergence_section=divergence_section,
            misuse_section=misuse_section,
        ))

    return TRACE_CONTEXT_TEMPLATE.format(case_summaries="\n".join(parts))


# ── 落盘 ──────────────────────────────────────────────────────


def trace_to_dict(trace: ExecutionTrace) -> dict:
    """将 ExecutionTrace 转为可 JSON 序列化的 dict。"""
    return asdict(trace)


def save_traces(traces_by_case: list[list[ExecutionTrace]], trace_dir: str | Path) -> None:
    """将轨迹落盘保存，每个 trial 一个文件。

    目录结构：
        trace_dir/
            case_0_trial_1.json
            case_0_trial_2.json
            case_1_trial_1.json
            ...
    """
    trace_dir = Path(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)

    for case_idx, case_traces in enumerate(traces_by_case):
        if not case_traces:
            continue
        for trace in case_traces:
            data = trace_to_dict(trace)
            out_file = trace_dir / f"case_{case_idx}_trial_{trace.trial}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

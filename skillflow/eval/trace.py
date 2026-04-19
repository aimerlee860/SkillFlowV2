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
class KeyEpisode:
    """具有诊断价值的关键执行片段。"""

    trial: int
    step_index: int
    reason: str  # 为什么这段有诊断价值
    tool_name: str = ""
    tool_args_preview: str = ""
    tool_result_preview: str = ""
    tool_error: str | None = None
    llm_output_preview: str = ""


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
    # 层次4：关键诊断片段（选择性保留高价值步骤）
    key_episodes: list[KeyEpisode] = field(default_factory=list)


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


def _extract_key_episodes(traces: list[ExecutionTrace], max_episodes: int = 5) -> list[KeyEpisode]:
    """从失败 trial 的轨迹中提取具有诊断价值的关键步骤片段。

    选取策略（按优先级）：
    1. 工具调用出错的步骤（最高优先）
    2. 分化点前后的步骤（pass/fail 路径分叉处）
    3. 连续重试同一工具的步骤（卡住信号）

    只处理失败的 trial，成功的 trial 不需要诊断。
    """
    episodes: list[KeyEpisode] = []
    failed = [t for t in traces if not t.passed]

    if not failed:
        return episodes

    passed = [t for t in traces if t.passed]

    for trace in failed:
        for i, step in enumerate(trace.steps):
            # 优先级1：工具出错
            if step.step_type == "tool_call" and step.tool_error:
                episodes.append(KeyEpisode(
                    trial=trace.trial,
                    step_index=i + 1,
                    reason="工具调用出错",
                    tool_name=step.tool_name,
                    tool_args_preview=_truncate(step.tool_args_preview, 200),
                    tool_result_preview=_truncate(step.tool_result_preview, 200),
                    tool_error=_truncate(step.tool_error, 200),
                ))

    # 优先级2：分化点附近步骤（需要 pass 和 fail 的 trial）
    if passed and failed:
        pass_tools = [
            (i, s) for i, s in enumerate(passed[0].steps)
            if s.step_type == "tool_call"
        ]
        for ft in failed[:1]:  # 只看第一个失败 trial
            fail_tools = [
                (i, s) for i, s in enumerate(ft.steps)
                if s.step_type == "tool_call"
            ]
            for (pi, ps), (fi, fs) in zip(pass_tools, fail_tools):
                if ps.tool_name != fs.tool_name:
                    # 分化点：保留 fail 侧的前后上下文
                    ctx_start = max(0, fi - 1)
                    ctx_end = min(len(ft.steps), fi + 2)
                    for ci in range(ctx_start, ctx_end):
                        s = ft.steps[ci]
                        if any(ep.step_index == ci + 1 and ep.trial == ft.trial for ep in episodes):
                            continue
                        episodes.append(KeyEpisode(
                            trial=ft.trial,
                            step_index=ci + 1,
                            reason=f"分化点附近（pass调用{ps.tool_name}，fail调用{fs.tool_name}）",
                            tool_name=s.tool_name,
                            tool_args_preview=_truncate(s.tool_args_preview, 200),
                            tool_result_preview=_truncate(s.tool_result_preview, 200),
                            tool_error=s.tool_error,
                            llm_output_preview=_truncate(s.llm_output_preview, 200) if s.step_type == "llm_call" else "",
                        ))
                    break

    # 优先级3：连续重试
    for trace in failed:
        tool_calls = [(i, s) for i, s in enumerate(trace.steps) if s.step_type == "tool_call"]
        prev_name = None
        repeat_count = 0
        for idx, (i, s) in enumerate(tool_calls):
            if s.tool_name == prev_name and s.tool_name:
                repeat_count += 1
                if repeat_count == 2:  # 第3次连续调用时记录
                    if not any(ep.step_index == i + 1 and ep.trial == trace.trial for ep in episodes):
                        episodes.append(KeyEpisode(
                            trial=trace.trial,
                            step_index=i + 1,
                            reason=f"{s.tool_name} 连续调用 {repeat_count + 1} 次",
                            tool_name=s.tool_name,
                            tool_args_preview=_truncate(s.tool_args_preview, 200),
                            tool_result_preview=_truncate(s.tool_result_preview, 200),
                        ))
            else:
                repeat_count = 0
            prev_name = s.tool_name

    # 去重（同 trial 同 step 只保留一个）并限制总数
    seen: set[tuple[int, int]] = set()
    unique: list[KeyEpisode] = []
    for ep in episodes:
        key = (ep.trial, ep.step_index)
        if key not in seen:
            seen.add(key)
            unique.append(ep)

    return unique[:max_episodes]


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

    # 层次4：关键诊断片段
    key_episodes = _extract_key_episodes(traces)

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
        key_episodes=key_episodes,
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
{episodes_section}
"""


def _format_episode(ep: KeyEpisode) -> str:
    """格式化单个关键片段。"""
    parts = [f"  Trial {ep.trial} 第{ep.step_index}步 [{ep.reason}]"]
    if ep.tool_name:
        parts.append(f"    工具: {ep.tool_name}")
        if ep.tool_args_preview:
            parts.append(f"    参数: {ep.tool_args_preview}")
    if ep.llm_output_preview:
        parts.append(f"    LLM输出: {ep.llm_output_preview}")
    if ep.tool_result_preview:
        parts.append(f"    结果: {ep.tool_result_preview}")
    if ep.tool_error:
        parts.append(f"    错误: {ep.tool_error}")
    return "\n".join(parts)


def format_trace_context(aggregates: list[CaseTraceAggregate]) -> str:
    """将聚合轨迹格式化为可注入 prompt 的文本。"""
    if not aggregates:
        return "（无执行轨迹数据）"

    parts = []
    for agg in aggregates:
        consensus_section = ""
        if agg.consensus_errors:
            lines = ["**共识错误**（多数 trial 共有）："]
            for e in agg.consensus_errors:
                lines.append(f"  - {_truncate(e, 100)}")
            consensus_section = "\n".join(lines)

        divergence_section = ""
        if agg.pass_fail_divergence:
            divergence_section = f"**分化点**：{agg.pass_fail_divergence}"

        misuse_section = ""
        if agg.consensus_tool_misuse:
            lines = ["**工具误用模式**："]
            for m in agg.consensus_tool_misuse:
                lines.append(f"  - {_truncate(m, 100)}")
            misuse_section = "\n".join(lines)

        episodes_section = ""
        if agg.key_episodes:
            lines = ["**关键失败步骤**（失败 trial 的具体出错细节）："]
            for ep in agg.key_episodes:
                lines.append(_format_episode(ep))
            episodes_section = "\n".join(lines)

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
            episodes_section=episodes_section,
        ))

    return TRACE_CONTEXT_TEMPLATE.format(case_summaries="\n".join(parts))


# ── Judge 专用格式化 ───────────────────────────────────────────


def _detect_process_signals(trace: ExecutionTrace) -> list[str]:
    """检测单次 trial 的过程异常信号（仅客观负面事实，不做主观判断）。"""
    signals: list[str] = []

    if not trace.skill_triggered:
        signals.append("技能未被触发")

    # 工具调用失败
    tool_errors: dict[str, list[str]] = {}
    for step in trace.steps:
        if step.step_type == "tool_call" and step.tool_error:
            tool_errors.setdefault(step.tool_name, []).append(step.tool_error)
    for name, errors in tool_errors.items():
        signals.append(f"工具 {name} 调用失败: {_truncate(errors[0], 80)}")

    # 带错误的连续重试：同一工具连续 ≥2 次，且其中至少 1 次有 tool_error
    tool_calls = [(i, s) for i, s in enumerate(trace.steps) if s.step_type == "tool_call"]
    if tool_calls:
        run_name = tool_calls[0][1].tool_name
        run_count = 1
        run_has_error = bool(tool_calls[0][1].tool_error)
        for idx in range(1, len(tool_calls)):
            _, step = tool_calls[idx]
            if step.tool_name == run_name:
                run_count += 1
                run_has_error = run_has_error or bool(step.tool_error)
            else:
                if run_count >= 2 and run_has_error and run_name:
                    signals.append(f"工具 {run_name} 连续调用 {run_count} 次（含失败重试）")
                run_name = step.tool_name
                run_count = 1
                run_has_error = bool(step.tool_error)
        if run_count >= 2 and run_has_error and run_name:
            signals.append(f"工具 {run_name} 连续调用 {run_count} 次（含失败重试）")

    return signals


def format_trace_for_judge(trace: ExecutionTrace) -> str:
    """将单次 trial 轨迹格式化为注入 judge prompt 的文本。

    Returns:
        格式化文本，trace.steps 为空时返回空字符串。
    """
    if not trace.steps:
        return ""

    skill_status = "已触发" if trace.skill_triggered else "未触发"
    header = (
        f"执行过程：共 {len(trace.steps)} 步，"
        f"{trace.total_tool_calls} 次工具调用，"
        f"{trace.total_llm_calls} 次 LLM 调用，"
        f"技能{skill_status}"
    )

    # 异常信号（条件展示）
    signals = _detect_process_signals(trace)
    signal_section = ""
    if signals:
        signal_section = "\n\n过程异常信号\n" + "\n".join(f"- {s}" for s in signals)

    # 逐步骤展示
    step_lines: list[str] = []
    for i, step in enumerate(trace.steps):
        if step.step_type == "tool_call":
            args_preview = _truncate(step.tool_args_preview, 100)
            if step.tool_error:
                status = f" → 失败: {_truncate(step.tool_error, 100)}"
            else:
                status = " → 成功"
            step_lines.append(f"{i+1}. [工具: {step.tool_name}] {args_preview}{status}")
        elif step.step_type == "llm_call":
            preview = _truncate(step.llm_output_preview, 100)
            step_lines.append(f"{i+1}. [LLM] {preview}")

    steps_section = "\n执行步骤\n" + "\n".join(step_lines)

    return header + signal_section + steps_section


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

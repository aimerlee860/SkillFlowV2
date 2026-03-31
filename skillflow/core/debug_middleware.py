"""Debug 中间件，用于追踪 agent 执行各阶段的详细状态和耗时。"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

logger = logging.getLogger("skillflow.debug")


def _extract_text_len(obj: Any) -> int:
    """从消息 content 中提取文本总长度，兼容 str 和 content blocks。"""
    content = getattr(obj, "content", "")
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for block in content:
            if isinstance(block, dict):
                total += len(block.get("text", ""))
            elif isinstance(block, str):
                total += len(block)
        return total
    return 0


def _extract_text_preview(obj: Any, max_len: int = 200) -> str:
    """从消息 content 中提取文本预览。"""
    content = getattr(obj, "content", "")
    if isinstance(content, str):
        return content[:max_len]
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)[:max_len]
    return ""


def _msg_summary(msg: Any) -> dict[str, Any]:
    """提取消息摘要信息。"""
    summary: dict[str, Any] = {"type": getattr(msg, "type", type(msg).__name__)}
    content_len = _extract_text_len(msg)
    summary["content_len"] = content_len
    if content_len > 0:
        summary["content_preview"] = _extract_text_preview(msg, 200)
    # tool_calls
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        summary["tool_calls"] = [
            {"name": tc.get("name"), "args_keys": list(tc.get("args", {}).keys())}
            for tc in tool_calls
        ]
    # usage_metadata
    usage = getattr(msg, "usage_metadata", None)
    if usage:
        summary["usage"] = usage
    return summary


def _estimate_tokens_from_messages(messages: list[Any]) -> dict[str, int]:
    """估算消息列表的 token 数（基于字符数 / 4）。"""
    total_chars = 0
    for msg in messages:
        total_chars += _extract_text_len(msg)
    est_tokens = total_chars // 4
    return {"estimated_tokens": est_tokens, "total_chars": total_chars}


def _estimate_tool_schema_tokens(tools: list[Any]) -> int:
    """估算 tool schemas 的 token 开销。"""
    total_json = 0
    for t in tools:
        if isinstance(t, dict):
            total_json += len(json.dumps(t, ensure_ascii=False))
        else:
            schema = getattr(t, "args_schema", None)
            if schema:
                try:
                    total_json += len(json.dumps(schema.schema(), ensure_ascii=False, default=str))
                except Exception:
                    total_json += 200  # 粗略估计
            else:
                total_json += 100
    return total_json // 4


class DebugMiddleware(AgentMiddleware):
    """记录 agent 执行全生命周期的 debug 日志。

    阶段覆盖：
      - before_agent: agent 启动
      - wrap_model_call: LLM 调用前后（含 token 统计）
      - wrap_tool_call: 工具调用前后（含耗时）
      - before_model / after_model: 模型调用前后状态
      - after_agent: agent 结束（含累计 token 统计）
    """

    def __init__(self, trial_id: str = "") -> None:
        self.trial_id = trial_id
        self._start_time: float = 0.0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._llm_calls: int = 0
        self._tool_calls: int = 0

    def _prefix(self) -> str:
        return f"[DEBUG{f' {self.trial_id}' if self.trial_id else ''}]"

    # ── Agent 生命周期 ──────────────────────────────────────────

    def before_agent(self, state: AgentState, runtime: Any) -> None:  # type: ignore[override]
        self._start_time = time.perf_counter()
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._llm_calls = 0
        self._tool_calls = 0
        msgs = state.get("messages", [])
        logger.info(
            "%s before_agent | msgs=%d | est=%s",
            self._prefix(),
            len(msgs),
            _estimate_tokens_from_messages(msgs),
        )

    def after_agent(self, state: AgentState, runtime: Any) -> None:  # type: ignore[override]
        elapsed = time.perf_counter() - self._start_time
        msgs = state.get("messages", [])
        logger.info(
            "%s after_agent | msgs=%d | llm_calls=%d | tool_calls=%d | "
            "total_in=%d | total_out=%d | elapsed=%.2fs",
            self._prefix(),
            len(msgs),
            self._llm_calls,
            self._tool_calls,
            self._total_input_tokens,
            self._total_output_tokens,
            elapsed,
        )

    # ── Model 调用前后（轻量 hook，不含计时）───────────────────────

    def before_model(self, state: AgentState, runtime: Any) -> None:  # type: ignore[override]
        msgs = state.get("messages", [])
        logger.debug(
            "%s before_model | msgs=%d | est=%s",
            self._prefix(),
            len(msgs),
            _estimate_tokens_from_messages(msgs),
        )

    def after_model(self, state: AgentState, runtime: Any) -> None:  # type: ignore[override]
        msgs = state.get("messages", [])
        last = msgs[-1] if msgs else None
        logger.debug(
            "%s after_model | msgs=%d | last=%s",
            self._prefix(),
            len(msgs),
            _msg_summary(last) if last else None,
        )

    # ── LLM 调用（wrap，含精确计时 + token 统计）──────────────────

    def wrap_model_call(  # type: ignore[override]
        self,
        request: ModelRequest,
        handler: Any,
    ) -> ModelResponse:
        self._llm_calls += 1
        call_idx = self._llm_calls
        prefix = f"{self._prefix()} LLM#{call_idx}"

        # ── 请求信息 ──
        msgs = request.messages
        sys_msg = request.system_message
        tools = request.tools

        sys_len = _extract_text_len(sys_msg) if sys_msg else 0
        sys_preview = _extract_text_preview(sys_msg, 150) if sys_msg else ""

        tool_names = []
        for t in tools:
            if isinstance(t, dict):
                tool_names.append(t.get("name", "?"))
            else:
                tool_names.append(getattr(t, "name", str(t)))

        msg_est = _estimate_tokens_from_messages(msgs)
        sys_est = sys_len // 4
        tool_est = _estimate_tool_schema_tokens(tools)
        total_est = msg_est["estimated_tokens"] + sys_est + tool_est

        logger.info(
            "%s REQUEST | sys_prompt_len=%d | msgs=%d | tools=%s | "
            "est_tokens=%d (msgs=%d + sys=%d + tool_schemas=%d)",
            prefix,
            sys_len,
            len(msgs),
            tool_names,
            total_est,
            msg_est["estimated_tokens"],
            sys_est,
            tool_est,
        )

        # system prompt 预览
        if sys_preview:
            logger.info(
                "%s   sys_prompt_preview = %s%s",
                prefix,
                sys_preview,
                "..." if sys_len > 150 else "",
            )

        # 逐条消息摘要
        for i, msg in enumerate(msgs):
            logger.debug(
                "%s   msg[%d] = %s",
                prefix,
                i,
                _msg_summary(msg),
            )

        # ── 执行 ──
        t0 = time.perf_counter()
        try:
            response = handler(request)
        except Exception as e:
            elapsed = time.perf_counter() - t0
            logger.error(
                "%s ERROR | elapsed=%.2fs | est_tokens=%d | error=%s",
                prefix, elapsed, total_est, e,
            )
            raise
        elapsed = time.perf_counter() - t0

        # ── 响应信息 ──
        result_msgs = response.result
        total_in = 0
        total_out = 0

        for rmsg in result_msgs:
            usage = getattr(rmsg, "usage_metadata", None)
            if usage:
                in_t = usage.get("input_tokens", 0)
                out_t = usage.get("output_tokens", 0)
                total_in += in_t
                total_out += out_t

        self._total_input_tokens += total_in
        self._total_output_tokens += total_out

        logger.info(
            "%s RESPONSE | elapsed=%.2fs | input_tokens=%d | output_tokens=%d | "
            "result_msgs=%d | cumulative_in=%d | cumulative_out=%d",
            prefix,
            elapsed,
            total_in,
            total_out,
            len(result_msgs),
            self._total_input_tokens,
            self._total_output_tokens,
        )

        for i, rmsg in enumerate(result_msgs):
            logger.debug(
                "%s   result[%d] = %s",
                prefix,
                i,
                _msg_summary(rmsg),
            )

        return response

    # ── Tool 调用（wrap，含计时 + 参数摘要）──────────────────────

    def wrap_tool_call(  # type: ignore[override]
        self,
        request: ToolCallRequest,
        handler: Any,
    ) -> ToolMessage:
        self._tool_calls += 1
        call_idx = self._tool_calls
        prefix = f"{self._prefix()} TOOL#{call_idx}"

        tc = request.tool_call
        tool_name = tc.get("name", "?")
        tool_args = tc.get("args", {})

        # 截断参数预览
        args_str = json.dumps(tool_args, ensure_ascii=False, default=str)
        args_preview = args_str[:500] if len(args_str) > 500 else args_str

        logger.info(
            "%s CALL | name=%s | args=%s",
            prefix,
            tool_name,
            args_preview,
        )

        t0 = time.perf_counter()
        try:
            result = handler(request)
        except Exception as e:
            elapsed = time.perf_counter() - t0
            logger.error(
                "%s ERROR | elapsed=%.2fs | error=%s",
                prefix, elapsed, e,
            )
            raise
        elapsed = time.perf_counter() - t0

        # 结果摘要
        if isinstance(result, ToolMessage):
            content = result.content
            if isinstance(content, str):
                result_preview = content[:300] if len(content) > 300 else content
            else:
                result_preview = str(content)[:300]
            status = getattr(result, "status", None)
            logger.info(
                "%s RESULT | elapsed=%.2fs | status=%s | content_len=%d | preview=%s",
                prefix,
                elapsed,
                status,
                len(content) if isinstance(content, str) else -1,
                result_preview,
            )
        else:
            logger.info(
                "%s RESULT | elapsed=%.2fs | type=%s",
                prefix,
                elapsed,
                type(result).__name__,
            )

        return result

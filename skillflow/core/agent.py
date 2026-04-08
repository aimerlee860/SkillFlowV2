"""Agent 构建器，封装 deepagents 的 create_deep_agent。"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends.local_shell import LocalShellBackend
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from .debug_middleware import DebugMiddleware
from .llm import get_llm


def _setup_debug_logging() -> None:
    """确保 debug logger 配置为 INFO 级别并输出到 stderr。"""
    dbg = logging.getLogger("skillflow.debug")
    if not dbg.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
        )
        dbg.addHandler(handler)
    dbg.setLevel(logging.DEBUG)


def build_agent(
    skills: list[str] | None = None,
    system_prompt: str | None = None,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    model: BaseChatModel | None = None,
    debug: bool = False,
    debug_trial_id: str = "",
) -> Any:
    """构建 deepagent。

    Args:
        skills: 技能路径列表（如 ["skill-creator"] 或 ["./skills/my-skill"]）
        system_prompt: 自定义系统提示词
        tools: 额外的工具列表
        model: 自定义模型，默认使用 get_llm()
        debug: 是否启用 debug 中间件
        debug_trial_id: debug 日志中的 trial 标识

    Returns:
        编译后的 LangGraph 图
    """
    middleware: list[AgentMiddleware] = []
    if debug:
        _setup_debug_logging()
        middleware.append(DebugMiddleware(trial_id=debug_trial_id))

    backend = LocalShellBackend(virtual_mode=False, inherit_env=True)

    return create_deep_agent(
        model=model or get_llm(),
        skills=skills,
        system_prompt=system_prompt,
        tools=tools,
        middleware=middleware,
        backend=backend,
    )


def run_agent(agent: Any, prompt: str) -> tuple[str, list]:
    """运行 agent 并提取最终响应文本和完整消息列表。

    Args:
        agent: 编译后的 LangGraph 图
        prompt: 用户输入

    Returns:
        (最终响应文本, 完整 messages 列表)
    """
    result = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]}
    )
    messages = result.get("messages", [])
    # 取最后一条 AI 消息
    for msg in reversed(messages):
        if hasattr(msg, "content") and msg.type == "ai":
            return msg.content, messages
    # fallback: 取最后一条消息的 content
    if messages:
        return str(messages[-1].content), messages
    return "", messages

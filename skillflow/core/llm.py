"""LLM 客户端工厂，单例模式。"""

from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter

from .config import get_config

_llm: BaseChatModel | None = None


def get_llm() -> BaseChatModel:
    """获取配置好的 LLM 客户端（单例）。"""
    global _llm
    if _llm is not None:
        return _llm

    config = get_config()
    kwargs: dict = {
        "base_url": config.base_url,
    }
    if config.api_key:
        kwargs["api_key"] = config.api_key

    # 速率限制
    if config.rpm:
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=config.rpm / 60.0,
            check_every_n_seconds=0.1,
            max_bucket_size=config.rpm,
        )
        kwargs["rate_limiter"] = rate_limiter

    _llm = init_chat_model(f"openai:{config.model_name}", **kwargs)
    return _llm

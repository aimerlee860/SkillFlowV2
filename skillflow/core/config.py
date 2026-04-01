"""环境变量加载，优先级：系统环境变量 > ./.env > ~/.skillflow/env"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values


@dataclass
class Config:
    base_url: str
    api_key: str | None
    model_name: str
    rpm: float | None  # 每分钟请求数限制
    min_interval: float | None  # 两次请求之间的最小间隔（秒）


_config: Config | None = None


def load_env() -> None:
    """按优先级加载环境变量，不覆盖已存在的系统环境变量。"""
    env_files = [
        Path.cwd() / ".env",
        Path.home() / ".skillflow" / "env",
    ]
    for env_file in env_files:
        if env_file.exists():
            for key, value in dotenv_values(env_file).items():
                if key not in os.environ and value is not None:
                    os.environ[key] = value


def get_config() -> Config:
    """获取配置，首次调用时自动加载环境变量。"""
    global _config
    if _config is not None:
        return _config

    load_env()

    base_url = os.environ.get("LLM_BASE_URL")
    model_name = os.environ.get("LLM_MODEL_NAME")
    api_key = os.environ.get("LLM_API_KEY")
    rpm = os.environ.get("LLM_RPM")
    min_interval = os.environ.get("LLM_MIN_INTERVAL")

    if not base_url:
        raise ValueError(
            "LLM_BASE_URL is required. Set it in system env, ./.env, or ~/.skillflow/env"
        )
    if not model_name:
        raise ValueError(
            "LLM_MODEL_NAME is required. Set it in system env, ./.env, or ~/.skillflow/env"
        )

    _config = Config(
        base_url=base_url,
        api_key=api_key,
        model_name=model_name,
        rpm=float(rpm) if rpm else None,
        min_interval=float(min_interval) if min_interval else None,
    )
    return _config

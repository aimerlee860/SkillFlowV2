"""测试用例生成器。"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from rich.console import Console

from ..core.llm import get_llm
from ..core.prompts import TEST_CASE_GEN_PROMPT
from ..core.utils import ensure_dir, load_json, load_text, save_json

console = Console()


def _timestamp() -> str:
    """返回当前时间戳字符串，格式 YYYYMMDDHHmm。"""
    return time.strftime("%Y%m%d%H%M")


def generate_test_cases(
    skill_path: str | Path,
    spec_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    ignore_cache: bool = False,
) -> tuple[list[dict], str]:
    """生成测试用例。

    Args:
        skill_path: 技能目录路径（读取 SKILL.md）
        spec_path: SPEC 文件路径（可选，补充上下文）
        output_dir: 输出目录
        ignore_cache: 是否忽略缓存重新生成

    Returns:
        (测试用例列表, 保存的文件名) — 文件名如 test_cases_202603312043.json
    """
    skill_content = load_text(Path(skill_path) / "SKILL.md")
    input_content = skill_content

    if spec_path and Path(spec_path).exists():
        input_content += "\n" + load_text(spec_path)

    # 生成时间戳文件名
    ts = _timestamp()
    filename = f"test_cases_{ts}.json"

    # 检查是否已有同时间戳的缓存
    if output_dir:
        cache_file = Path(output_dir) / filename
        if cache_file.exists() and not ignore_cache:
            console.print(f"[green]使用缓存的测试用例:[/green] {cache_file}")
            data = load_json(cache_file)
            return data["test_cases"], filename

    # 生成测试用例
    console.print("[blue]使用 LLM 生成测试用例...[/blue]")
    prompt = TEST_CASE_GEN_PROMPT.format(skill_content=input_content)
    llm = get_llm()
    response = llm.invoke(prompt)
    response_text = response.content if hasattr(response, "content") else str(response)

    # 解析 JSON
    test_cases_data = _extract_json(response_text)

    # 保存
    if output_dir:
        ensure_dir(output_dir)
        save_json(Path(output_dir) / filename, test_cases_data)
        console.print(f"[green]测试用例已保存:[/green] {Path(output_dir) / filename}")

    return test_cases_data["test_cases"], filename


def _extract_json(text: str) -> dict:
    """从 LLM 响应中提取 JSON。"""
    # 尝试提取 ```json ... ``` 代码块
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        text = match.group(1)
    return json.loads(text.strip())

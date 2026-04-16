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
    max_retries: int = 2,
) -> tuple[list[dict], str]:
    """生成测试用例。

    Args:
        skill_path: 技能目录路径（读取 SKILL.md）
        spec_path: SPEC 文件路径（可选，补充上下文）
        output_dir: 输出目录
        ignore_cache: 是否忽略缓存重新生成
        max_retries: 解析失败时的最大重试次数

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

    # 生成测试用例（带重试）
    console.print("[blue]使用 LLM 生成测试用例...[/blue]")
    prompt = TEST_CASE_GEN_PROMPT.format(skill_content=input_content)
    llm = get_llm()

    for attempt in range(max_retries + 1):
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, "content") else str(response)

        if not response_text or not response_text.strip():
            if attempt < max_retries:
                console.print(f"[yellow]LLM 返回空响应，重试 {attempt + 1}/{max_retries}[/yellow]")
                continue
            raise ValueError("LLM 返回了空响应，请检查 API 配置和网络连接")

        # 解析 JSON
        try:
            test_cases_data = _extract_json(response_text)
            break  # 解析成功，退出循环
        except ValueError as e:
            if attempt < max_retries:
                console.print(f"[yellow]JSON 解析失败，重试 {attempt + 1}/{max_retries}[/yellow]")
                console.print(f"[dim]错误: {e}[/dim]")
                # 添加更明确的格式要求
                retry_prompt = f"""上一次生成的 JSON 格式有问题，请重新生成。
错误信息: {str(e)[:200]}

请严格按照以下格式输出，确保 JSON 完整且有效：
```json
{{
  "test_cases": [
    {{
      "test_point": "简洁描述验证目标",
      "question": "简短的用户提问（控制在50字以内）"
    }}
  ]
}}
```

技能描述:
{input_content[:2000]}
"""
                prompt = retry_prompt
                continue
            raise

    # 保存
    if output_dir:
        ensure_dir(output_dir)
        save_json(Path(output_dir) / filename, test_cases_data)
        console.print(f"[green]测试用例已保存:[/green] {Path(output_dir) / filename}")

    return test_cases_data["test_cases"], filename


def _extract_json(text: str) -> dict:
    """从 LLM 响应中提取 JSON。"""
    stripped = text.strip()
    if not stripped:
        raise ValueError(f"LLM 响应为空，无法解析 JSON。原始响应: {text!r}")

    # 尝试提取 ```json ... ``` 代码块
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", stripped, re.DOTALL)
    if match:
        stripped = match.group(1).strip()

    # 尝试找到第一个 { 和最后一个 } 之间的内容
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        json_str = stripped[start : end + 1]
    else:
        json_str = stripped

    # 尝试直接解析
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # 尝试修复截断的 JSON
        fixed = _try_fix_truncated_json(json_str, e)
        if fixed:
            console.print("[yellow]JSON 已自动修复截断问题[/yellow]")
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

        raise ValueError(
            f"无法从 LLM 响应中解析 JSON: {e}\n响应前 500 字符: {stripped[:500]}"
        ) from e


def _try_fix_truncated_json(json_str: str, error: json.JSONDecodeError) -> str | None:
    """尝试修复截断的 JSON。

    常见情况：
    - 字符串未终止：找到最后一个完整的对象，截断后续内容
    - 数组未终止：补全 ] 和 }
    """
    if "Unterminated string" not in str(error):
        return None

    # 找到最后一个完整对象的开始位置
    # 格式通常是 { "test_cases": [ { ... }, { ... }, { ... (截断)

    # 尝试找到最后一个完整的 test_case 对象
    # 搜索模式: }, { 之间的分割点
    last_complete = -1
    depth = 0
    in_string = False
    escape = False

    for i, c in enumerate(json_str):
        if escape:
            escape = False
            continue
        if c == '\\' and in_string:
            escape = True
            continue
        if c == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 1:  # 回到 test_cases 数组层级，说明一个对象结束了
                last_complete = i

    if last_complete > 0:
        # 截断到最后一个完整对象，补全闭合符号
        truncated = json_str[:last_complete + 1]
        # 补全数组和中括号
        # 计算需要补多少层
        open_braces = truncated.count('{') - truncated.count('}')
        open_brackets = truncated.count('[') - truncated.count(']')

        # 只保留 test_cases 数组层级
        if open_braces >= 1:
            truncated += ']' * open_brackets + '}' * open_braces
            return truncated

    return None

"""测试用例生成器。"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from rich.console import Console

from ..core.llm import get_llm
from ..core.prompts import (
    TEST_CASE_CRITIC_PROMPT,
    TEST_CASE_GEN_PROMPT,
    TEST_CASE_REVISE_PROMPT,
)
from ..core.utils import ensure_dir, load_json, load_text, save_json

console = Console()


def _timestamp() -> str:
    """返回当前时间戳字符串，格式 YYYYMMDDHHmm。"""
    return time.strftime("%Y%m%d%H%M")


# --- 格式校验 ---

_CODE_INDICATORS = re.compile(r"(def |function |class |import |return |=>|\{$|\{$)", re.MULTILINE)


def _is_complex_question(question: str) -> bool:
    """判断 question 是否为包含代码/数据的复杂场景。"""
    if "\n" in question:
        return True
    if _CODE_INDICATORS.search(question):
        return True
    return False


def _validate_test_cases(test_cases: list[dict]) -> list[str]:
    """校验测试用例格式，返回错误列表。为空则表示全部通过。"""
    errors: list[str] = []

    if not test_cases:
        errors.append("测试用例列表为空")
        return errors

    for i, tc in enumerate(test_cases):
        prefix = f"用例 {i}"

        # 必填字段
        if not tc.get("test_point", "").strip():
            errors.append(f"{prefix}: test_point 缺失或为空")
        elif len(tc["test_point"]) > 50:
            errors.append(f"{prefix}: test_point 超过50字（当前{len(tc['test_point'])}字）")

        if not tc.get("question", "").strip():
            errors.append(f"{prefix}: question 缺失或为空")
        else:
            limit = 500 if _is_complex_question(tc["question"]) else 200
            if len(tc["question"]) > limit:
                errors.append(f"{prefix}: question 超过{limit}字（当前{len(tc['question'])}字）")

        checkpoints = tc.get("checkpoints")
        if checkpoints is None or not isinstance(checkpoints, list):
            errors.append(f"{prefix}: checkpoints 缺失或非数组")
        else:
            if len(checkpoints) < 2:
                errors.append(f"{prefix}: checkpoints 不足2条（当前{len(checkpoints)}条）")
            elif len(checkpoints) > 4:
                errors.append(f"{prefix}: checkpoints 超过4条（当前{len(checkpoints)}条）")
            for j, cp in enumerate(checkpoints):
                if not isinstance(cp, str) or not cp.strip():
                    errors.append(f"{prefix}: checkpoint[{j}] 为空或非字符串")
                elif len(cp) > 20:
                    errors.append(f"{prefix}: checkpoint[{j}] 超过20字（当前{len(cp)}字）")

    return errors


# --- LLM 调用辅助 ---

def _call_llm_json(prompt: str) -> dict:
    """调用 LLM 并解析 JSON 响应。"""
    llm = get_llm()
    response = llm.invoke(prompt)
    response_text = response.content if hasattr(response, "content") else str(response)

    if not response_text or not response_text.strip():
        raise ValueError("LLM 返回了空响应")

    return _extract_json(response_text)


def _validate_or_raise(test_cases_data: dict) -> dict:
    """格式校验：不通过则直接抛异常，由上游重试生成。"""
    test_cases = test_cases_data.get("test_cases", [])
    errors = _validate_test_cases(test_cases)
    if errors:
        raise ValueError(
            "格式校验失败:\n" + "\n".join(f"- {e}" for e in errors)
        )
    return {"test_cases": test_cases}


# --- Critic 校验 ---

def _run_critic(test_cases: list[dict]) -> list[dict]:
    """运行 Critic 校验，返回 actions 列表。"""
    test_cases_json = json.dumps({"test_cases": test_cases}, ensure_ascii=False, indent=2)
    prompt = TEST_CASE_CRITIC_PROMPT.format(test_cases_json=test_cases_json)

    result = _call_llm_json(prompt)
    return result.get("actions", [])


# --- LLM 修订 ---

def _revise_test_cases(
    test_cases: list[dict],
    actions: list[dict],
    skill_content: str,
) -> dict:
    """根据 Critic 意见修订测试用例。"""
    test_cases_json = json.dumps({"test_cases": test_cases}, ensure_ascii=False, indent=2)
    actions_text = "\n".join(
        f"- {'用例' + str(a.get('index', '?')) + ': ' if 'index' in a else '补充: '}"
        f"{a.get('action', '?')} — {a.get('reason', '')}"
        for a in actions
    )
    prompt = TEST_CASE_REVISE_PROMPT.format(
        test_cases_json=test_cases_json,
        critic_actions=actions_text,
        skill_content=skill_content[:3000],
    )

    return _call_llm_json(prompt)


# --- 主入口 ---

def generate_test_cases(
    skill_path: str | Path,
    spec_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    ignore_cache: bool = False,
    max_retries: int = 2,
    enable_critic: bool = True,
) -> tuple[list[dict], str]:
    """生成测试用例。

    Args:
        skill_path: 技能目录路径（读取 SKILL.md）
        spec_path: SPEC 文件路径（可选，补充上下文）
        output_dir: 输出目录
        ignore_cache: 是否忽略缓存重新生成
        max_retries: JSON 解析失败时的最大重试次数
        enable_critic: 是否启用 Critic 校验和修订

    Returns:
        (测试用例列表, 保存的文件名)
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

    # 阶段 1: LLM 生成（含格式校验）
    console.print("[blue]使用 LLM 生成测试用例...[/blue]")
    test_cases_data = _generate_initial(input_content, max_retries)
    console.print(f"[green]生成并校验通过，共 {len(test_cases_data['test_cases'])} 个用例[/green]")

    # 阶段 3: Critic 校验 + 修订
    if enable_critic and test_cases_data["test_cases"]:
        console.print("[blue]Critic 校验中...[/blue]")
        actions = _run_critic(test_cases_data["test_cases"])

        if actions:
            console.print(f"[yellow]Critic 发现 {len(actions)} 个问题，进行修订...[/yellow]")
            revised_data = _revise_test_cases(
                test_cases_data["test_cases"], actions, input_content
            )

            # 阶段 4: 第二道格式校验
            console.print("[blue]修订后格式校验...[/blue]")
            test_cases_data = _validate_or_raise(revised_data)
            console.print(f"[green]最终用例数: {len(test_cases_data['test_cases'])}[/green]")
        else:
            console.print("[green]Critic 校验通过，无需修订[/green]")

    # 保存
    if output_dir:
        ensure_dir(output_dir)
        save_json(Path(output_dir) / filename, test_cases_data)
        console.print(f"[green]测试用例已保存:[/green] {Path(output_dir) / filename}")

    return test_cases_data["test_cases"], filename


def _generate_initial(input_content: str, max_retries: int) -> dict:
    """初始生成测试用例，JSON 解析或格式校验失败时重试。"""
    prompt = TEST_CASE_GEN_PROMPT.format(skill_content=input_content)

    for attempt in range(max_retries + 1):
        try:
            result = _call_llm_json(prompt)
            if "test_cases" not in result:
                raise ValueError("响应中缺少 test_cases 字段")
            return _validate_or_raise(result)
        except ValueError as e:
            if attempt < max_retries:
                console.print(f"[yellow]生成失败，重试 {attempt + 1}/{max_retries}[/yellow]")
                console.print(f"[dim]错误: {e}[/dim]")
                retry_prompt = f"""上一次生成的测试用例存在问题，请重新生成。
错误信息: {str(e)[:200]}

请严格按照以下格式输出，确保 JSON 完整且有效：
```json
{{
  "test_cases": [
    {{
      "test_point": "简洁描述验证目标",
      "question": "模拟用户提问（基础场景200字以内，含代码/数据的复杂场景500字以内）",
      "checkpoints": ["应包含的关键要素1", "应包含的关键要素2"]
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

    raise ValueError("生成测试用例失败")  # pragma: no cover


# --- JSON 解析辅助 ---

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

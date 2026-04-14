"""技能创建主逻辑。"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from rich.console import Console

from ..core.agent import SKILLFLOW_SKILLS_DIR, build_agent, run_agent
from ..core.prompts import (
    LANG_CONSTRAINT_EN,
    LANG_CONSTRAINT_ZH,
    SPEC_TO_PROMPT_TEMPLATE,
)
from ..core.utils import load_text
from .spec_parser import SkillSpec, parse_spec

console = Console()


def _get_lang_constraint(lang: str) -> str:
    """根据 lang 参数返回语言约束提示词。"""
    if lang == "zh":
        return LANG_CONSTRAINT_ZH
    elif lang == "en":
        return LANG_CONSTRAINT_EN
    return ""


def spec_to_prompt(spec: SkillSpec, name: str, lang: str = "auto") -> str:
    """将 SkillSpec 转换为自然语言 prompt。

    Args:
        spec: 技能规范对象
        name: 用户指定的技能名称
        lang: 输出语言
    """
    scenes = "\n".join(f"- {s}" for s in spec.scenes)
    rules = "\n".join(f"- {r}" for r in spec.rules)
    examples = "\n".join(
        f"- 输入：{ex.get('input', '')}\n  输出：{ex.get('output', '')}"
        for ex in spec.examples
    )

    # 加载参考文件内容
    reference_section = ""
    if spec.references:
        ref_contents = []
        for ref_path in spec.references:
            p = Path(ref_path)
            if p.exists():
                ref_contents.append(f"### 参考文件: {ref_path}\n{load_text(p)}")
            else:
                ref_contents.append(f"### 参考文件: {ref_path}（文件不存在）")
        reference_section = "## 参考文件\n" + "\n\n".join(ref_contents)

    return SPEC_TO_PROMPT_TEMPLATE.format(
        name=name,
        description=spec.description,
        scenes=scenes,
        input_desc=spec.input_desc,
        output_desc=spec.output_desc,
        rules=rules,
        examples=examples,
        reference_section=reference_section,
        lang_constraint=_get_lang_constraint(lang),
    )


def _find_created_skill_dir(response: str, name: str) -> Path | None:
    """从响应中解析或搜索 skill-creator 创建的技能目录。

    Args:
        response: agent 的响应文本
        name: 期望的技能名称

    Returns:
        技能目录路径，找不到返回 None
    """
    # 尝试从响应中解析路径
    # 匹配常见路径格式
    patterns = [
        r"技能目录[：:]\s*([^\s\n]+)",
        r"skill directory[：:]\s*([^\s\n]+)",
        r"已创建[：:]\s*([^\s\n]+)",
        r"created[：:]\s*([^\s\n]+)",
        r"路径[：:]\s*([^\s\n]+)",
        r"path[：:]\s*([^\s\n]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            path_str = match.group(1).strip()
            p = Path(path_str)
            if p.exists() and p.is_dir() and (p / "SKILL.md").exists():
                return p

    # 搜索可能的位置
    search_dirs = [
        SKILLFLOW_SKILLS_DIR,  # ~/.skillflow/skills/
        Path.cwd() / "skills",
        Path.cwd(),  # 当前目录下可能创建的技能
    ]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for d in search_dir.iterdir():
            if d.is_dir() and (d / "SKILL.md").exists():
                # 检查 SKILL.md 中的 name 字段是否匹配
                skill_md = d / "SKILL.md"
                content = skill_md.read_text(encoding="utf-8")
                # 提取 frontmatter 中的 name
                fm_match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
                if fm_match:
                    for line in fm_match.group(1).splitlines():
                        if line.startswith("name:"):
                            skill_name = line.split(":", 1)[1].strip().strip("\"'")
                            if skill_name == name:
                                return d

    return None


def create_skill(
    spec_path: str | Path,
    output_dir: str | Path,
    name: str | None = None,
    lang: str = "auto",
) -> Path:
    """根据 SPEC 文件创建技能。

    Args:
        spec_path: SPEC YAML 文件路径
        output_dir: 技能输出目录
        name: 技能名称（用于 SKILL.md frontmatter）
        lang: 输出语言 (auto/zh/en)

    Returns:
        技能目录路径
    """
    spec_path = Path(spec_path)
    output_dir = Path(output_dir)

    # 从 output_dir 推断默认名称
    if name is None:
        name = output_dir.name

    console.print(f"[blue]解析 SPEC 文件:[/blue] {spec_path}")
    spec = parse_spec(spec_path)

    console.print(f"[blue]构建创建 prompt (lang={lang}, name={name})...[/blue]")
    prompt = spec_to_prompt(spec, name=name, lang=lang)

    console.print(f"[blue]加载 skill-creator 技能，创建 agent...[/blue]")
    skills_dir = str(SKILLFLOW_SKILLS_DIR)
    agent = build_agent(skills=[skills_dir])

    console.print("[blue]生成技能中（skill-creator 执行创建流程）...[/blue]")
    response, _ = run_agent(agent, prompt)

    # 查找 skill-creator 创建的技能目录
    created_dir = _find_created_skill_dir(response, name)

    if created_dir is None:
        console.print(f"[yellow]警告: 无法找到 skill-creator 创建的技能目录[/yellow]")
        console.print(f"[yellow]请检查 agent 响应中是否包含技能路径[/yellow]")
        console.print(f"[dim]响应内容:[/dim]")
        console.print(response[:500] + "..." if len(response) > 500 else response)
        raise RuntimeError("无法找到创建的技能目录")

    console.print(f"[blue]找到技能目录:[/blue] {created_dir}")

    # 移动到目标位置
    if created_dir.resolve() != output_dir.resolve():
        if output_dir.exists():
            console.print(f"[yellow]目标目录已存在，删除:[/yellow] {output_dir}")
            shutil.rmtree(output_dir)
        shutil.move(str(created_dir), str(output_dir))
        console.print(f"[green]技能已移动到:[/green] {output_dir}")
    else:
        console.print(f"[green]技能已在目标位置:[/green] {output_dir}")

    # 确保 SKILL.md 的 name 字段是用户指定的名称
    skill_md = output_dir / "SKILL.md"
    if skill_md.exists():
        content = skill_md.read_text(encoding="utf-8")
        # 提取 frontmatter 中的 name
        fm_match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
        if fm_match:
            frontmatter = fm_match.group(1)
            for line in frontmatter.splitlines():
                if line.startswith("name:"):
                    current_name = line.split(":", 1)[1].strip().strip("\"'")
                    if current_name != name:
                        console.print(f"[yellow]修正 name 字段: {current_name} -> {name}[/yellow]")
                        # 替换 name 字段
                        new_frontmatter = re.sub(
                            r"^name:\s*.*$",
                            f"name: {name}",
                            frontmatter,
                            count=1,
                            flags=re.MULTILINE
                        )
                        new_content = "---\n" + new_frontmatter + "\n---" + content[fm_match.end():]
                        skill_md.write_text(new_content, encoding="utf-8")
                    break

    console.print(f"[green]目录结构:[/green]")
    for item in sorted(output_dir.iterdir()):
        console.print(f"  ├── {item.name}/" if item.is_dir() else f"  ├── {item.name}")

    return output_dir

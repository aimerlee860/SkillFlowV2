"""技能创建主逻辑。"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from ..core.agent import build_agent, run_agent
from ..core.prompts import (
    LANG_CONSTRAINT_EN,
    LANG_CONSTRAINT_ZH,
    SPEC_TO_PROMPT_TEMPLATE,
)
from ..core.utils import ensure_dir, load_text, save_text
from .spec_parser import SkillSpec, parse_spec

console = Console()


def _get_lang_constraint(lang: str) -> str:
    """根据 lang 参数返回语言约束提示词。"""
    if lang == "zh":
        return LANG_CONSTRAINT_ZH
    elif lang == "en":
        return LANG_CONSTRAINT_EN
    return ""


def spec_to_prompt(spec: SkillSpec, lang: str = "auto") -> str:
    """将 SkillSpec 转换为自然语言 prompt。"""
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
        description=spec.description,
        scenes=scenes,
        input_desc=spec.input_desc,
        output_desc=spec.output_desc,
        rules=rules,
        examples=examples,
        reference_section=reference_section,
        lang_constraint=_get_lang_constraint(lang),
    )


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

    console.print(f"[blue]解析 SPEC 文件:[/blue] {spec_path}")
    spec = parse_spec(spec_path)

    console.print(f"[blue]构建创建 prompt (lang={lang})...[/blue]")
    prompt = spec_to_prompt(spec, lang=lang)

    console.print(f"[blue]加载 skill-creator 技能，创建 agent...[/blue]")
    agent = build_agent(skills=["skill-creator"])

    console.print("[blue]生成技能中...[/blue]")
    response = run_agent(agent, prompt)

    # 保存结果：创建标准技能目录结构
    ensure_dir(output_dir)
    for subdir in ("assets", "scripts", "references"):
        ensure_dir(output_dir / subdir)
    skill_file = output_dir / "SKILL.md"
    save_text(skill_file, response)

    console.print(f"[green]技能已创建:[/green] {skill_file}")
    console.print(f"[green]目录结构:[/green] {output_dir}/")
    console.print(f"  ├── SKILL.md")
    console.print(f"  ├── assets/")
    console.print(f"  ├── scripts/")
    console.print(f"  └── references/")
    return output_dir

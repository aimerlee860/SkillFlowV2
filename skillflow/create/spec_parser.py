"""SPEC 文件解析器。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..core.utils import load_yaml


@dataclass
class SkillSpec:
    description: str
    scenes: list[str] = field(default_factory=list)
    input_desc: str = ""
    output_desc: str = ""
    rules: list[str] = field(default_factory=list)
    examples: list[dict[str, str]] = field(default_factory=list)
    references: list[str] = field(default_factory=list)


def parse_spec(path: str | Path) -> SkillSpec:
    """解析 YAML SPEC 文件。"""
    data = load_yaml(path)

    io_data = data.get("input_output", data.get("input-output", {}))

    return SkillSpec(
        description=data.get("description", ""),
        scenes=data.get("scene", data.get("scenes", [])),
        input_desc=io_data.get("input", ""),
        output_desc=io_data.get("output", ""),
        rules=data.get("rule", data.get("rules", [])),
        examples=data.get("example", data.get("examples", [])),
        references=data.get("reference", data.get("references", [])),
    )

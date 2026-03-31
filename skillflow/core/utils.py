"""公共工具函数。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


def compute_hash(content: str) -> str:
    """计算内容的 SHA256 哈希值。"""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def load_yaml(path: str | Path) -> dict:
    """加载 YAML 文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path: str | Path, data: dict | list, indent: int = 2) -> None:
    """保存数据为 JSON 文件。"""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_json(path: str | Path) -> dict | list:
    """加载 JSON 文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_text(path: str | Path) -> str:
    """加载文本文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def save_text(path: str | Path, content: str) -> None:
    """保存文本文件。"""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在。"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_skill_dir(skill_path: str | Path) -> Path:
    """校验技能目录是否存在且包含 SKILL.md。

    Raises:
        FileNotFoundError: 目录不存在或缺少 SKILL.md
    """
    p = Path(skill_path).resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"技能目录不存在: {p}")
    skill_file = p / "SKILL.md"
    if not skill_file.is_file():
        raise FileNotFoundError(f"技能目录中缺少 SKILL.md: {skill_file}")
    return p

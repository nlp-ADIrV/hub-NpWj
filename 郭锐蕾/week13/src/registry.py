"""Skill 注册表：扫描 SKILL.md，解析 frontmatter，生成常驻索引。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .models import SkillMeta
from .token_utils import estimate_tokens


FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """轻量 YAML frontmatter 解析（不依赖 PyYAML）。"""
    m = FRONTMATTER_RE.match(text.strip())
    if not m:
        return {}, text
    raw_fm, body = m.group(1), m.group(2)
    meta: dict[str, Any] = {}
    current_key: str | None = None
    folded: list[str] = []

    def flush() -> None:
        nonlocal current_key, folded
        if current_key is None:
            return
        value = "\n".join(folded).strip()
        if current_key == "trigger":
            # trigger: a | b | c  或  trigger: "a, b"
            parts = re.split(r"[|,]", value)
            meta[current_key] = [p.strip().strip("\"'") for p in parts if p.strip()]
        else:
            meta[current_key] = value.strip("\"'")
        current_key, folded = None, []

    for line in raw_fm.splitlines():
        if not line.strip():
            continue
        # 折叠标量 continuation（description: >- 后的缩进行）
        if current_key and (line.startswith("  ") or line.startswith("\t")):
            folded.append(line.strip())
            continue
        km = re.match(r"^([A-Za-z0-9_-]+)\s*:\s*(.*)$", line)
        if not km:
            if current_key:
                folded.append(line.strip())
            continue
        flush()
        key, val = km.group(1), km.group(2).strip()
        if val in (">", ">-", "|", "|-"):
            current_key, folded = key, []
        elif val == "":
            current_key, folded = key, []
        else:
            current_key, folded = key, [val]
    flush()
    return meta, body


# 从描述/正文中额外抽取的短触发词（避免把整句例句塞进 triggers）
_EXTRA_KEYWORD_PATTERNS = [
    r"闪卡",
    r"flash\s*card",
    r"单词卡",
    r"画(?:个|一个)?(?:图|架构|流程|时序)?",
    r"架构图",
    r"流程图",
    r"时序图",
    r"思维导图",
    r"diagram",
    r"flowchart",
    r"审查代码",
    r"code\s*review",
    r"幻灯片",
    r"\bppt\b",
]


def _extract_triggers_from_body(body: str, description: str) -> list[str]:
    """从正文「触发场景」或 description 中抽取短触发词。"""
    blob = f"{description}\n{body}"
    triggers: list[str] = []
    for pat in _EXTRA_KEYWORD_PATTERNS:
        for m in re.finditer(pat, blob, flags=re.I):
            triggers.append(re.sub(r"\s+", " ", m.group(0)).strip())
    # description 里有意义的英文词（过滤停用词）
    stop = {
        "when", "the", "user", "asks", "for", "and", "with", "use", "from",
        "that", "this", "make", "html", "static", "english", "word",
    }
    for kw in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", description):
        if kw.lower() not in stop:
            triggers.append(kw.lower())
    seen: set[str] = set()
    out: list[str] = []
    for t in triggers:
        key = t.lower()
        if key not in seen and len(t) <= 24:
            seen.add(key)
            out.append(t)
    return out[:16]


class SkillRegistry:
    """
    扫描 skills_dir/*/SKILL.md，构建：
      1. SkillMeta 列表（供匹配）
      2. skills_index.md 文本（常驻层，目标 < 200 tokens）
    """

    def __init__(self, skills_dir: Path):
        self.skills_dir = Path(skills_dir)
        self.skills: dict[str, SkillMeta] = {}
        self.index_md: str = ""
        self.refresh()

    def refresh(self) -> None:
        self.skills.clear()
        if not self.skills_dir.exists():
            self.index_md = "# Skills Index\n\n（未发现任何 Skill）\n"
            return

        for skill_md in sorted(self.skills_dir.glob("*/SKILL.md")):
            raw = skill_md.read_text(encoding="utf-8")
            fm, body = parse_frontmatter(raw)
            name = str(fm.get("name") or skill_md.parent.name).strip()
            description = str(fm.get("description") or "").strip()
            version = str(fm.get("version") or "").strip()

            triggers: list[str] = []
            if isinstance(fm.get("trigger"), list):
                triggers = [str(t) for t in fm["trigger"]]
            elif isinstance(fm.get("trigger"), str) and fm["trigger"]:
                triggers = [t.strip() for t in re.split(r"[|,]", fm["trigger"]) if t.strip()]
            # 补充从正文/描述推断的触发词
            triggers.extend(_extract_triggers_from_body(body, description))
            # name 本身也是强触发
            triggers.insert(0, name)
            # 去重
            uniq: list[str] = []
            seen: set[str] = set()
            for t in triggers:
                k = t.lower()
                if k not in seen:
                    seen.add(k)
                    uniq.append(t)

            meta = SkillMeta(
                name=name,
                description=description or name,
                path=skill_md.parent,
                skill_md=skill_md,
                triggers=uniq,
                version=version,
                has_references=(skill_md.parent / "references").is_dir(),
                has_scripts=(skill_md.parent / "scripts").is_dir(),
            )
            self.skills[name] = meta

        lines = [
            "# Skills Index",
            "",
        ]
        for meta in self.skills.values():
            lines.append(meta.index_line())
        lines.append("")
        self.index_md = "\n".join(lines)

    def list_skills(self) -> list[SkillMeta]:
        return list(self.skills.values())

    def get(self, name: str) -> SkillMeta | None:
        return self.skills.get(name)

    def index_stats(self) -> dict[str, Any]:
        tokens = estimate_tokens(self.index_md)
        return {
            "skill_count": len(self.skills),
            "index_chars": len(self.index_md),
            "index_tokens": tokens,
            "under_200_tokens": tokens < 200,
            "skills": [
                {
                    "name": s.name,
                    "description": s.description[:80],
                    "triggers": s.triggers[:6],
                    "has_references": s.has_references,
                    "has_scripts": s.has_scripts,
                }
                for s in self.skills.values()
            ],
        }

    def write_index(self, path: Path | None = None) -> Path:
        out = path or (self.skills_dir / "SKILLS_INDEX.md")
        out.write_text(self.index_md, encoding="utf-8")
        return out

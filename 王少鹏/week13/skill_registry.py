"""
Skill 注册中心 —— 渐进式加载的两阶段实现

教学重点：
  Phase 0 (meta): 只解析每个 SKILL.md 的 frontmatter（name + description），
                  全部技能的 meta 汇总仅约 ~100 token，可始终放在 system prompt 里。
  Phase 1 (full): 只有 LLM 主动执行 read_skill(name=...) 时才把对应 SKILL.md
                  全文读入上下文（懒加载），省 token、聚焦注意力。

用法：
  from src.skill_registry import SkillRegistry
  reg = SkillRegistry()
  reg.meta_summary()          # -> "名字: 描述" 多行文本（塞进 system prompt）
  reg.load_full("flash-card") # -> 完整 SKILL.md 文本（懒加载全文）
"""

import re
from dataclasses import dataclass
from pathlib import Path

# 项目根目录 = src/ 的上一级（agent_memory_system/）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = PROJECT_ROOT / "skills"

# frontmatter：---\n ... \n---
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)

# YAML 块标量（block scalar）起始符：>（折叠）、|（字面量）及其 "-"/"+" 变体
_BLOCK_SCALAR = {">", ">-", "|", "|-"}


def _parse_frontmatter(text: str) -> dict:
    """从文本开头提取 frontmatter 并解析成 dict（只支持教学所需的简单子集）"""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}
    meta: dict[str, str] = {}
    block_key: str | None = None
    block_lines: list[str] = []

    for raw in m.group(1).splitlines():
        line = raw.rstrip()
        if block_key:
            # 块标量：后续行原样拼接
            if line.strip() == "":
                block_lines.append("")
            else:
                block_lines.append(line.lstrip())
            continue

        if not line.strip() or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if val in _BLOCK_SCALAR:
            block_key = key
            block_lines = []
            continue
        # 去掉首尾引号
        if len(val) >= 2 and val[0] in "\"'" and val[-1] == val[0]:
            val = val[1:-1]
        meta[key] = val

    if block_key:
        meta[block_key] = "\n".join(block_lines).strip()

    return meta


@dataclass
class SkillMeta:
    name: str          # frontmatter 里的 name
    description: str   # frontmatter 里的 description
    path: Path         # 技能目录（含 SKILL.md）
    has_scripts: bool  # 是否有 scripts/ 目录（工具型 vs 人格型）


class SkillRegistry:
    """扫描 skills/ 目录，维护每个 skill 的 meta；全文按需加载。"""

    def __init__(self, skills_dir: Path = SKILLS_DIR):
        self.skills_dir = Path(skills_dir)
        self._metas: dict[str, SkillMeta] = {}
        self._scan()

    def _scan(self):
        """遍历 skills/ 下每个含 SKILL.md 的子目录，解析 meta（Phase 0）。"""
        if not self.skills_dir.exists():
            return
        for child in sorted(self.skills_dir.iterdir()):
            skill_md = child / "SKILL.md"
            if not child.is_dir() or not skill_md.exists():
                continue
            fm = _parse_frontmatter(skill_md.read_text(encoding="utf-8"))
            name = fm.get("name", child.name)
            scripts_dir = child / "scripts"
            self._metas[name] = SkillMeta(
                name=name,
                description=fm.get("description", "").strip(),
                path=child,
                has_scripts=scripts_dir.is_dir(),
            )

    # ── 查询接口 ──────────────────────────────────────────────────────
    def names(self) -> list[str]:
        return list(self._metas)

    def exists(self, name: str) -> bool:
        return name in self._metas

    def meta(self, name: str) -> SkillMeta:
        return self._metas[name]

    def tool_skills(self) -> list[SkillMeta]:
        """有 scripts/ 目录的工具型技能（可被 executor 执行）"""
        return [m for m in self._metas.values() if m.has_scripts]

    # ── Phase 0：meta 汇总（轻量，常驻 system prompt）────────────────
    def meta_summary(self) -> str:
        """全部技能的「名字 + 描述」摘要，约 ~100 token"""
        lines = []
        for m in self._metas.values():
            desc = (m.description or "(无描述)").replace("\n", " ")
            lines.append(f"- {m.name}: {desc}")
        return "\n".join(lines) if lines else "(暂无可用技能)"

    # ── Phase 1：懒加载全文（LLM read_skill 时才调用）──────────────
    def load_full(self, name: str) -> str:
        """返回 SKILL.md 完整内容；未知技能抛 KeyError。"""
        if name not in self._metas:
            raise KeyError(f"未知技能: {name}")
        return (self._metas[name].path / "SKILL.md").read_text(encoding="utf-8")

    def skill_dir(self, name: str) -> Path:
        return self._metas[name].path

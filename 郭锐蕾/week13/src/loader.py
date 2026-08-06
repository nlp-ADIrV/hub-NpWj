"""渐进式加载器：按需读 SKILL.md / references / scripts 清单。"""

from __future__ import annotations

from pathlib import Path

from .models import SkillBody, SkillMeta
from .registry import parse_frontmatter


# 图表类型 → reference 文件名（对齐 baoyu-diagram）
REFERENCE_ALIASES: dict[str, str] = {
    "architecture": "architecture.md",
    "架构": "architecture.md",
    "flowchart": "flowchart.md",
    "流程": "flowchart.md",
    "sequence": "sequence.md",
    "时序": "sequence.md",
    "structural": "structural.md",
    "结构": "structural.md",
    "er": "structural.md",
    "class": "structural.md",
}


class ProgressiveLoader:
    """
    三层加载：
      Always   — 由 Registry 提供索引（本类不重复加载）
      OnDemand — load_skill_body()
      InContext— load_reference() / list_scripts()
    """

    def __init__(self) -> None:
        self._loaded_bodies: dict[str, SkillBody] = {}
        self._loaded_refs: dict[str, str] = {}

    def load_skill_body(self, meta: SkillMeta) -> SkillBody:
        if meta.name in self._loaded_bodies:
            return self._loaded_bodies[meta.name]
        raw = meta.skill_md.read_text(encoding="utf-8")
        fm, body = parse_frontmatter(raw)
        skill_body = SkillBody(meta=meta, frontmatter=fm, body=body, raw=raw)
        self._loaded_bodies[meta.name] = skill_body
        return skill_body

    def unload_skill(self, name: str) -> bool:
        """任务完成后释放触发层 / 执行层内容。"""
        gone = name in self._loaded_bodies
        self._loaded_bodies.pop(name, None)
        # 释放该 skill 相关 references
        prefix = f"{name}::"
        for key in list(self._loaded_refs):
            if key.startswith(prefix):
                del self._loaded_refs[key]
        return gone

    def unload_all(self) -> None:
        self._loaded_bodies.clear()
        self._loaded_refs.clear()

    def loaded_names(self) -> list[str]:
        return list(self._loaded_bodies.keys())

    def list_references(self, meta: SkillMeta) -> list[Path]:
        ref_dir = meta.path / "references"
        if not ref_dir.is_dir():
            return []
        return sorted(ref_dir.glob("*.md"))

    def list_scripts(self, meta: SkillMeta) -> list[Path]:
        scripts_dir = meta.path / "scripts"
        if not scripts_dir.is_dir():
            return []
        files: list[Path] = []
        for pattern in ("*.py", "*.ts", "*.js", "*.sh"):
            files.extend(scripts_dir.glob(pattern))
        return sorted(files)

    def infer_reference_name(self, user_message: str) -> str | None:
        msg = user_message.lower()
        for alias, filename in REFERENCE_ALIASES.items():
            if alias.lower() in msg or alias in user_message:
                return filename
        return None

    def load_reference(self, meta: SkillMeta, filename: str) -> str | None:
        key = f"{meta.name}::{filename}"
        if key in self._loaded_refs:
            return self._loaded_refs[key]
        path = meta.path / "references" / filename
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8")
        self._loaded_refs[key] = text
        return text

    def load_secondary_for_message(
        self, meta: SkillMeta, user_message: str
    ) -> list[tuple[str, str]]:
        """
        按用户消息推断需要的二级文档。
        返回 [(source_label, content), ...]
        """
        loaded: list[tuple[str, str]] = []
        filename = self.infer_reference_name(user_message)
        if filename:
            content = self.load_reference(meta, filename)
            if content:
                loaded.append((f"references/{filename}", content))
                return loaded

        # 未明确类型时：若只有一个 reference，可按需加载；多个则不自动全量加载
        refs = self.list_references(meta)
        if len(refs) == 1:
            content = self.load_reference(meta, refs[0].name)
            if content:
                loaded.append((f"references/{refs[0].name}", content))
        return loaded

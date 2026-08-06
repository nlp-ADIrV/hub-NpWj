"""Harness 核心数据模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class LoadLayer(str, Enum):
    """对应课件渐进式披露三层。"""

    ALWAYS = "always"          # 常驻层：技能索引
    ON_DEMAND = "on_demand"    # 触发层：完整 SKILL.md
    IN_CONTEXT = "in_context"  # 执行层：references / scripts 产出


@dataclass
class SkillMeta:
    """常驻索引条目：仅 frontmatter 级摘要。"""

    name: str
    description: str
    path: Path
    skill_md: Path
    triggers: list[str] = field(default_factory=list)
    version: str = ""
    has_references: bool = False
    has_scripts: bool = False

    def index_line(self) -> str:
        """生成常驻索引中的一行摘要（对齐课件：仅 name + 短描述）。"""
        short_desc = self.description.replace("\n", " ").strip()
        # 取第一句 / 前 36 字，控制 Always 层 token
        for sep in ("。", ".", "；", ";"):
            if sep in short_desc:
                short_desc = short_desc.split(sep)[0]
                break
        if len(short_desc) > 36:
            short_desc = short_desc[:33] + "..."
        return f"- [{self.name}]({self.name}/SKILL.md) — {short_desc}"


@dataclass
class SkillBody:
    """触发后加载的完整 Skill 定义。"""

    meta: SkillMeta
    frontmatter: dict[str, Any]
    body: str
    raw: str

    @property
    def char_count(self) -> int:
        return len(self.raw)


@dataclass
class ContextLayer:
    """注入 Context 的一层明细（便于可视化与 token 统计）。"""

    layer: LoadLayer
    name: str
    source: str
    content: str
    char_count: int = 0
    token_estimate: int = 0

    def __post_init__(self) -> None:
        from .token_utils import estimate_tokens

        self.char_count = len(self.content)
        self.token_estimate = estimate_tokens(self.content)


@dataclass
class MatchResult:
    skill: SkillMeta
    score: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class LifecycleEvent:
    """生命周期事件，供 SSE / CLI 展示。"""

    step: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class HarnessTurnResult:
    """一轮用户请求的完整处理结果。"""

    user_message: str
    matched: MatchResult | None
    layers: list[ContextLayer]
    system_prompt: str
    events: list[LifecycleEvent]
    execution_notes: list[str] = field(default_factory=list)
    secondary_loaded: list[str] = field(default_factory=list)
    script_outputs: list[dict[str, Any]] = field(default_factory=list)
    released: bool = False
    comparison: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_message": self.user_message,
            "matched": (
                {
                    "name": self.matched.skill.name,
                    "score": self.matched.score,
                    "reasons": self.matched.reasons,
                }
                if self.matched
                else None
            ),
            "layers": [
                {
                    "layer": ly.layer.value,
                    "name": ly.name,
                    "source": ly.source,
                    "char_count": ly.char_count,
                    "token_estimate": ly.token_estimate,
                    "preview": ly.content[:240] + ("..." if len(ly.content) > 240 else ""),
                }
                for ly in self.layers
            ],
            "events": [
                {"step": e.step, "message": e.message, "data": e.data} for e in self.events
            ],
            "execution_notes": self.execution_notes,
            "secondary_loaded": self.secondary_loaded,
            "script_outputs": self.script_outputs,
            "released": self.released,
            "comparison": self.comparison,
            "system_prompt_chars": len(self.system_prompt),
        }

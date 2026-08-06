"""Context 组装与生命周期：索引常驻 → 触发加载 → 执行驻留 → 释放。"""

from __future__ import annotations

from .loader import ProgressiveLoader
from .matcher import TriggerMatcher
from .models import ContextLayer, LifecycleEvent, LoadLayer, MatchResult
from .registry import SkillRegistry
from .token_utils import estimate_tokens


BASE_AGENT_PROMPT = """# Agent Harness 运行说明

你是一个支持 Skills 渐进式披露的 Agent。
- 下方「Skills Index」始终可见，仅含摘要。
- 仅当用户请求匹配到某个 Skill 时，完整 SKILL.md 才会注入。
- 执行期间可按 Skill 指示继续读取 references/ 或运行 scripts/。
- 任务结束后，完整 Skill 正文应从上下文释放，仅保留索引。
"""


class ContextAssembler:
    def __init__(
        self,
        registry: SkillRegistry,
        matcher: TriggerMatcher,
        loader: ProgressiveLoader,
    ):
        self.registry = registry
        self.matcher = matcher
        self.loader = loader

    def assemble(
        self,
        user_message: str,
        *,
        load_secondary: bool = True,
        force_skill: str | None = None,
    ) -> tuple[str, list[ContextLayer], MatchResult | None, list[LifecycleEvent], list[str]]:
        events: list[LifecycleEvent] = []
        layers: list[ContextLayer] = []
        secondary_loaded: list[str] = []

        # Step 01: 用户消息
        events.append(
            LifecycleEvent("01_user_message", "收到用户消息", {"message": user_message})
        )

        # Always: base + index
        layers.append(
            ContextLayer(
                layer=LoadLayer.ALWAYS,
                name="base_prompt",
                source="harness://base",
                content=BASE_AGENT_PROMPT,
            )
        )
        layers.append(
            ContextLayer(
                layer=LoadLayer.ALWAYS,
                name="skills_index",
                source="SKILLS_INDEX.md",
                content=self.registry.index_md,
            )
        )
        events.append(
            LifecycleEvent(
                "always_index",
                "常驻层已注入 Skills Index",
                {
                    "tokens": estimate_tokens(self.registry.index_md),
                    "skill_count": len(self.registry.skills),
                },
            )
        )

        # Step 02: 触发匹配
        matched: MatchResult | None = None
        if force_skill:
            meta = self.registry.get(force_skill)
            if meta:
                matched = MatchResult(skill=meta, score=99.0, reasons=["强制指定"])
        else:
            matched = self.matcher.match(user_message)

        if not matched:
            events.append(
                LifecycleEvent(
                    "02_no_match",
                    "未匹配到 Skill，仅保留常驻索引",
                    {"candidates": self.matcher.explain(user_message)},
                )
            )
            prompt = self._join(layers)
            return prompt, layers, None, events, secondary_loaded

        events.append(
            LifecycleEvent(
                "02_trigger_match",
                f"触发匹配命中：{matched.skill.name}",
                {
                    "score": matched.score,
                    "reasons": matched.reasons,
                    "candidates": self.matcher.explain(user_message),
                },
            )
        )

        # Step 03: 加载完整 Skill
        body = self.loader.load_skill_body(matched.skill)
        skill_section = (
            f"## Active Skill: {matched.skill.name}\n\n"
            f"**baseDir:** `{matched.skill.path.as_posix()}`\n\n"
            f"{body.raw}"
        )
        layers.append(
            ContextLayer(
                layer=LoadLayer.ON_DEMAND,
                name=f"skill:{matched.skill.name}",
                source=str(matched.skill.skill_md),
                content=skill_section,
            )
        )
        events.append(
            LifecycleEvent(
                "03_load_skill",
                f"已加载完整 Skill 定义：{matched.skill.name}",
                {
                    "chars": len(body.raw),
                    "tokens": estimate_tokens(body.raw),
                    "path": str(matched.skill.skill_md),
                },
            )
        )

        # Step 04 准备：二级加载（执行层）
        if load_secondary and matched.skill.has_references:
            for source, content in self.loader.load_secondary_for_message(
                matched.skill, user_message
            ):
                layers.append(
                    ContextLayer(
                        layer=LoadLayer.IN_CONTEXT,
                        name=f"ref:{source}",
                        source=source,
                        content=f"## Skill Reference: {source}\n\n{content}",
                    )
                )
                secondary_loaded.append(source)
            if secondary_loaded:
                events.append(
                    LifecycleEvent(
                        "04_secondary_load",
                        f"执行层按需加载 references：{', '.join(secondary_loaded)}",
                        {"files": secondary_loaded},
                    )
                )
            else:
                events.append(
                    LifecycleEvent(
                        "04_secondary_skip",
                        "存在 references/ 但未自动全量加载（避免 context 膨胀）；"
                        "仅在消息暗示具体类型时加载对应文档",
                        {
                            "available": [
                                p.name for p in self.loader.list_references(matched.skill)
                            ]
                        },
                    )
                )

        scripts = self.loader.list_scripts(matched.skill)
        if scripts:
            listing = "\n".join(f"- `{p.name}`" for p in scripts)
            script_note = f"## Available Scripts\n\n{listing}\n"
            layers.append(
                ContextLayer(
                    layer=LoadLayer.IN_CONTEXT,
                    name="scripts_listing",
                    source="scripts/",
                    content=script_note,
                )
            )
            events.append(
                LifecycleEvent(
                    "04_scripts_listed",
                    f"列出可执行脚本 {len(scripts)} 个（尚未执行）",
                    {"scripts": [p.name for p in scripts]},
                )
            )

        prompt = self._join(layers)
        return prompt, layers, matched, events, secondary_loaded

    def release(self, skill_name: str | None) -> LifecycleEvent:
        if skill_name:
            self.loader.unload_skill(skill_name)
            return LifecycleEvent(
                "05_release",
                f"已释放 Skill「{skill_name}」正文与 references，Context 恢复为仅索引",
                {"remaining_loaded": self.loader.loaded_names()},
            )
        self.loader.unload_all()
        return LifecycleEvent("05_release", "已释放全部按需加载内容", {})

    def compare_full_vs_progressive(self, layers: list[ContextLayer]) -> dict:
        """对比「全量加载所有 SKILL.md」vs「本轮渐进式加载」。"""
        progressive_tokens = sum(ly.token_estimate for ly in layers)
        progressive_chars = sum(ly.char_count for ly in layers)

        full_parts = [BASE_AGENT_PROMPT, self.registry.index_md]
        for meta in self.registry.list_skills():
            full_parts.append(meta.skill_md.read_text(encoding="utf-8"))
            # 全量还会把所有 references 塞进去（对照课件「全量加载」代价）
            for ref in (meta.path / "references").glob("*.md") if meta.has_references else []:
                full_parts.append(ref.read_text(encoding="utf-8"))
        full_text = "\n\n---\n\n".join(full_parts)
        full_tokens = estimate_tokens(full_text)
        saved = max(0, full_tokens - progressive_tokens)
        ratio = (saved / full_tokens * 100) if full_tokens else 0.0
        return {
            "full_load_tokens": full_tokens,
            "full_load_chars": len(full_text),
            "progressive_tokens": progressive_tokens,
            "progressive_chars": progressive_chars,
            "saved_tokens": saved,
            "saved_ratio_percent": round(ratio, 1),
            "index_only_tokens": estimate_tokens(self.registry.index_md),
        }

    @staticmethod
    def _join(layers: list[ContextLayer]) -> str:
        return "\n\n---\n\n".join(ly.content for ly in layers if ly.content.strip())

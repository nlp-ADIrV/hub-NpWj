"""SkillHarness：渐进式加载执行的统一门面。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .context import ContextAssembler
from .executor import SkillExecutor
from .loader import ProgressiveLoader
from .matcher import TriggerMatcher
from .models import HarnessTurnResult, LifecycleEvent
from .registry import SkillRegistry


class SkillHarness:
    """
    课件对齐的生命周期：
      01 用户发消息
      02 触发条件匹配
      03 加载 Skill 定义
      04 执行 Skill 流程（含二级 references / scripts）
      05 任务完成 / 释放
    """

    def __init__(self, skills_dir: Path | str, workspace: Path | str | None = None):
        self.skills_dir = Path(skills_dir)
        self.workspace = Path(workspace) if workspace else self.skills_dir.parent
        self.registry = SkillRegistry(self.skills_dir)
        self.loader = ProgressiveLoader()
        self.matcher = TriggerMatcher(self.registry)
        self.assembler = ContextAssembler(self.registry, self.matcher, self.loader)
        self.executor = SkillExecutor(self.loader, workspace=self.workspace)

    def refresh(self) -> None:
        self.registry.refresh()
        self.registry.write_index(self.workspace / "outputs" / "SKILLS_INDEX.md")

    def handle(
        self,
        user_message: str,
        *,
        mode: str = "demo",
        load_secondary: bool = True,
        force_skill: str | None = None,
        auto_release: bool = True,
        on_event: Callable[[LifecycleEvent], None] | None = None,
    ) -> HarnessTurnResult:
        events: list[LifecycleEvent] = []

        def track(ev: LifecycleEvent) -> None:
            events.append(ev)
            if on_event:
                on_event(ev)

        prompt, layers, matched, assemble_events, secondary = self.assembler.assemble(
            user_message,
            load_secondary=load_secondary,
            force_skill=force_skill,
        )
        for ev in assemble_events:
            track(ev)

        notes, script_outputs, exec_events = self.executor.execute(
            user_message,
            matched,
            mode=mode,
            on_event=track,
        )
        # execute 已通过 track 写入 events；exec_events 仅作兜底（无重复追加）
        _ = exec_events

        comparison = self.assembler.compare_full_vs_progressive(layers)

        released = False
        if auto_release:
            skill_name = matched.skill.name if matched else None
            release_ev = self.assembler.release(skill_name)
            track(release_ev)
            released = True

        return HarnessTurnResult(
            user_message=user_message,
            matched=matched,
            layers=layers,
            system_prompt=prompt,
            events=events,
            execution_notes=notes,
            secondary_loaded=secondary,
            script_outputs=script_outputs,
            released=released,
            comparison=comparison,
        )

    def status(self) -> dict[str, Any]:
        return {
            "skills_dir": str(self.skills_dir),
            "workspace": str(self.workspace),
            "index": self.registry.index_stats(),
            "loaded_bodies": self.loader.loaded_names(),
        }

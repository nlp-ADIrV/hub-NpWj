"""Skill 执行器：演示脚本调用；可选 LLM 模式。"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from .loader import ProgressiveLoader
from .models import LifecycleEvent, MatchResult, SkillMeta


EventCallback = Callable[[LifecycleEvent], None]


class SkillExecutor:
    """
    执行策略：
      1. dry_run / demo：按已知 Skill 跑确定性演示（不依赖 API Key）
      2. llm：把组装好的 system prompt + 用户消息交给 LLM（可选）
    """

    def __init__(self, loader: ProgressiveLoader, workspace: Path | None = None):
        self.loader = loader
        self.workspace = Path(workspace) if workspace else Path.cwd()

    def execute(
        self,
        user_message: str,
        matched: MatchResult | None,
        *,
        mode: str = "demo",
        on_event: EventCallback | None = None,
    ) -> tuple[list[str], list[dict[str, Any]], list[LifecycleEvent]]:
        notes: list[str] = []
        outputs: list[dict[str, Any]] = []
        events: list[LifecycleEvent] = []

        def emit(step: str, message: str, data: dict | None = None) -> None:
            ev = LifecycleEvent(step, message, data or {})
            events.append(ev)
            if on_event:
                on_event(ev)

        if not matched:
            notes.append("未匹配 Skill，跳过执行。")
            emit("04_execute_skip", "无匹配 Skill，不执行")
            return notes, outputs, events

        skill = matched.skill
        emit(
            "04_execute_start",
            f"开始执行 Skill：{skill.name}",
            {"mode": mode},
        )

        if mode == "llm":
            notes.append(self._run_llm(user_message, skill))
            emit("04_execute_llm", "已调用 LLM 完成一轮回复")
            return notes, outputs, events

        # demo 模式：按 skill 名称走确定性管线
        if skill.name == "flash-card":
            out = self._demo_flash_card(user_message, skill)
            outputs.append(out)
            notes.append(out.get("summary", "flash-card 执行完成"))
            emit("04_execute_script", "已运行 flash-card 脚本管线", out)
        elif skill.name == "baoyu-diagram":
            out = self._demo_baoyu_diagram(user_message, skill)
            outputs.append(out)
            notes.append(out.get("summary", "baoyu-diagram 执行完成"))
            emit("04_execute_plan", "已规划 baoyu-diagram 执行步骤（SVG 由 Agent/脚本生成）", out)
        else:
            out = self._demo_generic(user_message, skill)
            outputs.append(out)
            notes.append(out.get("summary", f"{skill.name} 执行计划已生成"))
            emit("04_execute_plan", f"已生成 {skill.name} 执行计划", out)

        return notes, outputs, events

    def _extract_word(self, message: str) -> str | None:
        # 「crazy 的闪卡」「flash card for resilient」
        m = re.search(
            r"(?:闪卡|flash\s*card|单词卡).*?([A-Za-z][A-Za-z'-]+)|"
            r"([A-Za-z][A-Za-z'-]+).*?(?:闪卡|flash\s*card|单词卡)|"
            r"(?:做|生成|make).*?([A-Za-z][A-Za-z'-]+)",
            message,
            re.I,
        )
        if m:
            for g in m.groups():
                if g:
                    return g.lower()
        words = re.findall(r"\b[A-Za-z]{3,}\b", message)
        stop = {"flash", "card", "make", "word", "the", "for", "html", "skill"}
        for w in words:
            if w.lower() not in stop:
                return w.lower()
        return None

    def _demo_flash_card(self, message: str, skill: SkillMeta) -> dict[str, Any]:
        word = self._extract_word(message) or "resilient"
        data_path = skill.path / "data" / f"{word}.json"
        script = skill.path / "scripts" / "make_flashcard.py"
        out_html = self.workspace / "outputs" / f"{word}.html"
        out_html.parent.mkdir(parents=True, exist_ok=True)

        if not data_path.exists():
            # 生成最小占位 JSON，保证管线可跑通
            sample = {
                "word": word,
                "phonetic": f"/{word}/",
                "pos": "n.",
                "definition": f"（演示）{word} 的中文释义",
                "examples": [
                    {"en": f"This is a demo sentence with {word}.", "zh": f"这是包含 {word} 的演示例句。"},
                    {"en": f"Students learn the word {word} today.", "zh": f"学生们今天学习单词 {word}。"},
                    {"en": f"Remember to review {word} often.", "zh": f"记得经常复习 {word}。"},
                ],
                "synonyms": ["example", "sample", "demo", "placeholder"],
            }
            data_path.parent.mkdir(parents=True, exist_ok=True)
            data_path.write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")
            created_data = True
        else:
            created_data = False

        result: dict[str, Any] = {
            "skill": "flash-card",
            "word": word,
            "data_path": str(data_path),
            "created_data": created_data,
            "html_path": str(out_html),
        }

        if script.exists():
            proc = subprocess.run(
                [sys.executable, str(script), str(data_path), "-o", str(out_html)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            result["returncode"] = proc.returncode
            result["stdout"] = (proc.stdout or "").strip()
            result["stderr"] = (proc.stderr or "").strip()
            result["summary"] = (
                f"flash-card：已为「{word}」生成 {out_html.name}"
                if proc.returncode == 0
                else f"flash-card 脚本失败：{result['stderr'][:200]}"
            )
        else:
            result["summary"] = "未找到 make_flashcard.py"
        return result

    def _demo_baoyu_diagram(self, message: str, skill: SkillMeta) -> dict[str, Any]:
        ref_name = self.loader.infer_reference_name(message)
        available = [p.name for p in self.loader.list_references(skill)]
        steps = [
            "确定 baseDir = skill 目录",
            "根据用户意图选择图表类型",
            f"按需读取 references/{ref_name or '<type>.md'}（二级加载）",
            "按 SKILL.md 设计系统生成单个 .svg",
            "可选：bun scripts/main.ts 转为 PNG",
        ]
        return {
            "skill": "baoyu-diagram",
            "inferred_reference": ref_name,
            "available_references": available,
            "steps": steps,
            "summary": (
                f"baoyu-diagram：推断类型 → {ref_name or '未指定（不自动全量加载 references）'}；"
                f"可用参考文档 {len(available)} 个"
            ),
        }

    def _demo_generic(self, message: str, skill: SkillMeta) -> dict[str, Any]:
        scripts = [p.name for p in self.loader.list_scripts(skill)]
        refs = [p.name for p in self.loader.list_references(skill)]
        return {
            "skill": skill.name,
            "message": message,
            "scripts": scripts,
            "references": refs,
            "steps": [
                "读取已注入的 SKILL.md 指令",
                "按需加载 references（若需要）",
                "按步骤调用 scripts / 外部工具",
                "产出结果并准备释放 Context",
            ],
            "summary": f"{skill.name}：已生成执行计划（scripts={len(scripts)}, refs={len(refs)}）",
        }

    def _run_llm(self, user_message: str, skill: SkillMeta) -> str:
        try:
            from .llm_config import get_chat_client
        except Exception as e:
            return f"LLM 不可用：{e}"

        # 调用方应已把 skill 注入 system；这里做最小补强
        body = self.loader.load_skill_body(skill)
        try:
            client, model = get_chat_client()
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"你正在执行 Skill「{skill.name}」。严格遵循下列说明书：\n\n{body.raw}"
                        ),
                    },
                    {"role": "user", "content": user_message},
                ],
                temperature=0.3,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            return f"LLM 调用失败：{e}"

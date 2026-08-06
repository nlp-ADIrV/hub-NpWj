"""触发匹配：关键词 / 名称 / 描述相似度 → 选出要加载的 Skill。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .models import MatchResult, SkillMeta
from .registry import SkillRegistry


@dataclass
class MatcherConfig:
    min_score: float = 1.0
    top_k: int = 3


def _normalize(text: str) -> str:
    return text.lower().strip()


def _tokenize(text: str) -> set[str]:
    """中英混合简单分词。"""
    text = _normalize(text)
    en = set(re.findall(r"[a-z0-9][a-z0-9_-]{1,}", text))
    # 抽出 2~6 字中文片段（滑动）
    zh_chars = re.findall(r"[\u4e00-\u9fff]+", text)
    zh: set[str] = set()
    for block in zh_chars:
        zh.add(block)
        for n in (2, 3, 4):
            for i in range(len(block) - n + 1):
                zh.add(block[i : i + n])
    return en | zh


class TriggerMatcher:
    """
    规则优先的触发匹配器。
    分数构成：
      - 触发词精确命中：+3
      - 触发词子串命中：+2
      - 名称命中：+4
      - 描述关键词重叠：+1 * overlap
    """

    def __init__(self, registry: SkillRegistry, config: MatcherConfig | None = None):
        self.registry = registry
        self.config = config or MatcherConfig()

    def match(self, user_message: str) -> MatchResult | None:
        ranked = self.rank(user_message)
        if not ranked:
            return None
        best = ranked[0]
        if best.score < self.config.min_score:
            return None
        return best

    def rank(self, user_message: str) -> list[MatchResult]:
        msg = user_message.strip()
        if not msg:
            return []
        msg_l = _normalize(msg)
        msg_tokens = _tokenize(msg)
        results: list[MatchResult] = []

        for skill in self.registry.list_skills():
            score = 0.0
            reasons: list[str] = []

            # 名称强匹配
            if skill.name.lower() in msg_l or skill.name.replace("-", " ") in msg_l:
                score += 4.0
                reasons.append(f"名称命中: {skill.name}")

            # 触发词
            for trig in skill.triggers:
                t = trig.lower().strip()
                if not t or len(t) < 2:
                    continue
                if t == msg_l or t in msg_l:
                    # 越长越精确
                    bonus = 3.0 + min(2.0, len(t) / 20)
                    score += bonus
                    reasons.append(f"触发词命中: {trig}")
                    break  # 同类只计一次最高
                # token 级
                trig_tokens = _tokenize(trig)
                overlap = msg_tokens & trig_tokens
                if len(overlap) >= 2:
                    score += 1.5
                    reasons.append(f"触发词部分重叠: {', '.join(list(overlap)[:4])}")

            # 描述关键词
            desc_tokens = _tokenize(skill.description)
            overlap = msg_tokens & desc_tokens
            # 过滤过短中文噪声
            overlap = {w for w in overlap if len(w) >= 2}
            if overlap:
                score += min(3.0, 0.6 * len(overlap))
                reasons.append(f"描述关键词重叠 ×{len(overlap)}")

            if score > 0:
                results.append(MatchResult(skill=skill, score=round(score, 2), reasons=reasons))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[: self.config.top_k]

    def explain(self, user_message: str) -> list[dict]:
        return [
            {
                "name": r.skill.name,
                "score": r.score,
                "reasons": r.reasons,
            }
            for r in self.rank(user_message)
        ]

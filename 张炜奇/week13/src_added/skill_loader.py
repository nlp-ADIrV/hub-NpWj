"""
Skill Harness — 渐进式技能加载系统

实现"渐进式加载执行 skills 的 harness"：
  1. 启动时扫描所有 skill，仅解析元信息（name, description）→ 轻量索引
  2. 用户输入时先关键词匹配 → 候选 skill
  3. 匹配成功后加载完整 SKILL.md → 注入 LLM Context
  4. LLM 按 SKILL.md 指令执行

设计理念：
  - 不把所有 skill 的内容全塞进 system prompt（浪费 token）
  - 只在用户真正需要时才加载对应 skill 的完整指令

使用方式：
  from src.skill_loader import SkillLoader

  loader = SkillLoader()
  loader.discover()                          # Phase 1: 扫描元信息

  matched = loader.match_skills(user_input)  # Phase 2: 关键词匹配
  if matched:
      prompt = loader.get_skill_for_context(matched[0].name)  # Phase 3: 加载完整指令
"""

import re
import logging
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# skills 目录：与 agent_memory_system 平级
# 从 src/skill_loader.py 向上两级到 agent_memory_system，再向上一级，然后进 skills/
_SKILLS_DIR = Path(__file__).resolve().parent.parent.parent / "skills"

# 提取触发短语的正则：
# 1. 中文引号中的短语（如 "给我做张 crazy 词的闪卡"）
_TRIGGER_QUOTED_ZH = re.compile(r'[""]([^""]{2,40})[""]')
# 2. 英文引号中的短语（如 "draw me a ..."）
_TRIGGER_QUOTED_EN = re.compile(r'"([^"]{2,40})"')
# 3. Use when / e.g. 后面的示例
_TRIGGER_EG = re.compile(r'(?:e\.g\.|例如|比如)\s*[：:]?\s*["""]?([^"""，,]{2,40})')

# 英文关键词过滤停用词
_STOP_WORDS = {
    'the', 'use', 'when', 'for', 'and', 'that', 'with', 'this', 'from',
    'your', 'are', 'any', 'all', 'its', 'not', 'can', 'has', 'have',
    'been', 'they', 'their', 'them', 'will', 'would', 'could', 'should',
    'may', 'also', 'such', 'just', 'like', 'other', 'more', 'some',
    'these', 'those', 'each', 'over', 'into', 'than', 'then', 'about',
    'what', 'which', 'who', 'whom', 'whose', 'how', 'where', 'when',
    'why', 'make', 'user', 'asks', 'type', 'skill', 'one',
}


@dataclass
class SkillMeta:
    """技能的轻量元信息（启动时加载，常驻内存，每条约 200 字符）"""
    name: str
    description: str
    skill_md_path: Path       # SKILL.md 的完整路径
    base_dir: Path            # skill 根目录（含 scripts/, data/, references/ 等）
    trigger_keywords: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.trigger_keywords:
            self.trigger_keywords = self._extract_triggers()

    def _extract_triggers(self) -> list[str]:
        """从 description 中提取触发关键词和短语"""
        desc = self.description
        keywords: list[str] = []

        # 1. 提取中文书名号/引号中的触发短语（最重要）
        for m in _TRIGGER_QUOTED_ZH.finditer(desc):
            phrase = m.group(1).strip()
            if len(phrase) >= 2:
                keywords.append(phrase)

        # 2. 提取英文引号中的短语
        for m in _TRIGGER_QUOTED_EN.finditer(desc):
            phrase = m.group(1).strip()
            if len(phrase) >= 3 and ' ' not in phrase:
                keywords.append(phrase)

        # 3. 提取 e.g. / 例如 后的示例
        for m in _TRIGGER_EG.finditer(desc):
            phrase = m.group(1).strip().rstrip('"').rstrip('"').rstrip('"')
            if len(phrase) >= 2:
                keywords.append(phrase)

        # 4. 提取英文关键词（至少4个字符，非停用词）
        en_words = re.findall(r'\b([a-zA-Z][a-zA-Z-]{3,20})\b', desc.lower())
        for w in en_words:
            if w not in _STOP_WORDS:
                keywords.append(w)

        # 5. 提取中文关键词（2-4个字的名词短语）
        zh_phrases = re.findall(r'[一-鿿]{2,4}(?:图|卡|表|系统|器)', desc)
        keywords.extend(zh_phrases)

        # 去重并保持有意义的顺序（短的在前面更通用）
        seen = set()
        unique = []
        for kw in keywords:
            kw_clean = kw.strip().lower()
            if kw_clean and kw_clean not in seen:
                seen.add(kw_clean)
                unique.append(kw_clean)

        return unique


@dataclass
class LoadedSkill:
    """完整加载的 skill（仅在匹配后才加载）"""
    meta: SkillMeta
    full_content: str           # SKILL.md 完整文件内容
    skill_prompt: str           # 去掉 frontmatter 后的正文（给 LLM 看的操作指令）


class SkillLoader:
    """渐进式技能加载器

    三个阶段：
      Phase 1 (discover):   扫描目录，只解析 frontmatter → 轻量索引
      Phase 2 (match):      用户输入 vs 触发关键词 → 候选列表
      Phase 3 (load):       加载完整 SKILL.md → 注入 Context
    """

    def __init__(self, skills_dir: Path = _SKILLS_DIR):
        self.skills_dir = Path(skills_dir)
        self._index: dict[str, SkillMeta] = {}      # name → meta
        self._loaded: dict[str, LoadedSkill] = {}    # name → full skill (lazy)
        self._discovered = False

    # ── Phase 1: 发现（启动时调用，只加载元信息）──────────────────────────────

    def discover(self) -> dict[str, SkillMeta]:
        """扫描 skills/ 目录，解析所有 SKILL.md 的 frontmatter 元信息。

        每个 skill 只提取 name + description（约 200 字符），
        不加载完整 SKILL.md 正文，做到真正的"轻量启动"。
        """
        if self._discovered:
            return self._index

        if not self.skills_dir.exists():
            logger.warning(f"Skills 目录不存在：{self.skills_dir}")
            self._discovered = True
            return {}

        count = 0
        for skill_dir in sorted(self.skills_dir.iterdir()):
            if not skill_dir.is_dir() or skill_dir.name.startswith('.'):
                continue

            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                logger.debug(f"跳过 {skill_dir.name}：无 SKILL.md")
                continue

            try:
                meta = self._parse_meta(skill_md, skill_dir)
                if meta:
                    self._index[meta.name] = meta
                    count += 1
                    logger.info(
                        f"  发现 skill [{meta.name}] "
                        f"触发词({len(meta.trigger_keywords)}个): "
                        f"{', '.join(meta.trigger_keywords[:5])}..."
                    )
            except Exception as e:
                logger.warning(f"解析 skill {skill_dir.name} 失败：{e}")

        self._discovered = True
        logger.info(f"Skill 渐进式索引构建完成：{count} 个 skill 已注册（仅元信息，完整指令按需加载）")
        return self._index

    def _parse_meta(self, skill_md_path: Path, base_dir: Path) -> SkillMeta | None:
        """仅解析 SKILL.md 的 frontmatter（~20行），不加载完整文件正文"""
        text = skill_md_path.read_text(encoding="utf-8")
        fm = self._parse_frontmatter(text)

        if not fm or "name" not in fm:
            logger.warning(f"{skill_md_path} 缺少 name 字段")
            return None

        return SkillMeta(
            name=fm.get("name", ""),
            description=fm.get("description", fm.get("description", "")),
            skill_md_path=skill_md_path,
            base_dir=base_dir,
        )

    @staticmethod
    def _parse_frontmatter(text: str) -> dict:
        """解析 YAML frontmatter（---...---），优先用 pyyaml，否则手动提取"""
        if not text.startswith('---'):
            return {}
        end = text.find('---', 3)
        if end == -1:
            return {}
        fm_text = text[3:end]

        try:
            import yaml
            result = yaml.safe_load(fm_text)
            return result if isinstance(result, dict) else {}
        except ImportError:
            pass
        except Exception:
            pass

        # Fallback：手动提取 key: value 对
        result = {}
        current_key = None
        current_lines: list[str] = []

        for line in fm_text.split('\n'):
            stripped = line.strip()
            # 检测顶层 key: value
            if stripped and not line[0].isspace() and ':' in stripped:
                # 保存上一个 key
                if current_key:
                    result[current_key] = ' '.join(current_lines).strip()
                key, _, val = stripped.partition(':')
                current_key = key.strip()
                val = val.strip()
                # 跳过 YAML 多行标记
                if val in ('>-', '>', '|-', '|'):
                    current_lines = []
                elif val:
                    current_lines = [val]
                else:
                    current_lines = []
            elif current_key and stripped:
                current_lines.append(stripped)

        if current_key:
            result[current_key] = ' '.join(current_lines).strip()

        return result

    # ── Phase 2: 匹配（每次用户输入时调用，轻量关键词扫描）──────────────────

    def match_skills(self, user_input: str) -> list[SkillMeta]:
        """
        用触发关键词匹配用户输入。
        返回匹配到的 skill 列表，按命中关键词数量降序排列。

        这是"渐进式"的关键：先用零成本的字符串匹配筛选，
        只有命中候选 skill 才进入 Phase 3 加载完整指令。
        """
        if not self._index:
            self.discover()

        user_lower = user_input.lower()
        scored: list[tuple[int, SkillMeta]] = []

        for name, meta in self._index.items():
            hits = 0
            matched_kws: list[str] = []
            for kw in meta.trigger_keywords:
                if kw.lower() in user_lower:
                    hits += 1
                    matched_kws.append(kw)

            # 长关键词（≥6字符）命中权重更高，加额外分
            long_hits = sum(1 for kw in matched_kws if len(kw) >= 6)
            total_score = hits + long_hits

            if total_score > 0:
                scored.append((total_score, meta))
                logger.debug(f"  skill [{name}] 命中 {hits} 个关键词: {matched_kws[:5]}")

        scored.sort(key=lambda x: x[0], reverse=True)
        return [meta for _, meta in scored]

    # ── Phase 3: 加载（匹配后调用，加载完整 SKILL.md）───────────────────────

    def load_full_skill(self, name: str) -> LoadedSkill | None:
        """加载完整 SKILL.md 内容（含完整执行流程），带缓存"""
        if name in self._loaded:
            return self._loaded[name]

        if name not in self._index:
            logger.warning(f"Skill '{name}' 未在索引中")
            return None

        meta = self._index[name]
        try:
            full_text = meta.skill_md_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"读取 {meta.skill_md_path} 失败：{e}")
            return None

        # 提取正文（去掉 frontmatter 后给 LLM 看的操作指令）
        skill_prompt = self._strip_frontmatter(full_text)

        loaded = LoadedSkill(
            meta=meta,
            full_content=full_text,
            skill_prompt=skill_prompt,
        )
        self._loaded[name] = loaded
        logger.info(f"  >> 完整加载 skill [{name}]：{len(skill_prompt)} 字符")
        return loaded

    @staticmethod
    def _strip_frontmatter(text: str) -> str:
        """去掉 YAML frontmatter，返回给 LLM 看的正文"""
        if not text.startswith('---'):
            return text
        end = text.find('---', 3)
        if end == -1:
            return text
        return text[end + 3:].strip()

    # ── 组装 Context ─────────────────────────────────────────────────────────

    def build_skills_summary_prompt(self) -> str:
        """
        构建"可用技能摘要"——注入 system prompt 的轻量部分。
        只含每个 skill 的名称 + 一句话描述（~100 chars/skill），
        不含完整执行指令。完整指令只在匹配后注入。
        """
        if not self._index:
            self.discover()

        if not self._index:
            return ""

        lines = [
            "## 可用技能 (Skills)",
            "",
            "你拥有以下可扩展技能。当用户请求匹配时，系统会自动加载该技能的完整执行指令。",
            "你不需要主动提及技能列表，只需在用户请求相关功能时自然地执行。",
            "",
        ]
        for name, meta in self._index.items():
            # 取第一句话作为简短描述
            short = meta.description.split('。')[0].split('.')[0].strip()
            if len(short) > 100:
                short = short[:97] + "..."
            lines.append(f"- **`{name}`**：{short}")

        lines.append("")
        lines.append("技能触发由系统后台自动检测，你只需正常回应用户即可。")
        return "\n".join(lines)

    def get_skill_context(self, name: str) -> str | None:
        """
        获取某个 skill 的完整执行指令，用于注入当前对话。
        包含：skill 名称、执行流程、脚本路径等全部内容。
        """
        loaded = self.load_full_skill(name)
        if not loaded:
            return None

        # 包装：告诉 LLM 当前激活了哪个 skill
        # 加入强制执行指令，避免 LLM 直接输出 HTML/结果而跳过脚本执行
        return (
            f"## 🎯 已激活技能：{loaded.meta.name}\n\n"
            f"**⚠️ 关键规则：你必须按照以下指令逐步执行。"
            f"如果指令中有脚本命令（如 python xxx.py），你必须在回复末尾用 ```bash 代码块输出完整的脚本命令，"
            f"让系统自动执行。不要只输出最终结果而跳过脚本执行步骤！**\n\n"
            f"{loaded.skill_prompt}\n\n"
            f"---\n"
            f"请按照以上技能指令执行。skill 根目录：{loaded.meta.base_dir}\n"
        )

    # ── 属性 ──────────────────────────────────────────────────────────────────

    @property
    def skill_count(self) -> int:
        if not self._index:
            self.discover()
        return len(self._index)

    @property
    def skill_names(self) -> list[str]:
        if not self._index:
            self.discover()
        return list(self._index.keys())

    def get_meta(self, name: str) -> SkillMeta | None:
        if not self._index:
            self.discover()
        return self._index.get(name)

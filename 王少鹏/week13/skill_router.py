"""技能路由系统 — 检测用户输入是否命中某个 Skill，加载技能内容"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SKILLS_DIR = Path(__file__).parent.parent / "skills"

# 技能注册表 — 扩展新 skill 只需在此添加一项
SKILLS = {
    "zhangxuefeng": {
        "name": "张雪峰视角",
        "keywords": [
            "张雪峰", "雪峰", "张雪峰视角", "雪峰视角",
            "张雪峰会怎么看", "用张雪峰的角度", "用张雪峰的视角",
            "切换到张雪峰", "张雪峰模式", "雪峰模式",
        ],
        "skill_path": SKILLS_DIR / "zhangxuefeng",
        "description": "用张雪峰的思维框架分析教育选择、职业规划等问题",
    },
}


def detect_skill(user_input: str) -> str | None:
    """检测用户输入是否命中某个 skill，返回 skill 名称或 None"""
    for skill_name, config in SKILLS.items():
        for keyword in config["keywords"]:
            if keyword in user_input:
                logger.info(f"命中 Skill [{skill_name}]：关键词 '{keyword}'")
                return skill_name
    return None


def load_skill_content(skill_name: str) -> list[dict]:
    """
    加载指定 skill 的核心内容。
    返回列表，每项含 type/name/label/content/source/char_count。
    仅加载 SKILL.md，不加载 references/（调研材料，非主动指令）。
    """
    config = SKILLS.get(skill_name)
    if not config:
        return []

    base_path = config["skill_path"]
    result = []

    skill_md_path = base_path / "SKILL.md"
    if skill_md_path.exists():
        content = skill_md_path.read_text(encoding="utf-8")
        result.append({
            "type": "skill",
            "name": skill_name,
            "label": f"🎭 激活技能: {config['name']}",
            "content": content,
            "source": str(skill_md_path),
            "char_count": len(content),
        })

    return result

"""
Skills ReAct Agent —— 渐进式加载 + 执行的最小 harness

教学重点：
  1. 系统提示里只放技能 meta（Phase 0），LLM 不知道任何技能细节
  2. LLM 通过 read_skill(name=...) 懒加载全文（Phase 1，由本 agent 处理）
  3. 每轮输出 Thought → Action / Final Answer，由 harness 解析并回填 Observation
  4. 未读取的技能不允许直接执行 —— 强制走「先读后调」流程

使用：
  python src/skills_agent.py            # 交互式
  python src/skills_agent.py "给我做一张 crazy 的闪卡"

与记忆系统衔接：CLI 的 /skill 命令与本模块共用 SkillsAgent，结果会写回会话数据库。
"""

import re
import sys
from pathlib import Path

# 强制 UTF-8 输出：技能内容常含音标/IPA 等非 GBK 字符，避免 GBK 控制台崩溃
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.skill_registry import SkillRegistry
from src.skill_executor import SkillExecutor
from src.llm_config import get_chat_client

MAX_STEPS = 8

SYSTEM_PROMPT = """你是「渐进式技能 Agent」。你可以使用若干技能（skills）来完成任务。

# 可用技能（只有名字与简介）
{skill_metas}

# 输出格式（每轮输出必须严格遵守，禁止输出其他内容）
每轮你只输出下面两种格式之一：

1) 还需要行动时：
Thought: 简短的思考
Action: <下面三种动作之一>
  read_skill(name="技能名")          # 先读技能全文再决定如何执行
  技能名(key=value, key2="value2")   # 执行技能（如 flash-card(data="...")）
  write_file(path="...", content="...")  # 保存 SVG/JSON 等文本产物

2) 任务完成时：
Final Answer: 给用户的最终回答

# 完整示例（给用户做 crazy 闪卡）
Thought: 用户要做闪卡，先了解 flash-card 技能怎么用
Action: read_skill(name="flash-card")
Observation: （系统返回 SKILL.md 全文）
Thought: 明白了，先准备数据，再执行脚本
Action: write_file(path="data/crazy.json", content="{{...}}")
Observation: （系统确认已写入 outputs/data/crazy.json）
Thought: 数据就绪，调用脚本生成 HTML
Action: flash-card(data="data/crazy.json")
Observation: （系统返回脚本输出）
Thought: 已完成
Final Answer: 已为你生成 crazy 的闪卡 HTML：outputs/crazy.html

# 硬性规则
1. 想用某个技能，必须先 Action: read_skill(name="...") 读取其 SKILL.md 全文，
   再决定如何执行；不允许没读过就调用技能。
2. read_skill 一次只读一个技能。
3. 动作参数必须符合 SKILL.md 里的约定。
4. 所有脚本都以 outputs/ 目录为工作目录运行；write_file 的相对路径也保存到
   outputs/ 下。脚本提到的「当前工作目录」即指它。
5. 不要输出 Action 以外的任何前缀或附属内容（如序号、Payload、解释），
   不要在同一轮里同时出现 Action 和 Final Answer。
6. 不要编造 Observation，不要重复已经执行过的动作。
"""

# 识别文本里的字段（辅助解析，纯字符串匹配即可）
_ACTION_RE = re.compile(r"Action\s*:\s*(.+)", re.IGNORECASE)
_FINAL_RE = re.compile(r"Final Answer\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)


def _step_outcome(step_text: str) -> tuple[str, str] | None:
    """
    从模型输出中提取本步意图，返回 ('final'|'action', 内容)。
    同一步里若 Final Answer 与 Action 并存，以先出现的为准。
    """
    positions = []
    fm = _FINAL_RE.search(step_text)
    if fm:
        positions.append((fm.start(), "final", fm.group(1).strip()))
    am = _ACTION_RE.search(step_text)
    if am:
        action = _extract_action(step_text, am)
        positions.append((am.start(), "action", action))
    if not positions:
        return None
    positions.sort(key=lambda x: x[0])
    return (positions[0][1], positions[0][2])


def _action_balanced(raw: str) -> bool:
    """粗略判断 Action 文本的引号/括号是否闭合（用于多行 Action 判断）"""
    depth, quote = 0, None
    for ch in raw:
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
    return depth == 0 and quote is None


def _extract_action(step_text: str, am: re.Match) -> str:
    """
    从匹配处截取完整的 Action 文本。若首行括号/引号未闭合（如 write_file
    的内容跨多行），就继续吞并后续行，直到闭合或遇到下一个标记。
    """
    seg = step_text[am.start(1):]
    lines = seg.splitlines()
    raw = lines[0]
    i = 1
    while not _action_balanced(raw) and i < len(lines):
        if re.match(r"\s*(Action|Observation|Final Answer)\s*:", lines[i], re.IGNORECASE):
            break
        raw += "\n" + lines[i]
        i += 1
    return raw.strip()


class SkillsAgent:
    def __init__(self, registry: SkillRegistry | None = None, executor: SkillExecutor | None = None):
        self.registry = registry or SkillRegistry()
        self.executor = executor or SkillExecutor(self.registry)
        self._read_skills: set[str] = set()  # 已加载全文的技能
        self._history: list[dict] = []       # 本轮消息历史（含 Observation）

    def reset(self):
        self._read_skills = set()
        self._history = []

    # ── 对外入口 ──────────────────────────────────────────────────────
    def run(self, query: str, verbose: bool = True) -> str:
        """执行一个用户请求，返回最终回答文本。"""
        self.reset()
        system_prompt = SYSTEM_PROMPT.format(skill_metas=self.registry.meta_summary())
        self._history = [{"role": "system", "content": system_prompt}]
        self._history.append({"role": "user", "content": query})
        if verbose:
            print(f"  \033[2m可用技能摘要:\033[0m {self.registry.names()}")

        client, model = get_chat_client()
        for step in range(1, MAX_STEPS + 1):
            response = self._ask(client, model)
            step_text = response.choices[0].message.content or ""
            if verbose:
                print(f"\n  ── 第 {step} 步 ──────────────────────")
                print(step_text.strip())

            outcome = _step_outcome(step_text)
            if outcome is None:
                if verbose:
                    print("  [!] 模型未输出 Action 或 Final Answer，提前结束")
                return step_text.strip()

            kind, content = outcome
            if kind == "final":
                self._history.append({"role": "assistant", "content": step_text})
                return content

            obs = self._handle_action(content, verbose)
            self._history.append({"role": "assistant", "content": step_text})
            self._history.append({"role": "user", "content": f"Observation: {obs}"})

        return "已到达最大步数，任务未完成。请重试或调整指令。"

    # ── 内部实现 ──────────────────────────────────────────────────────
    def _ask(self, client, model):
        return client.chat.completions.create(
            model=model, messages=self._history, temperature=0.3
        )

    def _handle_action(self, action: str, verbose: bool) -> str:
        """处理单个 Action，返回 Observation 文本。"""
        # 动作 1：read_skill —— 懒加载全文（兼容 name= 与位置参数两种写法）
        m = re.match(r"read_skill\s*\(\s*(?:name\s*=\s*)?[\"']([^\"']+)[\"']\s*\)", action)
        if m:
            name = m.group(1)
            if not self.registry.exists(name):
                return f"[错误] 未知技能：{name}。可用技能：{self.registry.names()}"
            if name in self._read_skills:
                return f"[提示] 技能 {name} 已加载过，请直接使用它。"
            full = self.registry.load_full(name)
            self._read_skills.add(name)
            full = self._normalize_skill_text(name, full)
            if verbose:
                print(f"  [read_skill] {name} 全文 {len(full)} 字符已载入")
            return f"以下是 {name} 的 SKILL.md 全文，请按它执行：\n\n{full}"

        # 动作 2：技能执行（或内置 write_file）
        # 解析技能名，检查是否已读全文（未读拦截）
        m2 = re.match(r"\s*([a-zA-Z][\w-]*)\s*\(", action)
        if not m2:
            return f"[错误] 无法识别的动作：{action}"
        target = m2.group(1)
        if target != "write_file":
            if target in self.registry.names() and target not in self._read_skills:
                return f"[拦截] 你还未读取技能 {target} 的全文，请先 Action: read_skill(name=\"{target}\")"
        result = self.executor.execute(action)
        if verbose:
            status = "OK" if result["ok"] else "FAIL"
            print(f"  [{status}] 执行 {result['action']}  {result['stdout'][:120]}")
            if result.get("outputs"):
                for o in result["outputs"]:
                    print(f"      file {o}")
            if result["error"]:
                print(f"      ! {result['error'][:200]}")
        if not result["ok"]:
            return f"[执行失败] {result['error'] or result['stderr'][:300]}"
        msg = result["stdout"].strip()
        if result["outputs"]:
            msg += "\n产物文件：" + ", ".join(result["outputs"])
        return msg or "[执行完成]"

    @staticmethod
    def _normalize_skill_text(name: str, text: str) -> str:
        """把技能原稿里的 .cursor/skills/<name> 路径指回本项目，并附运行须知"""
        text = text.replace(f".cursor/skills/{name}", f"skills/{name}")
        note = (
            "\n\n## 本环境运行须知\n"
            f"- 本项目的技能目录是 skills/{name}（不是 .cursor/skills/）。\n"
            "- 脚本以 outputs/ 为工作目录运行；write_file 的相对路径也保存到 outputs/ 下。\n"
            f"- 保存 {name} 需要的数据/JSON 时：先 Action: write_file(path=\"data/<文件名>.json\", content=\"...\")\n"
            f"- 执行 {name} 时，data 参数请传与 write_file 相同的相对路径（如 data/<文件名>.json），它相对 outputs/ 解析。\n"
        )
        return note + text


def main():
    import argparse
    parser = argparse.ArgumentParser(description="渐进式加载 Skills ReAct Agent")
    parser.add_argument("query", nargs="?", default=None, help="一句话任务（省略则进入交互模式）")
    args = parser.parse_args()

    agent = SkillsAgent()
    print(f"\nSkills Agent — 渐进式技能加载演示")
    print(f"可用技能：{agent.registry.names()}\n")

    if args.query:
        answer = agent.run(args.query)
        print(f"\n\n── 最终回答 ────────────────────\n{answer}")
        return

    while True:
        try:
            q = input("你：").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not q:
            continue
        if q in ("/exit", "/quit"):
            break
        answer = agent.run(q)
        print(f"\n\n── 最终回答 ────────────────────\n{answer}\n")


if __name__ == "__main__":
    main()

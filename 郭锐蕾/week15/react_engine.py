"""通用 ReAct 循环引擎

核心思想（ReAct = Reason + Act）：
  LLM 输出 Thought(推理) → Action(选工具) → Action Input(参数)，
  runner 执行工具得 Observation，再喂回 LLM 续写，直到 Final Answer。

设计要点：
  1. 主 agent 和 subagent 共用同一个引擎，区别只在 tools 字典：
     - 主 agent：{web_search, dispatch_subagents}
     - subagent：{web_search}
  2. 用 stop=["Observation:"] 让 LLM 在 Action Input 后停下，runner 执行工具
     再补 Observation 续写——ReAct 经典实现技巧
  3. 完整 trace 捕获每步 Thought/Action/ActionInput/Observation，供可视化/调试
  4. 解析兜底：LLM 拿到长结果常直接写报告不带 Final Answer: 前缀，
     _parse 检测到无 Action 但有实质文本时当作 Final Answer（避免空 action 死循环）

依赖：仅 llm_client + 工具函数，无外部库
"""
import re
import time
import logging
from typing import Callable, Optional

from llm_client import llm_chat

logger = logging.getLogger(__name__)

# 默认系统提示（subagent 用）。主 agent 会传入自定义 MAIN_SYSTEM 引导派发
DEFAULT_SYSTEM = """你是市场调研助手，能用以下工具联网搜索调研。

可用工具：
{tools_desc}

按如下格式严格输出（每轮一次 Thought/Action/Action Input）：
Thought: 你的推理，分析还需查什么
Action: 工具名
Action Input: 工具参数（字符串）

工具执行后会得到 Observation。多轮调用直到能给出完整答案，最后用：
Thought: 我已收集足够信息
Final Answer: 综合答案（带来源要点）

规则：
- Action 必须是上面列出的工具名之一
- Action Input 是该工具的参数字符串
- 每轮只调一次工具，等 Observation 再决定下一步"""


def build_tools_desc(tools: dict) -> str:
    """把 tools 字典格式化成工具说明。tools: {name: (fn, description)}"""
    return "\n".join(f"- {name}: {desc}" for name, (_, desc) in tools.items())


class ReActEngine:
    """通用 ReAct 循环。主 agent / subagent 各实例化一个。"""

    def __init__(self, agent_name: str, tools: dict,
                 max_steps: int = 6, model_tag: str = "deepseek-chat",
                 system_prompt: Optional[str] = None):
        """
        tools: {tool_name: (fn(arg, **kwargs)->str, description_str)}
        system_prompt: 自定义系统提示。None 时用 DEFAULT_SYSTEM。
                       {tools_desc} 占位符会被替换成工具说明。
        """
        self.agent_name = agent_name
        self.tools = tools
        self.max_steps = max_steps
        self.model_tag = model_tag
        self._system_template = system_prompt or DEFAULT_SYSTEM
        self.trace: list[dict] = []

    def run(self, question: str, on_step: Callable = None,
            shared_state: dict = None) -> dict:
        """
        执行 ReAct 循环。
        on_step(step_dict): 每步回调（CLI 实时流式 / SSE 用）。
        shared_state: 共享状态（主 agent 派发 subagent 时往里塞 subagent trace）。
        返回 {final_answer, trace, duration}。
        """
        self.trace = []
        t0 = time.time()
        system = self._system_template.format(tools_desc=build_tools_desc(self.tools))
        history = f"Question: {question}\n\n"
        final_answer = ""

        for step_idx in range(self.max_steps):
            # 调 LLM 生成下一步（停在 Observation: 前）
            llm_out = llm_chat(system, history, temperature=0.0,
                               max_tokens=768, stop=["Observation:"])
            thought, action, action_input = self._parse(llm_out)

            step = {"idx": step_idx, "agent": self.agent_name,
                    "thought": thought, "action": action,
                    "action_input": action_input, "observation": None,
                    "final": False, "done": False}

            if action == "Final Answer":
                step["final"] = True
                step["done"] = True
                final_answer = action_input
                self.trace.append(step)
                if on_step:
                    on_step(step)
                break

            # pre 执行：先发 step（observation=None），让 CLI 立刻看到决策
            if on_step:
                on_step(step)

            # 执行工具（dispatch_subagents 会很慢，要等所有 subagent 跑完）
            observation = self._exec_tool(action, action_input, shared_state)

            # post 执行：同一 idx 再发一次，带真实 observation
            step["observation"] = observation
            step["done"] = True
            self.trace.append(step)
            if on_step:
                on_step(step)

            # 续写历史（observation 截断避免 context 撑爆）
            history += llm_out + f"Observation: {observation[:1200]}\n"
        else:
            # 超过 max_steps，强制收尾
            final_answer = "（已达最大步数）" + (self.trace[-1].get("observation", "") or "")
            step = {"idx": self.max_steps, "agent": self.agent_name,
                    "thought": "达到步数上限", "action": "Final Answer",
                    "action_input": final_answer, "observation": None,
                    "final": True, "done": True}
            self.trace.append(step)
            if on_step:
                on_step(step)

        duration = round(time.time() - t0, 2)
        return {"final_answer": final_answer, "trace": self.trace, "duration": duration}

    def _parse(self, text: str) -> tuple[str, str, str]:
        """解析 Thought/Action/Action Input。Final Answer 时 action='Final Answer'。
        兜底：无 Action 也无 Final Answer 但有实质文本 → 当 Final Answer。"""
        thought = ""
        m = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.S)
        if m:
            thought = m.group(1).strip()[:400]

        # Final Answer 优先检测
        mfa = re.search(r"Final Answer:\s*(.*)", text, re.S)
        if mfa:
            return thought, "Final Answer", mfa.group(1).strip()

        # Action / Action Input
        ma = re.search(r"Action:\s*(.*)", text)
        mi = re.search(r"Action Input:\s*(.*)", text)
        if ma:
            action = ma.group(1).strip()
            action_input = (mi.group(1).strip() if mi else "")
            return thought, action, action_input

        # 兜底：有实质文本但无格式标记 → 当 Final Answer
        if text.strip():
            return thought or "综合调研结果给出报告", "Final Answer", text.strip()
        return thought, "", ""

    def _exec_tool(self, action: str, action_input: str, shared_state: dict) -> str:
        """执行工具，返回 observation 文本。未知工具返回错误说明。"""
        if action not in self.tools:
            return f"工具 '{action}' 不存在，可选: {list(self.tools.keys())}"
        fn, _ = self.tools[action]
        try:
            # 工具可能需要 shared_state（dispatch_subagents 用）
            if shared_state is not None:
                return str(fn(action_input, shared_state=shared_state))
            return str(fn(action_input))
        except Exception as e:
            return f"工具执行出错: {type(e).__name__}: {str(e)[:120]}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    from tools import web_search

    engine = ReActEngine(
        "test",
        tools={"web_search": (web_search, "联网搜索，参数是查询词")},
        max_steps=4,
    )
    r = engine.run("2024年中国新能源汽车销量是多少万辆？")
    print(f"\n答案: {r['final_answer'][:120]}")
    print(f"trace {len(r['trace'])} 步:")
    for s in r["trace"]:
        obs = (s.get("observation") or "")[:50]
        print(f"  [{s['idx']}] {s['action']}({s['action_input'][:40]}) → {obs}")

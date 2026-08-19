"""

通用 ReAct 循环引擎（主 agent / subagent 共用）

"""
import re
import time
import logging
from typing import Callable, Optional

from llm_client import llm_chat

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM = """你是软件开发规划助手，可以用以下工具完成规划任务。

可用工具：
{tools_desc}

每轮严格按以下格式输出（一轮只输出一个 Thought / Action / Action Input）：
Thought: 你的推理，分析还需要什么信息
Action: 工具名
Action Input: 工具参数（字符串）

程序执行工具后会返回 Observation。多轮循环直到信息足够，最后输出：
Thought: 信息已足够
Final Answer: 最终答案

规则：
- Action 必须从上方的工具列表里选
- 每轮只能调用一个工具，必须等 Observation 后再决定下一步
- 不要自己编造工具执行结果"""


def build_tools_desc(tools: dict) -> str:
    """把 {工具名: (函数, 描述)} 格式化成喂给 LLM 的工具说明。"""
    lines = [f"- {name}: {desc}" for name, (_, desc) in tools.items()]
    return "\n".join(lines)


class ReActLoop:
    """通用 ReAct 循环。主 agent / 每个 subagent 各实例化一个。"""

    def __init__(self, agent_name: str, tools: dict,
                 max_steps: int = 6, model_tag: str = "deepseek-chat",
                 system_prompt: Optional[str] = None):
        """
        tools: {工具名: (fn(action_input, **kw)->str, 工具描述)}
        system_prompt: 自定义系统提示（主 agent 传派发引导，subagent 传角色提示），
                       None 时用 DEFAULT_SYSTEM。含 {tools_desc} 占位符会被替换。
        """
        self.agent_name = agent_name
        self.tools = tools          # {name: (fn, desc)}
        self.max_steps = max_steps
        self.model_tag = model_tag
        self._system_template = system_prompt or DEFAULT_SYSTEM
        self.trace: list[dict] = []  # 本轮执行 trace（每步一个 dict）

    # ── 主循环 ──────────────────────────────────────────────────────────────
    def run(self, question: str, on_step: Callable = None,
            shared_state: dict = None) -> dict:
        """执行 ReAct 循环，返回 {final_answer, trace, duration}。

        on_step(step_dict): 每步回调。工具执行前回调一次（observation=None，
            让 CLI 立刻看到"它决定做什么"），执行后再回调一次（带真实 observation）。
        shared_state: 跨 agent 共享字典（主 agent 派发时往里写 subagent 的 trace）。
        """
        self.trace = []
        t0 = time.time()
        system = self._system_template.format(tools_desc=build_tools_desc(self.tools))
        history = f"Question: {question}\n\n"   # 累积 Thought/Action/Observation 的对话历史
        final_answer = ""

        for step_idx in range(self.max_steps):
            # 1. LLM 生成下一步（停在 "Observation:" 前，程序来补真实结果）
            llm_out = llm_chat(system, history, temperature=0.0,
                               max_tokens=2048, stop=["Observation:"])
            thought, action, action_input = self._parse(llm_out)

            step = {"idx": step_idx, "agent": self.agent_name,
                    "thought": thought, "action": action,
                    "action_input": action_input, "observation": None,
                    "final": action == "Final Answer"}

            if action == "Final Answer":
                final_answer = action_input
                self.trace.append(step)
                if on_step:
                    on_step(step)
                break

            # 2. 先回调一次（observation 还没出）：立刻看到决策
            if on_step:
                on_step(step)

            # 3. 程序真正执行工具（可能很慢：dispatch_subagents 要等所有子 agent 跑完）
            observation = self._exec_tool(action, action_input, shared_state)

            # 4. 补上 observation 再回调一次：同一步，内容完整
            step["observation"] = observation
            step["done"] = True
            self.trace.append(step)
            if on_step:
                on_step(step)

            history += llm_out + f"Observation: {observation[:1200]}\n"
        else:
            # max_steps 用完还没 Final Answer：用最后一步 observation 兜底收尾
            last_obs = (self.trace[-1].get("observation") or "") if self.trace else ""
            final_answer = "（达到最大步数，以下为已收集信息）\n" + last_obs
            step = {"idx": self.max_steps, "agent": self.agent_name,
                    "thought": "达到步数上限", "action": "Final Answer",
                    "action_input": final_answer, "observation": None, "final": True}
            self.trace.append(step)
            if on_step:
                on_step(step)

        duration = round(time.time() - t0, 2)
        return {"final_answer": final_answer, "trace": self.trace,
                "duration": duration}

    # ── 输出解析 ────────────────────────────────────────────────────────────
    @staticmethod
    def _cut_restart(text: str) -> str:
        """模型偶尔在输出 Action Input / Final Answer 后继续"自言自语"
        （生成第二段 Thought:）。把第二段起的尾巴切掉，只留第一段。"""
        return re.split(r"\s*Thought:", text, maxsplit=1)[0].strip()

    def _parse(self, text: str) -> tuple[str, str, str]:
        """解析 LLM 输出 → (thought, action, action_input)。

        Final Answer 时 action 记为 'Final Answer'，内容放 action_input。
        兜底：若没有 Action 也没有 Final Answer 标记，但有实质文本，当作 Final Answer
        （LLM 拿到子规划结果后常直接写文档、不带 Final Answer 前缀）。
        """
        thought = ""
        # 兼容模型把 Action/Final Answer 写在 Thought 同一行的写法（\s* 允许无换行）
        m = re.search(r"Thought:\s*(.*?)(?=\s*Action:|\s*Final Answer:|$)", text or "", re.S)
        if m:
            thought = m.group(1).strip()[:400]

        # Final Answer 优先检测（切掉后面可能接的第二段 Thought:）
        mfa = re.search(r"Final Answer:\s*(.*)", text or "", re.S)
        if mfa:
            return thought, "Final Answer", self._cut_restart(mfa.group(1))

        # Action 只取本行（避免把 Action Input 也吞进去）；Action Input 取到结尾
        ma = re.search(r"Action:\s*([^\n]*)", text or "")
        mi = re.search(r"Action Input:\s*(.*)", text or "", re.S)
        if ma:
            action = ma.group(1).strip()
            action_input = self._cut_restart(mi.group(1)) if mi else ""
            return thought, action, action_input

        if (text or "").strip():
            return thought or "综合信息给出答案", "Final Answer", self._cut_restart(text.strip())
        return thought, "", ""

    # ── 工具执行 ────────────────────────────────────────────────────────────
    def _exec_tool(self, action: str, action_input: str, shared_state: dict) -> str:
        """真正执行工具，返回 observation 文本。执行异常不抛出、转成错误文本，
        让 LLM 看到错误后自己决定换工具/换参数/直接作答。"""
        if action not in self.tools:
            return f"工具 '{action}' 不存在，可选: {list(self.tools.keys())}"
        fn, _ = self.tools[action]
        try:
            # dispatch_subagents 这类编排工具需要拿到 shared_state
            if shared_state is not None:
                return str(fn(action_input, shared_state=shared_state))
            return str(fn(action_input))
        except Exception as e:
            return f"工具执行出错: {type(e).__name__}: {str(e)[:120]}"

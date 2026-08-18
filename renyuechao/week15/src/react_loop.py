"""一个小而可注入的手写 ReAct 循环。"""

from __future__ import annotations

import re
import time
from typing import Any, Callable, Optional

from .llm_client import llm_chat


REACT_SYSTEM = """你是一个会使用工具的 ReAct Agent。
可用工具：
{tools_desc}

每轮严格输出：
Thought: 你的判断
Action: 工具名
Action Input: 工具参数

拿到足够信息后输出：
Final Answer: 最终答案
"""


def build_tools_desc(tools: dict[str, tuple[Callable[..., Any], str]]) -> str:
    """把工具注册表变成系统提示中的简短说明。"""

    if not tools:
        return "（无工具）"
    return "\n".join(f"- {name}: {description}" for name, (_, description) in tools.items())


class ReActLoop:
    """主 Agent 和 Subagent 共用的 ReAct 执行器。

    ``chat_fn`` 必须兼容 ``llm_chat`` 的完整调用签名；离线函数通常接收
    ``system_prompt``、``history`` 和 ``**kwargs``。
    """

    def __init__(
        self,
        agent_name: str,
        tools: dict[str, tuple[Callable[..., Any], str]],
        *,
        max_steps: int = 6,
        model_tag: str = "deepseek-chat",
        system_prompt: Optional[str] = None,
        chat_fn: Optional[Callable[..., str]] = None,
        observation_limit: int = 12000,
    ) -> None:
        self.agent_name = agent_name
        self.tools = tools
        self.max_steps = max_steps
        self.model_tag = model_tag
        self.chat_fn = chat_fn
        self.observation_limit = max(1, observation_limit)
        self._system_template = system_prompt or REACT_SYSTEM
        self.trace: list[dict[str, Any]] = []

    def run(
        self,
        question: str,
        on_step: Optional[Callable[[dict[str, Any]], None]] = None,
        shared_state: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """执行 ReAct，返回 ``final_answer``、``trace`` 和 ``duration``。"""

        self.trace = []
        started = time.perf_counter()
        system = self._system_template.format(tools_desc=build_tools_desc(self.tools))
        history = f"Question: {question}\n\n"
        final_answer = ""
        status = "succeeded"
        error: Optional[str] = None

        for step_idx in range(self.max_steps):
            try:
                llm_output = self._call_chat(system, history)
            except Exception as exc:
                error = f"LLM 调用失败: {type(exc).__name__}: {str(exc)[:160]}"
                status = "failed"
                step = {
                    "idx": step_idx,
                    "agent": self.agent_name,
                    "thought": "无法继续调用模型",
                    "action": "Final Answer",
                    "action_input": "",
                    "observation": error,
                    "final": True,
                }
                self.trace.append(step)
                if on_step:
                    on_step(step)
                break

            thought, action, action_input = self._parse(llm_output)
            step = {
                "idx": step_idx,
                "agent": self.agent_name,
                "thought": thought,
                "action": action,
                "action_input": action_input,
                "observation": None,
            }

            if action == "Final Answer":
                step["final"] = True
                final_answer = action_input
                self.trace.append(step)
                if on_step:
                    on_step(step)
                break

            step["final"] = False
            if on_step:
                on_step(step)
            observation = self._exec_tool(action, action_input, shared_state)
            step["observation"] = observation
            step["done"] = True
            self.trace.append(step)
            if on_step:
                on_step(step)
            history += llm_output + f"Observation: {observation[:self.observation_limit]}\n"
        else:
            last_observation = ""
            if self.trace:
                last_observation = self.trace[-1].get("observation") or ""
            status = "failed"
            error = "达到最大步数，未产生有效 Final Answer"
            step = {
                "idx": self.max_steps,
                "agent": self.agent_name,
                "thought": "达到步数上限",
                "action": "Final Answer",
                "action_input": "",
                "observation": f"{error}；最后一次 Observation: {last_observation}",
                "final": True,
            }
            self.trace.append(step)
            if on_step:
                on_step(step)

        return {
            "status": status,
            "final_answer": final_answer,
            "error": error,
            "trace": self.trace,
            "duration": round(time.perf_counter() - started, 4),
        }

    def _call_chat(self, system: str, history: str) -> str:
        """调用真实或注入的 chat 函数；运行期异常只传播一次，不盲目重试。"""

        chat = self.chat_fn or llm_chat
        output = chat(
            system,
            history,
            temperature=0.0,
            max_tokens=768,
            stop=["Observation:"],
        )
        return str(output or "")

    @staticmethod
    def _parse(text: str) -> tuple[str, str, str]:
        """解析 Thought/Action/Action Input，并提供无格式文本兜底。"""

        thought = ""
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.S)
        if thought_match:
            thought = thought_match.group(1).strip()[:400]

        final_match = re.search(r"Final Answer:\s*(.*)", text, re.S)
        if final_match:
            return thought, "Final Answer", final_match.group(1).strip()

        action_match = re.search(r"Action:\s*(.*)", text)
        input_match = re.search(r"Action Input:\s*(.*)", text, re.S)
        if action_match:
            action = action_match.group(1).strip()
            action_input = input_match.group(1).strip() if input_match else ""
            return thought, action, action_input

        if text.strip():
            return thought or "直接整理结果", "Final Answer", text.strip()
        return thought, "", ""

    def _exec_tool(
        self,
        action: str,
        action_input: str,
        shared_state: Optional[dict[str, Any]],
    ) -> str:
        if action not in self.tools:
            return f"工具 '{action}' 不存在，可选: {list(self.tools)}"
        function, _ = self.tools[action]
        try:
            if shared_state is not None:
                return str(function(action_input, shared_state=shared_state))
            return str(function(action_input))
        except Exception as exc:
            return f"工具执行出错: {type(exc).__name__}: {str(exc)[:160]}"

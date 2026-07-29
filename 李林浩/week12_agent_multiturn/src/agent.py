"""支持多轮对话的 ReAct Agent。

实现重点：
1. Session 保存 messages，实现跨轮上下文记忆；
2. 每轮问题继续复用历史消息，而不是重新开始；
3. 历史过长时进行轻量压缩，避免上下文无限增长；
4. 支持 OpenAI 兼容接口；未配置 API Key 时自动使用 MockLLM 演示。
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Generator

from tools import TOOLS

SYSTEM_PROMPT = """你是一个财务分析 ReAct Agent，可以使用工具回答问题。
严格按以下格式输出：
Thought: 当前思考
Action: 工具名
Action Input: {\"参数名\": \"参数值\"}

当信息足够时输出：
Thought: 已有足够信息
Final Answer: 最终答案

可用工具：
- knowledge_search(query): 检索演示财务知识库
- calculator(expr): 数学计算

多轮对话规则：
- 结合历史对话理解“它、这家公司、刚才两个公司、再算一下”等指代；
- 历史中已有数据时优先复用；
- 缺少信息时再调用工具。
"""

_FINAL_RE = re.compile(r"Final Answer:\s*(.+)", re.S)
_ACTION_RE = re.compile(r"Action:\s*(\w+)")
_INPUT_RE = re.compile(r"Action Input:\s*(\{.*?\})", re.S)
_THOUGHT_RE = re.compile(r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|$)", re.S)


@dataclass
class Session:
    session_id: str
    messages: list[dict] = field(default_factory=list)
    turns: list[dict] = field(default_factory=list)

    def compact_history(self, keep_last: int = 8) -> None:
        """轻量上下文压缩：保留摘要和最近消息。"""
        if len(self.messages) <= 20:
            return
        old = self.messages[:-keep_last]
        recent = self.messages[-keep_last:]
        summary_lines = []
        for msg in old:
            content = str(msg.get("content", "")).replace("\n", " ")[:160]
            summary_lines.append(f"{msg.get('role')}: {content}")
        summary = "【历史摘要】" + " | ".join(summary_lines[-10:])
        self.messages = [{"role": "user", "content": summary}] + recent


class MockLLM:
    """无 API Key 时的确定性演示模型，用于验证多轮状态管理。"""

    def complete(self, messages: list[dict]) -> str:
        q = messages[-1]["content"]
        history = "\n".join(str(m.get("content", "")) for m in messages[:-1])

        if q.startswith("Observation:"):
            obs = q.split("Observation:", 1)[1].strip()
            # 如果上一轮是差值计算，直接组织答案
            if re.fullmatch(r"-?\d+(\.\d+)?", obs):
                return f"Thought: 已有足够信息\nFinal Answer: 计算结果为 {obs} 个百分点。"
            return f"Thought: 已获得工具结果\nFinal Answer: 根据检索结果：{obs}"

        if "毛利率" in q and ("茅台" in q or "五粮液" in q):
            return f'Thought: 需要先查询财务数据\nAction: knowledge_search\nAction Input: {{"query": "{q}"}}'

        if "差" in q or "相差" in q or "高多少" in q:
            # 从历史中提取演示数据
            nums = [float(x) for x in re.findall(r"2023毛利率=(\d+\.\d+)", history)]
            # 历史中同一 Observation 可能出现多次，按数值去重后取两个公司数据
            unique_nums = []
            for n in nums:
                if n not in unique_nums:
                    unique_nums.append(n)
            if len(unique_nums) >= 2:
                a, b = unique_nums[0], unique_nums[1]
                return f'Thought: 历史中已有两个毛利率，直接计算差值\nAction: calculator\nAction Input: {{"expr": "{a}-{b}"}}'
            return f'Thought: 历史数据不足，先检索\nAction: knowledge_search\nAction Input: {{"query": "贵州茅台 五粮液 2023毛利率"}}'

        if "哪家" in q and "更高" in q and "毛利率" in history:
            return "Thought: 根据历史上下文可以直接回答\nFinal Answer: 贵州茅台的毛利率更高。"

        if "它" in q and "2023" in q and "营收" in q:
            company = "贵州茅台" if "贵州茅台" in history else "五粮液"
            return f'Thought: 结合历史指代，继续查询 {company}\nAction: knowledge_search\nAction Input: {{"query": "{company} 2023营收"}}'

        return "Thought: 当前问题可直接回答\nFinal Answer: 已结合当前会话历史理解该问题。"


class Agent:
    def __init__(self):
        self.mock = not bool(os.getenv("OPENAI_API_KEY"))
        if not self.mock:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL") or None,
            )
            self.model = os.getenv("AGENT_MODEL", "gpt-4.1-mini")
        else:
            self.client = None
            self.model = "mock-llm"
            self.mock_llm = MockLLM()

    def _complete(self, messages: list[dict]) -> str:
        if self.mock:
            return self.mock_llm.complete(messages)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
        )
        return (resp.choices[0].message.content or "").strip()

    def run(self, session: Session, question: str, max_steps: int = 6) -> Generator[dict, None, None]:
        session.compact_history()
        work_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + session.messages
        work_messages.append({"role": "user", "content": question})
        session.messages.append({"role": "user", "content": question})

        for step in range(1, max_steps + 1):
            output = self._complete(work_messages)
            thought_m = _THOUGHT_RE.search(output)
            final_m = _FINAL_RE.search(output)
            if final_m:
                answer = final_m.group(1).strip()
                session.messages.append({"role": "assistant", "content": answer})
                session.turns.append({"question": question, "answer": answer})
                yield {"step": step, "type": "final", "thought": thought_m.group(1).strip() if thought_m else "", "answer": answer}
                return

            action_m = _ACTION_RE.search(output)
            input_m = _INPUT_RE.search(output)
            if not action_m:
                yield {"step": step, "type": "error", "answer": f"无法解析模型输出: {output}"}
                return

            action = action_m.group(1)
            try:
                params = json.loads(input_m.group(1)) if input_m else {}
            except json.JSONDecodeError:
                params = {}
            tool = TOOLS.get(action)
            observation = tool(**params) if tool else f"未知工具: {action}"

            yield {
                "step": step,
                "type": "action",
                "thought": thought_m.group(1).strip() if thought_m else "",
                "action": action,
                "action_input": params,
                "observation": observation,
            }
            work_messages.append({"role": "assistant", "content": output})
            work_messages.append({"role": "user", "content": f"Observation: {observation}"})
            # 把工具结果也写入 Session，使下一轮可以复用历史数据
            session.messages.append({"role": "assistant", "content": output})
            session.messages.append({"role": "user", "content": f"Observation: {observation}"})

        yield {"step": max_steps + 1, "type": "max_steps", "answer": "超过最大推理步数"}

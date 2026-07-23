"""
多轮对话会话管理

核心设计：
  1. 单次 ReAct 循环内：Thought → Action → Observation（原有能力）
  2. 多轮用户对话间：只保留「用户问题 + Final Answer」摘要，注入下一轮上下文
  3. 不把历史工具调用轨迹全部塞进 prompt，控制 token，同时支持指代消解
     （如「它」「那家公司」「刚才的毛利率」）

使用方式：
  from conversation import ConversationSession
  session = ConversationSession(mode="manual")
  for step in session.chat("茅台2023年毛利率是多少？"):
      ...
  for step in session.chat("那五粮液呢？"):  # 能理解「那」指对比/同问法
      ...
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Generator, Literal


Mode = Literal["manual", "fc"]


@dataclass
class Turn:
    question: str
    answer: str


@dataclass
class ConversationSession:
    """一个用户会话：跨多轮追问保持上下文。"""

    mode: Mode = "manual"
    max_steps: int = 10
    max_history_turns: int = 6
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    turns: list[Turn] = field(default_factory=list)

    def clear(self) -> None:
        self.turns.clear()

    def build_history_messages(self) -> list[dict]:
        """
        构造可注入 LLM 的历史消息（仅 Q&A，不含工具轨迹）。

        格式与 OpenAI chat messages 一致，manual / fc 两套实现共用。
        """
        messages: list[dict] = []
        recent = self.turns[-self.max_history_turns :]
        for turn in recent:
            messages.append({"role": "user", "content": turn.question})
            messages.append({"role": "assistant", "content": turn.answer})
        return messages

    def chat(self, question: str) -> Generator[dict, None, None]:
        """
        执行一轮用户提问：跑完整 ReAct，并把 Final Answer 记入会话。

        yield 的 step dict 与原版 react_*.run() 保持一致，额外在 start 类事件外
        由调用方自行包装；本方法只透传 ReAct 步骤。
        """
        question = (question or "").strip()
        if not question:
            yield {
                "step": 0,
                "type": "error",
                "observation": "问题为空，请重新输入。",
            }
            return

        history = self.build_history_messages()

        if self.mode == "manual":
            from react_manual import run as react_run
        else:
            from react_function_calling import run as react_run

        answer: str | None = None
        for step in react_run(question, max_steps=self.max_steps, history=history):
            stype = step.get("type")
            if stype == "final":
                answer = step.get("answer") or ""
            elif stype in ("error", "max_steps"):
                answer = step.get("answer") or step.get("observation") or ""
            yield step

        if answer is not None:
            self.turns.append(Turn(question=question, answer=answer))

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "mode": self.mode,
            "max_steps": self.max_steps,
            "turn_count": len(self.turns),
            "turns": [{"question": t.question, "answer": t.answer} for t in self.turns],
        }


class SessionStore:
    """进程内会话仓库（Web 服务用）。生产环境可换成 Redis。"""

    def __init__(self) -> None:
        self._sessions: dict[str, ConversationSession] = {}

    def get_or_create(
        self,
        session_id: str | None,
        mode: Mode = "manual",
        max_steps: int = 10,
    ) -> ConversationSession:
        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]
            # 允许同一 session 切换实现方式
            session.mode = mode
            session.max_steps = max_steps
            return session

        session = ConversationSession(
            mode=mode,
            max_steps=max_steps,
            session_id=session_id or str(uuid.uuid4()),
        )
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> ConversationSession | None:
        return self._sessions.get(session_id)

    def clear(self, session_id: str) -> bool:
        session = self._sessions.get(session_id)
        if not session:
            return False
        session.clear()
        return True

    def delete(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

"""会话管理器：创建、获取、删除和列出多轮 Agent 会话。"""
from __future__ import annotations

import uuid
from agent import Session


class SessionManager:
    def __init__(self):
        self.sessions: dict[str, Session] = {}

    def create(self) -> Session:
        sid = f"sess_{uuid.uuid4().hex[:8]}"
        session = Session(session_id=sid)
        self.sessions[sid] = session
        return session

    def get(self, session_id: str) -> Session:
        if session_id not in self.sessions:
            raise KeyError(f"session 不存在: {session_id}")
        return self.sessions[session_id]

    def delete(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    def list_sessions(self) -> list[dict]:
        return [
            {"session_id": s.session_id, "turn_count": len(s.turns)}
            for s in self.sessions.values()
        ]

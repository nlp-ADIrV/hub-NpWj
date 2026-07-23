"""
多轮对话 FastAPI 服务（SSE 流式）

接口：
  POST /query/manual   - 手写版，带 session_id
  POST /query/fc       - Function Calling 版，带 session_id
  GET  /session/{id}   - 查看会话历史
  POST /session/{id}/clear - 清空历史
  DELETE /session/{id} - 删除会话
  GET  /health

使用方式：
  cd week12作业文件夹/src
  uvicorn serve:app --host 0.0.0.0 --port 8001
  # 建议用 8001，避免与原项目 8000 冲突
"""

from __future__ import annotations

import os
import sys
import json
import logging
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_HERE = Path(__file__).resolve().parent
_PARENT_SRC = _HERE.parent.parent / "src"
sys.path.insert(0, str(_PARENT_SRC))
sys.path.insert(0, str(_HERE))

from conversation import SessionStore  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STORE = SessionStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("预加载 FAISS 索引...")
    from tools import _load_rag
    await asyncio.to_thread(_load_rag)
    logger.info("多轮对话服务就绪")
    yield


app = FastAPI(title="ReAct Financial Agent — Multi-turn", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str
    max_steps: int = 10
    session_id: str | None = None


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _sse_chat(req: QueryRequest, mode: str):
    session = STORE.get_or_create(req.session_id, mode=mode, max_steps=req.max_steps)  # type: ignore[arg-type]
    queue: asyncio.Queue = asyncio.Queue()
    _SENTINEL = object()
    turns_before = len(session.turns)

    def _worker():
        try:
            for step_data in session.chat(req.question):
                queue.put_nowait(step_data)
        finally:
            queue.put_nowait(_SENTINEL)

    yield _sse({
        "type": "start",
        "question": req.question,
        "mode": mode,
        "session_id": session.session_id,
        "history_turns": turns_before,
    })

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _worker)

    while True:
        step_data = await queue.get()
        if step_data is _SENTINEL:
            break
        yield _sse(step_data)

    yield _sse({
        "type": "done",
        "session_id": session.session_id,
        "history_turns": len(session.turns),
    })


@app.post("/query/manual")
async def query_manual(req: QueryRequest):
    return StreamingResponse(
        _sse_chat(req, "manual"),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/query/fc")
async def query_fc(req: QueryRequest):
    return StreamingResponse(
        _sse_chat(req, "fc"),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    session = STORE.get(session_id)
    if not session:
        raise HTTPException(404, "session not found")
    return session.to_dict()


@app.post("/session/{session_id}/clear")
async def clear_session(session_id: str):
    if not STORE.clear(session_id):
        raise HTTPException(404, "session not found")
    return {"ok": True, "session_id": session_id}


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    if not STORE.delete(session_id):
        raise HTTPException(404, "session not found")
    return {"ok": True}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": os.getenv("AGENT_MODEL", "qwen-max"),
        "feature": "multi-turn",
    }


HTML_PATH = Path(__file__).resolve().parent.parent / "index.html"


@app.get("/")
async def root():
    if HTML_PATH.exists():
        return HTMLResponse(HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>index.html not found</h2>")

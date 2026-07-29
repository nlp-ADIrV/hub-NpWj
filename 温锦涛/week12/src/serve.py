"""
FastAPI HTTP 服务，提供流式 SSE 接口给 Web UI

接口：
  POST /session/create                    - 创建新会话
  GET  /session/{session_id}              - 获取会话信息
  POST /session/{session_id}/query        - 在会话中提问（流式 SSE）
  POST /query/manual                      - 旧版单轮，内部创建一次性 session
  POST /query/fc                          - 旧版单轮，内部创建一次性 session
  GET  /health                            - 健康检查

使用方式：
  uvicorn serve:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import json
import time
import uuid
import logging
import asyncio
from dataclasses import dataclass, field, asdict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── 常量 ──────────────────────────────────────────────────────────────────────
MAX_SESSIONS = 100
COMPRESS_THRESHOLD = 40
KEEP_RECENT_TURNS = 2


# ── Session 数据结构 ───────────────────────────────────────────────────────────

@dataclass
class TurnData:
    turn: int
    question: str
    final_answer: str
    summary: str = ""
    message_count: int = 0

@dataclass
class SessionData:
    session_id: str
    created_at: float
    turns: list[TurnData] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    compressed: bool = False

_sessions: dict[str, SessionData] = {}
_session_order: list[str] = []  # LRU order for eviction


def _evict_if_needed():
    if len(_sessions) >= MAX_SESSIONS:
        oldest = _session_order.pop(0)
        _sessions.pop(oldest, None)
        logger.info(f"淘汰最旧 session {oldest}")


def _touch_session(sid: str):
    if sid in _session_order:
        _session_order.remove(sid)
    _session_order.append(sid)


# ── 上下文压缩 ────────────────────────────────────────────────────────────────

def _generate_summary(old_messages: list[dict]) -> str:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    model = os.getenv("AGENT_MODEL", "qwen-max")
    text = "\n".join(
        f"[{m['role']}] {m.get('content', '')}" for m in old_messages
    )
    prompt = (
        "请将以下对话历史压缩为一段简洁的摘要，保留所有财务数值（毛利率、营收、百分比等）、"
        "股票代码、公司名称和工具调用结果。压缩后的摘要将作为后续对话的上下文：\n\n" + text
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        summary = resp.choices[0].message.content.strip()
        return f"【历史摘要】{summary}"
    except Exception as e:
        logger.warning(f"压缩摘要生成失败: {e}")
        return f"【历史摘要】（压缩失败，保留部分历史）{text[:500]}"


def _compress_session(session: SessionData):
    if len(session.messages) <= COMPRESS_THRESHOLD:
        return

    total = len(session.messages)
    keep_count = 0
    turn_count = len(session.turns)
    for t in reversed(session.turns):
        keep_count += t.message_count
        if len(session.turns) - session.turns.index(t) >= KEEP_RECENT_TURNS:
            break

    compress_end = total - max(keep_count, 20)
    if compress_end < 5:
        return

    old_msgs = session.messages[:compress_end]
    summary = _generate_summary(old_msgs)
    session.messages = (
        [{"role": "user", "content": summary}]
        + session.messages[compress_end:]
    )
    session.compressed = True
    logger.info(f"session {session.session_id} 已压缩: {total} -> {len(session.messages)} 条")


# ── 预加载 FAISS（启动时执行一次）────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("预加载 FAISS 索引和 Embedding 模型...")
    from tools import _load_rag
    await asyncio.to_thread(_load_rag)
    logger.info("预加载完成，服务就绪")
    yield


app = FastAPI(title="ReAct Financial Agent", lifespan=lifespan)


# ── 请求/响应模型 ─────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question:  str
    max_steps: int = 10


class SessionQueryRequest(BaseModel):
    question:  str
    max_steps: int = 10


# ── SSE 流式生成器 ────────────────────────────────────────────────────────────
def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream_session_react(
    session: SessionData,
    question: str,
    max_steps: int,
    mode: str,
):
    if mode == "manual":
        from react_manual import run as react_run
    else:
        from react_function_calling import run as react_run

    # 压缩检查（同步）
    _compress_session(session)

    turn_num = len(session.turns) + 1
    queue: asyncio.Queue = asyncio.Queue()
    _SENTINEL = object()
    start_time = time.time()
    action_step_count = 0
    final_answer = ""
    initial_msg_count = len(session.messages)

    def _worker():
        try:
            for step_data in react_run(
                question,
                max_steps=max_steps,
                history=session.messages,
            ):
                queue.put_nowait(step_data)
            queue.put_nowait(_SENTINEL)
        except Exception as e:
            queue.put_nowait({"type": "error", "observation": str(e)})
            queue.put_nowait(_SENTINEL)

    yield _sse({
        "type": "start",
        "session_id": session.session_id,
        "turn": turn_num,
        "question": question,
        "mode": mode,
    })

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _worker)

    while True:
        step_data = await queue.get()
        if step_data is _SENTINEL:
            break

        step_data["turn"] = turn_num

        if step_data.get("type") == "action":
            action_step_count += 1
            yield _sse(step_data)

        elif step_data.get("type") == "final":
            final_answer = step_data.get("answer", "")
            # 提取本轮新增的完整消息
            msgs = step_data.get("_messages")
            if msgs:
                session.messages.extend(msgs[initial_msg_count:])
            yield _sse(step_data)

        elif step_data.get("type") in ("max_steps", "error"):
            msgs = step_data.get("_messages")
            if msgs:
                session.messages.extend(msgs[initial_msg_count:])
            yield _sse(step_data)

        else:
            yield _sse(step_data)

    elapsed = time.time() - start_time
    new_msg_count = len(session.messages) - initial_msg_count
    summary = final_answer[:80] if final_answer else ""

    turn_data = TurnData(
        turn=turn_num,
        question=question,
        final_answer=final_answer,
        summary=summary,
        message_count=new_msg_count,
    )
    session.turns.append(turn_data)

    yield _sse({
        "type": "turn_done",
        "turn": turn_num,
        "question": question,
        "answer": final_answer[:200],
        "summary": summary,
        "action_steps": action_step_count,
        "elapsed_s": round(elapsed, 1),
    })

    yield _sse({"type": "done"})


# ── Session 路由 ──────────────────────────────────────────────────────────────

@app.post("/session/create")
async def create_session():
    _evict_if_needed()
    sid = f"sess_{uuid.uuid4().hex[:12]}"
    session = SessionData(session_id=sid, created_at=time.time())
    _sessions[sid] = session
    _session_order.append(sid)
    logger.info(f"创建 session {sid}")
    return {"session_id": sid, "created_at": session.created_at}


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    session = _sessions.get(session_id)
    if not session:
        return JSONResponse({"error": "session not found"}, status_code=404)
    _touch_session(session_id)
    return {
        "session_id": session.session_id,
        "created_at": session.created_at,
        "turn_count": len(session.turns),
        "turns": [asdict(t) for t in session.turns],
        "compressed": session.compressed,
    }


@app.post("/session/{session_id}/query")
async def query_session(session_id: str, req: SessionQueryRequest):
    session = _sessions.get(session_id)
    if not session:
        return JSONResponse({"error": "session not found"}, status_code=404)
    _touch_session(session_id)
    mode = "manual"  # default, can be extended
    return StreamingResponse(
        _stream_session_react(session, req.question, req.max_steps, mode),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/session/{session_id}/query/manual")
async def query_session_manual(session_id: str, req: SessionQueryRequest):
    session = _sessions.get(session_id)
    if not session:
        return JSONResponse({"error": "session not found"}, status_code=404)
    _touch_session(session_id)
    return StreamingResponse(
        _stream_session_react(session, req.question, req.max_steps, "manual"),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/session/{session_id}/query/fc")
async def query_session_fc(session_id: str, req: SessionQueryRequest):
    session = _sessions.get(session_id)
    if not session:
        return JSONResponse({"error": "session not found"}, status_code=404)
    _touch_session(session_id)
    return StreamingResponse(
        _stream_session_react(session, req.question, req.max_steps, "fc"),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── 旧端点向后兼容（内部创建一次性 session） ──────────────────────────────────

async def _stream_react_legacy(question: str, max_steps: int, mode: str):
    _evict_if_needed()
    sid = f"sess_{uuid.uuid4().hex[:12]}"
    session = SessionData(session_id=sid, created_at=time.time())
    _sessions[sid] = session
    _session_order.append(sid)

    async for event in _stream_session_react(session, question, max_steps, mode):
        yield event


@app.post("/query/manual")
async def query_manual(req: QueryRequest):
    return StreamingResponse(
        _stream_react_legacy(req.question, req.max_steps, "manual"),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/query/fc")
async def query_fc(req: QueryRequest):
    return StreamingResponse(
        _stream_react_legacy(req.question, req.max_steps, "fc"),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── 健康检查 ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": os.getenv("AGENT_MODEL", "qwen-max"),
        "sessions": len(_sessions),
    }


# ── 托管 index.html ──────────────────────────────────────────────────────────
HTML_PATH = Path(__file__).parent.parent / "index.html"

@app.get("/")
async def root():
    if HTML_PATH.exists():
        return HTMLResponse(HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>index.html not found</h2>")

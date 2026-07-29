"""
FastAPI HTTP 服务，提供流式 SSE 接口给 Web UI

接口：
  POST /query/manual  - 手写版 ReAct，流式返回每步
  POST /query/fc      - Function Calling 版，流式返回每步
  GET  /health        - 健康检查

使用方式：
  uvicorn serve:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import json
import logging
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

# 简单的内存级会话存储：{ session_id: messages_list }
SESSION_STORE = {} 

# ── SSE 流式生成器 ────────────────────────────────────────────────────────────
def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream_react(question: str, max_steps: int, mode: str):
    """
    同步生成器（react_run）在独立线程中逐步执行，
    每产出一步通过 asyncio.Queue 传递给异步 SSE 生成器，
    实现真正的边思考边推送。
    """
    if mode == "manual":
        from react_manual import run as react_run
    else:
        from react_function_calling import run as react_run

    queue: asyncio.Queue = asyncio.Queue()
    _SENTINEL = object()

    def _worker():
        try:
            for step_data in react_run(question, max_steps=max_steps):
                queue.put_nowait(step_data)
        finally:
            queue.put_nowait(_SENTINEL)

    yield _sse({"type": "start", "question": question, "mode": mode})

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _worker)

    while True:
        step_data = await queue.get()
        if step_data is _SENTINEL:
            break
        yield _sse(step_data)

    yield _sse({"type": "done"})


# ── 路由 ──────────────────────────────────────────────────────────────────────
@app.post("/query/manual")
async def query_manual(req: QueryRequest):
    return StreamingResponse(
        _stream_react(req.question, req.max_steps, "manual"),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/query/fc")
async def query_fc(req: QueryRequest):
    # 1. 处理 Session ID
    sid = req.session_id or str(uuid.uuid4())
    
    # 2. 获取历史上下文
    # 如果是新会话，初始化为空列表；否则读取历史
    # 注意：System Prompt 通常在 Agent 内部初始化，这里只存 User/Assistant 交互
    history = session_store.get(sid, [])
    
    # 3. 定义异步生成器来包装 Agent 调用
    async def event_generator():
        nonlocal history, sid
        
        # 发送头部信息，告诉前端现在的 Session ID
        yield f"data: {json.dumps({'type': 'meta', 'session_id': sid})}\n\n"
        
        try:
            # 调用 Agent (假设已修改为支持 history 参数)
            # 如果 react_function_calling.py 还没改好，这里暂时只能传空列表
            # 建议参考上一条回答修改 run 函数
            from react_function_calling import run as fc_run
            
            final_history = []
            for step in fc_run(question=req.question, max_steps=req.max_steps, history=history):
                yield f"data: {json.dumps(step, ensure_ascii=False)}\n\n"
                
                # 实时收集 Agent 产生的新消息 (如果 run 函数能实时返回的话)
                # 如果 run 函数是一次性返回所有步骤，则无法流式更新 history
                # *变通方案*：在 run 函数结束时 return messages
                
            # 4. 更新会话存储 (在循环结束后)
            # 这里需要 react_function_calling.py 的 run 函数支持返回最终 messages
            # 或者我们在本地根据 yield 的内容手动拼凑 messages (比较麻烦且容易出错)
            
            # *最简单的临时方案*：
            # 仅保存用户的问题，Agent 的回答在下一轮作为 context 传入比较复杂
            # 强烈建议采用“外部维护 messages”的模式
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model": os.getenv("AGENT_MODEL", "qwen-max")}


# ── 托管 index.html ──────────────────────────────────────────────────────────
HTML_PATH = Path(__file__).parent.parent / "index.html"

@app.get("/")
async def root():
    if HTML_PATH.exists():
        return HTMLResponse(HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>index.html not found</h2>")

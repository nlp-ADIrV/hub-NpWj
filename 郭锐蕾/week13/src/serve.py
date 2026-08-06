"""
FastAPI 可视化服务 — 渐进式 Skill 加载 Harness

启动：
  uvicorn src.serve:app --host 127.0.0.1 --port 8013

接口：
  GET  /              前端页面
  GET  /api/status    索引与注册状态
  POST /api/run       执行一轮（返回完整生命周期 JSON）
  GET  /api/index     常驻索引 Markdown
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.harness import SkillHarness

harness: SkillHarness | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global harness
    harness = SkillHarness(ROOT / "skills", workspace=ROOT)
    harness.refresh()
    yield


app = FastAPI(title="Progressive Skill Harness", lifespan=lifespan)


class RunRequest(BaseModel):
    message: str = Field(..., min_length=1)
    mode: str = "demo"
    force_skill: str | None = None
    auto_release: bool = True
    load_secondary: bool = True


@app.get("/")
def index_page():
    return FileResponse(ROOT / "index.html")


@app.get("/api/status")
def api_status():
    assert harness is not None
    return harness.status()


@app.get("/api/index")
def api_index():
    assert harness is not None
    return {
        "markdown": harness.registry.index_md,
        "stats": harness.registry.index_stats(),
    }


@app.post("/api/run")
def api_run(req: RunRequest):
    assert harness is not None
    result = harness.handle(
        req.message,
        mode=req.mode,
        force_skill=req.force_skill,
        auto_release=req.auto_release,
        load_secondary=req.load_secondary,
    )
    return JSONResponse(result.to_dict())


# 静态输出目录（生成的闪卡 HTML 等）
(outputs := ROOT / "outputs").mkdir(exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(outputs)), name="outputs")


def main():
    import uvicorn

    uvicorn.run("src.serve:app", host="127.0.0.1", port=8013, reload=False)


if __name__ == "__main__":
    main()

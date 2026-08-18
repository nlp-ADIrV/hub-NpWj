"""联网搜索工具 —— Tavily（标准库 urllib，零额外依赖）

设计要点：
- 用 urllib 而非 requests，遵循少依赖原则
- Tavily 为 LLM 优化：返回 answer(摘要) + results(带 url/content 的来源)
- 失败不抛异常，返回 {error} 字符串（让 ReAct 兜底处理）

环境变量：TAVILY_API_KEY
"""
import os
import json
import urllib.request
import logging

logger = logging.getLogger(__name__)

TAVILY_URL = "https://api.tavily.com/search"


def tavily_search(query: str, max_results: int = 5) -> dict:
    """调用 Tavily 搜索。返回 {answer, results, response_time}，失败返回 {error}。"""
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        return {"error": "未设置 TAVILY_API_KEY"}

    payload = {
        "api_key": key,
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
        "include_answer": True,
    }
    try:
        req = urllib.request.Request(
            TAVILY_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        # 精简：只留对 LLM 有用的字段，content 截断避免 context 过长
        results = [
            {"title": r.get("title", ""), "url": r.get("url", ""),
             "content": (r.get("content") or "")[:600]}
            for r in data.get("results", [])
        ]
        return {"answer": data.get("answer") or "",
                "results": results,
                "response_time": data.get("response_time")}
    except Exception as e:
        logger.warning(f"Tavily 搜索失败 '{query}': {e}")
        return {"error": f"{type(e).__name__}: {str(e)[:100]}"}


def format_search_result(r: dict) -> str:
    """把 Tavily 返回格式化成喂给 LLM 的文本。"""
    if "error" in r:
        return f"搜索失败: {r['error']}"
    parts = []
    if r.get("answer"):
        parts.append(f"摘要: {r['answer']}")
    for i, res in enumerate(r.get("results", []), 1):
        parts.append(f"[{i}] {res['title']}\n    {res['content'][:300]}")
    return "\n".join(parts) if parts else "无结果"


def web_search(query: str, **_kwargs) -> str:
    """web_search 工具函数：搜索 + 格式化，返回文本。
    **kwargs 兼容 ReAct 引擎注入的 shared_state（本工具不需要，忽略即可）。"""
    return format_search_result(tavily_search(query))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(web_search("2024年中国新能源汽车销量")[:400])

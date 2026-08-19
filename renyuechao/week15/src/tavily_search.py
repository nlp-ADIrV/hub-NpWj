"""Tavily 搜索工具。

它是可替换的边界：离线测试只传 fake tool，不会请求网络；真实运行时密钥来自
``TAVILY_API_KEY`` 环境变量。
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Any


logger = logging.getLogger(__name__)
TAVILY_URL = "https://api.tavily.com/search"


def tavily_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """调用 Tavily，失败时返回结构化错误而不是让 ReAct 循环崩溃。"""

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {"error": "未设置 TAVILY_API_KEY"}

    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
        "include_answer": True,
    }
    try:
        request = urllib.request.Request(
            TAVILY_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
        results = [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": (item.get("content") or "")[:600],
            }
            for item in data.get("results", [])
        ]
        return {
            "answer": data.get("answer") or "",
            "results": results,
            "response_time": data.get("response_time"),
        }
    except Exception as exc:  # pragma: no cover - 真实网络异常由环境决定
        logger.warning("Tavily 搜索失败: %s", type(exc).__name__)
        return {"error": f"{type(exc).__name__}: {str(exc)[:100]}"}


def format_search_result(result: dict[str, Any]) -> str:
    """把搜索结果格式化给 LLM，并保留每条来源 URL。"""

    if result.get("error"):
        return f"搜索失败: {result['error']}"

    parts: list[str] = []
    if result.get("answer"):
        parts.append(f"摘要: {result['answer']}")
    for index, item in enumerate(result.get("results", []), 1):
        title = item.get("title", "")
        url = item.get("url", "")
        content = (item.get("content") or "")[:300]
        parts.append(f"[{index}] {title}\n    URL: {url}\n    {content}")
    return "\n".join(parts) if parts else "无结果"


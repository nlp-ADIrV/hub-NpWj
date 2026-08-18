"""真实 LLM 客户端。

离线测试不经过本模块，而是给 :class:`src.react_loop.ReActLoop` 注入 ``chat_fn``。
密钥只从环境变量读取，避免把凭证写入示例代码或日志。
"""

from __future__ import annotations

import os
from typing import Iterable, Optional


def llm_chat(
    system_prompt: str,
    history: str,
    *,
    temperature: float = 0.0,
    max_tokens: int = 768,
    stop: Optional[Iterable[str]] = None,
) -> str:
    """调用兼容 OpenAI Chat Completions 的 DeepSeek 接口并返回文本。

    ``DEEPSEEK_API_KEY`` 未设置时会抛出不包含密钥内容的明确错误。这样真实运行
    可以快速发现配置问题，离线运行则可直接使用 ReActLoop 的 ``chat_fn``。
    """

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("未设置 DEEPSEEK_API_KEY；离线运行请注入 ReActLoop.chat_fn")

    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - 只在真实运行环境触发
        raise RuntimeError("真实运行需要安装 requirements.txt 中的 openai") from exc

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )
    response = client.chat.completions.create(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": history},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        stop=list(stop) if stop else None,
        extra_body={"thinking": {"type": "disabled"}},
    )
    message = response.choices[0].message.content
    return message or ""

"""

LLM 客户端

"""
import os
import time
import logging

from openai import OpenAI

logger = logging.getLogger(__name__)

DEEPSEEK_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"   # DeepSeek 对话模型（即官网的 deepseek-v4-flash）

_client = None


def get_client() -> OpenAI:
    """懒加载全局客户端（第一次调用才读环境变量）"""
    global _client
    if _client is None:
        key = os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise EnvironmentError("请先设置环境变量 DEEPSEEK_API_KEY")
        _client = OpenAI(api_key=key, base_url=DEEPSEEK_URL,
                         timeout=60.0, max_retries=0)  # 60s 超时 + 自管重试（网络抖动更快失败重来）
    return _client


def llm_chat(system, user, *, temperature=0.0, max_tokens=1024,
             stop=None, retries=3) -> str:
    """单轮 LLM 对话，返回文本。

    stop: 让模型生成到指定字符串前停下。ReAct 用它停在 "Observation:" 前——
          保证 Observation 是程序真实执行的结果，而不是模型自己编造的（防幻觉）。
    """
    for attempt in range(retries):
        try:
            resp = get_client().chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            logger.warning(f"LLM 调用失败({type(e).__name__})，{wait}s 后重试: {str(e)[:80]}")
            time.sleep(wait)

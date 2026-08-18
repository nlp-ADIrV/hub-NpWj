"""极简 LLM 客户端 —— DeepSeek（OpenAI 兼容接口）

设计要点：
- 单例 client，避免每次调用重建连接
- 指数退避重试（网络偶发失败兜底）
- stop 参数支持 ReAct 在 Observation 前截断

依赖：pip install openai
环境变量：DEEPSEEK_API_KEY
"""
import os
import time
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

DEEPSEEK_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

_client = None


def get_client() -> OpenAI:
    """懒加载单例 client。"""
    global _client
    if _client is None:
        key = os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise EnvironmentError("请先设置环境变量 DEEPSEEK_API_KEY")
        _client = OpenAI(api_key=key, base_url=DEEPSEEK_URL)
    return _client


def llm_chat(system: str, user: str, *, temperature: float = 0.0,
             max_tokens: int = 1024, stop=None, retries: int = 3) -> str:
    """单轮对话。stop 让 LLM 在指定 token 处停下（ReAct 用 "Observation:" 截断）。"""
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
            logger.warning(f"LLM 调用失败，{wait}s 后重试({attempt+1}/{retries}): {str(e)[:80]}")
            time.sleep(wait)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(llm_chat("你是助手", "用一句话介绍 ReAct 范式")[:200])

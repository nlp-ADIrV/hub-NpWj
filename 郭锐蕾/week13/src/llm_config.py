"""可选 LLM 配置（与 agent_memory_system 对齐，便于扩展）。"""

from __future__ import annotations

import os

PROVIDERS: dict[str, dict] = {
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "chat_model": "deepseek-chat",
        "display_name": "DeepSeek",
    },
    "qwen": {
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "chat_model": "qwen-plus",
        "display_name": "Qwen Plus (DashScope)",
    },
}


def get_provider() -> str:
    return os.getenv("LLM_PROVIDER", "deepseek").lower()


def get_chat_client():
    from openai import OpenAI

    provider = get_provider()
    if provider not in PROVIDERS:
        raise ValueError(f"未知 LLM_PROVIDER={provider}")
    cfg = PROVIDERS[provider]
    api_key = os.getenv(cfg["api_key_env"])
    if not api_key:
        raise EnvironmentError(f"需要环境变量 {cfg['api_key_env']}")
    return OpenAI(api_key=api_key, base_url=cfg["base_url"]), cfg["chat_model"]


def current_model_info() -> dict:
    provider = get_provider()
    cfg = PROVIDERS.get(provider, PROVIDERS["deepseek"])
    return {
        "provider": provider,
        "model": cfg["chat_model"],
        "display": cfg["display_name"],
    }

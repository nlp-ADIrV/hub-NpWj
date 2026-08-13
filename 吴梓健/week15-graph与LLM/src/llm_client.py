"""极简 LLM 客户端（新能源汽车对比调研 subagent 项目用）
DeepSeek deepseek-chat（OpenAI 兼容接口），原生支持 function calling。
依赖：pip install openai"""
import os, time, logging
from openai import OpenAI
import dotenv

dotenv.load_dotenv()

logger = logging.getLogger(__name__)
DEEPSEEK_URL = "https://llm-4k7qpb0jadl7z0zd.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
DEEPSEEK_MODEL = "glm-5.2" #qwen3.7-max
_client = None


def get_client():
    global _client
    if _client is None:
        key = os.getenv("MODELSCOPE_API_KEY")
        if not key:
            raise EnvironmentError("请设置 MODELSCOPE_API_KEY")
        _client = OpenAI(api_key=key, base_url=DEEPSEEK_URL)
    return _client


def llm_complete(messages, tools=None, *, temperature=0.0, max_tokens=4096, retries=3):
    """单轮 LLM 对话（API 原生 tool calling）。
    tools 传 OpenAI 兼容的 function calling 定义列表时，返回的 message 里
    会带 tool_calls（模型结构化返回 工具名+JSON参数），由 ReAct 循环执行。
    返回 resp.choices[0].message。"""
    for attempt in range(retries):
        try:
            kwargs = {"model": DEEPSEEK_MODEL, "messages": messages,
                      "temperature": temperature, "max_tokens": max_tokens}
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            resp = get_client().chat.completions.create(**kwargs)
            return resp.choices[0].message
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
            logger.warning(f"LLM 重试({attempt + 1}): {str(e)[:80]}")

if __name__ == 'main':
    get_client()
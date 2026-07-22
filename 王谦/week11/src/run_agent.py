"""
run_agent.py — 作业入口：两个拆分后的天气工具 + agent loop

教学重点：
  1. 两个独立工具 getcode / get_weather_by_coords（见 weather_tools.py）
  2. 【真正的 agent loop】——不是"一次 tool_call + 一次回填"的单轮，而是
     while 循环：模型输出 tool_call → 执行 → 回填 → 再问模型，直到模型不再
     调用工具、给出最终回答为止。
       - 问"宁德天气"：模型先调 getcode(宁德)→拿到经纬度→再调 get_weather_by_coords→回答（链式/循环）
       - 问"北京的经纬度"：模型只调一次 getcode 即可回答（单工具独立答）
       - 问"经度116.4 纬度39.9 的天气"：模型只调一次 get_weather_by_coords（单工具独立答）
     这三种形态共用同一个 loop，差异完全由模型自己决定——这就是 agent loop。

LLM 接口：参考项目里 mode_function_call/run_function_call.py 的 DeepSeek 调用
（OpenAI 兼容协议，openai SDK）。

使用：
  # 配置环境变量
  #   Windows:  set DEEPSEEK_API_KEY=sk-xxx
  #   Linux:    export DEEPSEEK_API_KEY=sk-xxx

  # 单个问题
  python homework_split_weather/run_agent.py -q "宁德总部今天的天气怎么样？"

  # 内置示例（演示链式/单工具三种形态）
  python homework_split_weather/run_agent.py --demo

  # 切到 DashScope 的 qwen-plus
  python homework_split_weather/run_agent.py --provider dashscope -q "北京的经纬度"
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

from openai import OpenAI
# 让 weather_tools 可被 import（直接 python 运行本脚本也能找到）
sys.path.insert(0, str(Path(__file__).parent))

from weather_tools import getcode, get_weather_by_coords 

PROVIDERS = {
    "dashscope": {
        "api_key": os.getenv("DASHSCOPE_API_KEY", ""),
        "base_url": os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "model": os.getenv("DASHSCOPE_MODEL", "qwen-plus"),
    },
    # 新增：OpenAI 配置
    "openai": {
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    },
}
def build_client(provider_name: str) -> OpenAI:
    """根据 provider_name 构建 OpenAI 客户端"""
    if provider_name not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider_name}")
    config = PROVIDERS[provider_name]
    return OpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"],
    ),config["model"]
#工具schema
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "getcode",
            "description": (
                "把城市名解析成经纬度（地理编码）。输入中文城市名如'北京'、'宁德'，"
                "返回该城市的纬度 latitude 和经度 longitude。"
                "当用户问'某城市的经纬度/坐标'时直接用本工具即可；"
                "当用户问'某城市天气'但本工具不含天气查询时，先用本工具拿到经纬度，"
                "再把经纬度传给 get_weather_by_coords 查天气（链式调用）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_by_coords",
            "description": (
                "按经纬度查询当前天气及未来3天预报。参数是数值型的纬度/经度。"
                "若用户已直接给出经纬度，直接调用本工具；"
                "若用户只给了城市名，请先调用 getcode 拿到经纬度，再调用本工具（链式）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lon": {
                        "type": "number",
                        "description": "经度"
                    },
                    "lat": {
                        "type": "number",
                        "description": "纬度"
                    }
                },
                "required": ["lon", "lat"]
            }
        }
    }
]

# 工具名 → 后端函数 的派发表（业务逻辑与协议层分离）
TOOL_DISPATCH = {
    "getcode": getcode,
    "get_weather_by_coords": get_weather_by_coords,
}

SYSTEM_PROMPT = (
    "你是一名天气助手，有两个工具可用：getcode（城市名→经纬度）和 "
    "get_weather_by_coords（经纬度→天气）。"
    "请按需调用工具，必要时可链式调用（先 getcode 拿经纬度，再 get_weather_by_coords 查天气）。"
    "只依据工具返回的数据作答，不要编造。"
)

# ── 【核心】agent loop：模型循环调用工具直到给出最终回答 ────────────────────

MAX_STEPS = 10  # 防御性兜底，避免模型无限循环

def run(client,model,question,verbose=True) -> dict:
    """
    agent loop：模型循环调用工具直到给出最终回答
    :param client: OpenAI 客户端
    :param model: 模型名称
    :param question: 用户问题
    :param verbose: 是否打印调试信息
    :return: 最终回答的字典，包含 'answer' 和 'tool_calls'
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]
    tool_call_log = []
    t0 = time.time()
    for step in range(1,MAX_STEPS+1):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            function_call="auto",
        )
        msg = resp.choices[0].message
        # 模型本轮不再调用工具 → 已经是最终回答，退出循环
        if not msg.tool_calls:
            if verbose:
                print(f"  → [llm] 最终回答（第{step}轮，共{time.time() - t0:.1f}s）")
            return {
                "answer": msg.content or "",
                "tool_calls": tool_call_log,
                "steps": step,
                "elapsed": time.time() - t0,
            }
        
        # 模型本轮调用工具 → 把工具调用记录下来，并追加到 messages 里
        messages.append(msg)
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            tool_call_log.append({"name": name, "args": args})
            if verbose:
                print(f"  → [llm] 第{step}轮调用工具 {name}，参数 {args}")
            # 调用后端函数
            if name not in TOOL_DISPATCH:
                raise ValueError(f"Unknown tool: {name}")
            try:
                result = TOOL_DISPATCH[name](**args)
            except TypeError as e:
                result = f"参数错误：{e}"
            except Exception as e:
                result = f"工具调用异常：{e}"
            preview = (result or "")[:120].replace("\n", " ")
            # 把工具调用结果回填到 messages 里，供模型下一轮使用
            if verbose:
                print(f"    ↩ {preview}{'...' if len(result or '') > 120 else ''}\n")
            # 以 role=tool 回填，tool_call_id 必须对上
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
    return {
        "answer": "（达到最大步数，模型仍未给出最终回答）",
        "tool_calls": tool_call_log,
        "steps": MAX_STEPS,
        "elapsed": time.time() - t0,
    }

# ── 入口 ───────────────────────────────────────────────────────────────────

DEMO_QUESTIONS = [
    "宁德今天的天气怎么样？",              # 链式：getcode → get_weather_by_coords
    "太原的经纬度是多少？",              # 单工具：只 getcode
    "北京这个地方天气如何？",  # 单工具：只 get_weather_by_coords
]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="作业：拆分天气工具 + agent loop")
    parser.add_argument("--question", "-q", help="单个问题")
    parser.add_argument("--demo", action="store_true", help="跑内置示例问题集")
    parser.add_argument("--provider", default="dashscope", choices=PROVIDERS.keys())
    parser.add_argument("--quiet", action="store_true", help="少输出")
    args = parser.parse_args()

    client, model = build_client(args.provider)
    print(f"[Split Weather Agent] provider={args.provider} model={model}\n")

    questions = DEMO_QUESTIONS if args.demo else ([args.question] if args.question else [DEMO_QUESTIONS[0]])
    for i, q in enumerate(questions, 1):
        print("=" * 60)
        print(f"Q{i}：{q}")
        print("=" * 60)
        result = run(client, model, q, verbose=not args.quiet)
        print("\n最终回答：")
        print(result["answer"])
        print(f"\n（工具调用 {len(result['tool_calls'])} 次，循环 {result['steps']} 轮，"
              f"耗时 {result['elapsed']:.1f}s）\n")


if __name__ == "__main__":
    main()

"""
run_weather_loop.py — 天气查询：单轮闭环 → 多轮循环调用（ReAct）

对比原版 mode_function_call/run_function_call.py：
  【原版·单轮】
    create →（若有 tool_calls）执行一次 → 再 create → 直接取最终回答
    第二次 create 即使还想调工具，宿主也不再继续执行。

  【本作业·循环】
    while 轮次 < MAX_ROUNDS:
        create
        若无 tool_calls → 结束，输出最终回答
        若有 tool_calls → 执行并 role=tool 回填 → 进入下一轮

为何天气场景特别适合演示循环：
  工具拆成 geocode_city → get_weather_by_coords 两步链式依赖，
  模型必须先拿坐标，再查天气；多城市提问时还会多轮重复该链。

使用方式：
  set DEEPSEEK_API_KEY=sk-xxx
  python week11作业/run_weather_loop.py -q "宁德今天天气怎么样？"
  python week11作业/run_weather_loop.py -q "对比北京和上海的天气"
  python week11作业/run_weather_loop.py --demo

依赖：
  pip install openai httpx
  环境变量：DEEPSEEK_API_KEY（默认）；或 --provider dashscope 时用 DASHSCOPE_API_KEY
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

# 本目录可直接 import weather_backend
sys.path.insert(0, str(Path(__file__).parent))

from weather_backend import geocode_city, get_weather_by_coords  # noqa: E402

# ── LLM 配置 ───────────────────────────────────────────────────────────────

PROVIDERS = {
    "deepseek": {
        "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    },
    "dashscope": {
        "api_key": os.environ.get("DASHSCOPE_API_KEY", ""),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
}

# 防止模型死循环无限调工具
MAX_ROUNDS = 8


def build_client(provider: str):
    cfg = PROVIDERS[provider]
    if not cfg["api_key"]:
        print(f"错误：未设置 {provider.upper()}_API_KEY", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"]), cfg["model"]


# ── 工具 Schema：刻意拆成两步，迫使模型走链式循环 ──────────────────────────

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "geocode_city",
            "description": (
                "将城市中文名解析为经纬度。查询天气前必须先调用本工具。"
                "返回 JSON，含 latitude、longitude、name、country、admin1。"
                "若查多个城市，请对每个城市分别调用一次。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市中文名，如 '宁德'、'北京'、'上海'",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_by_coords",
            "description": (
                "根据经纬度查询当前天气及未来3天预报。"
                "latitude/longitude（以及可选的 city_name/country/admin1）"
                "必须来自 geocode_city 的返回结果，不要自己编造坐标。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number", "description": "纬度"},
                    "longitude": {"type": "number", "description": "经度"},
                    "city_name": {"type": "string", "description": "城市显示名，来自 geocode_city"},
                    "country": {"type": "string", "description": "国家，来自 geocode_city"},
                    "admin1": {"type": "string", "description": "省/州，来自 geocode_city"},
                },
                "required": ["latitude", "longitude"],
            },
        },
    },
]

TOOL_DISPATCH = {
    "geocode_city": geocode_city,
    "get_weather_by_coords": get_weather_by_coords,
}

SYSTEM_PROMPT = (
    "你是天气助手。回答天气相关问题时，必须按两步调用工具：\n"
    "1) 先调用 geocode_city 获取城市经纬度；\n"
    "2) 再用返回的坐标调用 get_weather_by_coords 获取天气。\n"
    "不要编造坐标或天气数据。若用户一次问多个城市，对每个城市重复上述两步。"
    "拿到足够信息后再用自然语言总结回答。"
)


def execute_tool(name: str, args: dict) -> str:
    fn = TOOL_DISPATCH.get(name)
    if fn is None:
        return f"未知工具：{name}"
    try:
        return fn(**args)
    except TypeError as e:
        return f"参数错误：{e}"
    except Exception as e:
        return f"工具执行失败：{e}"


def run(client, model: str, question: str, verbose: bool = True) -> dict:
    """
    多轮循环调用（ReAct 风格）：
      每轮：LLM 决策 → 有 tool_calls 则执行并回填 → 再决策；
            无 tool_calls 则视为最终回答并退出。
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    tool_call_log = []
    round_idx = 0

    while round_idx < MAX_ROUNDS:
        round_idx += 1
        if verbose:
            print(f"  ── 第 {round_idx} 轮 ──")

        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        # 退出条件：本轮没有 tool_calls → 模型给出最终回答
        if not msg.tool_calls:
            answer = msg.content or ""
            elapsed = time.time() - t0
            if verbose:
                print(f"  → [llm] 最终回答（共 {round_idx} 轮，{elapsed:.1f}s）")
            return {
                "answer": answer,
                "tool_calls": tool_call_log,
                "rounds": round_idx,
                "elapsed": elapsed,
            }

        # 有工具调用：回填 assistant 消息，再逐个执行
        messages.append(msg)
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            tool_call_log.append({"round": round_idx, "name": name, "args": args})
            if verbose:
                print(f"  → [tool] {name}({args})")

            result = execute_tool(name, args)
            preview = (result or "")[:120].replace("\n", " ")
            if verbose:
                print(f"    ↩ {preview}{'...' if len(result or '') > 120 else ''}\n")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
        # 回到 while 顶部，进入下一轮 create —— 这就是「循环调用」

    # 超过最大轮次仍未结束：强制再要一次纯文本回答
    if verbose:
        print(f"  ⚠ 已达 MAX_ROUNDS={MAX_ROUNDS}，强制生成最终回答")
    resp = client.chat.completions.create(
        model=model,
        messages=messages + [{
            "role": "user",
            "content": "请根据已有工具结果直接给出最终回答，不要再调用工具。",
        }],
    )
    answer = resp.choices[0].message.content or ""
    elapsed = time.time() - t0
    return {
        "answer": answer,
        "tool_calls": tool_call_log,
        "rounds": round_idx,
        "elapsed": elapsed,
        "truncated": True,
    }


DEMO_QUESTIONS = [
    "宁德今天天气怎么样？",
    "对比北京和上海的天气",
    "查询福州的天气，如果找不到就试一下福州市",
]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="天气查询 · 多轮循环工具调用")
    parser.add_argument("--question", "-q", help="单个问题")
    parser.add_argument("--demo", action="store_true", help="跑内置示例问题集")
    parser.add_argument("--provider", default="deepseek", choices=PROVIDERS.keys())
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    client, model = build_client(args.provider)
    if not args.json:
        print(f"[Weather Loop] provider={args.provider} model={model} MAX_ROUNDS={MAX_ROUNDS}\n")

    questions = DEMO_QUESTIONS if args.demo else ([args.question] if args.question else [DEMO_QUESTIONS[0]])
    results = []
    for i, q in enumerate(questions, 1):
        if not args.json:
            print("=" * 60)
            print(f"Q{i}：{q}")
            print("=" * 60)
        result = run(client, model, q, verbose=not (args.quiet or args.json))
        result["question"] = q
        results.append(result)
        if not args.json:
            print("\n最终回答：")
            print(result["answer"])
            print(f"\n（工具调用 {len(result['tool_calls'])} 次，LLM 轮次 {result['rounds']}）\n")

    if args.json:
        print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False))


if __name__ == "__main__":
    main()

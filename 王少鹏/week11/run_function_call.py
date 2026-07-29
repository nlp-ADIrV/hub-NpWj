"""
run_function_call.py — 方式一：Function Call（模型原生函数调用）

教学重点：
  1. 手写 JSON Schema：每个工具的 name/description/parameters 都要开发者自己写
     ——这是 Function Call 的"接入成本"，schema 写得越清楚，模型调用越准
  2. 链式/嵌套调用（本版本核心）：与前一个"单轮闭环"版本不同，本版本将 run() 改为
     **多轮循环**——模型每轮都可以继续调工具或输出最终答案，形成工具调用的链条。
     例如：get_coordinates("宁德") → 取到经纬度 → get_weather_by_coords(lat, lon)
  3. 并行工具调用：单轮内模型仍可一次输出多个 tool_call，宿主逐个执行后一并回填
  4. 工具名 → 后端函数的 dispatch 表：业务逻辑（src/）与协议层（本文件）彻底分离

对比单轮 vs 链式：
  - 单轮版本：get_weather(city) 封装了地理编码+天气查询，模型一步到位
  - 链式版本：拆成 get_coordinates + get_weather_by_coords 两个原子工具，
    模型必须先拿到坐标才能查天气——多了"中间产物"的透明性，也降低了单工具复杂度。
    更重要的是：中间结果（经纬度）可以被下游的**其他链条**复用（如地图、时区、海拔查询）。

使用方式：
  # 配置环境变量
  #   Windows:  set DEEPSEEK_API_KEY=sk-xxx & set DASHSCOPE_API_KEY=sk-xxx
  #   Linux:    export DEEPSEEK_API_KEY=sk-xxx; export DASHSCOPE_API_KEY=sk-xxx

  # 单个问题
  python mode_function_call/run_function_call.py --question "宁德今天天气如何？"

  # 内置示例问题（演示链式工具调用）
  python mode_function_call/run_function_call.py --demo

依赖：
  pip install openai
  环境变量：DASHSCOPE_API_KEY（Embedding，rag_backend 内部用）
            DEEPSEEK_API_KEY（默认 LLM；可在 --provider dashscope 切到 qwen-plus）

与其它方式的关系：
  本文件的 LLM 多轮循环代码，和 mode_mcp/run_mcp.py、mode_cli/run_cli.py 几乎一样，
  差异只在"工具从哪来"和"调用怎么执行"——这正是三者对比的教学点。
"""

import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

# 把项目根目录加入 sys.path，让 src 可 import（直接 python 运行本脚本也能找到）
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_backend import search_annual_report, list_companies  # noqa: E402
from src.weather_backend import get_coordinates, get_weather_by_coords, get_weather  # noqa: E402

# ── LLM 配置 ───────────────────────────────────────────────────────────────

PROVIDERS = {
    "deepseek": {
        "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",  # 即 deepseek-v4-flash
    },
    "dashscope": {
        "api_key": os.environ.get("DASHSCOPE_API_KEY", ""),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
}


def build_client(provider: str):
    cfg = PROVIDERS[provider]
    if not cfg["api_key"]:
        print(f"错误：未设置 {provider.upper()}_API_KEY", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"]), cfg["model"]


# ── 【教学时刻 1】：手写工具的 JSON Schema ──────────────────────────────────
# Function Call 的核心接入成本：每个工具的参数 schema 必须开发者手写。
# description 直接决定模型"什么时候调这个工具、传什么参数"——写得越具体越准。

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_annual_report",
            "description": (
                "在A股年报语料库中检索与问题最相关的段落。"
                "知识库仅收录 5 家公司：贵州茅台(600519)/五粮液(000858)/"
                "宁德时代(300750)/海康威视(002415)/中国平安(601318)，"
                "年份仅 2021/2022/2023。不在库内的公司请勿调用本工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "检索问题，自然语言。重要：不要包含公司名和年份"
                            "（已由 stock_code/year 参数过滤），只用简短财务术语，"
                            "例如 '营收和净利润'、'研发投入'、'主营业务'。"
                            "把公司名写进 query 会稀释检索精度。"
                        ),
                    },
                    "stock_code": {
                        "type": "string",
                        "description": "可选，按公司过滤，如 '300750'。不传则跨公司检索",
                    },
                    "year": {
                        "type": "string",
                        "description": "可选，按年份过滤：'2021' / '2022' / '2023'",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回段落数，默认5，建议不超过10",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_companies",
            "description": "列出年报知识库中收录的所有公司、股票代码与可查年份。用于确认目标公司在库内。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_coordinates",
            "description": "查询城市的地理坐标（经纬度）和行政信息（国家、省份、城市名）。结果可作为 get_weather_by_coords 的输入。城市用中文名，如 '宁德'、'北京'。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市中文名，如 '宁德'"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_by_coords",
            "description": "根据经纬度查询当前天气及未来3天预报。经纬度通常由 get_coordinates 工具提供。",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number", "description": "纬度，如 26.67"},
                    "longitude": {"type": "number", "description": "经度，如 119.55"},
                },
                "required": ["latitude", "longitude"],
            },
        },
    },
]

# ── 【教学时刻 2】：工具名 → 后端函数的 dispatch 表 ─────────────────────────
# 业务逻辑在 src/，本文件只负责"协议层"——把模型生成的 tool_call 派发给后端函数。
# 新增工具只需：1) 在上面写 schema；2) 在这里加一行映射。这是 Function Call 的扩展方式。

TOOL_DISPATCH = {
    "search_annual_report": search_annual_report,
    "list_companies": list_companies,
    "get_coordinates": get_coordinates,
    "get_weather_by_coords": get_weather_by_coords,
    "get_weather": get_weather,  # 保留旧版本兼容（单步查天气仍然可用）
}


# ── 链式 / 嵌套调用 ────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "你是一名金融分析助手。你可以调用以下工具获取信息：\n"
    "1. search_annual_report — 检索A股年报原文\n"
    "2. list_companies — 查看知识库内的公司列表\n"
    "3. get_coordinates — 查询城市经纬度\n"
    "4. get_weather_by_coords — 根据经纬度查天气\n\n"
    "你可以在一次回答中调用多个工具，也可以根据前一个工具的结果继续调用下一个工具。\n"
    "例如：先 get_coordinates('宁德') 获取坐标，再 get_weather_by_coords(lat, lon) 查天气。\n\n"
    "回答年报问题时，只依据 search_annual_report 返回的段落作答，不要编造数据。"
    "如果用户问的公司不在知识库"
    "（贵州茅台/五粮液/宁德时代/海康威视/中国平安），请明确告知不在库内，不要臆测。"
)


def run(client, model: str, question: str, verbose: bool = True, max_rounds: int = 10) -> dict:
    """
    链式/嵌套调用：多轮循环，模型每轮都可以选择调工具或输出最终答案。

    与旧版"单轮闭环"的核心区别：
      - 旧版：一轮 tool_call → 执行 → 第二轮直接要求 final answer
      - 新版：N 轮循环，每轮模型可继续调工具，直到模型输出纯文本回答

    返回 {answer, tool_calls, elapsed, rounds} 用于对比器汇总。
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    tool_call_log = []
    total_rounds = 0
    answer = ""

    for round_idx in range(max_rounds):
        total_rounds = round_idx + 1

        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            answer = msg.content or ""
            if verbose:
                print(f"  -> [llm] final answer (round {total_rounds})")
            break

        # 模型想继续调工具
        messages.append(msg)
        if verbose:
            print(f"  -> [round {total_rounds}] tools:")

        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            tool_call_log.append({"name": name, "args": args, "round": total_rounds})
            if verbose:
                safe_args = str(args).encode("ascii", errors="replace").decode("ascii")
                print(f"    -> [tool] {name}({safe_args})")

            fn = TOOL_DISPATCH.get(name)
            if fn is None:
                result = f"未知工具：{name}"
            else:
                try:
                    result = fn(**args)
                except TypeError as e:
                    result = f"参数错误：{e}"
                except Exception as e:
                    result = f"工具执行失败：{e}"

            preview = (result or "")[:120].replace("\n", " ").encode("ascii", errors="replace").decode("ascii")
            if verbose:
                print(f"      <- {preview}{'...' if len(result or '') > 120 else ''}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    else:
        # max_rounds 耗尽仍未得到最终回答
        answer = answer or "(已达最大调用轮次，未能生成最终回答)"

    elapsed = time.time() - t0
    return {"answer": answer, "tool_calls": tool_call_log, "elapsed": elapsed, "rounds": total_rounds}


# ── 入口 ───────────────────────────────────────────────────────────────────

DEMO_QUESTIONS = [
    "宁德时代2023年营收和净利润是多少？",
    "宁德今天天气如何？",
    "宁德时代2023年营收和净利润是多少？另外总部宁德的天气如何？",
    "对比贵州茅台和五粮液2023年的营收。",
    "比亚迪2023年营收是多少？",  # 幻觉控制：比亚迪不在知识库
]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="方式一：Function Call（链式调用版）")
    parser.add_argument("--question", "-q", help="单个问题")
    parser.add_argument("--demo", action="store_true", help="跑内置示例问题集")
    parser.add_argument("--provider", default="deepseek", choices=PROVIDERS.keys())
    parser.add_argument("--max-rounds", type=int, default=10, help="最大工具调用轮次")
    parser.add_argument("--quiet", action="store_true", help="少输出（被 compare.py 调用时用）")
    parser.add_argument("--json", action="store_true", help="输出 JSON（供 compare.py 解析）")
    args = parser.parse_args()

    client, model = build_client(args.provider)
    if not args.json:
        sys.stderr.write(f"[Function Call - Chain] provider={args.provider} model={model} max_rounds={args.max_rounds}\n")

    questions = DEMO_QUESTIONS if args.demo else ([args.question] if args.question else [DEMO_QUESTIONS[0]])
    results = []
    for i, q in enumerate(questions, 1):
        if not args.json:
            safe_q = q.encode("ascii", errors="replace").decode("ascii")
            sys.stderr.write(f"=== Q{i}: {safe_q} ===\n")
        result = run(client, model, q, verbose=not (args.quiet or args.json), max_rounds=args.max_rounds)
        result["question"] = q
        results.append(result)
        if not args.json:
            safe_rounds = f"{result['rounds']} rounds, {len(result['tool_calls'])} tool calls"
            sys.stderr.write(f"  [info] {safe_rounds}\n")
            sys.stdout.write("\nAnswer:\n")
            safe_answer = result["answer"].encode("ascii", errors="replace").decode("ascii")
            sys.stdout.write(safe_answer + "\n\n")

    if args.json:
        print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False))


if __name__ == "__main__":
    main()

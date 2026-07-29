#!/usr/bin/env python3
"""第 11 周作业：拆分天气工具，并用循环实现链式 Function Call。

依赖：pip install openai httpx

运行示例：
    export OPENAI_API_KEY="你的 API Key"
    export OPENAI_BASE_URL="你的 BASE URL"
    export OPENAI_MODEL="gpt-5.6-sol"
    python3 week11_weather_tool_loop.py --demo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Callable

import httpx
from openai import OpenAI


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_CODE_MAP = {
    0: "晴天",
    1: "大致晴朗",
    2: "局部多云",
    3: "阴天",
    45: "雾",
    48: "冻雾",
    51: "小毛毛雨",
    53: "中毛毛雨",
    55: "大毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    80: "小阵雨",
    81: "中阵雨",
    82: "大阵雨",
    95: "雷暴",
    96: "雷暴伴小冰雹",
    99: "雷暴伴大冰雹",
}


def get_coordinates(city: str) -> dict[str, Any]:
    """把城市名称解析成经纬度，不查询天气。"""

    city = city.strip()
    if not city:
        return {"error": "城市名称不能为空"}

    def geocode(client: httpx.Client, name: str) -> list[dict[str, Any]]:
        response = client.get(
            GEOCODING_URL,
            params={"name": name, "count": 10, "language": "zh", "format": "json"},
        )
        response.raise_for_status()
        return response.json().get("results") or []

    try:
        with httpx.Client(timeout=10.0) as client:
            results = geocode(client, city)

            # 裸城市名可能命中同名村庄，沿用课堂代码的“追加市后重查”策略。
            is_low_admin = (
                all(
                    str(item.get("feature_code", "")).startswith("PPL")
                    and not str(item.get("feature_code", "")).startswith("PPLA")
                    for item in results
                )
                if results
                else True
            )
            has_suffix = any(city.endswith(suffix) for suffix in ("市", "县", "区", "镇"))
            if is_low_admin and not has_suffix:
                retry_results = geocode(client, city + "市")
                if retry_results:
                    results = retry_results
    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        return {"error": f"经纬度查询失败：{exc}"}

    if not results:
        return {"error": f"未找到城市 '{city}'，请检查城市名称"}

    def rank_location(item: dict[str, Any]) -> tuple[int, int]:
        feature_code = str(item.get("feature_code", ""))
        admin_priority = int(feature_code.startswith(("PPLA", "ADM")))
        population = item.get("population") or 0
        return admin_priority, population

    location = max(results, key=rank_location)
    return {
        "query_city": city,
        "resolved_name": location.get("name", city),
        "country": location.get("country", ""),
        "admin1": location.get("admin1", ""),
        "latitude": location["latitude"],
        "longitude": location["longitude"],
    }


def get_weather(latitude: float, longitude: float) -> dict[str, Any]:
    """根据经纬度查询当前天气和未来三天预报，不接受城市名称。"""

    if not -90 <= latitude <= 90:
        return {"error": "纬度必须在 -90 到 90 之间"}
    if not -180 <= longitude <= 180:
        return {"error": "经度必须在 -180 到 180 之间"}

    try:
        response = httpx.get(
            WEATHER_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": (
                    "temperature_2m,relative_humidity_2m,"
                    "wind_speed_10m,weather_code"
                ),
                "daily": (
                    "temperature_2m_max,temperature_2m_min,"
                    "precipitation_sum,weather_code"
                ),
                "timezone": "auto",
                "forecast_days": 3,
            },
            timeout=10.0,
        )
        response.raise_for_status()
    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        return {"error": f"天气查询失败：{exc}"}

    data = response.json()
    current = data["current"]
    daily = data["daily"]
    forecast = []
    for index, date in enumerate(daily["time"]):
        weather_code = daily["weather_code"][index]
        forecast.append(
            {
                "date": date,
                "weather": WEATHER_CODE_MAP.get(weather_code, f"天气代码 {weather_code}"),
                "max_temperature_c": daily["temperature_2m_max"][index],
                "min_temperature_c": daily["temperature_2m_min"][index],
                "precipitation_mm": daily["precipitation_sum"][index],
            }
        )

    current_code = current["weather_code"]
    return {
        "latitude": data.get("latitude", latitude),
        "longitude": data.get("longitude", longitude),
        "timezone": data.get("timezone", ""),
        "current": {
            "time": current["time"],
            "weather": WEATHER_CODE_MAP.get(current_code, f"天气代码 {current_code}"),
            "temperature_c": current["temperature_2m"],
            "relative_humidity_percent": current["relative_humidity_2m"],
            "wind_speed_kmh": current["wind_speed_10m"],
        },
        "forecast": forecast,
    }


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_coordinates",
            "description": (
                "根据城市名称查询经纬度。用户只问城市经纬度时只调用本工具；"
                "用户按城市查询天气时，必须先调用本工具取得 latitude 和 longitude，"
                "再把结果传给 get_weather。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称，例如北京、上海、宁德"}
                },
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "根据经纬度查询当前天气及未来三天预报。若用户只提供城市名称，"
                "不能猜测坐标，应先调用 get_coordinates，再使用其返回值调用本工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {
                        "type": "number",
                        "description": "纬度，范围 -90 到 90",
                    },
                    "longitude": {
                        "type": "number",
                        "description": "经度，范围 -180 到 180",
                    },
                },
                "required": ["latitude", "longitude"],
                "additionalProperties": False,
            },
        },
    },
]

TOOL_DISPATCH: dict[str, Callable[..., dict[str, Any]]] = {
    "get_coordinates": get_coordinates,
    "get_weather": get_weather,
}

SYSTEM_PROMPT = """你是天气助手，只依据工具返回的数据回答。
你有两个职责单一的工具：
1. get_coordinates：城市名转换为经纬度。
2. get_weather：根据经纬度查询天气。

严格遵守：
- 只问某城市经纬度：调用 get_coordinates 后直接回答，不查询天气。
- 已给出经纬度并询问天气：直接调用 get_weather。
- 根据城市询问天气：先调用 get_coordinates；看到其结果后，再调用 get_weather。
- 工具报错时如实说明，不得编造坐标或天气。"""


def execute_tool(name: str, arguments_text: str) -> str:
    """解析模型参数、执行工具，并统一返回 JSON 字符串。"""

    function = TOOL_DISPATCH.get(name)
    if function is None:
        return json.dumps({"error": f"未知工具：{name}"}, ensure_ascii=False)

    try:
        arguments = json.loads(arguments_text or "{}")
        result = function(**arguments)
    except json.JSONDecodeError as exc:
        result = {"error": f"工具参数不是合法 JSON：{exc}"}
    except TypeError as exc:
        result = {"error": f"工具参数错误：{exc}"}
    except Exception as exc:  # 防止单个工具异常中断整个调用循环
        result = {"error": f"工具执行失败：{exc}"}
    return json.dumps(result, ensure_ascii=False)


def run_tool_loop(
    client: OpenAI,
    model: str,
    question: str,
    max_rounds: int = 6,
    verbose: bool = True,
) -> dict[str, Any]:
    """循环调用模型和工具，直到模型给出最终文本回答。"""

    messages: list[Any] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    tool_call_log: list[dict[str, Any]] = []

    for round_number in range(1, max_rounds + 1):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            temperature=0,
        )
        message = response.choices[0].message

        # 没有 tool_calls 说明模型已经得到足够信息，循环结束。
        if not message.tool_calls:
            return {
                "answer": message.content or "",
                "tool_calls": tool_call_log,
                "rounds": round_number,
            }

        # 必须先回填带 tool_calls 的 assistant 消息，再逐条回填工具结果。
        messages.append(message)
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            arguments_text = tool_call.function.arguments or "{}"
            try:
                logged_arguments = json.loads(arguments_text)
            except json.JSONDecodeError:
                logged_arguments = arguments_text
            tool_call_log.append({"name": name, "arguments": logged_arguments})

            if verbose:
                print(f"[第 {round_number} 轮] 模型调用 {name}({arguments_text})")
            result = execute_tool(name, arguments_text)
            if verbose:
                print(f"[工具返回] {result}\n")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )

    return {
        "answer": f"达到最大工具调用轮数 {max_rounds}，为防止无限循环已停止。",
        "tool_calls": tool_call_log,
        "rounds": max_rounds,
    }


DEMO_QUESTIONS = [
    "北京的经纬度是多少？",
    "北纬 39.9042、东经 116.4074 当前天气如何？",
    "上海现在天气怎么样？请先查询坐标，再查询天气。",
]


def build_client() -> tuple[OpenAI, str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("请先设置环境变量 OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://ai.btcapp.net/v1")
    model = os.getenv("OPENAI_MODEL", "gpt-5.6-sol")
    # 部分第三方中转站会拦截 OpenAI SDK 的默认 User-Agent，允许用环境变量覆盖。
    user_agent = os.getenv("OPENAI_USER_AGENT", "curl/8.7.1")
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers={"User-Agent": user_agent},
    )
    return client, model


def run_self_test() -> int:
    """直接测试两个后端函数，不消耗大模型 Token。"""

    coordinates = get_coordinates("北京")
    print("get_coordinates('北京')：")
    print(json.dumps(coordinates, ensure_ascii=False, indent=2))
    if "error" in coordinates:
        return 1

    weather = get_weather(coordinates["latitude"], coordinates["longitude"])
    print("\nget_weather(latitude, longitude)：")
    print(json.dumps(weather, ensure_ascii=False, indent=2))
    return int("error" in weather)


def main() -> int:
    parser = argparse.ArgumentParser(description="天气工具拆分与循环 Function Call")
    parser.add_argument("--question", "-q", help="单个问题；不传时进入交互模式")
    parser.add_argument("--demo", action="store_true", help="运行三种工具调用示例")
    parser.add_argument("--self-test", action="store_true", help="只测试天气后端，不调用大模型")
    parser.add_argument("--max-rounds", type=int, default=6, help="最大模型调用轮数")
    args = parser.parse_args()

    if args.self_test:
        return run_self_test()

    try:
        client, model = build_client()
    except RuntimeError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    if args.demo:
        questions = DEMO_QUESTIONS
    elif args.question:
        questions = [args.question]
    else:
        questions = []

    if questions:
        for index, question in enumerate(questions, 1):
            print(f"{'=' * 60}\n问题 {index}：{question}\n{'=' * 60}")
            result = run_tool_loop(client, model, question, args.max_rounds)
            print(f"最终回答：{result['answer']}")
            print(f"模型调用轮数：{result['rounds']}，工具调用：{len(result['tool_calls'])}\n")
        return 0

    print(f"天气工具循环调用演示（模型：{model}），输入 exit 退出。")
    while True:
        try:
            question = input("\n问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit", "退出"}:
            break
        if question:
            result = run_tool_loop(client, model, question, args.max_rounds)
            print(f"回答> {result['answer']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import json
import os
import sys
from typing import Any

from openai import OpenAI

from weather_tools import TOOL_DISPATCH

MAX_TOOL_ROUNDS = 8

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_city",
            "description": (
                "根据城市名称查询候选地点和经纬度。"
                "当用户只提供城市名时，先调用此工具，再根据结果调用 query_weather。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如北京、上海、杭州",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_weather",
            "description": "根据经纬度查询当前天气和未来三天天气。",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {
                        "type": "number",
                        "description": "纬度",
                    },
                    "longitude": {
                        "type": "number",
                        "description": "经度",
                    },
                },
                "required": ["latitude", "longitude"],
            },
        },
    },
]

SYSTEM_PROMPT = """
你是天气查询助手。

处理天气问题时遵循以下规则：
1. 用户给出城市名但没有经纬度时，先调用 search_city。
2. 从 search_city 返回结果中选择最合理的城市候选，再调用 query_weather。
3. 如果用户一次询问多个城市，可以继续调用工具，直到所有城市都查询完成。
4. 工具返回信息不足时，可以进入下一轮继续调用工具。
5. 只有在信息足够时才输出最终答案，不要编造天气数据。
""".strip()


def create_client() -> tuple[OpenAI, str]:
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "未检测到 DASHSCOPE_API_KEY，请先设置环境变量后再运行。"
        )

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    return client, "qwen-plus"


def execute_tool(name: str, arguments: dict[str, Any]) -> str:
    function = TOOL_DISPATCH.get(name)
    if function is None:
        return f"未注册的工具：{name}"

    try:
        result = function(**arguments)
    except TypeError as exc:
        return f"工具参数错误：{exc}"
    except Exception as exc:
        return f"工具执行异常：{exc}"

    return str(result)


def run_agent(question: str, verbose: bool = True) -> str:
    client, model = create_client()

    messages: list[Any] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for round_index in range(1, MAX_TOOL_ROUNDS + 1):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        assistant_message = response.choices[0].message

        if verbose:
            print(f"\n[第 {round_index} 轮]")

        if not assistant_message.tool_calls:
            return assistant_message.content or ""

        messages.append(assistant_message)

        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name

            try:
                tool_args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                tool_args = {}

            if verbose:
                print(f"调用工具：{tool_name}")
                print(f"参数：{tool_args}")

            tool_result = execute_tool(tool_name, tool_args)

            if verbose:
                preview = tool_result.replace("\n", " ")[:180]
                print(f"结果：{preview}")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                }
            )

    return f"已达到最大工具调用轮数 {MAX_TOOL_ROUNDS}，本次任务终止。"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="支持循环工具调用的天气查询示例"
    )
    parser.add_argument(
        "-q",
        "--question",
        default="请查询北京今天的天气，并告诉我未来三天的温度情况。",
        help="天气查询问题",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="只输出最终回答",
    )
    args = parser.parse_args()

    try:
        answer = run_agent(args.question, verbose=not args.quiet)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    print("\n最终回答：")
    print(answer)


if __name__ == "__main__":
    main()

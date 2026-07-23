"""最小天气 Function Call 示例。

当前故意保留为“单轮工具调用”基线：模型调用一批工具，宿主回填结果后，
只再请求模型一次。week11 作业是在此基础上改造成多轮循环调用。
"""

import argparse
import json
import os
import sys

from openai import OpenAI

from weather_backend import get_weather


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"
MAX_TOOL_ROUND = 5


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的当前天气及未来 3 天预报。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市中文名，例如：北京、上海、宁德",
                    }
                },
                "required": ["city"],
            },
        },
    }
]

TOOL_DISPATCH = {"get_weather": get_weather}

SYSTEM_PROMPT = (
    "你是天气助手。回答天气问题时必须调用 get_weather 工具，"
    "并且只能根据工具返回的数据回答。"
)


def build_client() -> OpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError("请先设置环境变量 DEEPSEEK_API_KEY")
    return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)


def run(client: OpenAI, question: str) -> str:
    """循环执行最多5轮工具调用。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    round = 0

    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )
        # 第一次请求：让模型决定是否调用天气工具。
        message = response.choices[0].message

        if not message.tool_calls:
            return message.content or ""

        if round >= MAX_TOOL_ROUND:
            return f"Max Tool Call Reached {MAX_TOOL_ROUND}, Stop"

        round += 1
        print(
            f"[now round {round}] "
            f"Tool call number : {len(message.tool_calls)}"
        )

        # 必须先保存带 tool_calls 的 assistant 消息。
        messages.append(message)

        # 这个 for 只处理模型同一轮返回的一批工具调用，不是作业要求的多轮循环。
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name

            try:
                tool_arguments = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                result = f"工具参数不是合法 JSON：{exc}"
            else:
                print(f"[tool] {tool_name}({tool_arguments})")
                tool_function = TOOL_DISPATCH.get(tool_name)
                if tool_function is None:
                    result = f"未知工具：{tool_name}"
                else:
                    try:
                        result = str(tool_function(**tool_arguments))
                    except Exception as exc:  # 将错误回填给模型，而不是中断消息链。
                        result = f"工具执行失败：{exc}"

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )

    # # 第二次请求：当前基线在这里直接结束，不会继续处理新的 tool_calls。
    # response = client.chat.completions.create(
    #     model=MODEL,
    #     messages=messages,
    #     tools=TOOLS_SCHEMA,
    #     tool_choice="auto",
    # )
    # final_message = response.choices[0].message
    #
    # if final_message.tool_calls:
    #     return "模型仍想调用工具；请完成作业要求的多轮循环改造。"
    # return final_message.content or ""


def main() -> int:
    parser = argparse.ArgumentParser(description="天气 Function Call 最小示例")
    parser.add_argument(
        "--question",
        "-q",
        default="北京今天天气怎么样？",
        help="要询问的天气问题",
    )
    arguments = parser.parse_args()

    try:
        client = build_client()
        answer = run(client, arguments.question)
    except RuntimeError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1

    print("\n最终回答：")
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

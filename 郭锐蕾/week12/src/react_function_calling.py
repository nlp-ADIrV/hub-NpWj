"""
Function Calling API 版 ReAct Agent（多轮对话增强）

相对原版新增：
  - run(..., history=[...])：注入此前用户问答
  - System Prompt 增加多轮指代规则

使用方式：
  python react_function_calling.py
  python react_function_calling.py --question "茅台近一年股价涨跌幅如何？"
"""

from __future__ import annotations

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Generator

from openai import OpenAI

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_PARENT_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_PARENT_SRC) not in sys.path:
    sys.path.insert(0, str(_PARENT_SRC))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# 与原项目一致：优先 DashScope；若设置了 DEEPSEEK_API_KEY 且未设置 DASHSCOPE，可自行改配置
_api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
_base_url = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
    if os.getenv("DASHSCOPE_API_KEY")
    else "https://api.deepseek.com"
)
client = OpenAI(api_key=_api_key, base_url=_base_url)
MODEL = os.getenv("AGENT_MODEL", "qwen-max" if os.getenv("DASHSCOPE_API_KEY") else "deepseek-chat")

FC_SYSTEM_PROMPT = """你是一个专业的A股金融分析助手。
规则：
- 调用 financial_indicator 或 stock_price 之前，必须先用 company_lookup 获取股票代码
- 数字计算必须使用 calculator 工具，不能心算
- Final Answer 必须引用具体数据来源
- 如果没有合适工具能回答，直接说明原因
- 【多轮对话】可能包含此前问答。请理解「它」「那家公司」「刚才」「同样」等指代；
  若历史数据已足够回答，可直接作答，无需重复调用工具
"""


def run(
    question: str,
    max_steps: int = 10,
    history: list[dict] | None = None,
) -> Generator[dict, None, None]:
    from tools import TOOLS_MAP, TOOLS_SCHEMA

    messages: list[dict] = [{"role": "system", "content": FC_SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": question})

    for step in range(1, max_steps + 1):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            temperature=0,
        )
        msg = response.choices[0].message
        reason = response.choices[0].finish_reason

        if reason == "stop" or not msg.tool_calls:
            yield {
                "step": step,
                "type": "final",
                "thought": "",
                "answer": msg.content or "（模型返回空内容）",
            }
            return

        messages.append(msg)

        for tool_call in msg.tool_calls:
            tool_name = tool_call.function.name
            try:
                tool_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}

            tool_fn = TOOLS_MAP.get(tool_name)
            if tool_fn is None:
                observation = f"未知工具 '{tool_name}'"
            else:
                try:
                    observation = tool_fn(**tool_args)
                except TypeError as e:
                    observation = f"工具参数错误: {e}"

            yield {
                "step": step,
                "type": "action",
                "thought": "",
                "action": tool_name,
                "action_input": tool_args,
                "observation": str(observation),
            }

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(observation),
            })

    yield {
        "step": max_steps + 1,
        "type": "max_steps",
        "answer": f"已达最大步数 {max_steps}，未能得出最终答案",
    }


COLORS = {
    "thought": "\033[36m",
    "action": "\033[33m",
    "obs": "\033[32m",
    "final": "\033[35m",
    "error": "\033[31m",
    "reset": "\033[0m",
}


def _c(color: str, text: str) -> str:
    return f"{COLORS[color]}{text}{COLORS['reset']}"


def run_and_print(question: str, max_steps: int = 10, history: list[dict] | None = None):
    print(f"\n{'=' * 60}")
    print(f"问题: {question}")
    print(f"模型: {MODEL}  实现: Function Calling（多轮）")
    if history:
        print(f"历史轮数: {len(history) // 2}")
    print("=" * 60)

    start = time.time()

    for step_data in run(question, max_steps=max_steps, history=history):
        stype = step_data["type"]

        if stype == "action":
            print(f"\n[Step {step_data['step']}]")
            print(_c("thought", "🧠 Thought: （模型内部推理，Function Calling 版不可见）"))
            print(_c("action", f"🔧 Action:  {step_data['action']}"))
            print(_c("action", f"   Input:   {json.dumps(step_data['action_input'], ensure_ascii=False)}"))
            print(_c("obs", f"👁  Obs:     {step_data['observation'][:300]}"))

        elif stype == "final":
            elapsed = time.time() - start
            print(f"\n{'─' * 60}")
            print(_c("final", f"\n✅ Final Answer:\n{step_data['answer']}"))
            print(f"\n共 {step_data['step']} 步，耗时 {elapsed:.1f}s")

        elif stype in ("error", "max_steps"):
            print(_c("error", f"\n⚠️  {step_data.get('answer', '')}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--question",
        default="贵州茅台和五粮液2023年的毛利率哪家更高？差多少个百分点？",
    )
    parser.add_argument("--max_steps", type=int, default=10)
    args = parser.parse_args()
    run_and_print(args.question, args.max_steps)

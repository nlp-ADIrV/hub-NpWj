"""
手写 Prompt 解析版 ReAct Agent（多轮对话增强）

相对原版新增：
  - run(..., history=[...])：注入此前用户问答，支持追问与指代
  - System Prompt 增加多轮对话规则

使用方式：
  python react_manual.py
  python react_manual.py --question "茅台2023年毛利率是多少？"
"""

from __future__ import annotations

import os
import re
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Generator

from openai import OpenAI

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# 复用原项目 tools（向量库路径相对原项目根目录）
_PARENT_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_PARENT_SRC) not in sys.path:
    sys.path.insert(0, str(_PARENT_SRC))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
MODEL = os.getenv("AGENT_MODEL", "qwen-max")

SYSTEM_PROMPT = """你是一个专业的A股金融分析助手，可以使用以下工具来回答问题：

工具列表：
1. rag_search(query) - 在年报中语义检索文本内容（战略/财务数据/风险因素等）
2. company_lookup(name) - 将公司名称转换为股票代码
3. calculator(expr) - 计算数学表达式（支持四则运算和math函数）
4. financial_indicator(symbol) - 获取实时财务指标（PE/PB/ROE等）
5. stock_price(symbol, start_date, end_date) - 获取历史股价，日期格式YYYYMMDD

你必须严格按照以下格式交替输出，每次只能调用一个工具：

Thought: 分析当前状态，决定下一步做什么
Action: 工具名称
Action Input: {"参数名": "参数值"}

收到工具结果后继续推理，直到可以给出最终答案：

Thought: 已有足够信息
Final Answer: 完整的回答（含数据来源）

规则：
- 必须先用 company_lookup 获取股票代码，再调用 financial_indicator 或 stock_price
- 数字计算必须用 calculator，不能心算
- Final Answer 必须引用具体数据来源（哪份年报哪一页，或AkShare实时数据）
- 如果没有合适工具能回答，直接输出 Final Answer 说明原因
- 【多轮对话】消息中可能包含此前的问答。用户可能用「它」「那家公司」「刚才」「同样」等指代上文，
  请结合历史理解真实意图；若历史已有明确数据且足以回答，可直接 Final Answer，无需重复查工具
"""

_THOUGHT_RE = re.compile(r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|$)", re.DOTALL)
_ACTION_RE = re.compile(r"Action:\s*(\w+)")
_ACTION_INPUT_RE = re.compile(r"Action Input:\s*(\{.+?\})", re.DOTALL)
_FINAL_RE = re.compile(r"Final Answer:\s*(.+)", re.DOTALL)


def _parse_step(text: str) -> dict:
    final = _FINAL_RE.search(text)
    if final:
        thought_m = _THOUGHT_RE.search(text)
        return {
            "type": "final",
            "thought": thought_m.group(1).strip() if thought_m else "",
            "answer": final.group(1).strip(),
        }

    thought_m = _THOUGHT_RE.search(text)
    action_m = _ACTION_RE.search(text)
    input_m = _ACTION_INPUT_RE.search(text)

    if not action_m:
        return {"type": "unparseable", "raw": text}

    try:
        action_input = json.loads(input_m.group(1)) if input_m else {}
    except json.JSONDecodeError:
        action_input = {}

    return {
        "type": "action",
        "thought": thought_m.group(1).strip() if thought_m else "",
        "action": action_m.group(1).strip(),
        "action_input": action_input,
    }


def run(
    question: str,
    max_steps: int = 10,
    history: list[dict] | None = None,
    verbose: bool = True,
) -> Generator[dict, None, None]:
    """
    执行 ReAct 循环。

    history: 可选的多轮历史消息（user/assistant 交替），在当前问题之前注入。
    """
    from tools import TOOLS_MAP

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": question})

    for step in range(1, max_steps + 1):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0,
            stop=["Observation:"],
        )
        llm_output = response.choices[0].message.content.strip()
        parsed = _parse_step(llm_output)

        if parsed["type"] == "final":
            yield {
                "step": step,
                "type": "final",
                "thought": parsed["thought"],
                "answer": parsed["answer"],
            }
            return

        if parsed["type"] == "unparseable":
            yield {
                "step": step,
                "type": "error",
                "observation": f"格式解析失败，原始输出：{llm_output[:200]}",
            }
            return

        tool_name = parsed["action"]
        tool_args = parsed["action_input"]
        tool_fn = TOOLS_MAP.get(tool_name)

        if tool_fn is None:
            observation = f"未知工具 '{tool_name}'，可用工具：{list(TOOLS_MAP.keys())}"
        else:
            try:
                observation = tool_fn(**tool_args)
            except TypeError as e:
                observation = f"工具参数错误: {e}"

        yield {
            "step": step,
            "type": "action",
            "thought": parsed["thought"],
            "action": tool_name,
            "action_input": tool_args,
            "observation": str(observation),
        }

        messages.append({"role": "assistant", "content": llm_output})
        messages.append({
            "role": "user",
            "content": f"Observation: {observation}\n",
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
    print(f"模型: {MODEL}  实现: 手写Prompt解析（多轮）")
    if history:
        print(f"历史轮数: {len(history) // 2}")
    print("=" * 60)

    start = time.time()

    for step_data in run(question, max_steps=max_steps, history=history):
        stype = step_data["type"]

        if stype == "action":
            print(f"\n[Step {step_data['step']}]")
            print(_c("thought", f"🧠 Thought: {step_data['thought']}"))
            print(_c("action", f"🔧 Action:  {step_data['action']}"))
            print(_c("action", f"   Input:   {json.dumps(step_data['action_input'], ensure_ascii=False)}"))
            print(_c("obs", f"👁  Obs:     {step_data['observation'][:300]}"))

        elif stype == "final":
            elapsed = time.time() - start
            print(f"\n{'─' * 60}")
            if step_data.get("thought"):
                print(_c("thought", f"🧠 Thought: {step_data['thought']}"))
            print(_c("final", f"\n✅ Final Answer:\n{step_data['answer']}"))
            print(f"\n共 {step_data['step']} 步，耗时 {elapsed:.1f}s")

        elif stype in ("error", "max_steps"):
            print(_c("error", f"\n⚠️  {step_data.get('answer', step_data.get('observation', ''))}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--question",
        default="贵州茅台和五粮液2023年的毛利率哪家更高？差多少个百分点？",
    )
    parser.add_argument("--max_steps", type=int, default=10)
    args = parser.parse_args()
    run_and_print(args.question, args.max_steps)

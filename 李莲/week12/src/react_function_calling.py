"""
Function Calling API 版 ReAct Agent

教学重点：
  1. 与手写版对比：框架帮你处理格式解析，但 Thought 过程在内部不可见
  2. tool_choice="auto" 让模型自己决定调用哪个工具或直接回答
  3. finish_reason 判断：tool_calls 表示继续调用，stop 表示给出最终答案
  4. 相同工具集，相同问题，对比两种实现的稳定性和步骤数

使用方式：
  python react_function_calling.py
  python react_function_calling.py --question "茅台近一年股价涨跌幅如何？"
  python react_function_calling.py --question "..." --max_steps 8

依赖：
  pip install openai faiss-cpu sentence-transformers akshare
  export DASHSCOPE_API_KEY="sk-xxx"
"""

import os
import json
import time
import logging
import argparse
from typing import Generator

from openai import OpenAI

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# client = OpenAI(
#     api_key=os.getenv("DASHSCOPE_API_KEY"),
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )
# MODEL = os.getenv("AGENT_MODEL", "qwen-max")
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)
MODEL = os.getenv("AGENT_MODEL", "deepseek-v4-flash")

FC_SYSTEM_PROMPT = """你是一个专业的A股金融分析助手。
规则：
- 调用 financial_indicator 或 stock_price 之前，必须先用 company_lookup 获取股票代码
- 数字计算必须使用 calculator 工具，不能心算
- Final Answer 必须引用具体数据来源
- 如果没有合适工具能回答，直接说明原因
"""


def run(
    question: str, 
    max_steps: int = 10, 
    history: Optional[List[Dict[str, Any]]] = None
) -> Generator[Dict[str, Any], None, List[Dict[str, Any]]]:
    """
    执行 Function Calling 版 ReAct 循环，支持多轮对话上下文。

    Args:
        question: 用户当前的问题。
        max_steps: 单轮最大推理步数。
        history: (可选) 上一轮对话结束后的 messages 列表。如果提供，将在此基础上继续对话。

    Yields:
        dict: 包含 step, type, action, observation 等字段的结构化数据。
    
    Returns:
        list: 更新后的完整 messages 列表，用于下一轮对话。
    """
    from tools import TOOLS_MAP, TOOLS_SCHEMA

    # 1. 初始化或复用消息历史
    if history:
        # 如果有历史，直接复用，并将新问题追加为用户消息
        messages = history.copy()
        messages.append({"role": "user", "content": question})
    else:
        # 如果是新对话，初始化 System Prompt 和第一条用户消息
        messages = [
            {"role": "system", "content": FC_SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ]

    # 记录当前处理的是第几步（如果是多轮，step计数可能需要根据业务需求调整，这里保持单轮重置或全局递增均可，此处演示单轮重置）
    current_step = 1 

    try:
        while current_step <= max_steps:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=0,
            )
            msg    = response.choices[0].message
            reason = response.choices[0].finish_reason

            # --- 情况 A: 模型直接回答（无工具调用） ---
            if reason == "stop" or not msg.tool_calls:
                final_content = msg.content or "（模型返回空内容）"
                
                # 将 Assistant 的最终回复加入历史
                messages.append({"role": "assistant", "content": final_content})

                yield {
                    "step":   current_step,
                    "type":   "final",
                    "thought": "",
                    "answer": final_content,
                }
                # 对话结束，返回更新后的历史供下次使用
                return messages

            # --- 情况 B: 模型请求调用工具 ---
            messages.append(msg) # 将 Assistant 的工具调用请求加入历史

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
                    except Exception as e:
                        observation = f"工具执行异常: {str(e)}"

                # 产出当前步骤信息
                yield {
                    "step":         current_step,
                    "type":         "action",
                    "thought":      "", 
                    "action":       tool_name,
                    "action_input": tool_args,
                    "observation":  str(observation),
                }

                # 将工具执行结果加入历史
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tool_call.id,
                    "content":      str(observation),
                })
            
            current_step += 1

        # --- 情况 C: 达到最大步数 ---
        timeout_msg = f"已达最大步数 {max_steps}，未能得出最终答案"
        messages.append({"role": "assistant", "content": timeout_msg})
        
        yield {
            "step":   current_step,
            "type":   "max_steps",
            "answer": timeout_msg,
        }
        return messages

    except Exception as e:
        # 捕获其他异常，防止程序崩溃，并尝试返回当前状态
        error_msg = f"系统错误: {str(e)}"
        messages.append({"role": "assistant", "content": error_msg})
        yield {
            "step": current_step,
            "type": "error",
            "answer": error_msg
        }
        return messages


# ── CLI 打印（复用 react_manual 的彩色输出） ───────────────────────────────────

COLORS = {
    "thought": "\033[36m",
    "action":  "\033[33m",
    "obs":     "\033[32m",
    "final":   "\033[35m",
    "error":   "\033[31m",
    "reset":   "\033[0m",
}

def _c(color: str, text: str) -> str:
    return f"{COLORS[color]}{text}{COLORS['reset']}"


def run_and_print(question: str, max_steps: int = 10):
    print(f"\n{'='*60}")
    print(f"问题: {question}")
    print(f"模型: {MODEL}  实现: Function Calling")
    print('='*60)

    start = time.time()

    for step_data in run(question, max_steps=max_steps):
        stype = step_data["type"]

        if stype == "action":
            print(f"\n[Step {step_data['step']}]")
            # Thought 在 FC 版不可见，显示提示
            print(_c("thought", "🧠 Thought: （模型内部推理，Function Calling 版不可见）"))
            print(_c("action",  f"🔧 Action:  {step_data['action']}"))
            print(_c("action",  f"   Input:   {json.dumps(step_data['action_input'], ensure_ascii=False)}"))
            print(_c("obs",     f"👁  Obs:     {step_data['observation'][:300]}"))

        elif stype == "final":
            elapsed = time.time() - start
            print(f"\n{'─'*60}")
            print(_c("final", f"\n✅ Final Answer:\n{step_data['answer']}"))
            print(f"\n共 {step_data['step']} 步，耗时 {elapsed:.1f}s")

        elif stype in ("error", "max_steps"):
            print(_c("error", f"\n⚠️  {step_data.get('answer', '')}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question",  default="贵州茅台和五粮液2023年的毛利率哪家更高？差多少个百分点？")
    parser.add_argument("--max_steps", type=int, default=10)
    args = parser.parse_args()
    run_and_print(args.question, args.max_steps)

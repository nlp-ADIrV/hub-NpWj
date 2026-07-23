"""
支持多轮对话的 Function Calling ReAct Agent

教学重点：
  1. 与手写版对比：框架帮你处理格式解析，但 Thought 过程在内部不可见
  2. tool_choice="auto" 让模型自己决定调用哪个工具或直接回答
  3. finish_reason 判断：tool_calls 表示继续调用，stop 表示给出最终答案
  4. 相同工具集，相同问题，对比两种实现的稳定性和步骤数

使用方式：
  python react_function_calling.py                    # 进入多轮对话
  python react_function_calling.py --question "茅台近一年股价涨跌幅如何？"
  python react_function_calling.py --question "..." --max_steps 8

依赖：
  pip install -r ../requirements.txt
  export DEEPSEEK_API_KEY="..."
  export DASHSCOPE_API_KEY="..."
"""

import os
import json
import time
import logging
import argparse
import sys
from typing import Any, Generator

from openai import OpenAI

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

MODEL = os.getenv("AGENT_MODEL", "deepseek-v4-flash")


def build_client() -> OpenAI:
    """从环境变量创建客户端，避免在源码中保存 API Key。"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("请先设置环境变量 DEEPSEEK_API_KEY")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


FC_SYSTEM_PROMPT = """你是一个专业的A股金融分析助手。
规则：
- 结合之前的对话理解“它”“前者”“那家公司”等指代
- 调用 financial_indicator 或 stock_price 之前，必须先用 company_lookup 获取股票代码
- 数字计算必须使用 calculator 工具，不能心算
- Final Answer 必须引用具体数据来源
- 如果没有合适工具能回答，直接说明原因
"""


class ConversationSession:
    """保存最近若干轮已经完成的用户问题和最终回答。"""

    def __init__(self, max_history_turns: int = 6):
        if max_history_turns < 1:
            raise ValueError("max_history_turns 必须大于0")

        self.max_history_turns = max_history_turns
        self._turns: list[tuple[str, str]] = []

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    @property
    def history(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._turns)

    def commit(self, question: str, answer: str) -> None:
        self._turns.append((question, answer))
        self._turns = self._turns[-self.max_history_turns :]

    def build_messages(self, question: str) -> list[dict]:
        messages = [
            {
                "role": "system",
                "content": FC_SYSTEM_PROMPT,
            }
        ]

        for previous_question, previous_answer in self._turns:
            messages.append(
                {
                    "role": "user",
                    "content": previous_question,
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": previous_answer,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": question,
            }
        )
        return messages

    def reset(self) -> None:
        """清空当前会话。"""
        self._turns.clear()


def run(
    question: str,
    max_steps: int = 10,
    session: ConversationSession | None = None,
    api_client: Any = None,
    tools_map: dict | None = None,
    tools_schema: list | None = None,
) -> Generator[dict, None, None]:
    """
    执行 Function Calling 版 ReAct 循环，yield 每一步结构化结果

    格式与 react_manual.run() 保持一致，便于 evaluate.py 统一对比
    """
    if session is None:
        session = ConversationSession()
    if api_client is None:
        api_client = build_client()
    if tools_map is None or tools_schema is None:
        from tools import TOOLS_MAP, TOOLS_SCHEMA

        if tools_map is None:
            tools_map = TOOLS_MAP
        if tools_schema is None:
            tools_schema = TOOLS_SCHEMA

    messages = session.build_messages(question)

    for step in range(1, max_steps + 1):
        response = api_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools_schema,
            tool_choice="auto",
            temperature=0,
        )
        msg = response.choices[0].message

        # 模型决定直接回答（无工具调用）
        if not msg.tool_calls:
            answer = msg.content or "（模型返回空内容）"
            session.commit(question, answer)
            yield {
                "step":   step,
                "type":   "final",
                "thought": "",
                "answer": answer,
            }
            return

        # 模型请求调用工具
        messages.append(msg)

        for tool_call in msg.tool_calls:
            tool_name = tool_call.function.name
            try:
                tool_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}

            tool_fn = tools_map.get(tool_name)
            if tool_fn is None:
                observation = f"未知工具 '{tool_name}'"
            else:
                try:
                    observation = tool_fn(**tool_args)
                except TypeError as e:
                    observation = f"工具参数错误: {e}"

            step_result = {
                "step":         step,
                "type":         "action",
                "thought":      "",   # Function Calling 版 Thought 在模型内部，不可见
                "action":       tool_name,
                "action_input": tool_args,
                "observation":  str(observation),
            }
            yield step_result

            messages.append({
                "role":         "tool",
                "tool_call_id": tool_call.id,
                "content":      str(observation),
            })

    yield {
        "step":   max_steps + 1,
        "type":   "max_steps",
        "answer": f"已达最大步数 {max_steps}，未能得出最终答案",
    }


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


def run_and_print(
    question: str,
    max_steps: int = 10,
    session: ConversationSession | None = None,
    api_client: Any = None,
):
    print(f"\n{'='*60}")
    print(f"问题: {question}")
    print(f"模型: {MODEL}  实现: Function Calling")
    print('='*60)

    start = time.time()

    for step_data in run(
        question,
        max_steps=max_steps,
        session=session,
        api_client=api_client,
    ):
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


def print_history(session: ConversationSession) -> None:
    if not session.history:
        print("当前没有对话历史。")
        return

    for index, (question, answer) in enumerate(session.history, 1):
        print(f"\n[{index}] 你：{question}")
        print(f"    Agent：{answer}")


def interactive_chat(
    api_client: Any,
    max_steps: int = 10,
    history_turns: int = 6,
) -> None:
    session = ConversationSession(max_history_turns=history_turns)
    print("多轮金融 Agent 已启动。")
    print("命令：/history 查看历史，/reset 清空上下文，/exit 退出。")

    while True:
        try:
            question = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            return

        if not question:
            continue
        if question.lower() in {"/exit", "exit", "quit", "退出"}:
            print("再见。")
            return
        if question.lower() == "/reset":
            session.reset()
            print("已清空当前会话。")
            continue
        if question.lower() == "/history":
            print_history(session)
            continue

        try:
            run_and_print(
                question,
                max_steps=max_steps,
                session=session,
                api_client=api_client,
            )
        except Exception as exc:
            print(f"请求失败：{exc}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--question",
        default=None,
        help="只执行一个问题；不传时进入多轮对话",
    )
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--history_turns", type=int, default=6)
    args = parser.parse_args()

    try:
        api_client = build_client()
        if args.question:
            run_and_print(
                args.question,
                max_steps=args.max_steps,
                session=ConversationSession(args.history_turns),
                api_client=api_client,
            )
        else:
            interactive_chat(
                api_client,
                max_steps=args.max_steps,
                history_turns=args.history_turns,
            )
    except (RuntimeError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

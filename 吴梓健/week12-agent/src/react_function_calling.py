"""
Function Calling API 版 ReAct Agent（支持多轮对话）

教学重点：
  1. 与手写版对比：框架帮你处理格式解析，但 Thought 过程在模型内部不可见
  2. tool_choice="auto" 让模型自己决定调用哪个工具或直接回答
  3. finish_reason 判断：tool_calls 表示继续调用，stop 表示给出最终答案
  4. 多轮对话：用 ChatAgent 类维护跨轮对话历史，工具调用与结果自动累积为上下文
  5. 历史裁剪：超过 max_rounds 轮自动按"轮"裁剪，防止 token 爆炸且不破坏 tool_calls 配对

使用方式：
  # 多轮交互模式（默认）
  python react_function_calling.py
  python react_function_calling.py --max_steps 8 --max_rounds 6

  # 单轮模式（兼容旧脚本 / evaluate.py / serve.py）
  python react_function_calling.py --question "茅台近一年股价涨跌幅如何？"

依赖：
  pip install openai faiss-cpu sentence-transformers akshare
  export DASHSCOPE_API_KEY="sk-xxx"   # 或 DEEPSEEK_API_KEY
"""

import os
import json
import time
import logging
import argparse
from typing import Generator, List

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url= os.getenv("DASHSCOPE_BASE_URL","https://dashscope.aliyuncs.com/compatible-mode/v1"),
)
MODEL = os.getenv("AGENT_MODEL", "qwen-max")
# client = OpenAI(
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     base_url="https://api.deepseek.com",
# )
# MODEL = os.getenv("AGENT_MODEL", "deepseek-v4-flash")

FC_SYSTEM_PROMPT = """你是一个专业的A股金融分析助手。
规则：
- 调用 financial_indicator 或 stock_price 之前，必须先用 company_lookup 获取股票代码
- 数字计算必须使用 calculator 工具，不能心算
- Final Answer 必须引用具体数据来源
- 如果没有合适工具能回答，直接说明原因
- 当前是多轮对话，可以引用前文已获取的数据与结论，避免重复调用相同工具
"""


class ChatAgent:
    """
    带历史记忆的 Function Calling ReAct Agent

    - messages 在多轮之间累积，工具调用与结果自动成为后续上下文
    - 每轮 chat() 结束后按轮次裁剪历史，避免 token 爆炸
    - reset() 清空历史（保留 system prompt）
    """

    def __init__(
        self,
        max_steps: int = 10,
        max_rounds: int = 10,
        system_prompt: str = FC_SYSTEM_PROMPT,
    ):
        self.max_steps = max_steps
        self.max_rounds = max_rounds
        self.system_prompt = system_prompt
        self.messages: List[dict] = [{"role": "system", "content": system_prompt}]

    def reset(self):
        """清空对话历史，仅保留 system prompt"""
        self.messages = [{"role": "system", "content": self.system_prompt}]
        logger.info("对话历史已清空")

    def _trim_by_rounds(self):
        """
        按轮次裁剪：仅保留最近 max_rounds 轮 user 提问及其后续上下文

        以 user 消息作为轮次起点，裁剪边界天然落在"上一轮结束 → 下一轮 user 开始"，
        不会切断 assistant(tool_calls) ↔ tool 的配对，保证历史对 API 始终合法。
        """
        user_indices = [
            i for i, m in enumerate(self.messages)
            if m.get("role") == "user"
        ]
        if len(user_indices) <= self.max_rounds:
            return
        keep_from = user_indices[-self.max_rounds]
        # messages[0] 是 system prompt，必须保留
        self.messages = [self.messages[0]] + self.messages[keep_from:]
        logger.info(f"历史已裁剪至最近 {self.max_rounds} 轮")

    def chat(self, question: str, max_steps: int = None) -> Generator[dict, None, None]:
        """
        执行一轮 Function Calling ReAct 循环，yield 每一步结构化结果

        末尾自动：
          - 把最终答案 / 工具调用追加到 self.messages 作为后续上下文
          - 触发历史裁剪
        """
        from tools import TOOLS_MAP, TOOLS_SCHEMA

        steps = max_steps or self.max_steps
        self.messages.append({"role": "user", "content": question})

        for step in range(1, steps + 1):
            response = client.chat.completions.create(
                model=MODEL,
                messages=self.messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=0.3,        # 避免 temperature=0 导致的文本重复
                frequency_penalty=0.5,  # 惩罚重复 token，直接抑制"Final Answer"循环
            )
            msg    = response.choices[0].message
            reason = response.choices[0].finish_reason

            # 模型决定直接回答（无工具调用）
            if reason == "stop" or not msg.tool_calls:
                answer = msg.content or "（模型返回空内容）"
                # 把最终回答加入历史，供下一轮引用
                self.messages.append({"role": "assistant", "content": answer})
                yield {
                    "step":   step,
                    "type":   "final",
                    "thought": "",
                    "answer": answer,
                }
                self._trim_by_rounds()
                return

            # 模型请求调用工具：把整条 assistant 消息（含 tool_calls）入历史
            # 用 model_dump 转成 dict，便于后续 _trim_by_rounds 用 .get() 统一处理
            self.messages.append(msg.model_dump(exclude_none=True))

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

                step_result = {
                    "step":         step,
                    "type":         "action",
                    "thought":      "",   # Function Calling 版 Thought 在模型内部，不可见
                    "action":       tool_name,
                    "action_input": tool_args,
                    "observation":  str(observation),
                }
                yield step_result

                self.messages.append({
                    "role":         "tool",
                    "tool_call_id": tool_call.id,
                    "content":      str(observation),
                })

        # 达到最大步数仍未结束：追加一条兜底 assistant 消息，保持历史干净
        fallback = f"（已达最大步数 {steps}，本轮未能得出最终答案）"
        self.messages.append({"role": "assistant", "content": fallback})
        yield {
            "step":   steps + 1,
            "type":   "max_steps",
            "answer": f"已达最大步数 {steps}，未能得出最终答案",
        }
        self._trim_by_rounds()


# ── 向后兼容的单轮入口（serve.py / agent.py / evaluate.py 使用） ──────────────

def run(question: str, max_steps: int = 10) -> Generator[dict, None, None]:
    """
    单轮执行 Function Calling 版 ReAct 循环，yield 每一步结构化结果

    内部新建一个临时 ChatAgent，不在调用间保留历史。
    格式与 react_manual.run() 保持一致，便于 evaluate.py 统一对比。
    """
    agent = ChatAgent(max_steps=max_steps)
    yield from agent.chat(question, max_steps=max_steps)


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


def _print_step(step_data: dict, start: float):
    """打印一步的结构化结果（多轮 / 单轮共用）"""
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
        print(f"\n本轮 {step_data['step']} 步，耗时 {elapsed:.1f}s")

    elif stype in ("error", "max_steps"):
        print(_c("error", f"\n⚠️  {step_data.get('answer', '')}"))


def run_and_print(question: str, max_steps: int = 10):
    """单轮模式：执行一次并打印（兼容旧入口）"""
    print(f"\n{'='*60}")
    print(f"问题: {question}")
    print(f"模型: {MODEL}  实现: Function Calling")
    print('='*60)

    start = time.time()
    for step_data in run(question, max_steps=max_steps):
        _print_step(step_data, start)


# ── 多轮交互模式 ──────────────────────────────────────────────────────────────

HELP_TEXT = (
    "命令：\n"
    "  /clear  清空对话历史（保留 system prompt）\n"
    "  /help   显示帮助\n"
    "  exit / quit / 退出  结束对话"
)


def chat_loop(max_steps: int = 10, max_rounds: int = 10):
    """
    多轮交互式 CLI：在同一 ChatAgent 上连续提问，历史自动累积

    输入 exit/quit 退出；/clear 清空历史；/help 查看命令
    """
    agent = ChatAgent(max_steps=max_steps, max_rounds=max_rounds)

    print(f"\n{'='*60}")
    print(f"多轮对话模式  模型: {MODEL}  实现: Function Calling")
    print(f"每轮最大步数: {max_steps}   历史保留轮数: {max_rounds}")
    print(HELP_TEXT)
    print('='*60)

    while True:
        try:
            question = input("\n👤 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not question:
            continue

        cmd = question.lower()
        if cmd in ("exit", "quit", "q", "退出"):
            print("再见！")
            break
        if cmd in ("/clear", "/reset", "清空"):
            agent.reset()
            print("🧹 已清空对话历史")
            continue
        if cmd in ("/help", "帮助", "?"):
            print(HELP_TEXT)
            continue

        print("\n🤖 助手:")
        start = time.time()
        for step_data in agent.chat(question, max_steps=max_steps):
            _print_step(step_data, start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Function Calling 版 ReAct Agent（支持多轮对话）"
    )
    parser.add_argument(
        "--question",
        default=None,
        help="单轮模式：直接回答一个问题后退出；不指定则进入多轮交互模式",
    )
    parser.add_argument("--max_steps", type=int, default=10, help="每轮最大工具调用步数")
    parser.add_argument("--max_rounds", type=int, default=10, help="多轮模式中保留的历史轮数")
    args = parser.parse_args()

    if args.question:
        run_and_print(args.question, args.max_steps)
    else:
        chat_loop(args.max_steps, args.max_rounds)

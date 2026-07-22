"""
手写 Prompt 解析版 ReAct Agent（支持多轮对话）

教学重点：
  1. ReAct 核心循环：Thought → Action → Observation，逐步推理
  2. System Prompt 约束输出格式，Python 正则解析每一步
  3. 对话历史拼接方式：每轮结果追加到 prompt，形成上下文记忆
  4. 停止条件：模型输出 Final Answer 或达到最大步数
  5. 多轮对话：用 ChatAgent 类维护跨轮对话历史，Thought/Action/Observation 自动累积为上下文
  6. 历史裁剪：超过 max_rounds 轮自动按"轮"裁剪，防止 token 爆炸且不破坏上下文连续性

使用方式：
  # 多轮交互模式（默认）
  python react_manual.py
  python react_manual.py --max_steps 8 --max_rounds 6

  # 单轮模式（兼容旧脚本 / evaluate.py / serve.py）
  python react_manual.py --question "茅台和五粮液2023年毛利率差多少？"
  python react_manual.py --question "..." --max_steps 8 --verbose

依赖：
  pip install openai faiss-cpu sentence-transformers akshare
  export DASHSCOPE_API_KEY="sk-xxx"
"""

import os
import re
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

# ── LLM 客户端 ────────────────────────────────────────────────────────────────
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
)
MODEL = os.getenv("AGENT_MODEL", "qwen-max")
# client = OpenAI(
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     base_url="https://api.deepseek.com",
# )
# MODEL = os.getenv("AGENT_MODEL", "deepseek-v4-flash")

# 1. rag_search(query) - 在年报中语义检索文本内容（战略/财务数据/风险因素等）
# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """你是一个专业的A股金融分析助手，可以使用以下工具来回答问题：

工具列表：
1. company_lookup(name) - 将公司名称转换为股票代码
2. calculator(expr) - 计算数学表达式（支持四则运算和math函数）
3. financial_indicator(symbol) - 获取AkShare实时财务指标（PE/PB/ROE等），symbol为股票代码，格式为"600519"，不要带后缀
4. stock_price(symbol, start_date, end_date) - 获取历史股价，日期格式YYYYMMDD

你必须严格按照以下格式交替输出，每次只能调用一个工具：
'''
Thought: 分析当前状态，决定下一步做什么
Action: 工具名称
Action Input: {"参数名": "参数值"}
'''

收到工具结果后继续推理，直到可以给出最终答案：

Thought: 已有足够信息
Final Answer: 完整的回答（含数据来源）

规则：
- 必须先用 company_lookup 获取股票代码，再调用 financial_indicator 或 stock_price
- 数字计算必须用 calculator，不能心算
- Final Answer 必须引用具体数据来源（哪份年报哪一页，或AkShare实时数据）
- 如果没有合适工具能回答，直接输出 Final Answer 说明原因
- 当前是多轮对话，可以引用前文已获取的数据与结论，避免重复调用相同工具
"""

#贵州茅台2023年毛利率是多少？

#收到工具结果后继续推理，直到可以给出最终答案，
# 【重要！】输出最终答案必须严格按照以下格式输出：
# '''
# Thought: 已有足够信息
# Final Answer: 完整的回答（含数据来源）

#
# - Final Answer 的输出必须严格按照以下格式输出：
#   Thought: 已有足够信息
#   Final Answer: 完整的回答（含数据来源）
# ── 格式解析 ──────────────────────────────────────────────────────────────────
# 注意：冒号使用 [:：] 同时匹配英文半角冒号和中文全角冒号，
#       避免模型在多轮对话中输出全角标点导致解析失败
_THOUGHT_RE      = re.compile(r"Thought[:：]\s*(.+?)(?=\nAction[:：]|\nFinal Answer[:：]|$)", re.DOTALL)
_ACTION_RE       = re.compile(r"Action[:：]\s*(\w+)")
_ACTION_INPUT_RE = re.compile(r"Action Input[:：]\s*(\{.+?\})", re.DOTALL)
_FINAL_RE        = re.compile(r"Final Answer[:：]\s*(.+)", re.DOTALL)


def _parse_step(text: str) -> dict:
    """从 LLM 输出中解析一步的结构化内容"""
    final = _FINAL_RE.search(text)
    if final:
        thought_m = _THOUGHT_RE.search(text)
        return {
            "type":    "final",
            "thought": thought_m.group(1).strip() if thought_m else "",
            "answer":  final.group(1).strip(),
        }

    thought_m = _THOUGHT_RE.search(text)
    action_m  = _ACTION_RE.search(text)
    input_m   = _ACTION_INPUT_RE.search(text)

    if not action_m:
        return {"type": "unparseable", "raw": text}

    try:
        action_input = json.loads(input_m.group(1)) if input_m else {}
    except json.JSONDecodeError:
        action_input = {}

    return {
        "type":         "action",
        "thought":      thought_m.group(1).strip() if thought_m else "",
        "action":       action_m.group(1).strip(),
        "action_input": action_input,
    }


# ── ReAct Agent（支持多轮对话） ───────────────────────────────────────────────

class ChatAgent:
    """
    带历史记忆的手写 Prompt 解析版 ReAct Agent

    - messages 在多轮之间累积，Thought / Action / Observation 自动成为后续上下文
    - 每轮 chat() 结束后按轮次裁剪历史，避免 token 爆炸
    - reset() 清空历史（保留 system prompt）
    """

    def __init__(
        self,
        max_steps: int = 10,
        max_rounds: int = 10,
        system_prompt: str = SYSTEM_PROMPT,
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
        不会切断 assistant ↔ user(Observation) 的上下文连续性。
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
        执行一轮手写 Prompt 解析版 ReAct 循环，yield 每一步结构化结果

        末尾自动：
          - 把最终答案 / 工具调用与观测追加到 self.messages 作为后续上下文
          - 触发历史裁剪
        """
        from tools import TOOLS_MAP

        steps = max_steps or self.max_steps
        self.messages.append({"role": "user", "content": question})

        for step in range(1, steps + 1):
            response = client.chat.completions.create(
                model=MODEL,
                messages=self.messages,
                temperature=0,
                stop=["Observation:"],  # 让模型停在调用工具前
            )
            llm_output = response.choices[0].message.content.strip()
            parsed = _parse_step(llm_output)

            if parsed["type"] == "final":
                # 把最终回答加入历史，供下一轮引用
                self.messages.append({"role": "assistant", "content": llm_output})
                yield {
                    "step":    step,
                    "type":    "final",
                    "thought": parsed["thought"],
                    "answer":  parsed["answer"],
                }
                self._trim_by_rounds()
                return

            if parsed["type"] == "unparseable":
                # 格式解析失败：把错误反馈给历史，便于下一轮自我纠正
                error_msg = f"格式解析失败，原始输出：{llm_output[:200]}"
                self.messages.append({"role": "assistant", "content": llm_output})
                self.messages.append({"role": "user", "content": error_msg + "\n请严格按格式输出 Thought/Action/Action Input 或 Final Answer。"})
                yield {
                    "step":        step,
                    "type":        "error",
                    "observation": error_msg,
                }
                self._trim_by_rounds()
                continue

            # 执行工具
            tool_name  = parsed["action"]
            tool_args  = parsed["action_input"]
            tool_fn    = TOOLS_MAP.get(tool_name)

            if tool_fn is None:
                observation = f"未知工具 '{tool_name}'，可用工具：{list(TOOLS_MAP.keys())}"
            else:
                try:
                    observation = tool_fn(**tool_args)
                except TypeError as e:
                    observation = f"工具参数错误: {e}"

            step_result = {
                "step":         step,
                "type":         "action",
                "thought":      parsed["thought"],
                "action":       tool_name,
                "action_input": tool_args,
                "observation":  str(observation),
            }
            yield step_result

            # 将本步结果追加到对话历史，形成上下文记忆
            self.messages.append({"role": "assistant", "content": llm_output})
            self.messages.append({
                "role":    "user",
                "content": f"Observation: {observation}\n",
            })

        # 超出最大步数，强制终止：追加一条兜底 assistant 消息，保持历史干净
        fallback = f"（已达最大步数 {steps}，本轮未能得出最终答案）"
        self.messages.append({"role": "assistant", "content": fallback})
        yield {
            "step":   steps + 1,
            "type":   "max_steps",
            "answer": f"已达最大步数 {steps}，未能得出最终答案",
        }
        self._trim_by_rounds()


# ── 向后兼容的单轮入口（serve.py / agent.py / evaluate.py 使用） ──────────────

def run(question: str, max_steps: int = 10, verbose: bool = True) -> Generator[dict, None, None]:
    """
    单轮执行手写版 ReAct 循环，yield 每一步结构化结果

    内部新建一个临时 ChatAgent，不在调用间保留历史。
    格式与 react_function_calling.run() 保持一致，便于 evaluate.py 统一对比。
    """
    agent = ChatAgent(max_steps=max_steps)
    yield from agent.chat(question, max_steps=max_steps)


# ── CLI 打印 ──────────────────────────────────────────────────────────────────

COLORS = {
    "thought":  "\033[36m",   # cyan
    "action":   "\033[33m",   # yellow
    "obs":      "\033[32m",   # green
    "final":    "\033[35m",   # magenta
    "error":    "\033[31m",   # red
    "reset":    "\033[0m",
}

def _c(color: str, text: str) -> str:
    return f"{COLORS[color]}{text}{COLORS['reset']}"


def _print_step(step_data: dict, start: float):
    """打印一步的结构化结果（多轮 / 单轮共用）"""
    stype = step_data["type"]

    if stype == "action":
        print(f"\n[Step {step_data['step']}]")
        print(_c("thought", f"🧠 Thought: {step_data['thought']}"))
        print(_c("action",  f"🔧 Action:  {step_data['action']}"))
        print(_c("action",  f"   Input:   {json.dumps(step_data['action_input'], ensure_ascii=False)}"))
        print(_c("obs",     f"👁  Obs:     {step_data['observation'][:300]}"))

    elif stype == "final":
        elapsed = time.time() - start
        print(f"\n{'─'*60}")
        if step_data.get("thought"):
            print(_c("thought", f"🧠 Thought: {step_data['thought']}"))
        print(_c("final",  f"\n✅ Final Answer:\n{step_data['answer']}"))
        print(f"\n本轮 {step_data['step']} 步，耗时 {elapsed:.1f}s")

    elif stype in ("error", "max_steps"):
        print(_c("error", f"\n⚠️  {step_data.get('answer', step_data.get('observation', ''))}"))


def run_and_print(question: str, max_steps: int = 10):
    """单轮模式：执行一次并打印（兼容旧入口）"""
    print(f"\n{'='*60}")
    print(f"问题: {question}")
    print(f"模型: {MODEL}  实现: 手写Prompt解析")
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
    print(f"多轮对话模式  模型: {MODEL}  实现: 手写Prompt解析")
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
        description="手写 Prompt 解析版 ReAct Agent（支持多轮对话）"
    )
    parser.add_argument(
        "--question",
        default=None,
        help="单轮模式：直接回答一个问题后退出；不指定则进入多轮交互模式",
    )
    parser.add_argument("--max_steps", type=int, default=10, help="每轮最大工具调用步数")
    parser.add_argument("--max_rounds", type=int, default=10, help="多轮模式中保留的历史轮数")
    parser.add_argument("--verbose", action="store_true", help="兼容旧参数，无实际效果")
    args = parser.parse_args()

    if args.question:
        run_and_print(args.question, args.max_steps)
    else:
        chat_loop(args.max_steps, args.max_rounds)

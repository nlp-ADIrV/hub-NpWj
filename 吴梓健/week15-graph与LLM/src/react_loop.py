"""
通用 ReAct 循环引擎（API 原生 tool calling 版）

教学重点：
  1. ReAct = Reason + Act：LLM 输出 Thought 文本 + 原生 tool_calls（工具名 + JSON 参数），
     runner 执行工具，把结果作为 role="tool" 消息回灌，循环直到模型不再调工具
     （此时输出的文本即 Final Answer）。
  2. 主 agent 和 subagent 都是同一个 ReActLoop——区别只在 tools 列表：
     主 agent 有 web_search + dispatch_subagents，subagent 只有 web_search。
  3. 完整 trace 捕获：每步 Thought/Action/ActionInput/Observation 存下来，供可视化。

与「手搓 ReAct」（正则解析 Action:/Action Input: 文本 + stop=["Observation:"] 截断）不同：
  本版直接用 OpenAI 兼容 API 的 tools 参数，工具名与参数由模型结构化返回，
  没有 _parse 正则兜底、没有 stop 截断——是 API 自带的 tool_call，而非自己解析。
"""
import time, json, logging
from typing import Callable, Optional

from llm_client import llm_complete

logger = logging.getLogger(__name__)

REACT_SYSTEM = """你是新能源汽车参数调研助手，能用以下工具联网搜索调研。

可用工具：
{tools_desc}

先输出 Thought 简短说明推理，再调用合适的工具。工具执行后会得到 Observation。
重复调用直到信息足够，最后直接给出综合答案文本（无需声明 Final Answer 前缀）。

规则：
- 不要修改搜索的新能源汽车型号，按照用户输入的内容来搜索 
- 只能调用上面列出的工具
"""

#【重要！！！】

def build_tools_desc(tools: dict) -> str:
    """把 tools 字典格式化成系统提示里的工具说明。tools: {name: (fn, desc, params)}"""
    return "\n".join(f"- {name}: {desc}" for name, (fn, desc, params) in tools.items())


def build_openai_tools(tools: dict) -> list:
    """把内部工具声明转成 OpenAI 原生 function calling 的 tools 参数。"""
    return [{
        "type": "function",
        "function": {"name": name, "description": desc,
                     "parameters": _build_schema(params)},
    } for name, (fn, desc, params) in tools.items()]


def _build_schema(params: dict) -> dict:
    """由极简参数声明自动生成 JSON Schema，省掉重复手写：
    {"query": "str"} → 字符串参数；{"topics": "list[str]"} → 字符串数组。"""
    types = {"str": "string", "int": "integer", "float": "number",
             "bool": "boolean", "list[str]": "array"}
    props = {}
    for name, spec in params.items():
        props[name] = ({"type": "array", "items": {"type": "string"}}
                       if spec == "list[str]" else {"type": types.get(spec, "string")})
    return {"type": "object", "properties": props, "required": list(params)}


class ReActLoop:
    """通用 ReAct 循环（原生 tool calling 版）。主 agent / subagent 各实例化一个。"""

    def __init__(self, agent_name: str, tools: dict,
                 max_steps: int = 6, model_tag: str = "deepseek-chat",
                 system_prompt: Optional[str] = None):
        """
        tools: {tool_name: (fn(**args) -> str, description_str, params_spec)}
               params_spec 如 {"query": "str"} / {"topics": "list[str]"}，自动生成 schema。
               工具函数用普通具名参数书写（如 def web_search(query)），不用 **_ 吞参。
        system_prompt: 自定义系统提示（主 agent 用 MAIN_SYSTEM 引导派发）。
                       None 时用默认 REACT_SYSTEM。{tools_desc} 占位符会被替换。
        """
        self.agent_name = agent_name
        self.tools = tools
        self.max_steps = max_steps
        self.model_tag = model_tag
        self._system_template = system_prompt or REACT_SYSTEM
        self._openai_tools = build_openai_tools(tools)
        self.trace: list[dict] = []

    def run(self, question: str, on_step: Callable = None,
            shared_state: dict = None) -> dict:
        """
        执行 ReAct 循环。
        on_step(step_dict): 每步回调（SSE 流式用）。
        shared_state: 共享状态 dict（dispatch_subagents 往里塞 subagent trace）。
        返回 {final_answer, trace, duration}。
        """
        self.trace = []
        t0 = time.time()
        messages = [
            {"role": "system",
             "content": self._system_template.format(tools_desc=build_tools_desc(self.tools))},
            {"role": "user", "content": question},
        ]
        final_answer = ""
        step_idx = 0

        while step_idx < self.max_steps:
            msg = llm_complete(messages, tools=self._openai_tools, temperature=0.0)

            # ── 有原生 tool_call：逐条执行工具 → observation 回灌 ──
            if msg.tool_calls:
                messages.append({
                    "role": "assistant", "content": msg.content,
                    "tool_calls": [{
                        "id": c.id, "type": "function",
                        "function": {"name": c.function.name,
                                     "arguments": c.function.arguments},
                    } for c in msg.tool_calls],
                })
                for call in msg.tool_calls:
                    name = call.function.name
                    args = json.loads(call.function.arguments or "{}")
                    step = {"idx": step_idx, "agent": self.agent_name,
                            "thought": (msg.content or "").strip()[:400],
                            "action": name,
                            "action_input": json.dumps(args, ensure_ascii=False),
                            "observation": None, "final": False}
                    if on_step:
                        on_step(step)          # pre：先展示决策，工具执行后再更新

                    observation = self._exec_tool(name, args, shared_state)

                    step["observation"] = observation   # post：原地补 observation
                    step["done"] = True
                    self.trace.append(step)
                    if on_step:
                        on_step(step)
                    messages.append({"role": "tool", "tool_call_id": call.id,
                                     "content": observation[:1200]})
                    step_idx += 1
            else:
                # ── 无工具调用：本轮文本即 Final Answer ──
                final_answer = msg.content or "（模型未返回内容）"
                step = {"idx": step_idx, "agent": self.agent_name,
                        "thought": "信息已收集，综合输出", "action": "Final Answer",
                        "action_input": final_answer, "observation": None, "final": True}
                self.trace.append(step)
                if on_step:
                    on_step(step)
                break
        else:
            # 超过 max_steps：用已收集的上下文让模型补一轮最终答案，保证底部报告不缺失
            try:
                final_msg = llm_complete(messages, tools=None, temperature=0.0)
                final_answer = (final_msg.content or "").strip()
            except Exception:
                final_answer = ""
            if not final_answer:
                final_answer = "（已达最大步数，部分结果可能缺失）"
            step = {"idx": step_idx, "agent": self.agent_name,
                    "thought": "达到步数上限", "action": "Final Answer",
                    "action_input": final_answer, "observation": None, "final": True}
            self.trace.append(step)
            if on_step:
                on_step(step)

        return {"final_answer": final_answer, "trace": self.trace,
                "duration": round(time.time() - t0, 2)}

    def _exec_tool(self, name: str, args: dict, shared_state: dict) -> str:
        """执行工具：fn(**args) 直接解包模型返回的 JSON 参数（普通具名参数写法），
        工具若要访问共享状态，在声明处用闭包绑定即可，不需要 **_ 吞参技巧。"""
        if name not in self.tools:
            return f"工具 '{name}' 不存在，可选: {list(self.tools)}"
        fn, _, _ = self.tools[name]
        try:
            return str(fn(**args))
        except Exception as e:
            return f"工具执行出错: {type(e).__name__}: {str(e)[:120]}"


# ── 自测：单工具 ReAct 跑通 ──────────────────────────────────────────────────
if __name__ == "__main__":
    import logging as _l
    _l.basicConfig(level=_l.WARNING)
    from tavily_search import tavily_search, format_search_result

    def web_search(query: str):
        return format_search_result(tavily_search(query))

    loop = ReActLoop("test", tools={"web_search": (web_search, "联网搜索，参数是查询词", {"query": "str"})},
                     max_steps=4)
    r = loop.run("2025年比亚迪汉EV的续航和价格是多少？")
    print(f"\n答案: {r['final_answer'][:120]}")
    print(f"trace {len(r['trace'])} 步:")
    for s in r["trace"]:
        print(f"  [{s['idx']}] {s['action']}({s['action_input'][:40]}) → {(s.get('observation') or '')[:50]}")

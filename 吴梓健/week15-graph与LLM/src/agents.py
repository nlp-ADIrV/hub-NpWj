"""
主 Agent + 并行 Subagent 编排（API 原生 tool calling 版）

教学重点：
  1. 主 agent 自己是 ReAct 循环，有 2 个工具：
     - web_search：单次联网搜索（单一事实问题直接用）
     - dispatch_subagents：派发多个 subagent 并行调研（多车型参数对比问题用）
     主 agent 根据 query 自主决定用哪个——不是固定拓扑，是 LLM 自主路由。
  2. 工具传参走 API 原生 tool_call：模型直接返回结构化 JSON 参数
     （dispatch_subagents 的 topics 是字符串数组），不再用 "课题1|课题2"
     管道字符串 + split 解析，也不再需要 lambda q, **_: 这种吞参写法。
  3. 并行优势凸显：ThreadPoolExecutor 并行跑 N 个 subagent，wall-clock ≈ max
     而非 sum。每个 subagent 也是 ReAct 循环（只有 web_search），
     trace 全程捕获存入 shared_state，供可视化「点节点看 ReAct 过程」。

架构对应 PPT Part 6.3 的 Orchestrator-Workers 拓扑（动态：主 agent 决定派几个）。
"""

import os, time, json, logging, uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from react_loop import ReActLoop
from tavily_search import tavily_search, format_search_result

logger = logging.getLogger(__name__)

MAIN_SYSTEM = """你是新能源汽车调研主分析师，负责生成「多车型参数对比调研报告」。你有 2 个工具：
- web_search：联网搜索一次（参数=查询词）。仅用于单一事实可一次答出的问题
- dispatch_subagents：派发多个子调研员并行调研（参数=子课题字符串数组，
  如 ["比亚迪汉EV 价格续航动力参数", "特斯拉Model 3 价格续航动力参数"]）

【关键决策原则】
- 只要问题涉及 2 个及以上侧面（如「参数对比」「对比调研」「市场调研」「竞品分析」等），
  必须用 dispatch_subagents 把各侧面拆给子调研员并行处理，不要自己串行 web_search 多次。
  示例："对比比亚迪汉EV与特斯拉Model 3的价格、续航、动力"
  → Action: dispatch_subagents
    Action Input: {{"topics": ["比亚迪汉EV 价格续航动力参数", "特斯拉Model 3 价格续航动力参数"]}}
- 调研的新能源汽车型号不要擅自改动，必须按用户输入的车型名称去搜索
- 只有单一事实问题（如"2024年比亚迪销量"）才直接 web_search
- 拿到子调研结果后，综合成结构化对比报告

报告要求：按车型/维度组织参数对比表格，每个要点带来源，末尾给结论与不确定性说明。

【示例】
Question: 新能源汽车对比调研：比亚迪汉EV、特斯拉Model 3、蔚来ET5
Thought: 多车型多维度对比，必须派发子调研员并行收集各车型参数，不能自己串行搜索
Action: dispatch_subagents
Action Input: {{"topics": ["比亚迪汉EV 价格续航动力参数", "特斯拉Model 3 价格续航动力参数", "蔚来ET5 价格续航动力参数"]}}
Observation: 并行调研完成：3 个子调研员...（各子课题结果）
Thought: 已收齐三款车型的并行调研结果，整理成参数对比表
Final Answer: （分车型参数对比报告）"""

#【重要！！！】

def web_search(query: str) -> str:
    """web_search 工具实现：普通具名参数函数（不用 lambda q, **_: 吞参写法）。"""
    return format_search_result(tavily_search(query))


def _dispatch_subagents(topics: list, shared_state: dict = None,
                        on_subagent_step: Callable = None,
                        on_subagent_done: Callable = None,
                        on_dispatch: Callable = None,
                        serial: bool = False) -> str:
    """dispatch_subagents 工具实现。
    topics: 子课题字符串数组（API 原生 tool_call 直接给数组，不用 | 分隔文本解析）。
    派发 N 个 subagent 并行（ThreadPoolExecutor），收齐返回汇总文本。
    serial=True 时改成串行执行（eval A/B 对比用，凸显并行加速）。
    并行优势量化：wall_clock vs sum_durations。
    ⚠️ 用真实 subagent id 发 dispatch 事件（与 subagent_step 事件的 id 一致），
       否则前端拓扑节点和步骤对不上。"""
    subtopics = [t.strip() for t in (topics or []) if t and t.strip()][:6]
    if not subtopics:
        return "未解析出子课题"
    shared_state = shared_state if shared_state is not None else {}
    shared_state.setdefault("subagents", {})

    # 构造 {sid: (subagent, topic)}
    defs = {}
    for topic in subtopics:
        sid = f"sub_{uuid.uuid4().hex[:6]}"
        sub = ReActLoop(
            agent_name=sid,
            tools={"web_search": (web_search, "联网搜索，参数是查询词", {"query": "str"})},
            max_steps=8, model_tag="deepseek-chat(子)")
        defs[sid] = (sub, topic)

    # 记录派发（拓扑可视化用：主→N 个子节点）—— 用真实 subagent id
    dispatch_info = {"subtopics": subtopics,
                     "subagent_ids": list(defs)}
    shared_state.setdefault("dispatches", []).append(dispatch_info)
    if on_dispatch:
        on_dispatch(dispatch_info)   # 真实 id，前端加的节点和后续 subagent_step 对得上

    t0 = time.time()
    results = {}

    def _run_one(item):
        sid, (sub, topic) = item
        return sid, topic, sub.run(topic, on_step=(
            lambda step, sid=sid: on_subagent_step(sid, step) if on_subagent_step else None))

    def _collect(sid, res, topic):
        results[sid] = (topic, res)
        shared_state["subagents"][sid] = {
            "subtopic": topic, "trace": res["trace"],
            "duration": res["duration"], "final_answer": res["final_answer"]}
        if on_subagent_done:
            on_subagent_done(sid, res["duration"], topic)

    if serial:
        # 串行：一个接一个，凸显并行的意义（eval A/B 对比基线）
        for sid, item in defs.items():
            sid, topic, res = _run_one((sid, item))
            _collect(sid, res, topic)
    else:
        # 并行（凸显 subagent 并行优势的核心）
        with ThreadPoolExecutor(max_workers=len(defs)) as pool:
            futs = {pool.submit(_run_one, (sid, item)): sid for sid, item in defs.items()}
            for fut in as_completed(futs):
                sid, topic, res = fut.result()
                _collect(sid, res, topic)

    wall = round(time.time() - t0, 2)
    serial_sum = round(sum(r["duration"] for _, r in results.values()), 2)
    shared_state.setdefault("parallel_stats", []).append({
        "n_subagents": len(defs), "wall_clock": wall, "serial_sum": serial_sum,
        "speedup": round(serial_sum / wall, 2) if wall else 0})

    # 汇总文本
    parts = [f"【子课题: {topic}】(用时{r['duration']}s)\n{r['final_answer']}"
             for sid, (topic, r) in results.items()]
    stats = shared_state["parallel_stats"][-1]
    return (f"并行调研完成：{len(defs)} 个子调研员，wall-clock {wall}s "
            f"(串行需 {serial_sum}s，加速 {stats['speedup']}×)\n\n" + "\n\n".join(parts))


def run_research(question: str, on_main_step: Callable = None,
                 on_subagent_step: Callable = None,
                 on_subagent_done: Callable = None,
                 on_dispatch: Callable = None,
                 serial: bool = False) -> dict:
    """执行一次新能源汽车对比调研。返回 {final_answer, main_trace, subagents, parallel_stats}。
    serial=True 时 subagent 串行执行（eval A/B 对比基线）。"""
    shared_state = {"subagents": {}, "dispatches": [], "parallel_stats": []}

    def dispatch_subagents(topics: list):
        # 闭包绑定 shared_state/回调——工具函数是普通具名参数，
        # 不需要 original 里 fn(action_input, shared_state=...) 的 kwargs 分发写法
        return _dispatch_subagents(topics, shared_state=shared_state,
                                   on_subagent_step=on_subagent_step,
                                   on_subagent_done=on_subagent_done,
                                   on_dispatch=on_dispatch,
                                   serial=serial)

    main = ReActLoop(
        agent_name="main",
        tools={
            "web_search": (web_search, "联网搜索一次，参数=查询词", {"query": "str"}),
            "dispatch_subagents": (dispatch_subagents,
                                   "派发多个子调研员并行调研，参数=子课题字符串数组",
                                   {"topics": "list[str]"}),
        },
        max_steps=8,
        model_tag="deepseek-chat(主)",
        system_prompt=MAIN_SYSTEM,   # ← 传主 agent 的派发引导 prompt
    )
    result = main.run(question, on_step=on_main_step, shared_state=shared_state)
    return {
        "final_answer": result["final_answer"],
        "main_trace": result["trace"],
        "subagents": shared_state["subagents"],
        "parallel_stats": shared_state["parallel_stats"],
        "dispatches": shared_state["dispatches"],
    }


if __name__ == "__main__":
    import logging as _l
    _l.basicConfig(level=_l.WARNING)
    q = "新能源汽车对比调研：比亚迪汉EV、特斯拉Model 3、蔚来ET5 的价格、续航、动力参数对比"
    r = run_research(q)
    print(f"\n{'='*60}\n主 agent 动作: {[s['action'] for s in r['main_trace']]}")
    print(f"派发次数: {len(r['dispatches'])} | subagent 数: {len(r['subagents'])}")
    print(f"并行统计: {r['parallel_stats']}")
    print(f"\n报告头:\n{r['final_answer'][:200]}")

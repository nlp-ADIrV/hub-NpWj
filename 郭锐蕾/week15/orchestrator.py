"""主 Agent + 并行 Subagent 编排

核心范式：动态 Orchestrator-Workers（PPT 6.3）
  主 agent 自己是 ReAct 循环，有 2 个工具：
    - web_search：单次联网搜索（简单事实问题用）
    - dispatch_subagents：派发多个 subagent 并行调研（多侧面研究问题用）
  主 agent 根据 query 自行决定用哪个——不是固定拓扑，是 LLM 自主路由。

并行优势凸显：
  dispatch_subagents 一次派发 N 个 subagent，ThreadPoolExecutor 并行跑，
  wall-clock ≈ max(单 agent 时长)，而非 sum——这是 subagent 并行的核心价值。
  serial=True 模式退化为 for 循环（eval A/B 对比基线）。

每个 subagent 也是 ReAct 循环（只 web_search 工具），trace 全程捕获存入
shared_state，供 CLI 实时流式输出 / 可视化用。
"""
import time
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from react_engine import ReActEngine
from tools import web_search

logger = logging.getLogger(__name__)

# 主 agent 系统提示：引导自主路由 + worked example
# 关键：光说"必须 dispatch"无效，ReAct 需 worked example 教格式
MAIN_SYSTEM = """你是市场调研主分析师。你有 2 个工具：
- web_search：联网搜索一次（参数=查询词）。仅用于单一事实可一次答出的问题
- dispatch_subagents：派发多个子调研员并行调研（参数=用 | 分隔的多个子课题）

【关键决策原则】
- 只要问题涉及 2 个及以上侧面（如「市场调研」「竞品分析」「行业分析」「XX 概况/现状/趋势」等），
  必须用 dispatch_subagents 把各侧面拆给子调研员并行处理，不要自己串行 web_search 多次。
  示例："新能源汽车市场调研：销量、竞争、政策" → Action: dispatch_subagents
        Action Input: 2024年中国新能源汽车销量规模 | 主要厂商竞争格局 | 政策与补贴趋势
- 只有单一事实问题（如"2024年比亚迪销量"）才直接 web_search
- 拿到子调研结果后，综合成结构化报告

报告要求：分维度组织，每个要点带来源，末尾给结论与不确定性说明。

【示例】
Question: 2023中国咖啡市场调研：市场规模、主要品牌、消费趋势
Thought: 这是多维度市场调研（3个侧面），必须派发子调研员并行收集，不能自己串行搜索
Action: dispatch_subagents
Action Input: 2023年中国咖啡市场规模与增长 | 中国咖啡主要品牌竞争格局 | 中国咖啡消费趋势与人群
Observation: 并行调研完成：3 个子调研员...（各子课题结果）
Thought: 已收齐三个维度的并行调研结果，综合成报告
Final Answer: （分维度报告）"""


def dispatch_subagents(action_input: str, shared_state: dict = None,
                       on_subagent_step: Callable = None,
                       on_subagent_done: Callable = None,
                       on_dispatch: Callable = None,
                       serial: bool = False) -> str:
    """dispatch_subagents 工具实现。
    action_input: "子课题1 | 子课题2 | ..."（管道分隔）
    派发 N 个 subagent 并行（ThreadPoolExecutor），收齐返回汇总文本。
    serial=True 时改成串行执行（eval A/B 对比用，凸显并行加速）。
    并行优势量化：wall_clock vs serial_sum。
    ⚠️ 用真实 subagent id 发 dispatch 事件（与 subagent_step 事件的 id 一致），
       否则前端拓扑节点和步骤对不上。
    """
    subtopics = [s.strip() for s in action_input.split("|") if s.strip()][:6]
    if not subtopics:
        return "未解析出子课题"

    shared_state = shared_state if shared_state is not None else {}
    shared_state.setdefault("subagents", {})

    # 构造 (sid, engine, subtopic) 三元组
    defs = []
    for topic in subtopics:
        sid = f"sub_{uuid.uuid4().hex[:6]}"
        engine = ReActEngine(
            agent_name=sid,
            tools={"web_search": (web_search, "联网搜索，参数是查询词")},
            max_steps=4,
            model_tag="deepseek-chat(子)",
        )
        defs.append((sid, engine, topic))

    # 记录派发（用真实 subagent id，和后续 subagent_step 事件 id 一致）
    dispatch_info = {"subtopics": subtopics,
                     "subagent_ids": [sid for sid, _, _ in defs]}
    shared_state.setdefault("dispatches", []).append(dispatch_info)
    if on_dispatch:
        on_dispatch(dispatch_info)

    t0 = time.time()
    results = {}

    def _run_one(sid, engine, topic):
        """跑一个 subagent，每步回调透传 sid。"""
        def _on_step(step, _sid=sid):
            if on_subagent_step:
                on_subagent_step(_sid, step)
        return sid, engine.run(topic, on_step=_on_step)

    if serial:
        # 串行：一个接一个，凸显并行的意义（eval A/B 对比基线）
        for sid, engine, topic in defs:
            sid, res = _run_one(sid, engine, topic)
            results[sid] = (topic, res)
            shared_state["subagents"][sid] = {
                "subtopic": topic, "trace": res["trace"],
                "duration": res["duration"], "final_answer": res["final_answer"]}
            if on_subagent_done:
                on_subagent_done(sid, res["duration"], topic)
    else:
        # 并行（凸显 subagent 并行优势的核心）
        with ThreadPoolExecutor(max_workers=len(defs)) as pool:
            futs = {pool.submit(_run_one, sid, eng, top): sid for sid, eng, top in defs}
            for fut in as_completed(futs):
                sid, res = fut.result()
                topic = next(t for s, _, t in defs if s == sid)
                results[sid] = (topic, res)
                shared_state["subagents"][sid] = {
                    "subtopic": topic, "trace": res["trace"],
                    "duration": res["duration"], "final_answer": res["final_answer"]}
                if on_subagent_done:
                    on_subagent_done(sid, res["duration"], topic)

    wall = round(time.time() - t0, 2)
    serial_sum = round(sum(r["duration"] for _, r in results.values()), 2)
    stats = {"n_subagents": len(defs), "wall_clock": wall,
             "serial_sum": serial_sum,
             "speedup": round(serial_sum / wall, 2) if wall else 0}
    shared_state.setdefault("parallel_stats", []).append(stats)

    # 汇总文本（喂回主 agent 当 Observation，每个子结果截短避免 context 过长）
    parts = [f"【子课题: {topic}】(用时{r['duration']}s)\n{r['final_answer'][:500]}"
             for sid, (topic, r) in results.items()]
    return (f"并行调研完成：{len(defs)} 个子调研员，wall-clock {wall}s "
            f"(串行需 {serial_sum}s，加速 {stats['speedup']}×)\n\n"
            + "\n\n".join(parts))


def run_research(question: str, on_main_step: Callable = None,
                 on_subagent_step: Callable = None,
                 on_subagent_done: Callable = None,
                 on_dispatch: Callable = None,
                 serial: bool = False) -> dict:
    """执行一次市场调研。
    返回 {final_answer, main_trace, subagents, parallel_stats, dispatches}。
    serial=True 时 subagent 串行执行（eval A/B 对比基线）。
    """
    shared_state = {"subagents": {}, "dispatches": [], "parallel_stats": []}

    def dispatch_tool(action_input, shared_state=None):
        info = shared_state or {}
        return dispatch_subagents(
            action_input, shared_state=info,
            on_subagent_step=on_subagent_step,
            on_subagent_done=on_subagent_done,
            on_dispatch=on_dispatch,
            serial=serial)

    main = ReActEngine(
        agent_name="main",
        tools={
            "web_search": (web_search, "联网搜索一次，参数=查询词"),
            "dispatch_subagents": (dispatch_tool,
                                   "派发多个子调研员并行调研，参数=用 | 分隔的多个子课题"),
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
    logging.basicConfig(level=logging.WARNING)
    q = "2024年中国新能源汽车市场调研：销量规模、主要厂商竞争格局、政策趋势"
    r = run_research(q)
    print(f"\n{'='*60}\n主 agent 动作: {[s['action'] for s in r['main_trace']]}")
    print(f"派发次数: {len(r['dispatches'])} | subagent 数: {len(r['subagents'])}")
    print(f"并行统计: {r['parallel_stats']}")
    print(f"\n报告头:\n{r['final_answer'][:200]}")

"""
多任务并行助手 — 主 agent 下发 subagent 并行完成多项工作

场景（非市场调研）：
  - 技术/multi-topic 学习：「并行整理 Transformer / RAG / MCP 协议要点」
  - 方案对比：「对比 Rust / Go / Zig 在系统编程上的优劣」
  - 筹备类：「并行查：会议场地选项 | 餐饮方案 | 互动环节创意」

架构（Orchestrator-Workers，动态拓扑）：
  主 agent ReAct
    ├─ 单一简单问题 → web_search / 直接回答
    └─ 多个可并行子任务 → dispatch_workers("任务1 | 任务2 | ...")
                              ↓ ThreadPoolExecutor
                         worker1..N 各自 ReAct(web_search)
                              ↓ 汇总
                         主 agent 综合 → Final Answer

与市场调研 agents.py 完全独立，复用 react_loop / llm_client / tavily_search。
"""

import time
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from react_loop import ReActLoop
from tavily_search import tavily_search, format_search_result

logger = logging.getLogger(__name__)

MAIN_SYSTEM = """你是多任务并行调度主 agent。用户请求常含多个可独立并行处理的子目标。
可用工具：
- web_search：联网搜索一次（参数=查询词）。仅用于单一事实、一次搜索即可答出的问题
- dispatch_workers：派发多个 worker 并行处理（参数=用 | 分隔的多个子任务描述）

【决策原则】
- 只要请求含 2 个及以上可独立并行的子目标（如「分别查/对比/整理 A、B、C」「多主题」「多方面」），
  必须用 dispatch_workers，不要自己串行 web_search 多次。
- 只有单一简单问题才直接 web_search 或 Final Answer。
- 收齐各 worker 结果后，按任务类型综合输出：学习笔记→分节要点；对比→表格/优劣；筹备→清单。

【示例 1 — 多主题学习】
Question: 帮我并行整理：Transformer 核心原理 | RAG 工程实践 | MCP 协议是什么
Thought: 3 个独立专题，应并行派发 worker
Action: dispatch_workers
Action Input: Transformer 注意力机制与架构要点 | RAG 检索增强生成工程实践与常见坑 | MCP 协议定义与应用场景
Observation: 并行完成：3 个 worker...
Thought: 三节素材已齐，合并为学习大纲
Final Answer: （分三节结构化笔记）

【示例 2 — 技术对比】
Question: 对比 Rust、Go、Zig 在系统编程上的适用场景
Action: dispatch_workers
Action Input: Rust 系统编程优势与典型场景 | Go 系统编程优势与典型场景 | Zig 系统编程优势与典型场景

【示例 3 — 单事实，不派发】
Question: Python 3.12 什么时候发布的？
Action: web_search
Action Input: Python 3.12 release date"""

SUB_SYSTEM = """你是专题 worker，只负责完成分配给你的一个子任务。

可用工具：
{tools_desc}

按 ReAct 格式输出：
Thought: 分析还需查什么
Action: 工具名
Action Input: 参数

多轮搜索直到信息足够，最后：
Thought: 该子任务信息已够
Final Answer: 本子任务的完整结论（分点、带关键来源）

规则：只处理当前子任务，不要讨论其他子任务。"""


def _dispatch_workers(
    action_input: str,
    shared_state: dict | None = None,
    on_worker_step: Callable | None = None,
    on_worker_done: Callable | None = None,
    on_dispatch: Callable | None = None,
    serial: bool = False,
) -> str:
    """dispatch_workers 工具：解析子任务 → 并行/串行 ReAct → 汇总。"""
    subtasks = [s.strip() for s in action_input.split("|") if s.strip()][:6]
    if not subtasks:
        return "未解析出子任务，请用 | 分隔，例如：任务A | 任务B | 任务C"

    shared_state = shared_state if shared_state is not None else {}
    shared_state.setdefault("workers", {})

    defs: list[tuple[str, ReActLoop, str]] = []
    for task in subtasks:
        wid = f"worker_{uuid.uuid4().hex[:6]}"
        worker = ReActLoop(
            agent_name=wid,
            tools={
                "web_search": (
                    lambda q, **_: format_search_result(tavily_search(q)),
                    "联网搜索，参数=查询词",
                )
            },
            max_steps=4,
            model_tag="deepseek-chat(worker)",
            system_prompt=SUB_SYSTEM,
        )
        defs.append((wid, worker, task))

    dispatch_info = {
        "subtasks": subtasks,
        "subtopics": subtasks,  # 兼容现有拓扑 UI 字段名
        "subagent_ids": [wid for wid, _, _ in defs],
        "worker_ids": [wid for wid, _, _ in defs],
    }
    shared_state.setdefault("dispatches", []).append(dispatch_info)
    if on_dispatch:
        on_dispatch(dispatch_info)

    t0 = time.time()
    results: dict[str, tuple[str, dict]] = {}

    def _run_one(wid: str, worker: ReActLoop, task: str):
        return wid, worker.run(
            task,
            on_step=(lambda step, wid=wid: on_worker_step(wid, step) if on_worker_step else None),
        )

    def _record(wid: str, task: str, res: dict):
        results[wid] = (task, res)
        shared_state["workers"][wid] = {
            "subtask": task,
            "subtopic": task,
            "trace": res["trace"],
            "duration": res["duration"],
            "final_answer": res["final_answer"],
        }
        if on_worker_done:
            on_worker_done(wid, res["duration"], task)

    if serial:
        for wid, worker, task in defs:
            wid, res = _run_one(wid, worker, task)
            _record(wid, task, res)
    else:
        with ThreadPoolExecutor(max_workers=len(defs)) as pool:
            futs = {pool.submit(_run_one, wid, worker, task): wid for wid, worker, task in defs}
            for fut in as_completed(futs):
                wid, res = fut.result()
                task = next(t for w, _, t in defs if w == wid)
                _record(wid, task, res)

    wall = round(time.time() - t0, 2)
    serial_sum = round(sum(r["duration"] for _, r in results.values()), 2)
    stats = {
        "n_workers": len(defs),
        "n_subagents": len(defs),
        "wall_clock": wall,
        "serial_sum": serial_sum,
        "speedup": round(serial_sum / wall, 2) if wall else 0,
    }
    shared_state.setdefault("parallel_stats", []).append(stats)

    parts = [
        f"【{task}】(用时{r['duration']}s)\n{r['final_answer'][:600]}"
        for _, (task, r) in results.items()
    ]
    return (
        f"并行完成：{len(defs)} 个 worker，wall-clock {wall}s "
        f"(串行需 {serial_sum}s，加速 {stats['speedup']}×)\n\n" + "\n\n".join(parts)
    )


def run_parallel_task(
    question: str,
    on_main_step: Callable | None = None,
    on_worker_step: Callable | None = None,
    on_worker_done: Callable | None = None,
    on_dispatch: Callable | None = None,
    serial: bool = False,
) -> dict:
    """
    执行一次多任务并行请求。

    返回: {final_answer, main_trace, workers, parallel_stats, dispatches}
    """
    shared_state: dict = {"workers": {}, "dispatches": [], "parallel_stats": []}

    def dispatch_tool(action_input: str, shared_state: dict | None = None):
        state = shared_state or {}
        return _dispatch_workers(
            action_input,
            shared_state=state,
            on_worker_step=on_worker_step,
            on_worker_done=on_worker_done,
            on_dispatch=on_dispatch,
            serial=serial,
        )

    main = ReActLoop(
        agent_name="main",
        tools={
            "web_search": (
                lambda q, **_: format_search_result(tavily_search(q)),
                "联网搜索一次，参数=查询词",
            ),
            "dispatch_workers": (
                dispatch_tool,
                "派发多个 worker 并行处理，参数=用 | 分隔的多个子任务",
            ),
        },
        max_steps=8,
        model_tag="deepseek-chat(主)",
        system_prompt=MAIN_SYSTEM,
    )
    result = main.run(question, on_step=on_main_step, shared_state=shared_state)
    return {
        "final_answer": result["final_answer"],
        "main_trace": result["trace"],
        "workers": shared_state["workers"],
        "subagents": shared_state["workers"],  # 兼容可视化回调命名
        "parallel_stats": shared_state["parallel_stats"],
        "dispatches": shared_state["dispatches"],
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    q = "帮我并行整理：Transformer 核心原理 | RAG 工程实践要点 | MCP 协议是什么"
    r = run_parallel_task(q)
    print(f"\n{'=' * 60}")
    print(f"主 agent 动作: {[s['action'] for s in r['main_trace']]}")
    print(f"worker 数: {len(r['workers'])}")
    print(f"并行统计: {r['parallel_stats']}")
    print(f"\n汇总:\n{r['final_answer'][:400]}...")

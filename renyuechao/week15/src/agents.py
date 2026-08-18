"""最小的一层 Orchestrator-Workers 实现。

主 Agent 负责决定是否派发；``dispatch_subagents`` 负责把最多四个互相独立的
任务交给预注册 profile，并行执行后按计划顺序汇总。这里刻意不引入 DAG、数据库
或递归派发，便于在 week16 课程中观察完整链路。
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Optional

from .react_loop import ReActLoop
from .tavily_search import format_search_result, tavily_search


MAX_SUBAGENTS = 4
MAX_WORKERS = 4
MAX_RESULT_CHARS = 4000


MAIN_SYSTEM = """你是一个负责协调任务的主 Agent。
可用工具：
{tools_desc}

简单问题可以直接回答。遇到多个彼此独立的工作时，使用 dispatch_subagents 并行下发，
Action Input 必须是单行格式：profile::task | profile::task。
目前只允许 profile=researcher 或 profile=analyst，最多四项；不要让子 Agent 再派发。
拿到派发结果后，必须检查成功和失败项，再给出包含可用结果与缺失项的最终答案。

每轮严格输出 Thought、Action、Action Input；完成后输出 Final Answer。

示例：
Question: 分别调研产品 A 的功能和价格，并分析它的主要风险
Thought: 功能价格调研与风险分析彼此独立，适合交给两个不同 profile 并行完成
Action: dispatch_subagents
Action Input: researcher::调研产品 A 的功能与官方价格 | analyst::分析产品 A 的主要风险
Observation: 派发完成……
Thought: 已拿到两项结果，可以合并成功项并说明失败项
Final Answer: 综合结果……
"""


# profile 是轻量注册表，不是新的 Agent 框架；两个 profile 仍然复用同一个 ReActLoop。
PROFILE_CONFIG: dict[str, dict[str, Any]] = {
    "researcher": {
        "description": "针对一个子问题进行资料检索并给出事实和来源",
        "system_prompt": """你是 researcher 子 Agent，只负责完成分配给你的一个检索任务。
优先使用 web_search，整理关键事实、来源 URL 和不确定项；不要派发新的子 Agent。

可用工具：
{tools_desc}

每轮严格输出：
Thought: 你的判断
Action: web_search
Action Input: 搜索词

信息足够后输出：
Final Answer: 带来源 URL 的结论和不确定项
""",
        "max_steps": 4,
    },
    "analyst": {
        "description": "基于搜索结果进行对比、归纳和风险分析",
        "system_prompt": """你是 analyst 子 Agent，只负责完成分配给你的一个分析任务。
必要时使用 web_search，明确区分事实、推断和风险；不要派发新的子 Agent。

可用工具：
{tools_desc}

每轮严格输出：
Thought: 你的判断
Action: web_search
Action Input: 搜索词

信息足够后输出：
Final Answer: 区分事实、推断、风险和不确定项的分析
""",
        "max_steps": 4,
    },
}
def parse_dispatch_input(
    action_input: str,
    max_tasks: int = MAX_SUBAGENTS,
) -> list[dict[str, str]]:
    """解析 ``profile::task | profile::task``，并在解析阶段执行数量上限。

    未知或缺失 profile 仍会保留为任务，交给调度器生成明确的失败项；这样一项
    配置错误不会悄悄丢失，也不会让同批合法任务无法运行。
    """

    if not isinstance(action_input, str):
        return []
    try:
        limit = max(0, min(int(max_tasks), MAX_SUBAGENTS))
    except (TypeError, ValueError):
        limit = MAX_SUBAGENTS
    if limit == 0:
        return []

    tasks: list[dict[str, str]] = []
    for raw_item in action_input.split("|"):
        item = raw_item.strip()
        if not item:
            continue
        if "::" in item:
            profile, task = item.split("::", 1)
            profile, task = profile.strip(), task.strip()
        else:
            profile, task = "", item
        tasks.append({"profile": profile, "task": task})
        if len(tasks) >= limit:
            break
    return tasks


def _search_tool(query: str, **_: Any) -> str:
    return format_search_result(tavily_search(query))


def _worker_tools() -> dict[str, tuple[Callable[..., Any], str]]:
    return {"web_search": (_search_tool, "联网搜索一个查询词并返回带 URL 的结果")}


def _build_worker(
    profile: str,
    agent_id: str,
    *,
    chat_fn: Optional[Callable[..., str]] = None,
    observation_limit: int = 8000,
) -> ReActLoop:
    config = PROFILE_CONFIG[profile]
    return ReActLoop(
        agent_name=agent_id,
        tools=_worker_tools(),
        max_steps=int(config["max_steps"]),
        model_tag=f"{profile}(subagent)",
        system_prompt=config["system_prompt"],
        chat_fn=chat_fn,
        observation_limit=observation_limit,
    )


def _safe_error(exc: BaseException) -> str:
    # 错误只保留类型和短消息，避免把异常对象或潜在凭证写入结果。
    return f"{type(exc).__name__}: {str(exc)[:200]}"


def _failed_result(profile: str, task: str, agent_id: str, error: str) -> dict[str, Any]:
    return {
        "status": "failed",
        "profile": profile,
        "task": task,
        "final_answer": "",
        "error": error,
        "duration": 0.0,
        "agent_id": agent_id,
    }


def _normalise_worker_output(raw: Any) -> tuple[str, str, Optional[str]]:
    """统一 ReAct、fake worker 返回值，结果仍保持很小。"""

    if isinstance(raw, dict):
        status = str(raw.get("status", "succeeded"))
        answer = raw.get("final_answer", raw.get("answer", ""))
        error = raw.get("error")
        if status not in {"succeeded", "failed"}:
            error = error or f"不支持的 worker status: {status}"
            status = "failed"
        return status, str(answer or ""), str(error) if error else None
    return "succeeded", str(raw or ""), None


def _run_one(
    spec: dict[str, str],
    *,
    agent_id: str,
    chat_fn: Optional[Callable[..., str]],
    worker_fn: Optional[Callable[[str, str], Any]],
    observation_limit: int,
) -> dict[str, Any]:
    profile, task = spec["profile"], spec["task"]
    started = time.perf_counter()
    try:
        if worker_fn is not None:
            raw = worker_fn(profile, task)
        else:
            runner = _build_worker(
                profile,
                agent_id,
                chat_fn=chat_fn,
                observation_limit=observation_limit,
            )
            raw = runner.run(task, shared_state={})

        status, answer, error = _normalise_worker_output(raw)
        if status == "failed" and not error:
            error = "worker 返回失败但未提供 error"
        return {
            "status": status,
            "profile": profile,
            "task": task,
            "final_answer": answer,
            "error": error,
            "duration": round(time.perf_counter() - started, 4),
            "agent_id": agent_id,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "profile": profile,
            "task": task,
            "final_answer": "",
            "error": _safe_error(exc),
            "duration": round(time.perf_counter() - started, 4),
            "agent_id": agent_id,
        }


def _dispatch_subagents_impl(
    action_input: str,
    shared_state: Optional[dict[str, Any]] = None,
    serial: bool = False,
    chat_fn: Optional[Callable[..., str]] = None,
    worker_fn: Optional[Callable[[str, str], Any]] = None,
    max_subagents: int = MAX_SUBAGENTS,
    max_workers: int = MAX_WORKERS,
    observation_limit: int = 8000,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    specs = parse_dispatch_input(action_input, max_tasks=max_subagents)
    state = shared_state if shared_state is not None else {}
    state.setdefault("subagents", {})
    state.setdefault("dispatches", [])
    state.setdefault("parallel_stats", [])

    if not specs:
        return "未解析出可派发的子任务", [], {"n_subagents": 0, "wall_clock": 0.0, "serial_sum": 0.0, "speedup": 0.0}

    # 先生成真实 ID，后续状态和 worker 结果都复用同一批 ID。
    agent_ids = [f"sub_{index + 1}_{uuid.uuid4().hex[:6]}" for index in range(len(specs))]
    dispatch_info: dict[str, Any] = {
        "tasks": specs,
        "subagent_ids": agent_ids,
    }
    state["dispatches"].append(dispatch_info)

    results: list[Optional[dict[str, Any]]] = [None] * len(specs)
    known_indices: list[int] = []
    for index, spec in enumerate(specs):
        if spec["profile"] not in PROFILE_CONFIG:
            results[index] = _failed_result(
                spec["profile"],
                spec["task"],
                agent_ids[index],
                f"未知 profile '{spec['profile']}'；支持: {', '.join(PROFILE_CONFIG)}",
            )
        else:
            known_indices.append(index)

    started = time.perf_counter()
    if serial:
        for index in known_indices:
            results[index] = _run_one(
                specs[index],
                agent_id=agent_ids[index],
                chat_fn=chat_fn,
                worker_fn=worker_fn,
                observation_limit=observation_limit,
            )
    elif known_indices:
        bounded_workers = max(1, min(int(max_workers), MAX_WORKERS, len(known_indices)))
        with ThreadPoolExecutor(max_workers=bounded_workers, thread_name_prefix="subagent") as pool:
            future_to_index = {
                pool.submit(
                    _run_one,
                    specs[index],
                    agent_id=agent_ids[index],
                    chat_fn=chat_fn,
                    worker_fn=worker_fn,
                    observation_limit=observation_limit,
                ): index
                for index in known_indices
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as exc:  # 防止 Future 层面的异常打断整批
                    results[index] = {
                        "status": "failed",
                        "profile": specs[index]["profile"],
                        "task": specs[index]["task"],
                        "final_answer": "",
                        "error": _safe_error(exc),
                        "duration": 0.0,
                        "agent_id": agent_ids[index],
                    }

    final_results = [item or _failed_result("", "", agent_ids[i], "任务未返回结果") for i, item in enumerate(results)]
    for result in final_results:
        state["subagents"][result["agent_id"]] = dict(result)
    dispatch_info["results"] = final_results

    wall_clock = round(time.perf_counter() - started, 4)
    serial_sum = round(sum(float(item["duration"]) for item in final_results), 4)
    stats = {
        "n_subagents": len(final_results),
        "wall_clock": wall_clock,
        "serial_sum": serial_sum,
        "speedup": round(serial_sum / wall_clock, 2) if wall_clock else 0.0,
    }
    state["parallel_stats"].append(stats)

    lines = [
        f"派发完成：{len(final_results)} 个子任务，wall-clock {wall_clock}s "
        f"（串行估算 {serial_sum}s，加速 {stats['speedup']}x）"
    ]
    for index, result in enumerate(final_results, 1):
        lines.append(
            f"[{index}] status={result['status']} profile={result['profile']} "
            f"task={result['task']} duration={result['duration']}s"
        )
        if result["status"] == "succeeded":
            lines.append(f"final_answer: {result['final_answer'][:MAX_RESULT_CHARS]}")
        else:
            lines.append(f"error: {result['error']}")
    return "\n".join(lines), final_results, stats


def dispatch_subagents(
    action_input: str,
    shared_state: Optional[dict[str, Any]] = None,
    serial: bool = False,
    chat_fn: Optional[Callable[..., str]] = None,
    worker_fn: Optional[Callable[[str, str], Any]] = None,
    max_subagents: int = MAX_SUBAGENTS,
    max_workers: int = MAX_WORKERS,
    observation_limit: int = 8000,
) -> str:
    """工具形式的派发入口，返回可直接作为主 Agent Observation 的文本。"""

    summary, _, _ = _dispatch_subagents_impl(
        action_input,
        shared_state,
        serial,
        chat_fn,
        worker_fn,
        max_subagents,
        max_workers,
        observation_limit,
    )
    return summary


def dispatch_subagents_details(action_input: str, **kwargs: Any) -> dict[str, Any]:
    """给离线测试/教学代码使用的结构化派发入口。"""

    summary, results, stats = _dispatch_subagents_impl(action_input, **kwargs)
    return {"summary": summary, "results": results, "parallel_stats": stats}


def run_research(
    question: str,
    serial: bool = False,
    chat_fn: Optional[Callable[..., str]] = None,
    worker_fn: Optional[Callable[[str, str], Any]] = None,
    main_observation_limit: int = 20000,
) -> dict[str, Any]:
    """运行一次主 Agent 协调流程，并返回主 trace、子任务和并行统计。"""

    run_state: dict[str, Any] = {
        "subagents": {},
        "dispatches": [],
        "parallel_stats": [],
    }

    def dispatch_tool(
        action_input: str,
        shared_state: Optional[dict[str, Any]] = None,
    ) -> str:
        return dispatch_subagents(
            action_input,
            shared_state=shared_state if shared_state is not None else run_state,
            serial=serial,
            chat_fn=chat_fn,
            worker_fn=worker_fn,
            observation_limit=main_observation_limit,
        )

    main = ReActLoop(
        agent_name="main",
        tools={
            "web_search": (_search_tool, "联网搜索一个查询词"),
            "dispatch_subagents": (
                dispatch_tool,
                "按 profile::task | profile::task 格式并行派发最多四个独立任务",
            ),
        },
        max_steps=8,
        model_tag="deepseek-chat(main)",
        system_prompt=MAIN_SYSTEM,
        chat_fn=chat_fn,
        observation_limit=main_observation_limit,
    )
    result = main.run(question, shared_state=run_state)
    return {
        "status": result.get("status", "succeeded"),
        "final_answer": result["final_answer"],
        "error": result.get("error"),
        "main_trace": result["trace"],
        "subagents": run_state["subagents"],
        "dispatches": run_state["dispatches"],
        "parallel_stats": run_state["parallel_stats"],
    }


def _offline_worker(profile: str, task: str) -> str:
    """演示 worker：不同任务等待不同时间，并故意制造一个失败。"""

    durations = {"慢任务": 0.30, "快任务": 0.10, "中任务": 0.20}
    if "失败" in task:
        raise RuntimeError("离线演示的故意失败")
    delay = next((value for key, value in durations.items() if key in task), 0.15)
    time.sleep(delay)
    return f"{profile} 已完成：{task}"


def offline_demo() -> None:
    """无网络、无密钥演示并行和失败隔离。"""

    action_input = "researcher::慢任务 | analyst::快任务 | researcher::中任务 | analyst::失败任务"
    state: dict[str, Any] = {}
    summary = dispatch_subagents(action_input, shared_state=state, worker_fn=_offline_worker)
    print("离线并行 Subagent 演示（输出按计划顺序）")
    print(summary)
    print("\n结构化结果：")
    print(json.dumps(state["dispatches"][-1]["results"], ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="week16 最小并行 Subagent 示例")
    parser.add_argument("--offline-demo", action="store_true", help="运行无需网络和密钥的演示")
    args = parser.parse_args()
    if args.offline_demo:
        offline_demo()
    else:
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover - 入口由手工演示验证
    main()

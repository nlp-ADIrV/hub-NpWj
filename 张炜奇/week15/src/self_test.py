"""

离线自测：不需要任何 API Key，用假 LLM + 假搜索验证编排逻辑

"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ── 1. 先替换 LLM 客户端 ────────────────────────────────────────────────────
import llm_client

FAKE_MAIN_DISPATCH = [
    # 主 agent 第 1 次 LLM 调用：决定派发（4 个带角色标签的子课题）
    "Thought: 需要多个侧面并行研究，派发子规划师\n"
    "Action: dispatch_subagents\n"
    "Action Input: [需求分析] 小程序目标用户与核心功能 | [技术选型] 技术栈选型 "
    "| [架构设计] 模块划分与接口清单 | [风险与测试] 风险与测试要点",
    # 主 agent 第 2 次 LLM 调用：收到汇总 Observation 后给最终规划文档
    "Thought: 已收齐并行结果，综合作答\n"
    "Final Answer: 一、需求概述：……\n二、技术选型：……\n三、架构设计：……\n四、风险与测试：……",
]

FAKE_SYNTHESIS = ("（离线兜底综合）\n一、需求概述：……\n二、技术选型：……\n"
                  "三、架构设计：……\n四、风险与测试：……\n五、里程碑建议：……")

MAIN_MODE = ["dispatch"]   # dispatch=主 agent 按脚本派发；direct=模拟"模型不听话"直接作答


def fake_chat(system, user, **kw):
    """假 LLM：按 system prompt 区分调用方。"""
    if "直接输出文档正文" in system:          # 预路由路径的综合调用
        return FAKE_SYNTHESIS
    if "规划主管" in system:                  # 主 agent（LLM 路由路径）
        n = getattr(fake_chat, "main_n", 0)
        fake_chat.main_n = n + 1
        if MAIN_MODE[0] == "direct":
            return "Thought: 这个问题我已有足够知识\nFinal Answer: [离线直接作答] （未派发）"
        return FAKE_MAIN_DISPATCH[min(n, len(FAKE_MAIN_DISPATCH) - 1)]
    time.sleep(0.3)                           # subagent：模拟一次 LLM 调用耗时
    return f"Thought: 基于角色提示与已知信息作答\nFinal Answer: [离线假结果] {user[:40]}"


llm_client.llm_chat = fake_chat

# ── 2. 再替换搜索工具 ───────────────────────────────────────────────────────
import tavily_search


def fake_search(query, max_results=5):
    return {"answer": f"离线假摘要: {query}",
            "results": [{"title": "t", "url": "u", "content": "c"}],
            "response_time": 0.01}


tavily_search.tavily_search = fake_search

# ── 3. 最后 import agents（此时才绑定上面的假函数）──────────────────────────
import agents
import react_loop


def reset():
    fake_chat.main_n = 0
    MAIN_MODE[0] = "dispatch"


PLANNING_Q = "帮我做一个校园二手交易小程序的开发规划：需求分析、技术选型、架构设计、风险与测试"
FACT_Q = "校园二手交易小程序适合用什么前端框架"   # 非规划类 → 走 LLM 自主路由路径


def check_parallel_stats(r):
    ps = r["parallel_stats"][-1]
    assert ps["n_subagents"] == 4, f"应有 4 个子规划师，实际 {ps}"
    return ps


if __name__ == "__main__":
    # 场景1：非规划问题 → LLM 自主路由派发
    reset()
    r1 = agents.run_dev_plan(FACT_Q)
    assert len(r1["dispatches"]) == 1 and len(r1["subagents"]) == 4
    check_parallel_stats(r1)
    print("✓ 场景1 LLM 自主路由：非规划问题由主 agent 自行决定派发 4 个子规划师")
    print(f"  主 agent 动作: {[s['action'] for s in r1['main_trace']]}")

    # 场景2：规划问题 → 主控预路由（模型"不听话"也保证派发）
    reset()
    MAIN_MODE[0] = "direct"
    r2 = agents.run_dev_plan(PLANNING_Q)
    ps2 = check_parallel_stats(r2)
    assert len(r2["dispatches"]) == 1 and len(r2["subagents"]) == 4
    assert r2["main_trace"][0]["action"] == "dispatch_subagents"
    assert fake_chat.main_n == 0, "预路由路径不应调用主 agent 决策"
    print("✓ 场景2 主控预路由：规划问题确定性派发 4 个子规划师（不依赖模型自觉）")
    print(f"  派发子课题: {[t[:34] for t in r2['dispatches'][0]['subtopics']]}")
    print(f"  角色匹配: {r2['dispatches'][0]['roles']}")

    # 场景3：规划问题 serial=True → 串行基线（与场景2 对比并行加速）
    reset()
    MAIN_MODE[0] = "direct"
    r3 = agents.run_dev_plan(PLANNING_Q, serial=True)
    ps3 = check_parallel_stats(r3)
    print(f"  [并行] wall_clock {ps2['wall_clock']}s | serial_sum {ps2['serial_sum']}s | 加速 {ps2['speedup']}×")
    print(f"  [串行] wall_clock {ps3['wall_clock']}s | serial_sum {ps3['serial_sum']}s")

    # 场景4：guardrail=False → 实验纯 LLM 路由（模型直接作答，不派发属预期）
    reset()
    MAIN_MODE[0] = "direct"
    r4 = agents.run_dev_plan(PLANNING_Q, guardrail=False)
    assert len(r4["dispatches"]) == 0 and len(r4["subagents"]) == 0
    print("✓ 场景4 guardrail=False：纯 LLM 自主路由（本次模型直接作答未派发）")
    print("  —— 这就是真实 API 环境里发生过的现象，正是需要规则兜底的原因")
    print("  主 agent 动作:", [s['action'] for s in r4['main_trace']])

    # 场景5：CLI 回调接法回归测试（修复过 on_subagent_step 与 _print_step 参数个数不匹配的崩溃）
    import io
    import contextlib
    reset()
    MAIN_MODE[0] = "dispatch"
    with contextlib.redirect_stdout(io.StringIO()):
        r5 = agents.run_dev_plan(
            PLANNING_Q,
            on_main_step=agents._print_step,
            on_subagent_step=lambda sid, step: agents._print_step(step),  # 与 agents.py __main__ 同款接法
            on_dispatch=lambda info: None,
            on_subagent_done=lambda sid, dur, topic: None)
    assert len(r5["subagents"]) == 4 and len(r5["main_trace"]) == 2
    print("✓ 场景5 CLI 回调接法回归测试通过（_print_step 双参报错已修复）")

    # 场景6：单个 subagent 网络故障 → 其余照常，整体不崩（并行容错）
    reset()
    MAIN_MODE[0] = "dispatch"
    _orig_chat = react_loop.llm_chat

    def fake_flaky(system, user, **kw):
        if "[技术选型]" in user:
            raise RuntimeError("模拟网络故障")
        return _orig_chat(system, user, **kw)

    react_loop.llm_chat = fake_flaky
    try:
        r6 = agents.run_dev_plan(PLANNING_Q)
    finally:
        react_loop.llm_chat = _orig_chat
    assert len(r6["subagents"]) == 4
    failed = [v["final_answer"] for v in r6["subagents"].values() if "执行失败" in v["final_answer"]]
    assert len(failed) == 1, f"应恰有 1 个 subagent 失败，实际 {len(failed)}"
    print("✓ 场景6 并行容错：单个 subagent 故障不影响其他 3 个，整体不崩")

    # 场景7：解析兜底——模型在 Action Input 后"自说自话"（第二段 Thought:）应被切掉
    loop = react_loop.ReActLoop("t", tools={"web_search": (lambda q, **_: "ok", "desc")})
    thought, action, ai = loop._parse(
        "Thought: 需要搜索\nAction: web_search\n"
        "Action Input: FastAPI 最新版本Thought: 我再想想别的")
    assert action == "web_search" and ai == "FastAPI 最新版本", (action, ai)
    _, _, fa = loop._parse("Thought: 可以作答\nFinal Answer: 结论一\nThought: 补充结论二")
    assert fa == "结论一", fa
    print("✓ 场景7 解析兜底：Action Input / Final Answer 后的第二段 Thought 尾巴被正确切掉")

    print("\n全部场景通过")

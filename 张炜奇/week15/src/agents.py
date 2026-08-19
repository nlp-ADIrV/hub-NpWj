"""

主 Agent + 并行 Subagent 编排（编程开发规划助手）

"""
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from llm_client import llm_chat
from react_loop import ReActLoop
from tavily_search import tavily_search, format_search_result

logger = logging.getLogger(__name__)

# ── 主 agent 系统提示：决策原则 + worked example（LLM 自主路由路径用）───────
MAIN_SYSTEM = """你是资深软件开发规划主管。你有 2 个工具：
- web_search：联网搜索一次（参数=查询词）。仅用于单一技术事实问题
- dispatch_subagents：派发多个子规划师并行研究（参数=用 | 分隔的多个子课题，
  每个子课题前带角色标签，如 [需求分析] xxx、[技术选型] xxx）

【关键决策原则】
- 只要需求涉及开发方案的多个侧面（需求/选型/架构/风险/测试/上线等 2 个及以上），
  必须用 dispatch_subagents 把各侧面拆给子规划师并行处理，不要自己串行 web_search。
  子课题格式示例：
    [需求分析] 校园二手交易小程序的目标用户与核心功能
    [技术选型] 前后端与数据库技术栈对比
    [架构设计] 模块划分、数据流与接口清单
    [风险与测试] 关键风险、测试用例与上线计划
- 只有单一事实问题（如"FastAPI 最新版本是多少"）才直接 web_search
- 收齐子规划结果后，综合成一份结构化开发规划文档：
  一、需求概述 → 二、技术选型 → 三、架构设计 → 四、风险与测试 → 五、里程碑建议
  每个要点尽量带来源/依据，末尾给出不确定项说明。

【禁止行为】
- 严禁第一轮就直接输出 Final Answer。多侧面规划问题必须先拿到
  dispatch_subagents 的 Observation，再综合作答（先派发、后总结，顺序不能反）。

【示例】
Question: 帮我做一个校园二手交易小程序的开发规划
Thought: 这是多侧面的开发规划（需求/选型/架构/风险），必须派发子规划师并行研究
Action: dispatch_subagents
Action Input: [需求分析] 校园二手交易小程序的目标用户与核心功能 | [技术选型] 前后端与数据库技术栈选型 | [架构设计] 模块划分与接口清单 | [风险与测试] 关键风险与测试要点
Observation: 并行规划完成：4 个子规划师...（各子课题结果）
Thought: 已收齐四个侧面的并行结果，综合成开发规划文档
Final Answer: （结构化开发规划文档）"""

# ── subagent 角色系统提示（异构：按子课题关键词匹配）────────────────────────
# (角色名, 匹配关键词, 角色提示词模板)。dispatch 时按子课题文本匹配，匹配不到用通用角色。
SUB_ROLES = [
    ("需求分析", ["需求", "用户", "功能", "场景", "痛点", "目标用户"],
     """你是「需求分析」子规划师，负责的子课题与产品需求相关。
思考框架：目标用户画像 → 核心使用场景 → 功能清单及优先级(MoSCoW) → 同类产品现状（可联网查）。
输出：条理清晰的需求分析，功能点按优先级排列，注明哪些是 MVP 必需。"""),
    ("技术选型", ["技术选型", "选型", "技术栈", "框架", "数据库", "后端", "前端", "部署"],
     """你是「技术选型」子规划师，负责的子课题与技术栈选型相关。
思考框架：候选方案对比(优劣势/学习成本/生态/性能) → 结合项目规模给出推荐 → 说明推荐理由。
重要：优先联网搜索各方案的最新版本与社区热度，不要用过时信息。"""),
    ("架构设计", ["架构", "模块", "接口", "数据流", "数据库设计", "分层"],
     """你是「架构设计」子规划师，负责的子课题与系统架构相关。
思考框架：模块划分(职责单一) → 数据流与核心接口清单 → 数据库表结构要点 → 可扩展性考量。
输出：结构化架构说明，模块/接口用列表呈现。"""),
    ("风险与测试", ["风险", "测试", "上线", "安全", "合规", "运维", "维护"],
     """你是「风险与测试」子规划师，负责的子课题与风险/测试/上线相关。
思考框架：关键技术风险与业务风险 → 测试策略(单元/集成/验收)与关键测试用例 → 上线与运维计划 → 安全与合规注意点。
输出：风险按「发生概率×影响」排序，测试用例写成可执行条目。"""),
]
GENERIC_SUB_SYSTEM = """你是软件开发规划子规划师，负责你被分配的子课题。
思考框架：先明确该子课题要回答什么 → 必要时联网搜索获取依据 → 给出结构化、可执行的结论。
输出要具体，避免空话。"""

# ReAct 输出格式说明（角色提示词后统一追加，subagent 也要按格式输出）
REACT_TRAILER = """
可用工具：
{tools_desc}

按如下格式输出（每轮一次 Thought/Action/Action Input）：
Thought: 你的推理
Action: 工具名
Action Input: 工具参数
工具执行后你会收到 Observation，多轮直到信息足够，最后：
Thought: 信息已足够
Final Answer: 你的结论

规则：
- 最多调用工具 3 次，之后必须输出 Final Answer（避免搜索上瘾耗尽步数）"""

MAX_SUBAGENTS = 6   # 上限保护：防止一次派发过多子任务撑爆成本


def pick_sub_role(topic: str) -> str:
    """按子课题关键词匹配角色名（记录在派发信息里，供打印/可视化）。"""
    for role, keywords, _ in SUB_ROLES:
        if any(kw in topic for kw in keywords):
            return role
    return "通用"


def pick_sub_system(topic: str) -> str:
    """按子课题关键词匹配角色 system prompt（异构 subagent 的关键）。"""
    for _, keywords, template in SUB_ROLES:
        if any(kw in topic for kw in keywords):
            return template + REACT_TRAILER
    return GENERIC_SUB_SYSTEM + REACT_TRAILER


# ── 主控预路由───────────────────────────────────────────────────
# 规划类问题走确定性派发，单一事实问题仍走 LLM 自主路由。
PLAN_KEYWORDS = ["规划", "方案", "设计", "开发"]
ASPECT_KEYWORDS = ["需求", "选型", "架构", "风险", "测试", "上线", "运维", "技术"]

PLAN_DEFAULT_ASPECTS = [
    ("需求分析", "目标用户与核心功能"),
    ("技术选型", "前后端与数据库技术栈选型"),
    ("架构设计", "模块划分、数据流与接口清单"),
    ("风险与测试", "关键风险、测试用例与上线计划"),
]

SYNTHESIS_SYSTEM = """你是资深软件开发规划主管。根据【子规划师并行结果】输出一份结构化开发规划文档，包含：
一、需求概述 → 二、技术选型 → 三、架构设计 → 四、风险与测试 → 五、里程碑建议
要求：每个要点尽量带来源/依据；末尾给出不确定项说明。直接输出文档正文，不要输出 Thought/Action。"""


def _is_planning_query(question: str) -> bool:
    """预路由触发条件：含规划类词 + ≥2 个侧面词才算"多侧面规划问题"。
    单一事实问题（如"FastAPI 最新版本"）不触发，保持 LLM 自主路由。"""
    return (any(kw in question for kw in PLAN_KEYWORDS)
            and sum(kw in question for kw in ASPECT_KEYWORDS) >= 2)


def _project_name(question: str) -> str:
    """从问题里摘出项目主体，去掉"的开发规划：需求分析、技术选型…"这类尾巴。"""
    for sep in ("：", ":"):
        if sep in question:
            head, tail = question.split(sep, 1)
            if any(kw in tail for kw in ASPECT_KEYWORDS):
                question = head
                break
    for kw in ("的开发规划", "的规划", "开发规划", "规划",
               "的技术方案", "技术方案", "方案", "的设计", "设计"):
        if question.endswith(kw):
            question = question[: -len(kw)]
            break
    return question.strip(" 的，。 ")


def _planning_action_input(question: str) -> str:
    """规则兜底用：项目主体 + 默认四方面，生成 4 个带角色标签的子课题。"""
    project = _project_name(question)
    return " | ".join(f"[{role}] {project}——{aspect}"
                      for role, aspect in PLAN_DEFAULT_ASPECTS)


# ── dispatch_subagents 工具实现 ──────────────────────────────────────────────

def _dispatch_subagents(action_input: str, shared_state: dict = None,
                        on_subagent_step: Callable = None,
                        on_subagent_done: Callable = None,
                        on_dispatch: Callable = None,
                        serial: bool = False) -> str:
    """dispatch_subagents 工具实现。

    action_input: "[角色] 子课题1 | [角色] 子课题2 | ..."（管道分隔）
    N 个 subagent 用 ThreadPoolExecutor 并行跑；serial=True 退化为串行（A/B 基线）。
    并行收益量化：wall_clock(实际耗时) vs serial_sum(各子时长之和)。
    """
    subtopics = [s.strip() for s in action_input.split("|") if s.strip()][:MAX_SUBAGENTS]
    if not subtopics:
        return "未解析出子课题"
    shared_state = shared_state if shared_state is not None else {}
    shared_state.setdefault("subagents", {})

    # 构造 (sid, subagent, subtopic) 三元组，每个 subagent 按角色定制 system prompt
    defs = []
    for topic in subtopics:
        sid = f"sub_{uuid.uuid4().hex[:6]}"
        sub = ReActLoop(
            agent_name=sid,
            tools={"web_search": (lambda q, **_: format_search_result(tavily_search(q)),
                                  "联网搜索一次，参数=查询词")},
            max_steps=5,   # 5 步：够 3~4 次搜索 + 1 次最终作答（4 步会被纯搜索耗尽）
            model_tag="deepseek-chat(子)",
            system_prompt=pick_sub_system(topic),
        )
        defs.append((sid, sub, topic))

    # 记录派发（用真实 subagent id，后续 subagent_step 事件才能对得上）
    dispatch_info = {"subtopics": subtopics,
                     "roles": [pick_sub_role(t) for t in subtopics],
                     "subagent_ids": [sid for sid, _, _ in defs]}
    shared_state.setdefault("dispatches", []).append(dispatch_info)
    if on_dispatch:
        on_dispatch(dispatch_info)

    t0 = time.time()
    results = {}

    def _run_one(sid=sid, sub=sub, topic=topic):
        # 单个 subagent 的异常就地兜底：网络抖动/LLM 失败不影响其他并行任务
        try:
            return sid, sub.run(topic, on_step=(
                lambda step, sid=sid: on_subagent_step(sid, step) if on_subagent_step else None))
        except Exception as e:
            logger.warning(f"subagent {sid} 执行失败: {type(e).__name__}: {str(e)[:100]}")
            return sid, {"final_answer": f"（该子课题执行失败: {type(e).__name__}: {str(e)[:100]}）",
                         "trace": [], "duration": 0.0}

    if serial:
        # 串行：一个接一个（eval A/B 的基线）
        for sid, sub, topic in defs:
            sid, res = _run_one(sid, sub, topic)
            topic = next(t for s, _, t in defs if s == sid)
            results[sid] = (topic, res)
            shared_state["subagents"][sid] = {
                "subtopic": topic, "trace": res["trace"],
                "duration": res["duration"], "final_answer": res["final_answer"]}
            if on_subagent_done:
                on_subagent_done(sid, res["duration"], topic)
    else:
        # 并行：N 个 subagent 同时跑，墙钟从 sum 压到 ≈max
        with ThreadPoolExecutor(max_workers=len(defs)) as pool:
            futs = {pool.submit(_run_one, sid, sub, topic): sid for sid, sub, topic in defs}
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
    shared_state.setdefault("parallel_stats", []).append({
        "n_subagents": len(defs), "wall_clock": wall, "serial_sum": serial_sum,
        "speedup": round(serial_sum / wall, 2) if wall else 0})

    # 汇总文本喂回主 agent（每个子结果截短，避免主 agent context 过长）
    parts = [f"【子课题: {topic}】(用时 {r['duration']}s)\n{r['final_answer'][:1000]}"
             for sid, (topic, r) in results.items()]
    stats = shared_state["parallel_stats"][-1]
    return (f"并行规划完成：{len(defs)} 个子规划师，wall-clock {wall}s "
            f"(串行需 {serial_sum}s，加速 {stats['speedup']}×)\n\n" + "\n\n".join(parts))


# ── 对外入口 ─────────────────────────────────────────────────────────────────

def run_dev_plan(question: str, on_main_step: Callable = None,
                 on_subagent_step: Callable = None,
                 on_subagent_done: Callable = None,
                 on_dispatch: Callable = None,
                 serial: bool = False,
                 guardrail: bool = True) -> dict:
    """执行一次开发规划。返回 {final_answer, main_trace, subagents, parallel_stats, dispatches}。

    serial=True 时 subagent 串行执行（eval A/B 对比基线）。
    guardrail=True 时规划类问题走主控预路由（确定性派发）；
    False 时完全交给 LLM 自主路由（实验用，模型可能不派发）。
    """
    shared_state = {"subagents": {}, "dispatches": [], "parallel_stats": []}

    def dispatch_tool(action_input, shared_state=None):
        return _dispatch_subagents(action_input, shared_state=shared_state or {},
                                   on_subagent_step=on_subagent_step,
                                   on_subagent_done=on_subagent_done,
                                   on_dispatch=on_dispatch,
                                   serial=serial)

    # ── 路径 1：规划类问题 → 主控预路由（确定性派发 + 综合）────────────────
    if guardrail and _is_planning_query(question):
        logger.warning("规划类问题：走主控预路由（确定性派发子规划师）")
        action_input = _planning_action_input(question)
        dispatch_step = {"idx": 0, "agent": "main",
                         "thought": "主控预路由：多侧面规划必须派发子规划师并行研究",
                         "action": "dispatch_subagents",
                         "action_input": action_input,
                         "observation": None, "final": False}
        if on_main_step:
            on_main_step(dispatch_step)
        observation = dispatch_tool(action_input, shared_state=shared_state)
        dispatch_step["observation"] = observation
        dispatch_step["done"] = True
        if on_main_step:
            on_main_step(dispatch_step)

        try:
            final_answer = llm_chat(
                SYNTHESIS_SYSTEM,
                f"【问题】{question}\n\n【子规划师并行结果】\n{observation}",
                temperature=0.0, max_tokens=8192).strip()   # 文档较长，8K 上限防截断
        except Exception as e:
            logger.warning(f"综合调用失败，降级返回子规划师原始结果: {type(e).__name__}")
            final_answer = (f"（综合调用失败: {type(e).__name__}: {str(e)[:100]}，"
                            f"以下为子规划师原始结果）\n\n{observation}")
        final_step = {"idx": 1, "agent": "main",
                      "thought": "综合并行结果输出开发规划文档",
                      "action": "Final Answer", "action_input": final_answer,
                      "observation": None, "final": True}
        if on_main_step:
            on_main_step(final_step)
        return {
            "final_answer": final_answer,
            "main_trace": [dispatch_step, final_step],
            "subagents": shared_state["subagents"],
            "parallel_stats": shared_state["parallel_stats"],
            "dispatches": shared_state["dispatches"],
        }

    # ── 路径 2：非规划类问题 → 主 agent ReAct 自主路由 ──────────────────────
    main = ReActLoop(
        agent_name="main",
        tools={
            "web_search": (lambda q, **_: format_search_result(tavily_search(q)),
                           "联网搜索一次，参数=查询词"),
            "dispatch_subagents": (dispatch_tool,
                                   "派发多个子规划师并行研究，参数=用 | 分隔的多个子课题（可带[角色]标签）"),
        },
        max_steps=8,
        model_tag="deepseek-chat(主)",
        system_prompt=MAIN_SYSTEM,
    )
    result = main.run(question, on_step=on_main_step, shared_state=shared_state)
    return {
        "final_answer": result["final_answer"],
        "main_trace": result["trace"],
        "subagents": shared_state["subagents"],
        "parallel_stats": shared_state["parallel_stats"],
        "dispatches": shared_state["dispatches"],
    }


# ── CLI 打印辅助 ─────────────────────────────────────────────────────────────

def _print_step(step: dict):
    """打印一步 ReAct（主 agent 与 subagent 的回调共用）。"""
    tag = f"[{step['agent']}]"
    if step.get("final"):
        text = step["action_input"]
        preview = (text if len(text) <= 500
                   else text[:500] + f"…（CLI 预览截断，完整内容 {len(text)} 字已存 trace）")
        print(f"\n{tag} Final Answer:\n{preview}")
        return
    if step.get("observation") is None:
        # 工具执行前：先看到它决定做什么
        print(f"\n{tag} Thought: {step['thought'][:120]}")
        print(f"{tag} Action: {step['action']} ({step['action_input'][:100]})")
    else:
        # 工具执行后：补上 observation（截短显示）
        obs = (step["observation"] or "")[:200].replace("\n", " ")
        print(f"{tag} Observation: {obs}...")


# ── 自测 ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    question = "帮我做一个校园二手交易小程序的开发规划：需求分析、技术选型、架构设计、风险与测试"
    print(f"\n{'='*66}\n主 agent 开始处理: {question}\n{'='*66}")
    r = run_dev_plan(
        question,
        on_main_step=_print_step,
        # subagent 回调约定是 (sid, step) 两个参数，用 lambda 适配单参的 _print_step
        on_subagent_step=lambda sid, step: _print_step(step),
        on_dispatch=lambda info: print(
            f"\n>>> 派发 {len(info['subtopics'])} 个子规划师(并行): "
            f"{list(zip(info['roles'], [t if len(t) <= 20 else t[:20] + '…' for t in info['subtopics']]))}"),
        on_subagent_done=lambda sid, dur, topic: print(
            f"   [√] {sid} {topic[:24]} 完成, 用时 {dur}s"),
    )
    print(f"\n{'='*66}\n主 agent 动作序列: {[s['action'] for s in r['main_trace']]}")
    print(f"派发次数: {len(r['dispatches'])} | subagent 数: {len(r['subagents'])}")
    print(f"并行统计: {r['parallel_stats']}")
    print(f"\n{'='*66}\n最终规划文档:\n{'='*66}\n{r['final_answer']}")

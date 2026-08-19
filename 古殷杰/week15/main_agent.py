"""
主 Agent (Orchestrator Agent):
  接收用户任务 -> LLM 拆解为 N 个子任务 -> 并行调度 subagent -> 汇总返回

主 agent 自己也是一个 LLM 调用,但它不执行工具,只负责"规划"。
子任务通过结构化 JSON 输出交回给调度器。
"""
import json
import re

from llm import get_llm
from orchestrator import ParallelOrchestrator, SubTask, OrchestrationResult
from config import MAX_PARALLEL


MAIN_AGENT_SYSTEM_PROMPT = """你是主 Agent (Orchestrator),负责把用户的复杂任务拆解为多个独立的子任务,并交给 SubAgent 并行执行。

拆解原则:
1. 子任务之间应**独立可并行**(无依赖关系),这样才能同时跑
2. 每个子任务要**目标明确**,SubAgent 拿到任务就能直接开干
3. 子任务粒度适中:太小浪费调度成本,太大失去并行优势
4. 数量控制在 2-5 个

输出格式(严格遵守,只能输出 JSON):
{
  "subtasks": [
    {"task_id": "task_1", "description": "具体的子任务描述"},
    {"task_id": "task_2", "description": "..."}
  ]
}
不要输出任何 JSON 之外的内容。
"""


SUMMARY_SYSTEM_PROMPT = """你是主 Agent,负责把多个 SubAgent 的执行结果汇总成给用户的最终回复。

要求:
- 不要罗列所有中间过程
- 提炼关键结论,按逻辑组织
- 如有失败,简要说明原因
- 用清晰自然的中文输出
"""


class MainAgent:
    """主 Agent,负责任务拆解 + 并行调度 + 结果汇总。"""

    def __init__(self):
        self.llm = get_llm()
        self.orchestrator = ParallelOrchestrator(max_concurrent=MAX_PARALLEL)

    async def _decompose(self, user_task: str) -> list[SubTask]:
        """调用 LLM 把用户任务拆成子任务列表。"""
        resp = await self.llm.create_message(
            system=MAIN_AGENT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_task}],
            # 这里故意不传 tools,主 agent 只做规划
        )

        # 取文本内容
        text = ""
        for block in resp["content"]:
            if block["type"] == "text":
                text += block["text"]

        # 尝试解析 JSON(模型偶尔会包 ```json``` 围栏)
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            # Mock / 解析失败时,兜底拆成 1 个任务
            return [SubTask("task_1", user_task)]

        try:
            data = json.loads(m.group(0))
            subtasks = []
            for item in data.get("subtasks", [])[:MAX_PARALLEL]:
                subtasks.append(SubTask(
                    task_id=item.get("task_id", f"task_{len(subtasks)+1}"),
                    description=item.get("description", ""),
                ))
            if not subtasks:
                return [SubTask("task_1", user_task)]
            return subtasks
        except json.JSONDecodeError:
            return [SubTask("task_1", user_task)]

    async def _summarize(self, user_task: str, orch: OrchestrationResult) -> str:
        """把并行执行结果汇总成最终回答。"""
        # 把 subagent 结果拼成上下文
        ctx_lines = [f"用户原始任务: {user_task}", "", "各 SubAgent 的执行结果:"]
        for r in orch.results:
            ctx_lines.append(f"\n--- {r.task_id} ---")
            ctx_lines.append(f"任务: {r.task_desc}")
            ctx_lines.append(f"回答: {r.final_answer}")

        resp = await self.llm.create_message(
            system=SUMMARY_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": "\n".join(ctx_lines)}],
        )

        summary = ""
        for block in resp["content"]:
            if block["type"] == "text":
                summary += block["text"]
        return summary.strip()

    async def run(self, user_task: str) -> dict:
        """
        主入口:接收用户任务,完成"分解 -> 并行 -> 汇总"全流程。

        返回:
          {
            "user_task": 原始任务,
            "subtasks": 拆解出的子任务,
            "orchestration": 调度结果(含每个 subagent 的回答),
            "final_answer": 主 agent 汇总后的最终回答,
            "total_time": 总耗时(秒)
          }
        """
        import time
        t0 = time.time()

        # Step 1: 任务分解
        subtasks = await self._decompose(user_task)

        # Step 2: 并行调度
        orch = await self.orchestrator.run(subtasks)

        # Step 3: 汇总
        final_answer = await self._summarize(user_task, orch)

        return {
            "user_task": user_task,
            "subtasks": [{"task_id": t.task_id, "description": t.description} for t in subtasks],
            "orchestration": orch,
            "final_answer": final_answer,
            "total_time": time.time() - t0,
        }

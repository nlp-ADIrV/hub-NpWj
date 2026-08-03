"""The model/tool loop that progressively loads and executes skills."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .catalog import SkillCatalog
from .model import ChatModel
from .runtime import SkillRuntime, ToolExecutionError, TraceEvent


class HarnessError(RuntimeError):
    """Raised when the harness cannot safely finish a turn."""


@dataclass(frozen=True)
class RunResult:
    answer: str
    loaded_skills: tuple[str, ...]
    trace: tuple[TraceEvent, ...]
    tool_rounds: int


class SkillHarness:
    """Run one task with a fresh, disposable skill context."""

    def __init__(
        self,
        catalog: SkillCatalog,
        model: ChatModel,
        workspace: str | Path,
        *,
        max_tool_rounds: int = 8,
        history_message_limit: int = 12,
        script_timeout_seconds: float = 30.0,
    ) -> None:
        if max_tool_rounds < 1:
            raise ValueError("max_tool_rounds must be at least 1")
        self.catalog = catalog
        self.model = model
        self.workspace = Path(workspace).resolve()
        self.max_tool_rounds = max_tool_rounds
        self.history_message_limit = history_message_limit
        self.script_timeout_seconds = script_timeout_seconds

    def run(
        self,
        user_input: str,
        *,
        history: Sequence[Mapping[str, Any]] = (),
    ) -> RunResult:
        if not user_input.strip():
            raise HarnessError("user input must not be empty")

        runtime = SkillRuntime(
            self.catalog,
            self.workspace,
            script_timeout_seconds=self.script_timeout_seconds,
        )
        skill_index = self.catalog.prompt_index()
        runtime.trace.append(
            TraceEvent(
                "catalog_indexed",
                {
                    "skill_count": len(self.catalog),
                    "index_chars": len(skill_index),
                },
            )
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt(skill_index)}
        ]
        messages.extend(self._safe_history(history))
        messages.append({"role": "user", "content": user_input})

        tool_rounds = 0
        while True:
            runtime.trace.append(
                TraceEvent(
                    "model_called",
                    {
                        "tool_round": tool_rounds,
                        "loaded_skills": sorted(runtime.loaded_skills),
                    },
                )
            )
            try:
                reply = self.model.complete(messages, runtime.tool_schemas())
            except Exception as exc:
                raise HarnessError(
                    f"model call failed: {type(exc).__name__}: {exc}"
                ) from exc
            messages.append(reply.as_message())

            if not reply.tool_calls:
                answer = (reply.content or "").strip()
                if not answer:
                    raise HarnessError("model returned neither tool calls nor a final answer")
                runtime.trace.append(
                    TraceEvent(
                        "completed",
                        {
                            "tool_rounds": tool_rounds,
                            "loaded_skills": sorted(runtime.loaded_skills),
                        },
                    )
                )
                return RunResult(
                    answer=answer,
                    loaded_skills=tuple(sorted(runtime.loaded_skills)),
                    trace=tuple(runtime.trace),
                    tool_rounds=tool_rounds,
                )

            if tool_rounds >= self.max_tool_rounds:
                raise HarnessError(
                    f"tool round limit exceeded ({self.max_tool_rounds})"
                )
            tool_rounds += 1

            for tool_call in reply.tool_calls:
                tool_output = self._execute_tool_call(runtime, tool_call.name, tool_call.arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_output,
                    }
                )

    def _execute_tool_call(
        self,
        runtime: SkillRuntime,
        tool_name: str,
        raw_arguments: str,
    ) -> str:
        try:
            parsed = json.loads(raw_arguments or "{}")
            if not isinstance(parsed, dict):
                raise ToolExecutionError("tool arguments must decode to a JSON object")
            return runtime.dispatch(tool_name, parsed)
        except (json.JSONDecodeError, ToolExecutionError) as exc:
            runtime.trace.append(
                TraceEvent(
                    "tool_error",
                    {
                        "tool": tool_name,
                        "error": str(exc),
                    },
                )
            )
            return json.dumps(
                {
                    "ok": False,
                    "error": str(exc),
                },
                ensure_ascii=False,
            )
        except Exception as exc:
            runtime.trace.append(
                TraceEvent(
                    "tool_error",
                    {
                        "tool": tool_name,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
            )
            return json.dumps(
                {
                    "ok": False,
                    "error": f"unexpected tool failure: {type(exc).__name__}",
                },
                ensure_ascii=False,
            )

    def _safe_history(
        self,
        history: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, str]]:
        safe_messages: list[dict[str, str]] = []
        for item in history[-self.history_message_limit :]:
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and isinstance(content, str):
                safe_messages.append({"role": role, "content": content})
        return safe_messages

    @staticmethod
    def _system_prompt(skill_index: str) -> str:
        return f"""你是一个支持渐进式 Skills 的执行型 Agent。

当前 Context 只包含 skill 元数据索引，不包含任何完整 skill 指令：
<available_skills>{skill_index}</available_skills>

必须遵守以下执行协议：
1. 先判断用户任务是否匹配索引中的 skill。若匹配，必须先调用 load_skill，不能猜测其正文。
2. load_skill 返回完整指令和资源文件名。严格按已加载指令执行。
3. 只有已加载指令明确需要某个参考文件时，才调用 read_skill_resource；不要批量读取资源。
4. 只有对应 skill 已加载后，才可调用 run_skill_script。
5. 需要生成脚本输入或文本产物时，使用 write_artifact，并绑定已加载的 skill；
   路径必须位于 artifacts/<skill-name>/。
6. 有依赖关系的工具调用分轮执行：先取得上一步结果，再发起下一步。
7. 可按任务需要加载多个 skill；无匹配 skill 时直接回答。
8. 工具失败时根据错误修正参数，禁止虚构成功结果。
9. 完成后给出简洁结果，明确实际产生的文件路径。

完整 skill 内容只服务于当前任务；任务结束后 harness 会释放这些工具消息。"""

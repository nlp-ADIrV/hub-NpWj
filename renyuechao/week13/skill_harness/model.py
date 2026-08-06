"""LLM adapter types for an OpenAI-compatible chat completion API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str

    def as_message_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            },
        }


@dataclass(frozen=True)
class AssistantReply:
    content: str | None
    tool_calls: tuple[ToolCall, ...] = ()

    def as_message(self) -> dict[str, Any]:
        message: dict[str, Any] = {
            "role": "assistant",
            "content": self.content,
        }
        if self.tool_calls:
            message["tool_calls"] = [
                tool_call.as_message_record() for tool_call in self.tool_calls
            ]
        return message


class ChatModel(Protocol):
    def complete(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> AssistantReply:
        """Return one assistant message, including zero or more tool calls."""


class OpenAIChatModel:
    """Thin adapter around the OpenAI Python SDK."""

    def __init__(self, client: Any, model: str, *, temperature: float = 0.0) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature

    @classmethod
    def from_env(
        cls,
        *,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> "OpenAIChatModel":
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai is not installed; run `python -m pip install -r requirements.txt`"
            ) from exc

        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        model_name = 'deepseek-v4-flash' #model or os.getenv("LLM_MODEL")
        base_url = 'https://api.deepseek.com' #os.getenv("LLM_BASE_URL")

        if not api_key:
            raise RuntimeError("set LLM_API_KEY (or OPENAI_API_KEY) before running")
        if not model_name:
            raise RuntimeError("set LLM_MODEL or pass --model before running")

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)
        return cls(client, model_name, temperature=temperature)

    def complete(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
    ) -> AssistantReply:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=list(messages),
            tools=list(tools),
            tool_choice="auto",
            temperature=self.temperature,
        )
        message = response.choices[0].message
        parsed_calls = tuple(
            ToolCall(
                id=tool_call.id,
                name=tool_call.function.name,
                arguments=tool_call.function.arguments or "{}",
            )
            for tool_call in (message.tool_calls or ())
        )
        content = message.content if isinstance(message.content, str) else None
        return AssistantReply(content=content, tool_calls=parsed_calls)

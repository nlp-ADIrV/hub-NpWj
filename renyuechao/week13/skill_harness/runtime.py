"""Runtime tools exposed to the model after the compact skill index."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .catalog import SkillCatalog, SkillCatalogError, SkillSummary


class ToolExecutionError(RuntimeError):
    """A safe, model-visible tool failure."""


@dataclass(frozen=True)
class TraceEvent:
    phase: str
    details: dict[str, Any]


class SkillRuntime:
    """Per-turn state: selected skills, deferred resources and execution trace."""

    MAX_ARTIFACT_CHARS = 200_000
    MAX_RESOURCE_CHARS = 100_000
    MAX_TOOL_OUTPUT_CHARS = 16_000
    MAX_SCRIPT_ARGS = 64
    MAX_SCRIPT_ARG_CHARS = 4_096

    def __init__(
        self,
        catalog: SkillCatalog,
        workspace: str | Path,
        *,
        script_timeout_seconds: float = 30.0,
    ) -> None:
        self.catalog = catalog
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir = (self.workspace / "artifacts").resolve()
        if not self.artifacts_dir.is_relative_to(self.workspace):
            raise ToolExecutionError(
                "workspace artifacts/ resolves outside the workspace"
            )
        self.script_timeout_seconds = script_timeout_seconds
        self.loaded_skills: set[str] = set()
        self.trace: list[TraceEvent] = []

    @staticmethod
    def tool_schemas() -> tuple[dict[str, Any], ...]:
        return (
            {
                "type": "function",
                "function": {
                    "name": "load_skill",
                    "description": (
                        "Load the complete instructions for one relevant skill. "
                        "Call this before using any other tool for that skill."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Exact skill name from the available skill index.",
                            }
                        },
                        "required": ["name"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_skill_resource",
                    "description": (
                        "Read one text reference or data file from an already-loaded "
                        "skill. Use only when its instructions require that file."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "skill_name": {"type": "string"},
                            "path": {
                                "type": "string",
                                "description": (
                                    "Path relative to the skill root, for example "
                                    "'references/flowchart.md'."
                                ),
                            },
                        },
                        "required": ["skill_name", "path"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_artifact",
                    "description": (
                        "Write UTF-8 text under artifacts/<skill_name>/ for an "
                        "already-loaded skill. Use this for inputs and text artifacts."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "skill_name": {"type": "string"},
                            "path": {
                                "type": "string",
                                "description": (
                                    "Workspace-relative path beginning with "
                                    "'artifacts/<skill_name>/'."
                                ),
                            },
                            "content": {"type": "string"},
                        },
                        "required": ["skill_name", "path", "content"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_skill_script",
                    "description": (
                        "Run a trusted script from scripts/ in an already-loaded skill. "
                        "Arguments are passed directly without a shell."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "skill_name": {"type": "string"},
                            "script": {
                                "type": "string",
                                "description": (
                                    "Path below the skill's scripts/ directory."
                                ),
                            },
                            "args": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Ordered command-line arguments.",
                            },
                        },
                        "required": ["skill_name", "script", "args"],
                        "additionalProperties": False,
                    },
                },
            },
        )

    def dispatch(self, tool_name: str, arguments: Mapping[str, Any]) -> str:
        handlers = {
            "load_skill": self._load_skill,
            "read_skill_resource": self._read_skill_resource,
            "write_artifact": self._write_artifact,
            "run_skill_script": self._run_skill_script,
        }
        handler = handlers.get(tool_name)
        if handler is None:
            raise ToolExecutionError(f"unknown tool: {tool_name}")
        result = handler(arguments)
        return json.dumps(result, ensure_ascii=False)

    def _load_skill(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        name = _required_string(arguments, "name")
        if name in self.loaded_skills:
            return {"ok": True, "name": name, "already_loaded": True}

        try:
            instructions = self.catalog.load_instructions(name)
            resources = self.catalog.resource_manifest(name)
        except SkillCatalogError as exc:
            raise ToolExecutionError(str(exc)) from exc

        self.loaded_skills.add(name)
        self.trace.append(
            TraceEvent(
                "skill_loaded",
                {
                    "name": name,
                    "instruction_chars": len(instructions),
                    "resource_count": len(resources),
                },
            )
        )
        return {
            "ok": True,
            "name": name,
            "instructions": instructions,
            "available_resources": list(resources),
        }

    def _read_skill_resource(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        skill_name = _required_string(arguments, "skill_name")
        relative_path = _required_string(arguments, "path")
        summary = self._require_loaded(skill_name)
        resource = _resolve_inside(summary.root, relative_path)

        if not resource.is_file():
            raise ToolExecutionError(
                f"skill resource does not exist or is not a file: {relative_path}"
            )
        try:
            content = resource.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ToolExecutionError(
                f"skill resource is not UTF-8 text: {relative_path}"
            ) from exc
        if len(content) > self.MAX_RESOURCE_CHARS:
            raise ToolExecutionError(
                f"skill resource is too large ({len(content)} chars, "
                f"limit {self.MAX_RESOURCE_CHARS})"
            )

        self.trace.append(
            TraceEvent(
                "resource_loaded",
                {
                    "skill": skill_name,
                    "path": relative_path,
                    "chars": len(content),
                },
            )
        )
        return {
            "ok": True,
            "skill_name": skill_name,
            "path": relative_path,
            "content": content,
        }

    def _write_artifact(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        skill_name = _required_string(arguments, "skill_name")
        self._require_loaded(skill_name)
        relative_path = _required_string(arguments, "path")
        content = _required_string(arguments, "content", allow_empty=True)
        if len(content) > self.MAX_ARTIFACT_CHARS:
            raise ToolExecutionError(
                f"artifact is too large ({len(content)} chars, "
                f"limit {self.MAX_ARTIFACT_CHARS})"
            )

        target = _resolve_inside(self.workspace, relative_path)
        skill_artifacts_dir = (self.artifacts_dir / skill_name).resolve()
        if not skill_artifacts_dir.is_relative_to(self.artifacts_dir):
            raise ToolExecutionError(
                f"artifacts/{skill_name}/ resolves outside artifacts/"
            )
        if not target.is_relative_to(skill_artifacts_dir):
            raise ToolExecutionError(
                f"artifact path must be below artifacts/{skill_name}/"
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        self.trace.append(
            TraceEvent(
                "artifact_written",
                {
                    "skill": skill_name,
                    "path": target.relative_to(self.workspace).as_posix(),
                    "chars": len(content),
                },
            )
        )
        return {
            "ok": True,
            "skill_name": skill_name,
            "path": target.relative_to(self.workspace).as_posix(),
            "chars": len(content),
        }

    def _run_skill_script(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        skill_name = _required_string(arguments, "skill_name")
        script_path = _required_string(arguments, "script")
        raw_args = arguments.get("args")
        if not isinstance(raw_args, list) or not all(
            isinstance(item, str) for item in raw_args
        ):
            raise ToolExecutionError("args must be a list of strings")
        if len(raw_args) > self.MAX_SCRIPT_ARGS:
            raise ToolExecutionError(
                f"too many script arguments (limit {self.MAX_SCRIPT_ARGS})"
            )
        if any(len(item) > self.MAX_SCRIPT_ARG_CHARS for item in raw_args):
            raise ToolExecutionError(
                "a script argument exceeds "
                f"{self.MAX_SCRIPT_ARG_CHARS} characters"
            )

        summary = self._require_loaded(skill_name)
        script = _resolve_inside(summary.root, script_path)
        scripts_root = (summary.root / "scripts").resolve()
        if not script.is_relative_to(scripts_root):
            raise ToolExecutionError("script must be below the skill's scripts/ directory")
        if not script.is_file():
            raise ToolExecutionError(f"skill script does not exist: {script_path}")

        runner = _script_runner(script)
        command = [*runner, str(script), *raw_args]
        try:
            completed = subprocess.run(
                command,
                cwd=self.workspace,
                env=_safe_subprocess_env(),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.script_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolExecutionError(
                f"skill script timed out after {self.script_timeout_seconds:g}s"
            ) from exc

        stdout = _truncate(completed.stdout, self.MAX_TOOL_OUTPUT_CHARS)
        stderr = _truncate(completed.stderr, self.MAX_TOOL_OUTPUT_CHARS)
        self.trace.append(
            TraceEvent(
                "script_executed",
                {
                    "skill": skill_name,
                    "script": script_path,
                    "exit_code": completed.returncode,
                },
            )
        )
        return {
            "ok": completed.returncode == 0,
            "skill_name": skill_name,
            "script": script_path,
            "exit_code": completed.returncode,
            "stdout": stdout,
            "stderr": stderr,
        }

    def _require_loaded(self, name: str) -> SkillSummary:
        if name not in self.loaded_skills:
            raise ToolExecutionError(
                f"skill {name!r} is not loaded; call load_skill first"
            )
        try:
            return self.catalog.get(name)
        except SkillCatalogError as exc:
            raise ToolExecutionError(str(exc)) from exc


def _required_string(
    arguments: Mapping[str, Any],
    key: str,
    *,
    allow_empty: bool = False,
) -> str:
    value = arguments.get(key)
    if not isinstance(value, str):
        raise ToolExecutionError(f"{key} must be a string")
    if not allow_empty and not value.strip():
        raise ToolExecutionError(f"{key} must not be empty")
    return value


def _resolve_inside(base: Path, relative_path: str) -> Path:
    candidate_path = Path(relative_path)
    if candidate_path.is_absolute():
        raise ToolExecutionError("absolute paths are not allowed")
    candidate = (base / candidate_path).resolve()
    if not candidate.is_relative_to(base.resolve()):
        raise ToolExecutionError(f"path escapes allowed root: {relative_path}")
    return candidate


def _script_runner(script: Path) -> list[str]:
    suffix = script.suffix.lower()
    if suffix == ".py":
        return [sys.executable]
    if suffix == ".sh":
        return ["/bin/bash"]
    if suffix in {".js", ".mjs"}:
        node = shutil.which("node")
        if not node:
            raise ToolExecutionError("node is required to run this skill script")
        return [node]
    if suffix == ".ts":
        bun = shutil.which("bun")
        if not bun:
            raise ToolExecutionError("bun is required to run TypeScript skill scripts")
        return [bun]
    raise ToolExecutionError(
        f"unsupported skill script type {suffix!r}; allowed: .py, .sh, .js, .mjs, .ts"
    )


def _safe_subprocess_env() -> dict[str, str]:
    allowed_names = {
        "PATH",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TMPDIR",
        "TEMP",
        "TMP",
        "SYSTEMROOT",
    }
    child_env = {
        name: value for name, value in os.environ.items() if name in allowed_names
    }
    child_env["PYTHONIOENCODING"] = "utf-8"
    return child_env


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    omitted = len(value) - limit
    return f"{value[:limit]}\n... <truncated {omitted} chars>"

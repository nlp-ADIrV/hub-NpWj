"""CLI entry point for the progressive skill harness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from skill_harness import (
    HarnessError,
    OpenAIChatModel,
    SkillCatalog,
    SkillCatalogError,
    SkillHarness,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Progressively load and execute local SKILL.md packages."
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="One-shot user request. Omit it to enter interactive mode.",
    )
    parser.add_argument(
        "--skills-dir",
        type=Path,
        default=PROJECT_ROOT / "skills",
        help="Directory containing */SKILL.md packages.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Workspace used for artifacts and skill script execution.",
    )
    parser.add_argument("--model", help="Model name; defaults to LLM_MODEL.")
    parser.add_argument(
        "--max-tool-rounds",
        type=int,
        default=8,
        help="Maximum tool-call rounds per user task.",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Print progressive loading events to stderr.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List frontmatter metadata without starting the LLM.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        catalog = SkillCatalog(args.skills_dir)
        if args.list:
            for summary in catalog.summaries():
                print(f"{summary.name}: {' '.join(summary.description.split())}")
            return 0

        model = OpenAIChatModel.from_env(model=args.model)
        harness = SkillHarness(
            catalog,
            model,
            args.workspace,
            max_tool_rounds=args.max_tool_rounds,
        )

        if args.prompt:
            result = harness.run(" ".join(args.prompt))
            print(result.answer)
            if args.trace:
                print_trace(result.trace)
            return 0

        return interactive_loop(harness, trace=args.trace)
    except (HarnessError, SkillCatalogError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def interactive_loop(harness: SkillHarness, *, trace: bool) -> int:
    print("渐进式 Skills Harness（/skills 查看索引，/exit 退出）")
    history: list[dict[str, str]] = []

    while True:
        try:
            user_input = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not user_input:
            continue
        if user_input == "/exit":
            return 0
        if user_input == "/skills":
            for summary in harness.catalog.summaries():
                print(f"- {summary.name}: {' '.join(summary.description.split())}")
            continue

        try:
            result = harness.run(user_input, history=history)
        except HarnessError as exc:
            print(f"error: {exc}", file=sys.stderr)
            continue

        print(f"Agent：{result.answer}")
        if trace:
            print_trace(result.trace)

        # Only final dialogue is retained. Loaded SKILL.md/tool results are released.
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": result.answer})


def print_trace(events) -> None:
    for event in events:
        details = json.dumps(event.details, ensure_ascii=False, sort_keys=True)
        print(f"[trace] {event.phase} {details}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())

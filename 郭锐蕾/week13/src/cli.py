"""CLI：演示渐进式 Skill 加载生命周期。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.harness import SkillHarness


def _print_event(ev) -> None:
    print(f"  [{ev.step}] {ev.message}")


def main(argv: list[str] | None = None) -> int:
    # Windows 控制台避免中文/符号打印崩溃
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="渐进式 Skill Harness CLI")
    parser.add_argument(
        "message",
        nargs="?",
        default="",
        help="用户消息，例如：给我做张 crazy 的闪卡",
    )
    parser.add_argument(
        "--skills-dir",
        type=Path,
        default=ROOT / "skills",
        help="Skills 目录",
    )
    parser.add_argument("--mode", choices=["demo", "llm"], default="demo")
    parser.add_argument("--force-skill", default=None)
    parser.add_argument("--no-release", action="store_true")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    parser.add_argument("--list", action="store_true", help="列出已注册 Skills 与索引统计")
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="只展示全量 vs 渐进式 token 对比（使用默认示例消息）",
    )
    args = parser.parse_args(argv)

    harness = SkillHarness(args.skills_dir, workspace=ROOT)
    harness.refresh()

    if args.list:
        status = harness.status()
        print("=== Skills Index Stats ===")
        print(json.dumps(status["index"], ensure_ascii=False, indent=2))
        print("\n=== Index Markdown ===")
        print(harness.registry.index_md)
        return 0

    message = args.message
    if args.compare_only and not message:
        message = "给我做张 crazy 的闪卡"
    if not message:
        parser.error("请提供用户消息，或使用 --list / --compare-only")

    print(f"\n用户消息: {message}\n")
    print("── 生命周期 ──")
    result = harness.handle(
        message,
        mode=args.mode,
        force_skill=args.force_skill,
        auto_release=not args.no_release,
        on_event=_print_event,
    )

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0

    print("\n── 匹配结果 ──")
    if result.matched:
        print(f"  Skill: {result.matched.skill.name}  score={result.matched.score}")
        print(f"  reasons: {', '.join(result.matched.reasons)}")
    else:
        print("  （无匹配）")

    print("\n── Context 分层 ──")
    for ly in result.layers:
        print(
            f"  [{ly.layer.value:10}] {ly.name:28} "
            f"{ly.token_estimate:5} tok  {ly.char_count:6} chars  ← {ly.source}"
        )

    print("\n── Token 对比（全量 vs 渐进）──")
    c = result.comparison
    print(f"  全量加载:   {c.get('full_load_tokens')} tokens")
    print(f"  本轮渐进:   {c.get('progressive_tokens')} tokens")
    print(f"  节省:       {c.get('saved_tokens')} tokens ({c.get('saved_ratio_percent')}%)")
    print(f"  仅索引:     {c.get('index_only_tokens')} tokens")

    if result.secondary_loaded:
        print("\n── 二级加载 ──")
        for s in result.secondary_loaded:
            print(f"  + {s}")

    if result.execution_notes:
        print("\n── 执行备注 ──")
        for n in result.execution_notes:
            print(f"  - {n}")

    if result.script_outputs:
        print("\n── 脚本输出 ──")
        print(json.dumps(result.script_outputs, ensure_ascii=False, indent=2))

    print(f"\n释放完成: {result.released}")
    print(f"当前仍驻留的 body: {harness.loader.loaded_names() or '[]'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

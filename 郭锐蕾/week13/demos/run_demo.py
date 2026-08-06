"""一键演示：多条用户消息走完整渐进式生命周期。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.harness import SkillHarness

CASES = [
    "给我做张 crazy 的闪卡",
    "画一个系统架构图",
    "帮我做个 PPT slides",
    "请帮我 code review 这段改动",
    "随便聊聊今天天气",
]


def main() -> None:
    harness = SkillHarness(ROOT / "skills", workspace=ROOT)
    harness.refresh()
    print(harness.registry.index_md)
    print("=" * 60)

    summary = []
    for msg in CASES:
        print(f"\n>>> {msg}")
        result = harness.handle(msg, mode="demo")
        name = result.matched.skill.name if result.matched else None
        c = result.comparison
        print(
            f"    match={name}  progressive={c['progressive_tokens']}tok  "
            f"full={c['full_load_tokens']}tok  saved={c['saved_ratio_percent']}%  "
            f"released={result.released}"
        )
        summary.append(
            {
                "message": msg,
                "matched": name,
                "progressive_tokens": c["progressive_tokens"],
                "full_load_tokens": c["full_load_tokens"],
                "saved_ratio_percent": c["saved_ratio_percent"],
                "secondary_loaded": result.secondary_loaded,
                "notes": result.execution_notes,
            }
        )

    out = ROOT / "outputs" / "demo_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n汇总已写入 {out}")


if __name__ == "__main__":
    main()

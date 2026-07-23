"""
多轮对话 CLI 入口

使用方式：
  cd week12作业文件夹/src
  python agent.py                     # 交互式多轮（默认 manual）
  python agent.py --mode fc           # Function Calling 多轮
  python agent.py --question "..."    # 单次提问后退出
  python agent.py --demo              # 跑一段追问演示

交互命令：
  /clear   清空会话历史
  /history 查看已记录的问答轮次
  /mode manual|fc  切换实现
  /quit    退出
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# 本目录优先（conversation / react_*），再挂原项目 tools
_HERE = Path(__file__).resolve().parent
_PARENT_SRC = _HERE.parent.parent / "src"
sys.path.insert(0, str(_PARENT_SRC))
sys.path.insert(0, str(_HERE))

from conversation import ConversationSession  # noqa: E402


DEMO_TURNS = [
    "贵州茅台2023年的毛利率是多少？",
    "那五粮液呢？同样问2023年毛利率。",
    "两家差多少个百分点？用计算器算一下。",
]


def _print_banner(session: ConversationSession) -> None:
    print("\n" + "=" * 60)
    print("ReAct Financial Agent — 多轮对话模式")
    print(f"session: {session.session_id[:8]}...  mode: {session.mode}")
    print("命令: /clear  /history  /mode manual|fc  /quit")
    print("=" * 60)


def _run_one(session: ConversationSession, question: str) -> None:
    import json
    import time

    from react_manual import _c

    print(f"\n{'=' * 60}")
    print(f"问题: {question}")
    print(
        f"实现: {'手写Prompt解析' if session.mode == 'manual' else 'Function Calling'}  "
        f"历史轮数: {len(session.turns)}"
    )
    print("=" * 60)

    start = time.time()
    for step_data in session.chat(question):
        stype = step_data["type"]
        if stype == "action":
            print(f"\n[Step {step_data['step']}]")
            thought = step_data.get("thought") or "（模型内部推理，Function Calling 版不可见）"
            print(_c("thought", f"🧠 Thought: {thought}"))
            print(_c("action", f"🔧 Action:  {step_data['action']}"))
            print(_c("action", f"   Input:   {json.dumps(step_data['action_input'], ensure_ascii=False)}"))
            print(_c("obs", f"👁  Obs:     {step_data['observation'][:300]}"))
        elif stype == "final":
            elapsed = time.time() - start
            print(f"\n{'─' * 60}")
            if step_data.get("thought"):
                print(_c("thought", f"🧠 Thought: {step_data['thought']}"))
            print(_c("final", f"\n✅ Final Answer:\n{step_data['answer']}"))
            print(f"\n共 {step_data['step']} 步，耗时 {elapsed:.1f}s  | 会话已累积 {len(session.turns)} 轮")
        elif stype in ("error", "max_steps"):
            print(_c("error", f"\n⚠️  {step_data.get('answer', step_data.get('observation', ''))}"))


def interactive(mode: str, max_steps: int) -> None:
    session = ConversationSession(mode=mode, max_steps=max_steps)  # type: ignore[arg-type]
    _print_banner(session)

    while True:
        try:
            raw = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break

        if not raw:
            continue

        if raw in ("/quit", "/exit", "quit", "exit"):
            print("再见。")
            break
        if raw == "/clear":
            session.clear()
            print("已清空会话历史。")
            continue
        if raw == "/history":
            if not session.turns:
                print("（暂无历史）")
            else:
                for i, t in enumerate(session.turns, 1):
                    print(f"\n--- 第 {i} 轮 ---")
                    print(f"Q: {t.question}")
                    print(f"A: {t.answer[:200]}{'...' if len(t.answer) > 200 else ''}")
            continue
        if raw.startswith("/mode"):
            parts = raw.split()
            if len(parts) == 2 and parts[1] in ("manual", "fc"):
                session.mode = parts[1]  # type: ignore[assignment]
                print(f"已切换为 {session.mode}")
            else:
                print("用法: /mode manual  或  /mode fc")
            continue

        _run_one(session, raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="多轮对话 ReAct Financial Agent")
    parser.add_argument("--mode", choices=["manual", "fc"], default="manual")
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--question", default=None, help="单次提问（非交互）")
    parser.add_argument("--demo", action="store_true", help="运行追问演示后退出")
    args = parser.parse_args()

    if args.demo:
        session = ConversationSession(mode=args.mode, max_steps=args.max_steps)  # type: ignore[arg-type]
        _print_banner(session)
        for q in DEMO_TURNS:
            print(f"\n>>> 演示提问: {q}")
            _run_one(session, q)
        return

    if args.question:
        session = ConversationSession(mode=args.mode, max_steps=args.max_steps)  # type: ignore[arg-type]
        _run_one(session, args.question)
        return

    interactive(args.mode, args.max_steps)


if __name__ == "__main__":
    main()

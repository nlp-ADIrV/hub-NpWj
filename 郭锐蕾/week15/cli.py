"""CLI 实时流式输出 + 主入口

直观展示「主 agent 自主路由 + subagent 并行」全过程：
  - 主 agent 的 Thought / Action / Action Input 实时打印
  - dispatch 时高亮显示派发了几个子课题
  - 多个 subagent 的 ReAct 过程并行交错输出（带 sid badge）
  - 最终报告 + 并行加速统计

用法：
  python cli.py "你的调研问题"
  python cli.py                    # 跑内置示例
  python cli.py --serial "问题"    # 串行模式（对比基线）
"""
import sys
import os
import time
import threading
import logging
import argparse

from orchestrator import run_research


# ANSI 颜色（Windows 10+ 终端支持）
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"
    BLUE = "\033[34m"
    RED = "\033[31m"
    GRAY = "\033[90m"


# Windows 启用 ANSI 转义
if sys.platform == "win32":
    os.system("")

# 多线程打印锁：subagent 并行输出会交错，需加锁保证每行完整
_print_lock = threading.Lock()

# 给每个 subagent 分配固定颜色（按 sid 出现顺序循环）
_SUB_COLORS = [C.CYAN, C.MAGENTA, C.BLUE, C.YELLOW, C.GREEN, C.RED]


def _c(text, color) -> str:
    return f"{color}{text}{C.RESET}"


def _print(*args):
    with _print_lock:
        print(*args, flush=True)


def _badge(text, color) -> str:
    return _c(f"[{text}]", color)


def _sub_color(sid: str, color_map: dict) -> str:
    """给 sid 分配固定颜色（同一 subagent 全程同色，便于追踪）。"""
    if sid not in color_map:
        color_map[sid] = _SUB_COLORS[len(color_map) % len(_SUB_COLORS)]
    return color_map[sid]


def on_main_step_cli(step: dict):
    """主 agent 每步回调。"""
    idx = step["idx"]
    thought = step.get("thought") or ""
    action = step.get("action") or ""
    action_input = step.get("action_input") or ""
    done = step.get("done", False)
    final = step.get("final", False)

    badge = _badge(f"main #{idx}", C.BOLD + C.GREEN)

    if final:
        _print(f"\n{badge} {_c('Final Answer', C.BOLD + C.GREEN)}")
        _print(f"  {action_input[:300]}{'...' if len(action_input) > 300 else ''}")
        return

    if not done:
        # pre 执行：展示决策
        _print(f"\n{badge} {_c('Thought', C.YELLOW)}: {thought[:200]}")
        _print(f"  {_c('Action', C.CYAN)}: {action}")
        _print(f"  {_c('Action Input', C.CYAN)}: {action_input[:150]}")
        if action == "dispatch_subagents":
            n = len([s for s in action_input.split("|") if s.strip()])
            _print(f"  {_c(f'>>> 派发 {n} 个子调研员并行执行 <<<', C.BOLD + C.MAGENTA)}")
    else:
        # post 执行：展示 observation（截断）
        obs = step.get("observation") or ""
        _print(f"  {_c('Observation', C.DIM)}: {obs[:200]}{'...' if len(obs) > 200 else ''}")


def make_subagent_callbacks(color_map: dict):
    """构造 subagent 的 dispatch / step / done 回调闭包。"""

    def on_dispatch(info: dict):
        ids = info["subagent_ids"]
        topics = info["subtopics"]
        _print(f"\n{_c('=' * 60, C.GRAY)}")
        _print(_c(f"dispatch_subagents 派发 {len(ids)} 个子调研员（并行执行）",
                  C.BOLD + C.MAGENTA))
        for sid, topic in zip(ids, topics):
            color = _sub_color(sid, color_map)
            _print(f"  {_badge(sid, color)} {topic}")
        _print(_c("=" * 60, C.GRAY))

    def on_subagent_step(sid: str, step: dict):
        color = _sub_color(sid, color_map)
        idx = step["idx"]
        badge = _badge(f"{sid} #{idx}", color)
        action = step.get("action") or ""
        thought = step.get("thought") or ""
        action_input = step.get("action_input") or ""
        done = step.get("done", False)
        final = step.get("final", False)

        if final:
            _print(f"{badge} {_c('Final', C.BOLD + color)}: {action_input[:120]}")
            return
        if not done:
            _print(f"{badge} {_c('Thought', C.DIM)}: {thought[:100]}")
            _print(f"{badge} {_c('Action', color)}: {action} | {action_input[:80]}")
        else:
            obs = step.get("observation") or ""
            _print(f"{badge} {_c('Obs', C.DIM)}: "
                   f"{obs[:100]}{'...' if len(obs) > 100 else ''}")

    def on_subagent_done(sid: str, duration: float, topic: str):
        color = _sub_color(sid, color_map)
        _print(f"{_badge(sid, color)} {_c(f'done ({duration}s)', C.GREEN)}")

    return on_dispatch, on_subagent_step, on_subagent_done


def main():
    parser = argparse.ArgumentParser(description="市场调研 Subagent 并行系统 (CLI)")
    parser.add_argument("question", nargs="?", default=None,
                        help="调研问题（不填则跑内置示例）")
    parser.add_argument("--serial", action="store_true",
                        help="串行模式（对比基线，凸显并行加速）")
    parser.add_argument("--verbose", action="store_true",
                        help="显示详细日志")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    question = (args.question
                or "2024年中国新能源汽车市场调研：销量规模、主要厂商竞争格局、政策趋势")

    _print(_c("=" * 60, C.GRAY))
    _print(_c("  市场调研 Subagent 并行系统", C.BOLD + C.CYAN))
    _print(_c("=" * 60, C.GRAY))
    _print(f"{_badge('mode', C.YELLOW)} {'串行(对比基线)' if args.serial else '并行'}")
    _print(f"{_badge('query', C.YELLOW)} {question}")
    _print(_c("-" * 60, C.GRAY))

    color_map = {}
    on_dispatch, on_subagent_step, on_subagent_done = make_subagent_callbacks(color_map)

    t0 = time.time()
    result = run_research(
        question,
        on_main_step=on_main_step_cli,
        on_subagent_step=on_subagent_step,
        on_subagent_done=on_subagent_done,
        on_dispatch=on_dispatch,
        serial=args.serial,
    )
    total = round(time.time() - t0, 2)

    # 最终报告
    _print(f"\n{_c('=' * 60, C.GRAY)}")
    _print(_c("  最终调研报告", C.BOLD + C.GREEN))
    _print(_c("=" * 60, C.GRAY))
    _print(result["final_answer"])

    # 并行加速统计
    _print(f"\n{_c('=' * 60, C.GRAY)}")
    _print(_c("  并行加速统计", C.BOLD + C.MAGENTA))
    _print(_c("=" * 60, C.GRAY))
    _print(f"{_badge('main', C.GREEN)} 步数: {len(result['main_trace'])} | "
           f"派发次数: {len(result['dispatches'])} | "
           f"subagent 数: {len(result['subagents'])}")
    for st in result["parallel_stats"]:
        _print(f"  {_c('subagents', C.CYAN)}: {st['n_subagents']} | "
               f"{_c('wall', C.YELLOW)}: {st['wall_clock']}s | "
               f"{_c('serial_sum', C.DIM)}: {st['serial_sum']}s | "
               f"{_c('speedup', C.BOLD + C.GREEN)}: {st['speedup']}×")
    _print(f"{_badge('total', C.BOLD + C.GREEN)} 端到端墙钟: {total}s")


if __name__ == "__main__":
    main()

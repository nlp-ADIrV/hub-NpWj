"""Parallel vs Serial 量化对比

使用方式：
  python src/eval_compare.py --limit 2   # 只跑前 2 题（约 4~5 分钟）
  python src/eval_compare.py             # 跑全部 4 题（约 8~10 分钟）
  
"""

import sys
import time
import json
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

EVAL_QUESTIONS = [
    "帮我做一个校园二手交易小程序的开发规划：需求分析、技术选型、架构设计、风险与测试",
    "帮我做一个个人博客系统的开发规划：需求分析、技术选型、架构设计、上线与运维",
    "帮我做一个宠物领养平台的开发规划：需求分析、技术选型、架构设计、风险与测试",
    "帮我做一个食堂外卖点餐小程序的开发规划：需求分析、技术选型、架构设计、风险与测试",
]


def run_one(question, serial):
    """跑一次规划，返回 {wall, n_subagents, dispatch_wall, serial_sum, speedup, dispatched}。"""
    import agents
    t0 = time.time()
    r = agents.run_dev_plan(question, serial=serial)
    wall = time.time() - t0
    ps = r["parallel_stats"][-1] if r["parallel_stats"] else None
    return {
        "wall": round(wall, 2),
        "n_subagents": ps["n_subagents"] if ps else 0,
        "dispatch_wall": ps["wall_clock"] if ps else 0,
        "serial_sum": ps["serial_sum"] if ps else 0,
        "speedup": ps["speedup"] if ps else 0,
        "dispatched": len(r["dispatches"]) > 0,
    }


def main():
    parser = argparse.ArgumentParser(description="parallel vs serial 对比")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题（0=全部）")
    args = parser.parse_args()
    qs = EVAL_QUESTIONS[:args.limit] if args.limit else EVAL_QUESTIONS

    results = []
    for i, q in enumerate(qs):
        logger.warning(f"[{i+1}/{len(qs)}] {q}")
        p = run_one(q, serial=False)
        s = run_one(q, serial=True)
        results.append({"question": q, "parallel": p, "serial": s})
        print(f"  {q[:20]:<22} 并行 {p['wall']}s vs 串行 {s['wall']}s "
              f"(subagent {p['n_subagents']} 个, dispatch 加速 {p['speedup']}×)")

    avg_p = sum(r["parallel"]["wall"] for r in results) / len(results)
    avg_s = sum(r["serial"]["wall"] for r in results) / len(results)
    avg_spd = sum(r["parallel"]["speedup"] for r in results) / len(results)

    print(f"\n{'='*62}\nParallel vs Serial 对比（{len(results)} 题）\n{'='*62}")
    print(f"{'指标':<20}{'并行(ThreadPool)':<18}{'串行(for循环)':<18}")
    print(f"{'平均墙钟(s)':<20}{avg_p:<18.2f}{avg_s:<18.2f}")
    print(f"{'平均 dispatch 加速':<20}{f'{avg_spd:.2f}×':<18}{'—':<18}")
    print(f"\n结论：subagent 并行把 N 个独立子任务的墙钟从 sum 压到 ≈max。")
    print(f"注意：总墙钟加速一般小于 dispatch 加速——主 agent 的规划+综合是串行段(Amdahl 定律)。")

    out = {"summary": {"avg_parallel_s": round(avg_p, 2),
                       "avg_serial_s": round(avg_s, 2),
                       "avg_speedup": round(avg_spd, 2)},
           "details": results}
    out_path = Path(__file__).parent.parent / "outputs" / "eval_compare.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已写入 {out_path}")


if __name__ == "__main__":
    main()

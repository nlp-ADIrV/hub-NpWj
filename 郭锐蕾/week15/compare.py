"""Parallel vs Serial A/B 对比

跑同一问题两遍：一遍并行（ThreadPoolExecutor），一遍串行（for 循环），
量化 subagent 并行的墙钟加速。

诚实教学点：并行收益只在可并行的子任务部分，主 agent 自身的规划/综合
串行段不并行化，拉低总加速比（Amdahl 定律）。

用法：
  python compare.py "你的调研问题"
  python compare.py                # 跑内置示例
"""
import time
import logging
import argparse

from orchestrator import run_research

logging.basicConfig(level=logging.WARNING)


def run_one(question: str, serial: bool) -> dict:
    mode = "串行" if serial else "并行"
    print(f"\n>>> 开始 {mode} 执行 ...")
    t0 = time.time()
    r = run_research(question, serial=serial)
    total = round(time.time() - t0, 2)
    print(f"<<< {mode} 完成，端到端墙钟 {total}s")
    r["total_wall"] = total
    return r


def main():
    parser = argparse.ArgumentParser(description="Parallel vs Serial A/B 对比")
    parser.add_argument("question", nargs="?", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    question = args.question or "2023中国咖啡市场调研：市场规模、主要品牌、消费趋势"

    print("=" * 60)
    print("  Parallel vs Serial A/B 对比")
    print("=" * 60)
    print(f"问题: {question}")

    r_par = run_one(question, serial=False)
    r_ser = run_one(question, serial=True)

    # 汇总对比表
    print("\n" + "=" * 60)
    print("  A/B 结果对比")
    print("=" * 60)
    print(f"{'指标':<20} {'并行':>15} {'串行':>15}")
    print("-" * 60)

    par_stat = r_par["parallel_stats"][0] if r_par["parallel_stats"] else {}
    ser_stat = r_ser["parallel_stats"][0] if r_ser["parallel_stats"] else {}

    def fmt(v):
        return str(v) if v is not None and v != "" else "—"

    print(f"{'subagent 数':<20} {fmt(par_stat.get('n_subagents')):>15} "
          f"{fmt(ser_stat.get('n_subagents')):>15}")
    print(f"{'dispatch 墙钟(s)':<20} {fmt(par_stat.get('wall_clock')):>15} "
          f"{fmt(ser_stat.get('wall_clock')):>15}")
    print(f"{'serial_sum(s)':<20} {fmt(par_stat.get('serial_sum')):>15} "
          f"{fmt(ser_stat.get('serial_sum')):>15}")
    print(f"{'dispatch 加速(×)':<20} {fmt(par_stat.get('speedup')):>15} "
          f"{fmt(ser_stat.get('speedup')):>15}")
    print(f"{'端到端墙钟(s)':<20} {fmt(r_par.get('total_wall')):>15} "
          f"{fmt(r_ser.get('total_wall')):>15}")

    print("\n" + "-" * 60)
    if par_stat.get("wall_clock") and ser_stat.get("wall_clock"):
        print(f"  dispatch 加速: {par_stat['speedup']}×  "
              f"(并行墙钟 {par_stat['wall_clock']}s vs 串行墙钟 "
              f"{ser_stat['wall_clock']}s)")
    if r_par.get("total_wall") and r_ser.get("total_wall"):
        e2e = round(r_ser["total_wall"] / r_par["total_wall"], 2)
        print(f"  端到端加速: {e2e}×  "
              f"(并行总 {r_par['total_wall']}s vs 串行总 {r_ser['total_wall']}s)")
        print(f"\n  注：端到端加速 < dispatch 加速，因为主 agent 的规划/综合段"
              f"是串行的（Amdahl 定律）")


if __name__ == "__main__":
    main()

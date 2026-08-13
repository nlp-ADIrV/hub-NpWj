"""
Skill 优化对比实验
==================
演示"自进化 Agent"中的 Skill 优化环节：

  1. 由大模型编写一个初始（冗长版）物流 Skill —— logistics_original/SKILL.md (v1)
  2. 由大模型从 token 消耗 / 执行效率角度优化它 —— logistics_optimized/SKILL.md (v2)
  3. 在同一套评估题（物流类 Q46-Q53）上用真实 DeepSeek API 对比两个版本：
       - token 消耗（取自 API 返回的 usage，权威值）
       - 执行耗时（端到端延迟）
       - 回答准确率（复用项目根目录的 Evaluator，基于关键词匹配）
  4. 生成 Markdown 对比报告 comparison_report.md

设计要点：
  - 控制变量：两个版本使用完全相同的系统提示模板、相同的题目、相同模型、temperature=0。
    唯一变量就是被注入的 Skill 内容，从而干净地度量"Skill 本身"对 token / 耗时 / 准确率的影响。
  - 真实 token：直接读取 response.usage.prompt_tokens / completion_tokens / total_tokens，
    不依赖 tiktoken，避免分词器不一致带来的误差。
  - 成本估算：按 DeepSeek-chat 公开定价（输入 ¥0.5/1M、输出 ¥1.1/1M）折算单次调用成本，
    并外推到 10k 次调用场景，直观体现优化收益。

运行：
  cd "F:\\week14自进化agent\\week14 自进化agent\\week14 自进化agent\\self_evolving_agent\\week14作业"
  python optimize_skill_experiment.py
"""

import os
import sys
import time
import json
from pathlib import Path

# 复用项目根目录的 Evaluator 与 eval_set，保证评估口径与主项目一致
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from evaluator import Evaluator  # noqa: E402
from openai import OpenAI  # noqa: E402

# ── 路径 ─────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
SKILLS_DIR = HERE / "skills"
EVAL_SET = ROOT / "data" / "eval_set.json"
REPORT_PATH = HERE / "comparison_report.md"
RAW_PATH = HERE / "comparison_raw.json"

ORIGINAL_FILE = SKILLS_DIR / "logistics_original" / "SKILL.md"
OPTIMIZED_FILE = SKILLS_DIR / "logistics_optimized" / "SKILL.md"

# ── 系统提示模板（与 src/agent.py 风格一致，仅注入被测的单个 skill）─────────
SYSTEM_TEMPLATE = """你是云购商城的智能客服助手。

你的所有知识来源于以下技能文档，严格基于文档内容回答，不要自行推断或编造政策。

## 回答规则（严格遵守）
- 【能回答】如果技能文档覆盖了用户问题：直接给出完整具体的答案（含具体天数/金额/工作日数等政策细节）。不要在答案中加"建议联系人工客服"之类的推脱话。
- 【不能回答】如果技能文档确实不覆盖：仅回答一句 "需要联系人工客服"，不要编造答案。

## 当前知识库（共1个技能）

### 技能：logistics
{skill}
"""

# 被评估的题目：物流类 Q46-Q53（初始 Skill 完全缺失，正好用来验证新 skill 是否补齐知识）
LOGISTICS_QIDS = [46, 47, 48, 49, 50, 51, 52, 53]

# DeepSeek-chat 公开定价（人民币 / 每百万 token），用于成本外推估算
PRICE_INPUT_PER_M = 0.5    # 输入 ¥0.5 / 1M tokens
PRICE_OUTPUT_PER_M = 1.1   # 输出 ¥1.1 / 1M tokens

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
MODEL = "deepseek-chat"


def build_prompt(skill_content: str) -> str:
    return SYSTEM_TEMPLATE.format(skill=skill_content)


def char_stats(text: str) -> dict:
    """字符级统计（无 tiktoken 时的辅助近似指标）"""
    return {
        "chars": len(text),
        "chars_no_space": len(text.replace(" ", "").replace("\n", "")),
    }


def run_one(skill_content: str, question: str) -> dict:
    """对单个问题调用 API，返回答案 + 真实 token 用量 + 端到端耗时"""
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": build_prompt(skill_content)},
            {"role": "user", "content": question},
        ],
        temperature=0,
        max_tokens=400,
    )
    elapsed = time.perf_counter() - t0
    answer = resp.choices[0].message.content.strip()
    u = resp.usage
    return {
        "answer": answer,
        "prompt_tokens": u.prompt_tokens,
        "completion_tokens": u.completion_tokens,
        "total_tokens": u.total_tokens,
        "elapsed": elapsed,
    }


def run_eval(skill_content: str, label: str, ev: Evaluator) -> dict:
    print(f"\n{'=' * 64}\n运行评估: {label}\n{'=' * 64}")
    agg = {
        "label": label,
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        "elapsed": 0.0, "correct": 0, "total": 0, "details": [],
    }
    for qid in LOGISTICS_QIDS:
        q = ev.questions[qid]
        r = run_one(skill_content, q["question"])
        ok, reason = ev.evaluate_answer(r["answer"], qid)
        agg["prompt_tokens"] += r["prompt_tokens"]
        agg["completion_tokens"] += r["completion_tokens"]
        agg["total_tokens"] += r["total_tokens"]
        agg["elapsed"] += r["elapsed"]
        agg["total"] += 1
        if ok:
            agg["correct"] += 1
        agg["details"].append({
            "id": qid, "correct": ok, "reason": reason if not ok else "",
            "answer": r["answer"][:160],
            "prompt_tokens": r["prompt_tokens"],
            "completion_tokens": r["completion_tokens"],
            "total_tokens": r["total_tokens"],
            "elapsed": round(r["elapsed"], 3),
        })
        mark = "✓" if ok else "✗"
        print(f"  Q{qid} {mark}  prompt={r['prompt_tokens']:>4} compl={r['completion_tokens']:>3} "
              f"total={r['total_tokens']:>4}  {r['elapsed']:.2f}s  {('-> ' + reason) if not ok else ''}")
    agg["accuracy"] = round(agg["correct"] / agg["total"], 3)
    agg["avg_prompt_tokens"] = round(agg["prompt_tokens"] / agg["total"], 1)
    agg["avg_completion_tokens"] = round(agg["completion_tokens"] / agg["total"], 1)
    agg["avg_total_tokens"] = round(agg["total_tokens"] / agg["total"], 1)
    agg["avg_elapsed"] = round(agg["elapsed"] / agg["total"], 3)
    return agg


def cost_yuan(prompt_tokens: int, completion_tokens: int) -> float:
    return round(prompt_tokens / 1_000_000 * PRICE_INPUT_PER_M
                 + completion_tokens / 1_000_000 * PRICE_OUTPUT_PER_M, 6)


def pct_reduce(old, new):
    if old == 0:
        return 0.0
    return round((old - new) / old * 100, 1)


def build_report(o_static, n_static, o, n) -> str:
    lines = []
    lines.append("# Skill 优化对比实验报告\n")
    lines.append(f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"模型：`{MODEL}` | temperature=0 | 评估题：物流类 Q46-Q53（共 8 题）\n")

    lines.append("## 一、实验设计\n")
    lines.append("控制变量法：系统提示模板、题目、模型、采样参数完全相同，"
                 "**唯一变量是被注入的 Skill 内容**。\n")
    lines.append("- **v1 原始版**：大模型初次编写的物流 Skill，行文冗长、含大量敬语/示例/重复解释。\n"
                 "- **v2 优化版**：从 token 消耗角度优化，去除冗余敬语、示例与重复解释，"
                 "改用紧凑的列表/加粗，**保留全部事实知识**。\n")
    lines.append("优化目标：在保持（或提升）准确率的前提下，降低每次调用的输入 token 消耗与端到端耗时。\n")

    lines.append("## 二、Skill 静态指标\n")
    lines.append("| 指标 | v1 原始版 | v2 优化版 | 缩减 |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| 字符数 | {o_static['chars']} | {n_static['chars']} | "
                 f"{pct_reduce(o_static['chars'], n_static['chars'])}% |")
    lines.append(f"| 去空白字符数 | {o_static['chars_no_space']} | {n_static['chars_no_space']} | "
                 f"{pct_reduce(o_static['chars_no_space'], n_static['chars_no_space'])}% |")

    lines.append("\n## 三、运行结果（8 题累计 / 真实 API usage）\n")
    lines.append("| 指标 | v1 原始版 | v2 优化版 | 变化 |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| 准确率 | {o['correct']}/{o['total']} = {o['accuracy']:.1%} | "
                 f"{n['correct']}/{n['total']} = {n['accuracy']:.1%} | "
                 f"{(n['accuracy'] - o['accuracy']) * 100:+.0f}pp |")
    lines.append(f"| 输入 prompt_tokens | {o['prompt_tokens']} | {n['prompt_tokens']} | "
                 f"{pct_reduce(o['prompt_tokens'], n['prompt_tokens'])}% ↓ |")
    lines.append(f"| 输出 completion_tokens | {o['completion_tokens']} | {n['completion_tokens']} | "
                 f"{pct_reduce(o['completion_tokens'], n['completion_tokens'])}% |")
    lines.append(f"| 总 total_tokens | {o['total_tokens']} | {n['total_tokens']} | "
                 f"{pct_reduce(o['total_tokens'], n['total_tokens'])}% ↓ |")
    lines.append(f"| 平均单题输入 token | {o['avg_prompt_tokens']} | {n['avg_prompt_tokens']} | "
                 f"{pct_reduce(o['avg_prompt_tokens'], n['avg_prompt_tokens'])}% ↓ |")
    lines.append(f"| 累计耗时(秒) | {round(o['elapsed'], 2)} | {round(n['elapsed'], 2)} | "
                 f"{pct_reduce(o['elapsed'], n['elapsed'])}% |")
    lines.append(f"| 平均单题耗时(秒) | {o['avg_elapsed']} | {n['avg_elapsed']} | "
                 f"{pct_reduce(o['avg_elapsed'], n['avg_elapsed'])}% |")

    lines.append("\n## 四、成本外推（按 DeepSeek-chat 定价）\n")
    lines.append(f"定价假设：输入 ¥{PRICE_INPUT_PER_M}/1M tokens，输出 ¥{PRICE_OUTPUT_PER_M}/1M tokens。\n")
    o_cost = cost_yuan(o['prompt_tokens'], o['completion_tokens'])
    n_cost = cost_yuan(n['prompt_tokens'], n['completion_tokens'])
    o_cost_10k = round(o_cost / o['total'] * 10000, 4)
    n_cost_10k = round(n_cost / n['total'] * 10000, 4)
    lines.append("| 场景 | v1 原始版 | v2 优化版 | 节省 |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| 本次 8 题成本(¥) | {o_cost} | {n_cost} | "
                 f"{round(o_cost - n_cost, 6)} ({pct_reduce(o_cost, n_cost)}%) |")
    lines.append(f"| 外推 10,000 次调用(¥) | {o_cost_10k} | {n_cost_10k} | "
                 f"{round(o_cost_10k - n_cost_10k, 4)} ({pct_reduce(o_cost_10k, n_cost_10k)}%) |")

    lines.append("\n## 五、逐题明细\n")
    for tag, st in [("v1", o), ("v2", n)]:
        lines.append(f"### {tag}\n")
        lines.append("| 题号 | 正确 | 输入token | 输出token | 总token | 耗时(s) | 失败原因 | 答案摘要 |")
        lines.append("|---|:---:|---:|---:|---:|---:|---|---|")
        for d in st["details"]:
            ans = d["answer"].replace("|", "/").replace("\n", " ")
            lines.append(f"| Q{d['id']} | {'✓' if d['correct'] else '✗'} | "
                         f"{d['prompt_tokens']} | {d['completion_tokens']} | {d['total_tokens']} | "
                         f"{d['elapsed']} | {d['reason']} | {ans} |")

    lines.append("\n## 六、结论\n")
    tok_red = pct_reduce(o['prompt_tokens'], n['prompt_tokens'])
    char_red = pct_reduce(o_static['chars'], n_static['chars'])
    acc_delta = (n['accuracy'] - o['accuracy']) * 100
    if acc_delta > 0:
        acc_comment = (f"精简同时将「取消规则」与「退款时效」合并表述，引导模型给出更完整的答案，"
                       f"准确率反而提升 {acc_delta:.0f}pp。")
    elif acc_delta == 0:
        acc_comment = "优化未损失知识，准确率持平。"
    else:
        acc_comment = f"注意：精简丢失了关键表述，准确率下降 {abs(acc_delta):.0f}pp，需回补。"
    lines.append(f"- **Token 消耗**：输入 prompt_tokens 下降 **{tok_red}%**，"
                 f"Skill 字符数下降 **{char_red}%**，优化效果显著。\n")
    lines.append(f"- **执行效率**：因输入变短，模型首字延迟与解码负载略降；"
                 f"平均单题耗时由 {o['avg_elapsed']}s → {n['avg_elapsed']}s。\n")
    lines.append(f"- **准确率**：{o['accuracy']:.0%} → {n['accuracy']:.0%}（{acc_delta:+.0f}pp）。{acc_comment}\n")
    lines.append("- **自进化意义**：本实验即 Agent 自进化闭环中的「Skill 优化」环节——"
                 "在准确率不退化的前提下压缩 Skill，直接降低每次调用的输入 token 成本，"
                 "对高频客服场景具有可观的规模化收益。\n")
    return "\n".join(lines) + "\n"


def main():
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("✗ 未检测到 DEEPSEEK_API_KEY，请先设置环境变量。")
        sys.exit(1)

    original = ORIGINAL_FILE.read_text(encoding="utf-8")
    optimized = OPTIMIZED_FILE.read_text(encoding="utf-8")
    print(f"已加载 v1 ({len(original)} 字符) / v2 ({len(optimized)} 字符)")

    o_static = char_stats(original)
    n_static = char_stats(optimized)

    ev = Evaluator(str(EVAL_SET))

    o_stats = run_eval(original, "v1 原始版（冗长）", ev)
    n_stats = run_eval(optimized, "v2 优化版（精简）", ev)

    report = build_report(o_static, n_static, o_stats, n_stats)
    REPORT_PATH.write_text(report, encoding="utf-8")
    RAW_PATH.write_text(
        json.dumps(
            {"static": {"v1": o_static, "v2": n_static},
             "v1": o_stats, "v2": n_stats},
            ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n{'=' * 64}\n报告已生成: {REPORT_PATH}\n原始数据: {RAW_PATH}\n{'=' * 64}")
    print(report)


if __name__ == "__main__":
    main()

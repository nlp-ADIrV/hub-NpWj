#!/usr/bin/env python3
"""
会议纪要格式化辅助脚本
将非结构化的会议记录文本提取关键要素，输出标准 Markdown 格式纪要。

用法：
    python format_notes.py <输入文件> [--output <输出文件>] [--type <会议类型>]

会议类型：general(通用) / weekly(周会) / review(评审) / brainstorm(头脑风暴)
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path


# 口语化噪音词，整理时剔除
FILLER_WORDS = [
    "嗯", "啊", "那个", "这个", "就是说", "然后呢", "对吧",
    "怎么说呢", "基本上", "其实吧", "就是说吧", "懂吧",
]


def clean_text(text: str) -> str:
    """剔除口语化噪音词，压缩多余空白。"""
    cleaned = text
    for word in FILLER_WORDS:
        cleaned = cleaned.replace(word, "")
    # 压缩连续空白
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def extract_todos(text: str) -> list[dict]:
    """
    从文本中启发式提取待办事项。
    匹配模式：责任人 + 动词 + 任务 + (可选)时间
    """
    todos = []
    # 匹配 "张三负责..." "由李四跟进..." "王五要在周五前..."
    patterns = [
        r"([\u4e00-\u9fa5]{2,4})\s*(?:负责|跟进|完成|落实|推进|确认|整理|输出|对接)",
        r"(?:由|让|请|安排)\s*([\u4e00-\u9fa5]{2,4})\s*",
    ]
    # 时间提取
    time_patterns = [
        r"(\d{4}[-/年]\d{1,2}[-/月]\d{1,2})",
        r"(本周|下周|这周|月底|下周[一二三四五六日天])",
        r"(\d{1,2}月\d{1,2}[日号])",
    ]

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        owner = None
        for pat in patterns:
            m = re.search(pat, line)
            if m:
                owner = m.group(1)
                break
        if not owner:
            continue
        deadline = None
        for pat in time_patterns:
            m = re.search(pat, line)
            if m:
                deadline = m.group(1)
                break
        todos.append({
            "task": clean_text(line),
            "owner": owner,
            "deadline": deadline or "[待确认]",
        })
    return todos


def extract_decisions(text: str) -> list[str]:
    """提取决议语句：包含'决定/同意/确认/通过'等关键词的句子。"""
    keywords = ["决定", "同意", "确认", "通过", "一致认为", "达成共识", "敲定"]
    decisions = []
    for line in text.splitlines():
        line = clean_text(line)
        if any(kw in line for kw in keywords):
            decisions.append(line)
    return decisions


def render_general(title, raw_text, todos, decisions):
    """渲染通用会议纪要模板。"""
    today = datetime.now().strftime("%Y-%m-%d")
    lines = [
        f"# 会议纪要：{title}",
        "",
        f"**时间**：{today}",
        "**地点**：[待填写]",
        "**主持人**：[待填写]",
        "**记录人**：[待填写]",
        "**参会人员**：[待填写]",
        "",
        "---",
        "",
        "## 讨论要点",
        "",
    ]
    # 把清理后的原文按行作为要点（粗糙处理，实际应人工校对）
    for line in raw_text.splitlines():
        line = clean_text(line)
        if line and len(line) > 5:
            lines.append(f"- {line}")
    lines.append("")
    if decisions:
        lines.append("## 决议")
        lines.append("")
        for d in decisions:
            lines.append(f"> {d}")
        lines.append("")
    if todos:
        lines.append("## 待办事项")
        lines.append("")
        for i, t in enumerate(todos, 1):
            lines.append(f"- [ ] {t['task']} — **{t['owner']}** — {t['deadline']}")
        lines.append("")
        lines.append("| 序号 | 任务 | 责任人 | 截止时间 | 状态 |")
        lines.append("|------|------|--------|----------|------|")
        for i, t in enumerate(todos, 1):
            lines.append(f"| {i} | {t['task'][:20]} | {t['owner']} | {t['deadline']} | 待开始 |")
        lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("> ⚠️ 本纪要由脚本初稿生成，请人工校对后发送。")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="会议纪要格式化辅助工具")
    parser.add_argument("input", help="输入文件路径（会议记录文本）")
    parser.add_argument("--output", "-o", help="输出文件路径，默认输出到 stdout")
    parser.add_argument(
        "--type", "-t", default="general",
        choices=["general", "weekly", "review", "brainstorm"],
        help="会议类型，默认 general"
    )
    parser.add_argument("--title", default="未命名会议", help="会议标题")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误：文件不存在 {input_path}", file=sys.stderr)
        sys.exit(1)

    raw_text = input_path.read_text(encoding="utf-8")
    cleaned = clean_text(raw_text)
    todos = extract_todos(cleaned)
    decisions = extract_decisions(cleaned)

    # 目前仅实现通用模板渲染，其他类型可按需扩展
    if args.type == "general":
        output = render_general(args.title, raw_text, todos, decisions)
    else:
        # 其他类型暂回退到通用模板
        output = render_general(args.title, raw_text, todos, decisions)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"✅ 纪要已生成：{args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()

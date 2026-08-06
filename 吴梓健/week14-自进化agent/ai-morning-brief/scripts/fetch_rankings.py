#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AI排行榜数据抓取脚本(零依赖, 仅标准库)

抓取两个数据源并输出结构化 rankings.json 供 generate_report.py 使用:
  1. LMArena 镜像站 arena.atease.dev (文本/代码/Agent等11个分类, 各取Top10)
  2. Artificial Analysis Intelligence Index (经 benchlm.ai 镜像, Top 20)

用法:
    python fetch_rankings.py [-o rankings.json] [--top 10]
"""
import argparse
import json
import re
import sys
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

CATEGORIES = [
    ("text", "LMArena 文本对话榜"),
    ("code", "LMArena 代码生成榜"),
    ("agent", "LMArena Agent 榜"),
    ("vision", "LMArena 视觉多模态榜"),
    ("search", "LMArena 搜索增强榜"),
    ("document", "LMArena 文档理解榜"),
    ("text-to-image", "LMArena 文生图榜"),
    ("text-to-video", "LMArena 文生视频榜"),
    ("image-to-video", "LMArena 图生视频榜"),
    ("image-edit", "LMArena 图像编辑榜"),
    ("video-edit", "LMArena 视频编辑榜"),
]

ARENA_BASE = "https://arena.atease.dev"
BENCHLM_URL = "https://benchlm.ai/benchmarks/artificialanalysis"


def _get(url):
    req = urllib.request.Request(url, headers=UA)
    return urllib.request.urlopen(req, timeout=30).read()


def fetch_arena(cat, title, top):
    data = json.loads(_get("{}/data/{}.json".format(ARENA_BASE, cat)))
    meta = data.get("meta", {})
    models = []
    for m in data.get("models", [])[:top]:
        item = {
            "rank": m.get("rank"),
            "range": m.get("rankSpread"),
            "name": m.get("model"),
            "org": m.get("vendor"),
            "score": m.get("score_val"),
            "ci": m.get("score_ci"),
            "votes": m.get("votes_val"),
        }
        lic = m.get("license") or ""
        if lic and lic != "proprietary":
            item["license"] = lic
        models.append(item)
    return {
        "title": "{} · Top {}".format(title, min(top, len(models))),
        "source": "LMArena（镜像 arena.atease.dev）",
        "url": meta.get("source_url") or ARENA_BASE,
        "as_of": meta.get("last_updated", ""),
        "models": models,
    }


def fetch_aa(top):
    html = _get(BENCHLM_URL).decode("utf-8", "ignore")
    i = html.find("Benchmark score table")
    if i < 0:
        raise ValueError("未找到 AA 指数表格")
    seg = html[i:]
    row_pat = re.compile(
        r'class="w-6[^"]*">\s*(\d+)\s*</span>'
        r'.*?href="/models/[^"]*">\s*([^<]+?)\s*</a>'
        r'.*?text-muted-foreground">\s*([^<]+?)\s*(?:<!-- -->)'
    )
    score_pat = re.compile(r'text-foreground">\s*([\d.]+%)')
    models = []
    for m in row_pat.finditer(seg):
        sm = score_pat.search(seg, m.end())
        if not sm:
            break
        models.append(
            {
                "rank": int(m.group(1)),
                "name": m.group(2).strip(),
                "org": m.group(3).strip(),
                "score": sm.group(1),
            }
        )
        if len(models) >= top:
            break
    dm = re.search(r'dateTime="(\d{4}-\d{2}-\d{2})"', html)
    return {
        "title": "Artificial Analysis 智能指数 · Top {}".format(len(models)),
        "source": "Artificial Analysis（经 BenchLM 镜像）",
        "url": BENCHLM_URL,
        "as_of": dm.group(1) if dm else "",
        "models": models,
    }


def main():
    ap = argparse.ArgumentParser(description="AI排行榜抓取")
    ap.add_argument("-o", "--output", default="rankings.json")
    ap.add_argument("--top", type=int, default=10, help="LMArena 每榜取前N名")
    args = ap.parse_args()

    rankings, errs = [], []
    try:
        rankings.append(fetch_aa(20))
    except Exception as e:
        errs.append("AA指数: {}".format(e))
    for cat, title in CATEGORIES:
        try:
            rankings.append(fetch_arena(cat, title, args.top))
        except Exception as e:
            errs.append("{}: {}".format(cat, e))

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"rankings": rankings}, f, ensure_ascii=False, indent=1)

    print("OK: {} 个榜单 -> {}".format(len(rankings), args.output))
    for e in errs:
        print("WARN " + e, file=sys.stderr)
    if not rankings:
        sys.exit("错误: 所有榜单抓取失败")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AI早报 HTML 生成脚本

用法:
    python generate_report.py <news.json> [-o 输出路径] [--title 标题] [--rankings rankings.json]

榜单数据加载优先级: --rankings 参数 > 当前目录 rankings.json > news.json 内联字段。
rankings.json 由 fetch_rankings.py 抓取生成, 结构:
{
  "rankings": [
    {"title": "榜单标题", "source": "来源", "url": "来源链接", "as_of": "2026-08-04",
     "models": [{"rank": 1, "range": "1 4", "name": "模型名", "org": "厂商",
                 "score": 1509, "ci": 6, "votes": 17799, "license": "open weight"}]}
  ]
}
(range/ci/votes/license 可选; score 为数字时渲染进度条)

新闻 JSON 格式:
{
  "date": "2026-08-04",          # 可选, 默认今天
  "news": [
    {
      "title": "新闻标题",
      "summary": "中文详情摘要",
      "url": "https://原文地址",
      "source": "来源媒体名",     # 可选
      "category": "模型发布",      # 可选分类
      "score": 9                   # 重磅程度 1-10, 用于排序和徽章
    }
  ]
}
仅依赖 Python 标准库。
"""
import argparse
import datetime
import html
import json
import os
import sys


def tier_of(score):
    """根据分数返回 (徽章文字, CSS 类名)"""
    if score >= 9:
        return "重磅", "tier-hot"
    if score >= 7:
        return "重要", "tier-important"
    return "关注", "tier-normal"


def render_card(idx, item):
    title = html.escape(str(item.get("title", "无标题")))
    summary = html.escape(str(item.get("summary", "")))
    url = html.escape(str(item.get("url", "#")), quote=True)
    source = html.escape(str(item.get("source", "")))
    category = html.escape(str(item.get("category", "")))
    score = item.get("score", 5)
    try:
        score = int(score)
    except (TypeError, ValueError):
        score = 5
    tier_text, tier_cls = tier_of(score)

    badges = ['<span class="badge {}">{}</span>'.format(tier_cls, tier_text)]
    if category:
        badges.append('<span class="badge badge-cat">{}</span>'.format(category))
    badge_html = "".join(badges)

    source_html = ""
    if source:
        source_html = (
            '<a class="source" href="{}" target="_blank" rel="noopener">'
            "来源：{} &rarr;</a>".format(url, source)
        )

    return """
    <article class="card">
      <div class="card-rank">{:02d}</div>
      <div class="card-body">
        <div class="badges">{}</div>
        <h2 class="card-title">
          <a href="{}" target="_blank" rel="noopener">{}</a>
        </h2>
        <p class="card-summary">{}</p>
        {}
      </div>
    </article>""".format(idx, badge_html, url, title, summary, source_html)


def render_leaderboard(ranking):
    """渲染 AI 排行榜模块；无数据时返回空字符串。

    表格列按数据动态显示:
    排名 | 排名区间 | 模型 | 得分(带进度条) | 95%置信区间 | 投票数 | 开发厂商
    """
    if not isinstance(ranking, dict) or not ranking.get("models"):
        return ""
    title = html.escape(str(ranking.get("title", "AI 排行榜 · 实时排名")))
    source = html.escape(str(ranking.get("source", "")))
    url = html.escape(str(ranking.get("url", "#")), quote=True)
    as_of = html.escape(str(ranking.get("as_of", "")))
    models = ranking["models"][:20]

    has_range = any(m.get("range") for m in models)
    has_ci = any(m.get("ci") is not None for m in models)
    has_votes = any(m.get("votes") is not None for m in models)

    # 数字得分计算进度条宽度(相对本榜最高/最低分)
    nums = []
    for m in models:
        s = m.get("score")
        if isinstance(s, (int, float)):
            nums.append(float(s))
    lo, hi = (min(nums), max(nums)) if nums else (0.0, 1.0)
    span = (hi - lo) or 1.0

    head = "<th>排名</th>"
    if has_range:
        head += "<th>排名区间</th>"
    head += "<th>模型</th><th>得分</th>"
    if has_ci:
        head += "<th style='text-align:center'>95%置信区间</th>"
    if has_votes:
        head += "<th style='text-align:right'>投票数</th>"
    head += "<th>开发厂商</th>"

    rows = ""
    for m in models:
        try:
            rank = int(m.get("rank", 0))
        except (TypeError, ValueError):
            rank = 0
        medal = {1: "medal-1", 2: "medal-2", 3: "medal-3"}.get(rank, "")

        lic = m.get("license")
        lic_html = '<span class="lb-lic">开源</span>' if lic else ""
        name_cell = "{}{}".format(html.escape(str(m.get("name", ""))), lic_html)

        s = m.get("score")
        if isinstance(s, (int, float)):
            width = 30 + 70 * (float(s) - lo) / span
            score_cell = (
                '<span class="lb-score-num">{}</span>'
                '<span class="lb-bar"><i style="width:{:.1f}%"></i></span>'
            ).format(html.escape(str(s)), width)
        else:
            score_cell = '<span class="lb-score-num">{}</span>'.format(
                html.escape(str(s if s is not None else ""))
            )

        rows += '<tr><td class="lb-rank {}">{}</td>'.format(medal, rank)
        if has_range:
            rng = html.escape(str(m.get("range", "") or ""))
            rows += '<td class="lb-range">{}</td>'.format(rng)
        rows += '<td class="lb-name">{}</td><td class="lb-score">{}</td>'.format(
            name_cell, score_cell
        )
        if has_ci:
            ci = m.get("ci")
            ci_text = "&plusmn;{}".format(ci) if ci is not None else ""
            rows += '<td class="lb-ci">{}</td>'.format(ci_text)
        if has_votes:
            v = m.get("votes")
            v_text = "{:,}".format(v) if isinstance(v, (int, float)) else ""
            rows += '<td class="lb-votes">{}</td>'.format(v_text)
        rows += '<td class="lb-org">{}</td></tr>'.format(
            html.escape(str(m.get("org", "")))
        )

    source_link = ""
    if source:
        source_link = (
            '<a class="lb-source" href="{}" target="_blank" rel="noopener">'
            "数据来源：{} &rarr;</a>".format(url, source)
        )
    as_of_html = '<span class="lb-asof">更新于 {}</span>'.format(as_of) if as_of else ""

    return """
  <section class="leaderboard">
    <div class="lb-header">
      <h2>{}</h2>
      <div class="lb-meta">{}{}</div>
    </div>
    <div class="lb-scroll">
    <table class="lb-table">
      <thead><tr>{}</tr></thead>
      <tbody>{}</tbody>
    </table>
    </div>
  </section>""".format(title, as_of_html, source_link, head, rows)


CSS = """
:root {
  --bg: #f5f6fa;
  --card-bg: #ffffff;
  --ink: #1f2330;
  --muted: #6b7280;
  --accent: #2455d3;
  --hot: #d43a3a;
  --important: #e07b12;
  --normal: #2f7fd1;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: "PingFang SC", "Microsoft YaHei", "Segoe UI", sans-serif;
  background: var(--bg);
  color: var(--ink);
  line-height: 1.65;
}
.wrap { max-width: 860px; margin: 0 auto; padding: 32px 20px 60px; }
header.hero {
  background: linear-gradient(135deg, #1b2a5e 0%, #2455d3 60%, #3a7bff 100%);
  color: #fff;
  border-radius: 16px;
  padding: 36px 32px;
  margin-bottom: 28px;
}
header.hero h1 { font-size: 28px; letter-spacing: 1px; }
header.hero .meta { margin-top: 10px; font-size: 14px; opacity: .85; }
.tabs {
  display: flex;
  gap: 10px;
  margin-bottom: 22px;
  position: sticky;
  top: 0;
  z-index: 10;
  padding: 10px 0;
  background: var(--bg);
}
.tab-btn {
  flex: 1;
  padding: 12px 16px;
  font-size: 15px;
  font-weight: 600;
  border: 1px solid #dde2ee;
  border-radius: 10px;
  background: var(--card-bg);
  color: var(--muted);
  cursor: pointer;
  transition: all .15s ease;
  font-family: inherit;
}
.tab-btn:hover { border-color: var(--accent); color: var(--accent); }
.tab-btn.active {
  background: var(--accent);
  border-color: var(--accent);
  color: #fff;
  box-shadow: 0 3px 10px rgba(36, 85, 211, .3);
}
.tab-pane { display: none; }
.tab-pane.active { display: block; }
.card {
  display: flex;
  gap: 18px;
  background: var(--card-bg);
  border-radius: 12px;
  padding: 20px 22px;
  margin-bottom: 16px;
  box-shadow: 0 1px 3px rgba(20, 30, 60, .08);
  transition: transform .15s ease, box-shadow .15s ease;
}
.card:hover { transform: translateY(-2px); box-shadow: 0 6px 18px rgba(20, 30, 60, .12); }
.card-rank {
  font-size: 26px;
  font-weight: 700;
  color: #c3cadb;
  min-width: 44px;
  font-family: Georgia, serif;
}
.card-body { flex: 1; }
.badges { margin-bottom: 8px; }
.badge {
  display: inline-block;
  font-size: 12px;
  padding: 2px 10px;
  border-radius: 999px;
  color: #fff;
  margin-right: 8px;
  vertical-align: middle;
}
.tier-hot { background: var(--hot); }
.tier-important { background: var(--important); }
.tier-normal { background: var(--normal); }
.badge-cat { background: #eef1f8; color: #44507a; }
.card-title { font-size: 18px; font-weight: 600; margin-bottom: 6px; }
.card-title a { color: var(--ink); text-decoration: none; }
.card-title a:hover { color: var(--accent); text-decoration: underline; }
.card-summary { font-size: 14.5px; color: #3d4353; margin-bottom: 10px; }
.source { font-size: 13px; color: var(--accent); text-decoration: none; }
.source:hover { text-decoration: underline; }
.leaderboard {
  background: var(--card-bg);
  border-radius: 12px;
  padding: 22px 24px;
  margin-bottom: 24px;
  box-shadow: 0 1px 3px rgba(20, 30, 60, .08);
  border-top: 3px solid var(--accent);
}
.lb-header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 12px;
}
.lb-header h2 { font-size: 18px; color: var(--ink); }
.lb-meta { font-size: 13px; color: var(--muted); }
.lb-asof { margin-right: 12px; }
.lb-source { color: var(--accent); text-decoration: none; }
.lb-source:hover { text-decoration: underline; }
.lb-scroll { overflow-x: auto; }
table.lb-table { width: 100%; border-collapse: collapse; font-size: 14.5px; }
.lb-table th {
  text-align: left;
  color: var(--muted);
  font-weight: 500;
  font-size: 13px;
  padding: 6px 8px;
  border-bottom: 1px solid #e5e8f0;
}
.lb-table td { padding: 9px 8px; border-bottom: 1px solid #f0f2f7; }
.lb-table tbody tr:last-child td { border-bottom: none; }
.lb-rank { font-weight: 700; width: 44px; color: #8a91a5; }
.medal-1 { color: #d4a017; }
.medal-2 { color: #7d8595; }
.medal-3 { color: #b0713a; }
.lb-range { color: var(--muted); font-size: 13px; white-space: nowrap; }
.lb-name { font-weight: 600; white-space: nowrap; }
.lb-lic {
  display: inline-block;
  margin-left: 6px;
  font-size: 11px;
  font-weight: 500;
  color: #1a7f37;
  background: #e6f4ea;
  border-radius: 4px;
  padding: 1px 5px;
  vertical-align: 1px;
}
.lb-score { white-space: nowrap; }
.lb-score-num {
  color: var(--accent);
  font-weight: 700;
  font-variant-numeric: tabular-nums;
}
.lb-bar {
  display: inline-block;
  width: 72px;
  height: 6px;
  margin-left: 8px;
  background: #e8ecf6;
  border-radius: 999px;
  vertical-align: 2px;
  overflow: hidden;
}
.lb-bar i {
  display: block;
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, #3a7bff, #2455d3);
}
.lb-ci {
  text-align: center;
  color: var(--muted);
  font-size: 13px;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
.lb-votes {
  text-align: right;
  color: var(--muted);
  font-size: 13px;
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
.lb-org { color: var(--muted); white-space: nowrap; }
footer {
  margin-top: 36px;
  text-align: center;
  font-size: 13px;
  color: var(--muted);
}
@media (max-width: 560px) {
  .card { flex-direction: column; gap: 6px; }
  .card-rank { font-size: 18px; min-width: auto; }
}
"""


def load_rankings(cli_path, data):
    """榜单数据优先级: --rankings 参数 > 自动发现 rankings.json > JSON 内联字段"""
    if cli_path:
        with open(cli_path, "r", encoding="utf-8") as f:
            return json.load(f).get("rankings") or []
    if os.path.exists("rankings.json"):
        try:
            with open("rankings.json", "r", encoding="utf-8") as f:
                return json.load(f).get("rankings") or []
        except (json.JSONDecodeError, OSError):
            pass
    inline = data.get("rankings")
    if inline is None:
        legacy = data.get("ranking")
        inline = [legacy] if legacy else []
    return inline


def build_html(data, title, rankings):
    news = data.get("news", [])
    # 按 score 降序排序
    def _score(item):
        try:
            return int(item.get("score", 5))
        except (TypeError, ValueError):
            return 5

    news = sorted(news, key=_score, reverse=True)
    date_str = data.get("date") or datetime.date.today().isoformat()
    boards = [r for r in rankings if isinstance(r, dict) and r.get("models")]
    leaderboard = "".join(render_leaderboard(r) for r in boards)

    cards = "".join(render_card(i + 1, item) for i, item in enumerate(news))
    if not cards:
        cards = '<p style="text-align:center;color:#6b7280;">今日暂无收录新闻</p>'
    if not leaderboard:
        leaderboard = '<p style="text-align:center;color:#6b7280;">今日暂无榜单数据</p>'

    return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} - {date}</title>
<style>{css}</style>
</head>
<body>
<div class="wrap">
  <header class="hero">
    <h1>{title}</h1>
    <div class="meta">{date} &middot; 收录 {count} 条 AI 新闻 &middot; {boards} 个排行榜</div>
  </header>
  <div class="tabs">
    <button class="tab-btn active" data-tab="tab-news" onclick="showTab('tab-news')">今日要闻（{count}）</button>
    <button class="tab-btn" data-tab="tab-rankings" onclick="showTab('tab-rankings')">AI 排行榜（{boards}）</button>
  </div>
  <div id="tab-news" class="tab-pane active"><main>{cards}</main></div>
  <div id="tab-rankings" class="tab-pane">{leaderboard}</div>
  <footer>本早报由 AI 自动搜集整理生成 &middot; 点击标题可跳转原文</footer>
</div>
<script>
function showTab(id) {{
  document.querySelectorAll('.tab-pane').forEach(function (p) {{
    p.classList.toggle('active', p.id === id);
  }});
  document.querySelectorAll('.tab-btn').forEach(function (b) {{
    b.classList.toggle('active', b.dataset.tab === id);
  }});
}}
</script>
</body>
</html>""".format(
        title=html.escape(title),
        date=html.escape(date_str),
        count=len(news),
        boards=len(boards),
        css=CSS,
        leaderboard=leaderboard,
        cards=cards,
    )


def main():
    parser = argparse.ArgumentParser(description="AI早报 HTML 生成器")
    parser.add_argument("input", help="新闻数据 JSON 文件路径")
    parser.add_argument("-o", "--output", help="输出 HTML 路径 (默认 output/AI早报_日期.html)")
    parser.add_argument("--title", default="AI 新闻早报", help="页面标题")
    parser.add_argument("--rankings", help="榜单数据 JSON (默认自动读取 rankings.json)")
    args = parser.parse_args()

    try:
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        sys.exit("错误: 找不到输入文件 {}".format(args.input))
    except json.JSONDecodeError as e:
        sys.exit("错误: JSON 解析失败 - {}".format(e))

    if not isinstance(data, dict) or not isinstance(data.get("news"), list):
        sys.exit('错误: JSON 必须为 {"news": [...]} 结构')

    try:
        rankings = load_rankings(args.rankings, data)
    except (json.JSONDecodeError, OSError) as e:
        sys.exit("错误: 榜单数据读取失败 - {}".format(e))

    out = args.output
    if not out:
        date_str = data.get("date") or datetime.date.today().isoformat()
        out = os.path.join("output", "AI早报_{}.html".format(date_str.replace("-", "")))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    html_text = build_html(data, args.title, rankings)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html_text)

    print("OK: 已生成 {} ({} 条新闻, {} 个榜单)".format(out, len(data["news"]), len(rankings)))


if __name__ == "__main__":
    main()

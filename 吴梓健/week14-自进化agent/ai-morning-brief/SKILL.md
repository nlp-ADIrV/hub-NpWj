---
name: ai-morning-brief
description: 搜集最近24小时内的AI新闻，按重磅程度排序筛选，生成带原文链接的HTML早报页面。当用户要求制作AI早报、AI日报、AI新闻汇总、生成新闻报告页面，或提到"早报""日报""AI新闻汇总"时使用。
---

# AI 早报制作

搜集 1 天内的 AI 新闻 → 按重磅程度筛选排序 → 写入 JSON → 用 Python 脚本生成 HTML（不要在对话中生成 HTML 代码，节省 token）。

## 工作流程

复制以下清单并跟踪进度：

```
- [ ] 第1步：多轮搜索搜集新闻
- [ ] 第2步：去重、打分、筛选 Top 10-15
- [ ] 第3步：写入 news_data.json
- [ ] 第4步：运行脚本生成 HTML
- [ ] 第5步：向用户报告结果
```

## 第1步：搜集新闻

使用 `WebSearch` 工具，**timeRange 必须设为 `OneDay`**，执行多轮搜索覆盖中英文源。至少执行以下查询（可增补）：

- `AI news today major announcement`
- `OpenAI OR Anthropic OR Google DeepMind announcement`
- `AI model release OR AI funding OR AI regulation`
- `人工智能 最新消息 发布`
- `AI大模型 新闻 今天`

对结果中疑似重要但信息不完整的条目，可用 `WebFetch` 打开原文确认细节（标题、事实、发布时间、关键数据）。**重要新闻尽量抓取原文详情**，为后续撰写充分详情积累素材。丢弃超过 24 小时的旧闻和软文广告。

**采集 AI 排行榜实时排名（必做）**——已脚本化，一条命令完成，无需手动抓取网页：

```powershell
& "D:\ProgramData\miniconda3\python.exe" .qoder/skills/ai-morning-brief/scripts/fetch_rankings.py -o rankings.json
```

脚本零依赖（标准库 urllib），自动抓取并输出结构化 `rankings.json`：
- Artificial Analysis Intelligence Index Top 20（经 `benchlm.ai` 镜像，正则解析 HTML）
- LMArena 11 个分类各 Top 10（经国内可访问的镜像站 `arena.atease.dev` 的 `/data/<分类>.json` 接口）：文本、代码、Agent、视觉、搜索、文档、文生图、文生视频、图生视频、图像编辑、视频编辑

输出 `OK: 12 个榜单 -> rankings.json` 即成功；`WARN` 行表示个别榜单失败但不影响整体。**榜单数据不要写进 news_data.json**，generate_report.py 会自动读取 rankings.json。仅当脚本报错"所有榜单抓取失败"时，才回退用 `WebSearch`（timeRange `OneWeek`）查询 `Artificial Analysis intelligence index ranking` 与 `LMArena <分类> leaderboard top models` 手工补齐并写入 news_data.json 的 `rankings` 字段。

## 第2步：打分与筛选

为每条新闻打 `score`（1-10），标准：

| 分数 | 标准 | 示例 |
|------|------|------|
| 9-10 | 重磅 | 重大模型/产品发布、巨额融资并购、国家级AI政策、重大安全事件 |
| 7-8 | 重要 | 知名公司动态、开源模型更新、行业报告、人事变动 |
| 5-6 | 关注 | 一般产品更新、研究论文、小型融资 |

去重（同一事件多来源只保留一条，优先权威来源）。**只保留 score ≥ 5 的新闻，取 Top 10-15 条**。

`category` 从以下选取：模型发布 / 公司动态 / 政策监管 / 投融资 / 开源生态 / 研究突破 / 行业应用。

## 第3步：写入 JSON

用 Write 工具在工作区根目录创建 `news_data.json`，严格遵循此结构：

```json
{
  "date": "YYYY-MM-DD",
  "news": [
    {
      "title": "中文新闻标题（简洁有力）",
      "summary": "3-5句中文详情（约120-200字）",
      "url": "原文真实链接",
      "source": "来源媒体",
      "category": "模型发布",
      "score": 9
    }
  ]
}
```

**news_data.json 只写新闻**，榜单由 `rankings.json`（脚本生成）单独提供，避免重复占用 token。

**summary 写作要求**：很多原文链接是英文，读者不会点进去阅读，因此 summary 必须用中文**自包含地讲清新闻全貌**，覆盖三个层次：① 发生了什么（时间、主体、事件）；② 关键细节与数据（参数规模、金额、人名、政策条款等）；③ 影响或意义。禁止只写一两句话的概括。

注意：`url` 必须是搜索/抓取中实际获得的原文链接，禁止编造。

## 第4步：生成 HTML

运行（脚本零依赖，仅需 Python 标准库；本机 `python` 命令为商店占位符，须用 conda 解释器全路径）：

```powershell
& "D:\ProgramData\miniconda3\python.exe" .qoder/skills/ai-morning-brief/scripts/generate_report.py news_data.json
```

- 榜单数据自动从当前目录 `rankings.json` 读取（无需传参；也可用 `--rankings 路径` 显式指定）
- 页面分为「今日要闻」「AI 排行榜」两个选项卡；榜单表格含排名区间、得分进度条、95%置信区间、投票数、开发厂商等列，按数据自动增减
- 默认输出：`output/AI早报_YYYYMMDD.html`
- 可选参数：`-o 自定义路径.html`、`--title "自定义标题"`
- 输出 `OK: ...` 即成功

## 第5步：报告结果

告知用户：生成文件路径、收录条数、Top 3 重磅新闻标题。HTML 为自包含静态页面，用户直接双击即可在浏览器打开，**默认不要启动本地服务器**；仅当用户明确要求 IDE 内预览时才启动静态服务器并用 RunPreview 提供预览（用完即停）。

# Week13 作业：渐进式 Skill 加载执行 Harness

对齐课件《Skills》Part 3 **渐进式披露（Progressive Disclosure）**，实现一套可运行的 Skill Harness：

> 常驻索引（&lt; ~200 tokens）→ 触发匹配 → 按需加载完整 `SKILL.md` → 执行期二级加载 `references/` / 调用 `scripts/` → 任务完成释放

并用项目内真实 Skill（`flash-card`、`baoyu-diagram`）+ 课件示例 Skill（`pptx`、`code-review`）做端到端演示。

## 目录结构

```
week13作业/
├── ARCHITECTURE.md          # 设计说明（对照课件三层）
├── README.md
├── requirements.txt
├── index.html               # 可视化前端
├── src/
│   ├── registry.py          # 扫描 SKILL.md → 常驻索引
│   ├── matcher.py           # 触发匹配
│   ├── loader.py            # 渐进式加载 / 释放
│   ├── context.py           # Context 组装 + 全量对比
│   ├── executor.py          # demo / llm 执行
│   ├── harness.py           # 统一门面（生命周期）
│   ├── cli.py               # 命令行
│   └── serve.py             # FastAPI 服务
├── skills/                  # 可被扫描的 Skills
│   ├── flash-card/
│   ├── baoyu-diagram/
│   ├── pptx/
│   └── code-review/
├── demos/run_demo.py        # 批量演示
└── outputs/                 # 索引快照、闪卡 HTML、汇总 JSON
```

## 快速开始

```bash
cd week13作业
pip install -r requirements.txt

# 查看常驻索引
python -m src.cli --list

# 跑一轮（flash-card 会真实调用脚本生成 HTML）
python -m src.cli "给我做张 crazy 的闪卡"

# 架构图：触发 baoyu-diagram，并二级加载 architecture.md
python -m src.cli "画一个系统架构图"

# 批量演示
python demos/run_demo.py

# 可视化 UI
python -m src.serve
# 浏览器打开 http://127.0.0.1:8013
```

可选 LLM 模式（需 API Key）：

```bash
set LLM_PROVIDER=deepseek
set DEEPSEEK_API_KEY=sk-xxx
python -m src.cli "审查这段代码" --mode llm
```

## 生命周期（对照课件）

| 步骤 | 含义 | 实现 |
|------|------|------|
| 01 用户发消息 | 进入 harness | `SkillHarness.handle` |
| 02 触发条件匹配 | 关键词 / 名称 / 描述打分 | `TriggerMatcher` |
| 03 加载 Skill 定义 | 仅加载命中的 `SKILL.md` | `ProgressiveLoader.load_skill_body` |
| 04 执行 Skill 流程 | 二级 references + scripts | `SkillExecutor` |
| 05 完成 / 释放 | 卸载正文，恢复仅索引 | `ContextAssembler.release` |

## Token 对比

每轮结果都包含 `comparison`：

- `full_load_tokens`：把所有 Skill 正文 + 全部 references 一次性塞进 Context
- `progressive_tokens`：本轮实际注入（索引 + 命中 Skill + 按需 reference）
- `saved_ratio_percent`：节省比例（课件目标：大规模 Skill 库可省 60–90%）

## 设计要点

1. **索引常驻**：`SKILLS_INDEX.md` 每 Skill 一行摘要，不塞完整说明书。
2. **触发才加载**：未命中时 Context 只有 base + 索引。
3. **二级渐进**：`baoyu-diagram` 的 `references/*.md` 不自动全量加载；仅当消息暗示「架构/流程/时序…」时加载对应文档。
4. **执行后释放**：`auto_release=True`（默认）清空 loader 中的 body/refs。
5. **无 Key 可跑**：默认 `demo` 模式即可完整体验匹配 / 加载 / 脚本 / 对比。

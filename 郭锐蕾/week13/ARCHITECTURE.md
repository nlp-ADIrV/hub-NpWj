# ARCHITECTURE — 渐进式 Skill Harness

## 1. 问题：为什么需要 Harness？

课件指出实际 Agent Context 中，工具定义 + 系统 Prompt 常占 **65%+**。若把数十个 Skill 全文一次性塞进 system prompt：

- Token 成本倍增
- 有效推理空间被挤占
- 误触发与注意力分散上升

**Skills 范式**用「模块化认知单元 + 渐进式披露」解决该矛盾。

## 2. 三层 Context（课件 Part 3）

```
┌──────────────────────────────────────────────────────────┐
│                 Context Window（本轮）                     │
│                                                          │
│  Always (<~200 tok)     On Demand (500–2000)   In Context│
│  ┌────────────────┐    ┌─────────────────┐   ┌─────────┐│
│  │ Skills Index   │ →  │ 命中的 SKILL.md │ → │ refs /  ││
│  │ 每 Skill 一行  │    │ 完整指令+知识   │   │ scripts ││
│  └────────────────┘    └─────────────────┘   └─────────┘│
│                              │                           │
│                              ▼ 任务结束                   │
│                         释放正文，恢复仅索引               │
└──────────────────────────────────────────────────────────┘
```

| 层 | 内容 | 本仓库模块 |
|----|------|-----------|
| Always | `SKILLS_INDEX.md` 摘要 | `registry.py` |
| On Demand | 命中 Skill 的完整 `SKILL.md` | `loader.py` + `context.py` |
| In Context | `references/*`、脚本清单/产物 | `loader.load_secondary_*` + `executor.py` |

## 3. 组件关系

```
User Message
    │
    ▼
SkillHarness.handle()
    │
    ├── TriggerMatcher.match()          # 02 触发匹配
    │
    ├── ContextAssembler.assemble()     # Always + OnDemand + 可选 InContext
    │     ├── Registry.index_md
    │     ├── Loader.load_skill_body()
    │     └── Loader.load_secondary_for_message()
    │
    ├── SkillExecutor.execute()         # 04 执行（demo / llm）
    │
    └── ContextAssembler.release()      # 05 释放
```

## 4. 匹配策略

规则优先（教学可解释）：

1. Skill `name` 出现在消息中 → 高分
2. frontmatter `trigger` / 正文触发例句命中 → 高分
3. `description` 关键词重叠 → 加分
4. 取 Top-1 且超过 `min_score`

开放问题（课件）：多 Skill 冲突时如何仲裁——本实现返回 candidates 列表，默认取最高分，可用 `--force-skill` 覆盖。

## 5. 与示例 Skill 的对应

| Skill | 形态 | Harness 行为 |
|-------|------|--------------|
| `flash-card` | 纯代码工具 | 匹配 → 加载 SKILL → 跑 `make_flashcard.py` → 释放 |
| `baoyu-diagram` | 知识复合 | 匹配 → 加载 SKILL → **按类型**读 `references/*.md` → 给出执行计划 → 释放 |
| `pptx` / `code-review` | 课件示意 | 匹配 → 加载 → 生成执行计划 → 释放 |

## 6. 全量 vs 渐进对比

`compare_full_vs_progressive()`：

- **全量**：base + 索引 + 所有 `SKILL.md` + 所有 `references/*.md`
- **渐进**：本轮实际 `layers` 之和

用于课堂量化展示课件中的「20 Skills：12k → ~1k」类收益（本仓库 Skill 数量较少，节省比例随库增大而接近课件量级）。

## 7. 非目标

- 不实现完整 OpenClaw Gateway / Lane Queue
- 不替代 `agent_memory_system` 的四层记忆（MEMORY.md 长期记忆 ≠ Skills 索引；本项目索引文件名为 `SKILLS_INDEX.md` 以免混淆）
- LLM 模式为可选项；核心教学路径是 **demo 模式可见的加载/释放**

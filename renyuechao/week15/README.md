# Week15：最小并行 Subagent Agent

这是基于课程项目 `market_research_subagents` 思路的最小可运行版本：一个手写
ReAct 主 Agent 判断是否需要拆题，把一批互相独立的任务下发给预注册的
`researcher` / `analyst` Subagent；Subagent 通过 `ThreadPoolExecutor` 并行执行，
主 Agent 收到按计划顺序排列的结果后汇总。

本版本刻意只实现一层 fan-out / fan-in，不包含 DAG、数据库、消息队列、Web UI、
递归 Subagent 或新的 Agent 框架。

## 目录

```text
week15/
├── README.md
├── requirements.txt       # 真实 LLM 运行才需要 openai
├── src/
│   ├── agents.py          # profile、解析、并行派发、主 Agent、离线演示
│   ├── react_loop.py      # 可注入 chat_fn 的手写 ReAct
│   ├── llm_client.py      # DeepSeek/OpenAI-compatible 客户端
│   └── tavily_search.py   # Tavily 搜索及带 URL 的格式化
└── tests/
    ├── test_agents.py     # ReAct、并行、顺序和失败隔离
    └── test_llm_client.py # 模型 ID、thinking 与缺失 Key
```

## 处理流程

```text
用户问题
  ↓
主 ReActLoop
  ├─ 简单问题：web_search 或直接回答
  └─ 综合问题：dispatch_subagents
       ↓
  解析 profile::task | profile::task（最多 4 项）
       ↓
  researcher / analyst ReActLoop 通过有界线程池并行执行
       ↓
  每个任务返回 status/profile/task/final_answer/error/duration
       ↓
  按原始计划顺序生成 Observation
       ↓
  主 Agent 汇总成功项，并明确列出失败项
```

Subagent 不会再次派发 Subagent。任务之间如果存在先后依赖，应在主 Agent 的下一
轮中再派发，而不是把有依赖的任务放进同一批并行任务。

## 离线演示

在本目录执行：

```bash
python3 -m src.agents --offline-demo
```

演示使用 fake worker，不需要网络、API key 或 `openai`。它会运行三个不同耗时的
任务（慢/快/中），再运行一个故意失败的任务。输出顺序仍然是计划顺序，且失败项
不会阻断其他任务；结果中会显示 wall-clock 和串行耗时估算。

## 测试

```bash
python3 -m unittest discover -s tests -v
```

测试覆盖：

- `profile::task` 解析、未知 profile 保留、四项上限；
- ReActLoop 注入 fake `chat_fn` 以及可配置 Observation 截断；
- 真实线程并行的墙钟耗时明显小于串行，并验证结果按计划顺序；
- 单个 worker 抛异常和未知 profile 的失败隔离；
- LLM 异常、运行期 `TypeError` 不会盲目重试，未得到 `Final Answer` 会标记失败；
- DeepSeek 请求使用 `deepseek-v4-flash` 并关闭 thinking；
- 主 Agent 派发后读取聚合 Observation 并给出最终答案。

## 真实运行

先安装唯一的运行依赖：

```bash
python3 -m pip install -r requirements.txt
```

然后在 shell 环境中设置（不要写入源码、README 或日志）：

```bash
export DEEPSEEK_API_KEY="..."
export TAVILY_API_KEY="..."
# 可选：
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
export DEEPSEEK_MODEL="deepseek-v4-flash"
```

当前默认使用 `deepseek-v4-flash`，并在请求中显式设置
`thinking.type=disabled`，以保持手写 ReAct 的输出格式稳定。

代码中的 `run_research(question)` 是主入口；如果要进行离线单测，给它传入
`chat_fn`（签名需兼容 `llm_chat`，可用 `**kwargs` 接收调用参数），并可用
`worker_fn(profile, task)` 替代真实 Subagent。实际派发工具的
输入例子：

```text
researcher::调研产品 A | analyst::比较产品 A 和 B | researcher::查找官方定价
```

未知 profile 会形成 `status=failed` 的结果，最多四项的限制在解析阶段执行。线程
池上限固定为 4；结果收集使用任务索引写回，因此不会因为快任务先完成而改变汇总
顺序。

## 已验证范围与边界

已在本目录验证：

- 12 个标准库离线测试全部通过；
- DeepSeek `deepseek-v4-flash` 真实 smoke test 通过，thinking 已显式关闭；
- Tavily 真实搜索 smoke test 通过并返回来源 URL；
- 一次真实双 Subagent 端到端测试通过：主 Agent 执行
  `dispatch_subagents → Final Answer`，两个子任务均成功；总耗时约 24.9 秒，并行阶段
  约 17.8 秒，子任务时长合计约 27.6 秒。本次观测加速约 1.55 倍，仅代表该次调用，
  不是稳定性能承诺。

尚未验证：供应商限流、超时与重试、Token/费用统计、长时间运行时的任务取消/恢复、
生产级持久化、Web/SSE 展示，以及 Subagent 之间有依赖时的 DAG 调度。真实搜索结果
仍可能优先返回第三方来源，需要后续增加官方域名优先和引用质量检查。

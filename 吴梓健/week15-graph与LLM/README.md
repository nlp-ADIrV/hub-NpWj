# 新能源汽车参数对比调研 · Subagent 并行系统

基于 `market_research_subagents` 架构的改写版：主 agent（ReAct 循环，工具 `web_search` + `dispatch_subagents`）
自主决定是否派发多个 subagent 并行联网调研各车型/维度的参数，收齐后综合成结构化对比报告。
前端提供交互界面：左侧拓扑图（节点动态生长）+ 右侧任意节点 ReAct 过程流 + 最终对比报告。

## 与原型项目的关键差异

| 原型（手搓 ReAct） | 本版（API 原生 tool_call） |
|---|---|
| 正则解析 `Action:/Action Input:` 文本 + `stop=["Observation:"]` 截断 | 直接用 OpenAI 兼容 API 的 `tools` 参数，模型结构化返回工具名 + JSON 参数，无正则手搓 |
| 工具声明 `lambda q, **_: fn(q)`，`**_` 吞参写法 | 普通具名参数函数（如 `def web_search(query)`），需要共享状态时用闭包绑定 |
| 派发传参 `"课题1\|课题2"` 管道字符串 + split 解析 | 模型 tool_call 直接给 `topics` 字符串数组，`fn(**args)` 解包执行 |

## 环境准备

```bash
cd homework
pip install -r requirements.txt
export DEEPSEEK_API_KEY="sk-xxx"     # 主/subagent 的 LLM 推理（原生 tool calling）
export TAVILY_API_KEY="tvly-xxx"     # 联网搜索
```

## 运行

```bash
# 1) CLI 跑一次调研（自测）
python src/agents.py

# 2) HTTP 服务 + 可视化界面
uvicorn src.serve:app --host 0.0.0.0 --port 8002
# 浏览器开 http://localhost:8002

# 3) Parallel vs Serial 量化对比
python src/eval_compare.py --limit 2
```

## 目录结构

```
homework/
├── src/
│   ├── llm_client.py       # LLM 客户端，支持原生 function calling
│   ├── tavily_search.py    # Tavily 搜索（urllib 零依赖）
│   ├── react_loop.py       # 通用 ReAct 引擎（原生 tool_call，主/subagent 共用）
│   ├── agents.py           # 主 agent + dispatch_subagents(topics 数组) 并行派发
│   ├── serve.py            # FastAPI + SSE 流式
│   └── eval_compare.py     # parallel vs serial A/B
├── static/
│   ├── index.html          # 左拓扑右 trace 切换 主流程 UI（含对比表格渲染）
│   └── viz/topology.js     # SVG 拓扑动画
└── outputs/
```

# Week12 作业：为 ReAct Agent 增加多轮对话

在原有 **单轮 ReAct（Thought → Action → Observation）** 之上，增加 **跨用户轮次的会话记忆**，支持追问与指代消解。

## 与原项目的关系

| 能力 | 原项目 `src/` | 本目录 |
|------|---------------|--------|
| 单次问题内的工具循环 | ✅ | ✅（保留） |
| 多轮用户追问上下文 | ❌ 每次 `run()` 从零开始 | ✅ `ConversationSession` |
| Web 会话 | 无 session | `session_id` + 历史接口 |
| 工具 / FAISS | — | **复用** 上级目录 `../src/tools.py` 与 `../vectorstore/` |

不修改原项目代码；本目录为独立作业实现。

## 设计要点

1. **两层记忆分开**
   - **轮内**：完整 Thought / Action / Observation 轨迹（与原版相同）
   - **轮间**：只保存 `用户问题 + Final Answer`，注入下一轮 prompt  
     → 控制 token，同时让模型理解「它 / 那家 / 刚才」

2. **核心 API**

```python
from conversation import ConversationSession

session = ConversationSession(mode="manual")  # 或 "fc"
for step in session.chat("茅台2023年毛利率是多少？"):
    ...
for step in session.chat("那五粮液呢？"):  # 带历史
    ...
```

3. **`run(question, history=...)`**  
   `react_manual.py` / `react_function_calling.py` 新增可选 `history` 参数。

## 目录

```
week12作业文件夹/
├── README.md
├── index.html              # 多轮对话 Web UI
└── src/
    ├── conversation.py     # 会话与 SessionStore
    ├── react_manual.py     # 手写解析 + history
    ├── react_function_calling.py
    ├── agent.py            # 交互式 CLI
    └── serve.py            # FastAPI（建议端口 8001）
```

## 使用前准备

与原项目相同：

```powershell
$env:DASHSCOPE_API_KEY = "sk-xxx"
$env:AGENT_MODEL = "qwen-max"   # 可选
pip install -r ..\requirements.txt
```

## 运行方式

### 1. 交互式多轮 CLI（推荐体验）

```powershell
cd week12作业文件夹\src
python agent.py
python agent.py --mode fc
python agent.py --demo          # 三段追问演示后退出
```

交互命令：`/clear` `/history` `/mode manual|fc` `/quit`

### 2. Web 服务

```powershell
cd week12作业文件夹\src
uvicorn serve:app --host 0.0.0.0 --port 8001
```

浏览器打开 http://localhost:8001

SSE 请求体示例：

```json
{
  "question": "那五粮液呢？",
  "max_steps": 10,
  "session_id": "上次返回的 uuid"
}
```

额外接口：

- `GET /session/{id}` — 查看历史
- `POST /session/{id}/clear` — 清空
- `DELETE /session/{id}` — 删除会话

### 3. 作为模块

```python
import sys
sys.path.insert(0, r".../week12作业文件夹/src")
sys.path.insert(0, r".../react_financial_agent/src")  # tools

from conversation import ConversationSession
session = ConversationSession(mode="manual")
list(session.chat("贵州茅台2023年毛利率？"))
list(session.chat("五粮液呢？"))
print(session.to_dict())
```

## 建议验证路径

1. 问：「贵州茅台2023年毛利率是多少？」
2. 追问：「那五粮液呢？」→ 应理解同指标、不同公司
3. 再问：「两家差多少个百分点？」→ 可复用上文数字或再调 `calculator`
4. `/clear` 后再问「它的毛利率？」→ 应无法消解或要求澄清

## 说明

- 默认最多保留最近 **6** 轮问答（`max_history_turns`），可在 `ConversationSession` 中调整。
- Web 会话存在进程内存；重启服务会丢失，作业演示足够。

# Week12 作业：为 Agent 流程增加多轮对话能力

## 一、作业目标

本项目在基础 ReAct Agent 流程上增加 **多轮对话能力**。核心目标是让 Agent 不再把每次提问视为独立请求，而是能够保存同一 Session 中的历史问题、工具调用结果和最终回答，并在后续追问中继续使用这些上下文。

例如：

```text
用户：查询贵州茅台和五粮液2023年的毛利率。
Agent：贵州茅台 91.96%，五粮液 75.79%。

用户：哪家更高？
Agent：贵州茅台更高。

用户：两家公司相差多少个百分点？
Agent：16.17 个百分点。
```

第二、三轮没有重新给出完整公司名称和数据，Agent 需要根据历史上下文理解追问。

## 二、项目结构

```text
week12_agent_multiturn/
├── README.md
├── requirements.txt
└── src/
    ├── agent.py             # ReAct Agent + 多轮上下文核心逻辑
    ├── session_manager.py   # Session 创建、获取、删除和列表管理
    ├── tools.py             # 示例知识检索工具和计算器工具
    ├── main.py              # 交互式命令行程序
    └── demo_multiturn.py    # 自动多轮对话演示
```

## 三、核心设计

### 1. Session 保存会话状态

每个 Session 维护两类信息：

- `messages`：模型上下文，包括用户问题、Agent 输出和工具 Observation；
- `turns`：每轮最终问答，用于历史查看。

新一轮问题到来时，Agent 会将之前的 `messages` 一起传入模型，因此能够理解“它”“刚才两家公司”“再算一下”等上下文指代。

### 2. ReAct 流程保持不变

每一轮内部仍然遵循：

```text
Thought -> Action -> Observation -> Thought -> Final Answer
```

区别在于第一轮结束后的信息不会丢失，而是继续保存在 Session 中供下一轮使用。

### 3. 工具结果也进入历史

只保存用户问题和最终回答是不够的。后续追问可能需要继续利用上一轮工具查出的数据，所以本项目把 Action 和 Observation 也加入 Session 历史。

### 4. 上下文压缩

当历史消息超过一定长度时，`compact_history()` 会把较早消息压缩为简要历史摘要，同时保留最近消息，避免上下文无限增长。这是长对话中常见的 Memory 管理思路。

### 5. MockLLM 降级模式

为了让作业在没有大模型 API Key 的情况下也能直接运行，本项目自带一个简单的 `MockLLM`。

- 未配置 `OPENAI_API_KEY`：自动使用 MockLLM；
- 配置了 `OPENAI_API_KEY`：调用 OpenAI 兼容 Chat Completions API。

MockLLM 主要用于验证 Session、多轮历史和工具结果复用逻辑，不代表真实大模型能力。

## 四、运行方法

### 方式一：直接运行自动演示

```bash
cd src
python demo_multiturn.py
```

无需 API Key 即可运行。

### 方式二：启动交互式多轮对话

```bash
cd src
python main.py
```

支持命令：

```text
/history   查看当前 Session 历史
/new       新建 Session
/quit      退出
```

### 方式三：接入真实大模型

先安装依赖：

```bash
pip install -r requirements.txt
```

配置 OpenAI 兼容 API：

```bash
export OPENAI_API_KEY="你的API Key"
export AGENT_MODEL="gpt-4.1-mini"
```

如果使用其他兼容接口，还可以设置：

```bash
export OPENAI_BASE_URL="https://your-api.example.com/v1"
```

然后运行：

```bash
cd src
python main.py
```

## 五、与单轮 Agent 的主要区别

单轮 Agent 通常每次执行都重新创建：

```python
messages = [system_prompt, current_question]
```

本项目改为：

```python
messages = [system_prompt] + session.messages + [current_question]
```

每轮完成后再把本轮新增内容写回：

```python
session.messages.append(user_message)
session.messages.append(agent_action)
session.messages.append(tool_observation)
session.messages.append(final_answer)
```

因此同一个 Session 内的下一次调用可以获得之前的完整上下文，实现真正的多轮对话。

## 六、作业总结

本次作业完成了以下功能：

1. 为 Agent 增加 Session 会话机制；
2. 支持同一 Session 下连续多轮提问；
3. 保存用户消息、Agent 推理动作、工具结果和最终回答；
4. 支持上下文指代和历史数据复用；
5. 增加会话历史查看与新会话创建；
6. 增加简单的上下文压缩机制；
7. 提供 MockLLM，保证代码无需 API Key 也能演示多轮流程。

通过这些改造，Agent 从原来的单轮“问题 -> 回答”模式升级为具有短期会话记忆的多轮 Agent 流程。

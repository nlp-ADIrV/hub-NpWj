# 多轮对话功能实现说明

## 修改文件

`src/react_function_calling.py`

## 核心原理

原始代码每次调用 `run()` 都会新建空的 `messages` 列表，导致 Agent 每轮都"失忆"。多轮对话的核心改动就是让 `messages` 在多次问答之间**持久化**，使模型能看到完整的历史上下文。

```
单轮（改前）：每轮新建 messages → 不记得上一轮
多轮（改后）：messages 跨轮累积 → [system, Q1, A1, Q2, A2, Q3, ...]
```

## 具体改动（4处）

1. **`run()` 函数增加 `messages` 参数**：默认为 `None`（不传时自动新建，保持向后兼容），传入已有列表时在历史基础上追加本轮问题

2. **Final Answer 写入 messages**：模型给出最终答案后，将 `assistant` 角色的回答追加到对话历史，使下一轮能"记住"本轮说了什么

3. **新增 `chat()` 函数**：循环接收用户输入、维护持久 `messages`、调用 `run(question, messages=messages)`，支持 `exit` 退出

4. **新增 `--chat` 命令行参数**：不加时行为不变（单轮），加 `--chat` 进入多轮对话模式

## 运行方式

```bash
python react_function_calling.py --chat          # 多轮对话
python react_function_calling.py --question "..." # 单轮（原有功能不变）
```

## 关键知识点

- **messages 持久化**：对话历史在多次问答间保留，实现上下文记忆
- **System Prompt 只加一次**：初始化时加入，后续轮次不重复
- **向后兼容**：`messages` 默认 `None` 保持原有调用方不受影响

## 问题
- **网络环境问题**：Max retries exceeded 说明连接能建立但响应超时，可能是因为开启了 VPN 导致的网络不稳定
- **代码优化**：后续代码可以进一步优化

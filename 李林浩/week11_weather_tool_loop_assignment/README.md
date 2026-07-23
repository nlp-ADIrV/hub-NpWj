# 天气查询工具循环调用

## 一、作业说明

本项目将天气查询从一次工具调用改造成了循环工具调用。

原来的单次调用流程通常是：

```text
用户问题
  -> 模型判断是否调用工具
  -> 执行一次工具
  -> 返回结果
  -> 模型回答
```

本项目改造后的流程是：

```text
用户问题
  -> 模型生成工具调用
  -> 执行工具并把结果回填给模型
  -> 模型再次判断是否还需要调用工具
  -> 继续执行下一轮工具调用
  -> 直到模型不再请求工具后输出最终回答
```

因此，一个天气问题可以自动完成下面的连续步骤：

```text
城市名称
  -> search_city 查询城市坐标
  -> query_weather 根据坐标查询天气
  -> 模型整理最终答案
```

当用户一次查询多个城市时，模型也可以在后续轮次继续调用工具，而不是在第一次工具执行后立即结束。

---

## 二、项目结构

```text
week11_weather_tool_loop_assignment/
├── main.py
├── weather_tools.py
├── requirements.txt
└── README.md
```

文件说明：

- `main.py`：Function Calling 主程序，负责模型请求、工具执行和循环控制。
- `weather_tools.py`：天气相关工具，包括城市坐标查询和天气查询。
- `requirements.txt`：项目依赖。
- `README.md`：运行说明和实现思路。

---

## 三、核心改造

循环调用的核心代码位于 `main.py`：

```python
for round_index in range(1, MAX_TOOL_ROUNDS + 1):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )

    assistant_message = response.choices[0].message

    if not assistant_message.tool_calls:
        return assistant_message.content or ""

    messages.append(assistant_message)

    for tool_call in assistant_message.tool_calls:
        tool_result = execute_tool(tool_name, tool_args)

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result,
            }
        )
```

关键点有两个：

1. 模型请求工具后，程序不会直接结束，而是把工具结果加入 `messages`。
2. 程序进入下一轮，再次请求模型，由模型判断继续调用工具还是生成最终回答。

项目设置：

```python
MAX_TOOL_ROUNDS = 8
```

用于限制最大循环轮数，防止异常情况下无限调用。

---

## 四、安装依赖

建议使用 Python 3.10 或更高版本。

```bash
pip install -r requirements.txt
```

---

## 五、配置 API Key

项目默认使用阿里云百炼兼容 OpenAI 接口。

macOS / Linux：

```bash
export DASHSCOPE_API_KEY="你的 API Key"
```

Windows PowerShell：

```powershell
$env:DASHSCOPE_API_KEY="你的 API Key"
```

---

## 六、运行

查询单个城市：

```bash
python main.py -q "查询杭州今天的天气和未来三天预报"
```

查询多个城市：

```bash
python main.py -q "分别查询北京和上海的天气，并比较哪个城市今天更热"
```

只输出最终回答：

```bash
python main.py -q "查询广州天气" --quiet
```

默认问题：

```bash
python main.py
```

---

## 七、示例调用过程

执行：

```bash
python main.py -q "查询北京天气"
```

程序可能经历以下过程：

```text
第 1 轮
调用 search_city
获得北京的经纬度

第 2 轮
调用 query_weather
获得当前天气和三天天气预报

第 3 轮
模型不再请求工具
输出最终回答
```

这说明天气查询已经不再局限于一次工具调用，而是由模型根据上一轮工具结果决定下一步操作。

---

## 八、实现特点

本项目把城市解析和天气查询拆分为两个独立工具：

- `search_city(city)`
- `query_weather(latitude, longitude)`

这种拆分可以更直观地验证循环工具调用机制，同时保留了继续扩展空气质量、天气预警等工具的空间。

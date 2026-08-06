import time
import json
from openai import OpenAI
import os

# ==========================================
# 0. 配置千问客户端 (使用 OpenAI 兼容模式)
# ==========================================
# 请替换为你的阿里云百炼 API Key
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 指定千问最强模型
MODEL_NAME = "qwen-max" 

# ==========================================
# 1. 原始的低效 Skill (冗长的 Prompt + 臃肿的 Schema)
# ==========================================
ORIGINAL_PROMPT = """
你是一个非常厉害的新闻编辑。现在我需要你帮我一个忙。请仔细阅读下面我提供给你的新闻文章。
读完之后，我希望你能帮我写一个不超过200字的总结，告诉大家这篇新闻主要讲了什么。
另外，请你仔细寻找文章中出现的重要人物的名字，还有新闻发生的具体地点。
请确保不要漏掉任何一个重要的信息。并且请按照下面规定的JSON格式输出给我，谢谢你的合作。
"""

ORIGINAL_SCHEMA = {
    "type": "object",
    "properties": {
        "article_summary_text_description": {
            "type": "string",
            "description": "请在这里填写这篇新闻文章的主要内容的概括总结，字数限制在200字以内。"
        },
        "important_person_names_mentioned": {
            "type": "array",
            "items": {"type": "string"},
            "description": "文章中提到的所有重要人物的姓名的列表"
        },
        "event_happening_locations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "新闻事件发生所在的城市或国家等地点"
        }
    },
    "required": ["article_summary_text_description", "important_person_names_mentioned", "event_happening_locations"]
}

# 测试用例文本
TEST_TEXT = "2023年9月15日，苹果公司CEO蒂姆·库克在加州库比蒂诺的史蒂夫·乔布斯剧院发布了最新的iPhone 15系列。本次发布会吸引了全球数百万科技爱好者的目光。库克表示，新一代芯片将彻底改变移动端游戏体验。"

# ==========================================
# 2. 让模型自我优化的 Meta-Skill
# ==========================================
def optimize_skill(prompt, schema):
    print(f"🤖 正在让 {MODEL_NAME} 优化 Skill 中...")
    meta_prompt = f"""
    你是一个资深的 Prompt 工程师。请优化以下大模型技能的 System Prompt 和 JSON Schema。
    优化目标：
    1. 大幅减少 Token 消耗（剔除所有礼貌用语和冗余解释）。
    2. 提高执行效率（使用缩写、结构化的指令，减少模型推理负担）。
    3. JSON 键名（Key）必须极度简短，去掉冗长的描述。
    
    原始 Prompt: {prompt}
    原始 Schema: {json.dumps(schema, ensure_ascii=False)}
    
    请严格返回 JSON 格式，包含 "optimized_prompt" (字符串) 和 "optimized_schema" (对象) 两个字段。
    """
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": meta_prompt}],
        response_format={"type": "json_object"}
    )
    
    result = json.loads(response.choices[0].message.content)
    return result["optimized_prompt"], result["optimized_schema"]

# ==========================================
# 3. 执行并统计性能的工具函数 (升级为 tools 语法)
# ==========================================
def run_and_evaluate(name, prompt, schema, text):
    print(f"\n▶ 开始测试: {name}")
    start_time = time.time()
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"提取以下文本：\n{text}"}
        ],
        # 使用最新的 tools 格式替代废弃的 functions
        tools=[{
            "type": "function",
            "function": {
                "name": "extract_info",
                "parameters": schema
            }
        }],
        tool_choice={"type": "function", "function": {"name": "extract_info"}}
    )
    
    end_time = time.time()
    latency = end_time - start_time
    usage = response.usage
    
    # 提取千问返回的 Tool Call 参数
    tool_call = response.choices[0].message.tool_calls[0]
    output_data = tool_call.function.arguments
    
    print(f"⏱️ 耗时: {latency:.2f} 秒")
    print(f"🪙 Input Tokens: {usage.prompt_tokens}")
    print(f"🪙 Output Tokens: {usage.completion_tokens}")
    print(f"🪙 Total Tokens: {usage.total_tokens}")
    return usage.total_tokens, latency

# ==========================================
# 4. 主函数：运行对比流水线
# ==========================================
if __name__ == "__main__":
    # 步骤 1: 评估原始 Skill
    orig_tokens, orig_time = run_and_evaluate("【未优化的原始 Skill】", ORIGINAL_PROMPT, ORIGINAL_SCHEMA, TEST_TEXT)
    
    # 步骤 2: 让千问自己重写代码
    opt_prompt, opt_schema = optimize_skill(ORIGINAL_PROMPT, ORIGINAL_SCHEMA)
    print("\n✅ 优化完成！")
    print(f"✨ 优化后的 Prompt: {opt_prompt}")
    print(f"✨ 优化后的 Schema: {json.dumps(opt_schema, ensure_ascii=False, indent=2)}")
    
    # 步骤 3: 评估优化后的 Skill
    opt_tokens, opt_time = run_and_evaluate("【优化后的极简 Skill】", opt_prompt, opt_schema, TEST_TEXT)
    
    # 步骤 4: 输出对比报告
    print("\n📊 =============== 优化效果报告 ===============")
    print(f"Token 消耗: {orig_tokens} -> {opt_tokens} (减少了 {((orig_tokens-opt_tokens)/orig_tokens)*100:.1f}%)")
    print(f"执行耗时: {orig_time:.2f}s -> {opt_time:.2f}s (提速了 {((orig_time-opt_time)/orig_time)*100:.1f}%)")
    print("==============================================")
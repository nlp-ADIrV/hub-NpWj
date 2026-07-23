from openai import OpenAI

client = OpenAI(
    api_key="YOUR_API_KEY",
    base_url="https://api.deepseek.com"
)

def llm_predict(s1, s2):
    prompt = f"""
判断两个句子是否语义相同，只回答0或1。

句子1: {s1}
句子2: {s2}

答案：
"""

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    text = resp.choices[0].message.content.strip()

    return 1 if "1" in text else 0



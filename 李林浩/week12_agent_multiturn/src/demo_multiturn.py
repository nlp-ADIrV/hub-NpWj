"""自动演示多轮上下文能力。"""
from agent import Agent
from session_manager import SessionManager

QUESTIONS = [
    "查询贵州茅台和五粮液2023年的毛利率。",
    "哪家更高？",
    "两家公司相差多少个百分点？",
]

manager = SessionManager()
session = manager.create()
agent = Agent()

print(f"session_id={session.session_id}\n")
for i, q in enumerate(QUESTIONS, 1):
    print(f"===== Round {i} =====")
    print("User:", q)
    for item in agent.run(session, q):
        if item["type"] == "action":
            print(f"Action: {item['action']} -> {item['observation']}")
        elif item["type"] == "final":
            print("Assistant:", item["answer"])
    print()

print("保存的对话轮数:", len(session.turns))
print("保存的消息条数:", len(session.messages))

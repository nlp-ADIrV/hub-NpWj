"""命令行多轮对话 Demo。"""
from agent import Agent
from session_manager import SessionManager


def print_step(item: dict) -> None:
    if item["type"] == "action":
        print(f"  Thought: {item['thought']}")
        print(f"  Action: {item['action']} {item['action_input']}")
        print(f"  Observation: {item['observation']}")
    elif item["type"] == "final":
        print(f"Assistant: {item['answer']}")
    else:
        print(item)


def main():
    manager = SessionManager()
    agent = Agent()
    session = manager.create()

    print("=== Week12 多轮对话 Agent ===")
    print(f"Session: {session.session_id}")
    print("输入 /history 查看历史，/new 新建会话，/quit 退出。")
    print("未设置 OPENAI_API_KEY 时会自动使用 MockLLM，可直接演示。\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question == "/quit":
            break
        if question == "/new":
            session = manager.create()
            print(f"已创建新会话: {session.session_id}")
            continue
        if question == "/history":
            if not session.turns:
                print("当前暂无历史。")
            for i, turn in enumerate(session.turns, 1):
                print(f"[{i}] Q: {turn['question']}\n    A: {turn['answer']}")
            continue

        for item in agent.run(session, question):
            print_step(item)


if __name__ == "__main__":
    main()

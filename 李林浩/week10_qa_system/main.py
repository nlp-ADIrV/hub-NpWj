import argparse
from pathlib import Path

from src.qa_system import LocalQASystem


def parse_args():
    parser = argparse.ArgumentParser(description="基于本地文件的问答系统")
    parser.add_argument(
        "--data",
        default="data",
        help="资料目录，默认使用 data",
    )
    parser.add_argument(
        "-q",
        "--question",
        default=None,
        help="直接输入一个问题；不填写时进入交互模式",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="返回的检索结果数量，默认 3",
    )
    return parser.parse_args()


def print_result(result):
    print("\n回答：")
    print(result["answer"])

    print("\n参考文件：")
    for item in result["sources"]:
        print(f"- {item['source']}")

    print()


def main():
    args = parse_args()
    data_dir = Path(args.data)

    qa = LocalQASystem(data_dir=data_dir)
    qa.build()

    print(f"已加载 {qa.document_count} 个文件，建立 {qa.chunk_count} 个文本块。")

    if args.question:
        result = qa.answer(args.question, top_k=args.top_k)
        print_result(result)
        return

    print("输入问题开始问答，输入 exit 或 quit 结束。\n")

    while True:
        try:
            question = input("问题：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n程序结束。")
            break

        if question.lower() in {"exit", "quit"}:
            print("程序结束。")
            break

        if not question:
            continue

        result = qa.answer(question, top_k=args.top_k)
        print_result(result)


if __name__ == "__main__":
    main()

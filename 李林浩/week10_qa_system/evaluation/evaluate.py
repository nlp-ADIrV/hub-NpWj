import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.qa_system import LocalQASystem


def main():
    qa = LocalQASystem(PROJECT_ROOT / "data")
    qa.build()

    question_file = Path(__file__).parent / "questions.json"
    questions = json.loads(question_file.read_text(encoding="utf-8"))

    hit_count = 0

    for index, item in enumerate(questions, start=1):
        results = qa.retrieve(item["question"], top_k=1)
        sources = [result["source"] for result in results]
        hit = item["expected_source"] in sources

        if hit:
            hit_count += 1

        print(f"[{index}] {item['question']}")
        print(f"期望文件：{item['expected_source']}")
        print(f"检索结果：{', '.join(sources)}")
        print(f"Hit@1：{'Yes' if hit else 'No'}")
        print("-" * 50)

    score = hit_count / len(questions) if questions else 0.0
    print(f"最终 Hit@1：{score:.2%} ({hit_count}/{len(questions)})")


if __name__ == "__main__":
    main()

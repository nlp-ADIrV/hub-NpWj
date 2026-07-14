"""汇总人民日报 NER 上 Linear 与 CRF 的评估结果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "outputs" / "logs"


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="人民日报 NER 实验结果汇总")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    linear = load_json(LOG_DIR / f"eval_peoples_daily_linear_{args.split}.json")
    crf = load_json(LOG_DIR / f"eval_peoples_daily_crf_{args.split}.json")

    print("\n" + "=" * 78)
    print(f"人民日报 NER 序列标注结果汇总（split={args.split}）")
    print("=" * 78)
    print(f"{'模型':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'非法序列':>10}")
    print("-" * 66)

    for name, result in [("BERT + Linear", linear), ("BERT + CRF", crf)]:
        if result is None:
            print(f"{name:<20} {'未找到评估结果':>42}")
            continue
        illegal = result["illegal_stats"]["total_illegal"]
        print(
            f"{name:<20} "
            f"{result['precision']:>10.4f} "
            f"{result['recall']:>10.4f} "
            f"{result['f1']:>10.4f} "
            f"{illegal:>10d}"
        )

    print("=" * 78)
    print("提示：若结果缺失，请先运行 src/train.py 与 src/evaluate.py。")


if __name__ == "__main__":
    main()

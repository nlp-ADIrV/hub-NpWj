"""
LLM API zero-shot 文本匹配对比（DashScope 兼容 OpenAI SDK）。

适用：BQ Corpus、LCQMC 或任意同格式 JSONL 数据集。
使用：
  export DASHSCOPE_API_KEY="sk-xxx"
  python src/llm_compare_dashscope.py --data_dir data/bq_corpus_sample --num_samples 20
  python src/llm_compare_dashscope.py --data_dir data/lcqmc_sample --model qwen-plus
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

from data_utils import load_dataset, balanced_sample
from metrics import compute_binary_metrics

DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

PROMPT_TEMPLATE = """请判断以下两个句子是否表达相同的意思。只回答“是”或“否”，不要输出解释。
句子1：{s1}
句子2：{s2}
回答："""


def parse_answer(text: str) -> int:
    text = text.strip()
    if "是" in text and "否" not in text:
        return 1
    if "否" in text:
        return 0
    return -1


def call_llm(client, s1: str, s2: str, model: str) -> Tuple[int, str]:
    prompt = PROMPT_TEMPLATE.format(s1=s1, s2=s2)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content.strip()
        return parse_answer(raw), raw
    except Exception as e:
        return -1, f"ERROR: {e}"


def main():
    parser = argparse.ArgumentParser(description="LLM zero-shot 文本匹配评估")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--model", default="qwen-plus")
    parser.add_argument("--sleep_sec", type=float, default=0.2)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("未设置 DASHSCOPE_API_KEY。请先运行：export DASHSCOPE_API_KEY='sk-xxx'")

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=DASHSCOPE_URL)

    data_dir = Path(args.data_dir)
    dataset = data_dir.name
    rows = balanced_sample(load_dataset(data_dir, args.split), args.num_samples, args.seed)

    results = []
    valid_labels, valid_preds = [], []
    for i, r in enumerate(rows, 1):
        pred, raw = call_llm(client, r["sentence1"], r["sentence2"], args.model)
        results.append({**r, "pred": pred, "raw": raw})
        if pred != -1:
            valid_labels.append(r["label"])
            valid_preds.append(pred)
        if i % 10 == 0:
            m = compute_binary_metrics(valid_labels, valid_preds)
            print(f"[{i}/{len(rows)}] valid={m['n']} acc={m['accuracy']:.4f} parse_fail={i-m['n']}")
        time.sleep(args.sleep_sec)

    metrics = compute_binary_metrics(valid_labels, valid_preds)
    save = {
        "dataset": dataset,
        "split": args.split,
        "method": f"llm_zero_shot_{args.model}",
        "num_samples": len(rows),
        "metrics": metrics,
        "parse_fail": len(rows) - metrics["n"],
        "results": results,
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"llm_zero_shot_{dataset}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(save, f, ensure_ascii=False, indent=2)
    print(json.dumps(save["metrics"], ensure_ascii=False, indent=2))
    print(f"结果已保存：{out_path}")


if __name__ == "__main__":
    main()

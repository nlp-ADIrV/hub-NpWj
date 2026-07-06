"""
Qwen LoRA SFT 文本匹配评估脚本（可选扩展）。
需要先完成 train_sft_qwen_lora.py，并安装 transformers / peft。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from tqdm import tqdm

from data_utils import load_dataset
from metrics import compute_binary_metrics


def parse_pred(text: str) -> int:
    if "不相似" in text:
        return 0
    if "相似" in text:
        return 1
    return -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--max_new_tokens", type=int, default=8)
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float32, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, args.adapter_dir).to(device).eval()

    rows = load_dataset(args.data_dir, args.split)
    preds, labels, records = [], [], []
    for r in tqdm(rows):
        messages = [
            {"role": "system", "content": "你是一个语义匹配助手。判断两句话语义是否相同，只输出【相似】或【不相似】，不要输出其他内容。"},
            {"role": "user", "content": f"句子A：{r['sentence1']}\n句子B：{r['sentence2']}\n是否相似："},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        gen = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        pred = parse_pred(gen)
        if pred != -1:
            labels.append(r["label"])
            preds.append(pred)
        records.append({**r, "pred": pred, "raw": gen})

    metrics = compute_binary_metrics(labels, preds)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sft_predictions_{Path(args.data_dir).name}.csv"
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(metrics)


if __name__ == "__main__":
    main()

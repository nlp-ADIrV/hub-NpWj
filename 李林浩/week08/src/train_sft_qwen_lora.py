"""
Qwen2-0.5B-Instruct LoRA SFT 文本匹配训练脚本（可选扩展）。

说明：该脚本对应作业中的“生成式 SFT 方法”。默认不随提交包运行，因为需要本地基座模型、GPU、transformers、peft。
输入数据格式与 baselines 一致：train.jsonl / validation.jsonl，每行包含 sentence1、sentence2、label。

示例：
  python src/train_sft_qwen_lora.py \
    --model_path /path/to/Qwen2-0.5B-Instruct \
    --data_dir data/bq_corpus \
    --output_dir outputs/sft_bq_adapter \
    --num_train 5000 --epochs 3
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from data_utils import load_dataset

SYSTEM_PROMPT = "你是一个语义匹配助手。判断两句话语义是否相同，只输出【相似】或【不相似】，不要输出其他内容。"
LABEL_MAP = {1: "【相似】", 0: "【不相似】"}


class SFTDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        label_text = LABEL_MAP[item["label"]]
        prompt_text = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"句子A：{item['sentence1']}\n句子B：{item['sentence2']}\n是否相似："},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        response_ids = self.tokenizer.encode(label_text, add_special_tokens=False) + [self.tokenizer.eos_token_id]
        input_ids = (prompt_ids + response_ids)[: self.max_length]
        labels = ([-100] * len(prompt_ids) + response_ids)[: self.max_length]
        return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels)}


def collate_fn(batch, pad_id):
    max_len = max(x["input_ids"].size(0) for x in batch)
    input_ids, labels, attention = [], [], []
    for x in batch:
        n = x["input_ids"].size(0)
        pad = max_len - n
        input_ids.append(torch.cat([x["input_ids"], torch.full((pad,), pad_id)]))
        labels.append(torch.cat([x["labels"], torch.full((pad,), -100)]))
        attention.append(torch.cat([torch.ones(n), torch.zeros(pad)]).long())
    return {"input_ids": torch.stack(input_ids).long(), "labels": torch.stack(labels).long(), "attention_mask": torch.stack(attention).long()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_train", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import get_peft_model, LoraConfig, TaskType

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_raw = load_dataset(args.data_dir, "train")
    val_raw = load_dataset(args.data_dir, "validation")[:500]
    pos = [r for r in train_raw if r["label"] == 1]
    neg = [r for r in train_raw if r["label"] == 0]
    if args.num_train > 0:
        n_each = min(args.num_train // 2, len(pos), len(neg))
        train_raw = random.sample(pos, n_each) + random.sample(neg, n_each)
        random.shuffle(train_raw)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_loader = DataLoader(SFTDataset(train_raw, tokenizer, args.max_length), batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id))
    val_loader = DataLoader(SFTDataset(val_raw, tokenizer, args.max_length), batch_size=args.batch_size * 2, shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id))

    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float32, trust_remote_code=True)
    lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.05, bias="none")
    model = get_peft_model(model, lora_config).to(device)
    model.print_trainable_parameters()

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs = []
    best = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_tokens = 0.0, 0
        optimizer.zero_grad()
        t0 = time.time()
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            (loss / args.grad_accum).backward()
            if (step + 1) % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            tokens = (batch["labels"] != -100).sum().item()
            total_loss += loss.item() * tokens
            total_tokens += tokens
        train_loss = total_loss / max(total_tokens, 1)

        model.eval()
        val_loss, val_tokens = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = model(**batch).loss
                tokens = (batch["labels"] != -100).sum().item()
                val_loss += loss.item() * tokens
                val_tokens += tokens
        val_loss = val_loss / max(val_tokens, 1)
        logs.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "elapsed_s": round(time.time() - t0, 2)})
        print(logs[-1])
        if val_loss < best:
            best = val_loss
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    with (output_dir / "train_log.json").open("w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

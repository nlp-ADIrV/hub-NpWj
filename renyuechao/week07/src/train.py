"""在人民日报 NER 数据集上训练序列标注模型。

示例：
  python src/train.py --epochs 3 --max_length 160
  python src/train.py --use_crf --epochs 3 --max_length 160
  python src/train.py --max_train_samples 200 --max_eval_samples 100 --epochs 1
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from seqeval.metrics import f1_score
from torch.optim import AdamW
from tqdm import tqdm
from transformers import BertTokenizer, get_linear_schedule_with_warmup

from dataset import DATA_DIR, build_dataloaders, build_label_schema, resolve_default_bert_path
from model import build_model


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def decode_batch(
    model,
    batch: dict[str, torch.Tensor],
    id2label: dict[int, str],
    device: torch.device,
    model_type: str,
) -> tuple[list[list[str]], list[list[str]], torch.Tensor | None]:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    token_type_ids = batch["token_type_ids"].to(device)
    labels = batch["labels"].to(device)

    if model_type == "crf":
        _, loss = model(input_ids, attention_mask, token_type_ids, labels)
        pred_ids_list = model.decode(input_ids, attention_mask, token_type_ids)
    else:
        logits, loss = model(input_ids, attention_mask, token_type_ids, labels)
        pred_ids_list = logits.argmax(dim=-1).cpu().tolist()

    gold_ids_list = labels.cpu().tolist()
    all_preds: list[list[str]] = []
    all_golds: list[list[str]] = []

    for i, gold_ids in enumerate(gold_ids_list):
        pred_ids = pred_ids_list[i]
        pred_seq: list[str] = []
        gold_seq: list[str] = []
        for j, gold_id in enumerate(gold_ids):
            if gold_id == -100:
                continue
            gold_seq.append(id2label[gold_id])
            pred_seq.append(id2label.get(pred_ids[j] if j < len(pred_ids) else 0, "O"))
        all_golds.append(gold_seq)
        all_preds.append(pred_seq)

    return all_preds, all_golds, loss


def evaluate_epoch(
    model,
    loader,
    id2label: dict[int, str],
    device: torch.device,
    model_type: str,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    all_preds: list[list[str]] = []
    all_golds: list[list[str]] = []

    with torch.no_grad():
        for batch in loader:
            preds, golds, loss = decode_batch(model, batch, id2label, device, model_type)
            total_loss += float(loss.item()) if loss is not None else 0.0
            all_preds.extend(preds)
            all_golds.extend(golds)

    return total_loss / max(len(loader), 1), f1_score(all_golds, all_preds)


def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    device: torch.device,
    grad_accum: int,
    epoch: int,
    epochs: int,
) -> float:
    model.train()
    optimizer.zero_grad()
    total_loss = 0.0

    progress = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
    for step, batch in enumerate(progress, 1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        _, loss = model(input_ids, attention_mask, token_type_ids, labels)
        (loss / grad_accum).backward()
        total_loss += loss.item()

        if step % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        progress.set_postfix(loss=f"{loss.item():.4f}")

    if len(loader) % grad_accum != 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return total_loss / max(len(loader), 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="人民日报 NER 序列标注训练")
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--bert_path", type=Path, default=resolve_default_bert_path())
    parser.add_argument("--model_type", choices=["linear", "crf"], default="linear")
    parser.add_argument("--use_crf", action="store_true", help="等价于 --model_type crf")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=160)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head_lr_mult", type=float, default=5.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=0, help="调试用，0 表示全量")
    parser.add_argument("--max_eval_samples", type=int, default=0, help="调试用，0 表示全量")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.use_crf:
        args.model_type = "crf"

    set_seed(args.seed)
    labels, label2id, id2label = build_label_schema()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"数据集：人民日报 NER ({args.data_dir})")
    print(f"标签：{labels}")
    print(f"设备：{device}")
    print(f"BERT 路径：{args.bert_path}")

    tokenizer = BertTokenizer.from_pretrained(str(args.bert_path))
    train_loader, val_loader, _ = build_dataloaders(
        tokenizer=tokenizer,
        label2id=label2id,
        batch_size=args.batch_size,
        max_length=args.max_length,
        data_dir=args.data_dir,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )

    model = build_model(args.model_type, str(args.bert_path), len(labels), args.dropout).to(device)

    head_params = list(model.classifier.parameters())
    if args.model_type == "crf":
        head_params += list(model.crf.parameters())
    optimizer = AdamW(
        [
            {"params": model.bert.parameters(), "lr": args.lr},
            {"params": head_params, "lr": args.lr * args.head_lr_mult},
        ],
        weight_decay=0.01,
    )
    total_steps = max(1, len(train_loader) * args.epochs // max(args.grad_accum, 1))
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_tag = f"peoples_daily_{args.model_type}"
    ckpt_path = CKPT_DIR / f"best_{run_tag}.pt"
    log_path = LOG_DIR / f"train_{run_tag}.json"

    best_f1 = 0.0
    records = []
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, args.grad_accum, epoch, args.epochs
        )
        val_loss, val_f1 = evaluate_epoch(model, val_loader, id2label, device, args.model_type)
        elapsed = time.time() - start
        print(
            f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_f1={val_f1:.4f} | time={elapsed:.1f}s"
        )
        records.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "val_entity_f1": round(val_f1, 6),
                "elapsed_s": round(elapsed, 1),
            }
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                {
                    "epoch": epoch,
                    "model_type": args.model_type,
                    "state_dict": model.state_dict(),
                    "val_entity_f1": val_f1,
                    "label2id": label2id,
                    "id2label": id2label,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"  保存最优 checkpoint: {ckpt_path}")

    log_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"训练完成，最优 val_f1={best_f1:.4f}")
    print(f"训练日志：{log_path}")
    print(f"下一步：python src/evaluate.py --model_type {args.model_type} --split test")


if __name__ == "__main__":
    main()

"""评估人民日报 NER 序列标注模型。

示例：
  python src/evaluate.py --split test
  python src/evaluate.py --use_crf --split test
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import BertTokenizer

from dataset import DATA_DIR, build_dataloaders, build_label_schema, resolve_default_bert_path
from model import build_model


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "outputs" / "checkpoints"
LOG_DIR = ROOT / "outputs" / "logs"


def count_illegal_sequences(pred_seqs: list[list[str]]) -> dict:
    """统计 BIO 非法序列数量。"""
    stats = {"illegal_start": 0, "illegal_transition": 0, "total_seqs": len(pred_seqs)}

    for seq in pred_seqs:
        if not seq:
            continue
        if seq[0].startswith("I-"):
            stats["illegal_start"] += 1

        for i in range(1, len(seq)):
            prev = seq[i - 1]
            curr = seq[i]
            if not curr.startswith("I-"):
                continue
            curr_type = curr[2:]
            if prev == "O":
                stats["illegal_transition"] += 1
            elif prev.startswith("B-") or prev.startswith("I-"):
                prev_type = prev[2:]
                if prev_type != curr_type:
                    stats["illegal_transition"] += 1

    stats["total_illegal"] = stats["illegal_start"] + stats["illegal_transition"]
    return stats


def run_inference(model, loader, id2label: dict[int, str], device: torch.device, model_type: str):
    model.eval()
    all_preds: list[list[str]] = []
    all_golds: list[list[str]] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            if model_type == "crf":
                pred_ids_list = model.decode(input_ids, attention_mask, token_type_ids)
            else:
                logits, _ = model(input_ids, attention_mask, token_type_ids)
                pred_ids_list = logits.argmax(dim=-1).cpu().tolist()

            labels_list = labels.cpu().tolist()
            for i, gold_ids in enumerate(labels_list):
                gold_seq: list[str] = []
                pred_seq: list[str] = []
                pred_ids = pred_ids_list[i]

                for j, gold_id in enumerate(gold_ids):
                    if gold_id == -100:
                        continue
                    gold_seq.append(id2label[gold_id])
                    pred_seq.append(id2label.get(pred_ids[j] if j < len(pred_ids) else 0, "O"))

                all_golds.append(gold_seq)
                all_preds.append(pred_seq)

    return all_preds, all_golds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="人民日报 NER 序列标注评估")
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--bert_path", type=Path, default=resolve_default_bert_path())
    parser.add_argument("--model_type", choices=["linear", "crf"], default="linear")
    parser.add_argument("--use_crf", action="store_true", help="等价于 --model_type crf")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--max_eval_samples", type=int, default=0, help="调试用，0 表示全量")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.use_crf:
        args.model_type = "crf"

    labels, label2id, id2label = build_label_schema()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_tag = f"peoples_daily_{args.model_type}"
    ckpt_path = CKPT_DIR / f"best_{run_tag}.pt"

    if not ckpt_path.exists():
        print(f"找不到 checkpoint: {ckpt_path}")
        print(f"请先运行: python src/train.py --model_type {args.model_type}")
        return

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    max_length = checkpoint.get("args", {}).get("max_length", 160)

    tokenizer = BertTokenizer.from_pretrained(str(args.bert_path))
    _, val_loader, test_loader = build_dataloaders(
        tokenizer=tokenizer,
        label2id=label2id,
        batch_size=args.batch_size,
        max_length=max_length,
        data_dir=args.data_dir,
        max_eval_samples=args.max_eval_samples,
    )
    loader = val_loader if args.split == "validation" else test_loader

    model = build_model(args.model_type, str(args.bert_path), len(labels)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    print(
        f"加载 checkpoint: epoch={checkpoint['epoch']}, "
        f"val_f1={checkpoint['val_entity_f1']:.4f}"
    )

    all_preds, all_golds = run_inference(model, loader, id2label, device, args.model_type)
    precision = precision_score(all_golds, all_preds)
    recall = recall_score(all_golds, all_preds)
    f1 = f1_score(all_golds, all_preds)
    illegal_stats = count_illegal_sequences(all_preds)

    print("\n" + "=" * 70)
    print(f"模型：BERT + {args.model_type.upper()} | 数据集：人民日报 NER | split={args.split}")
    print("=" * 70)
    print(f"Entity-level Precision: {precision:.4f}")
    print(f"Entity-level Recall:    {recall:.4f}")
    print(f"Entity-level F1:        {f1:.4f}")
    print("\n【逐类型指标】")
    print(classification_report(all_golds, all_preds, digits=4))
    print("【非法 BIO 序列统计】")
    print(f"  总序列数：{illegal_stats['total_seqs']}")
    print(f"  非法开头：{illegal_stats['illegal_start']}")
    print(f"  非法转移：{illegal_stats['illegal_transition']}")
    print(f"  合计非法：{illegal_stats['total_illegal']}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LOG_DIR / f"eval_{run_tag}_{args.split}.json"
    out_path.write_text(
        json.dumps(
            {
                "dataset": "peoples_daily",
                "model": f"BERT+{args.model_type.upper()}",
                "split": args.split,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "illegal_stats": illegal_stats,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n评估结果已保存：{out_path}")


if __name__ == "__main__":
    main()

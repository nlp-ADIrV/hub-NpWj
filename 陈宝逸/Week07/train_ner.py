"""
People's Daily Chinese NER — Sequence Labeling with BERT

Trains a BERT-base-Chinese model for named entity recognition
(PER / ORG / LOC in BIO tagging scheme).

Usage:
    python train_ner.py
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from tqdm import tqdm

from transformers import AutoTokenizer, BertModel, get_linear_schedule_with_warmup

# ──────────────────────────────────────────────────────────────
# 1. Configuration
# ──────────────────────────────────────────────────────────────


@dataclass
class Config:
    # Paths
    data_dir: str = "data/peoples_daily"
    output_dir: str = "output/ner_model"
    label_names_path: str = "data/peoples_daily/label_names.json"

    # Model
    bert_model_name: str = "bert-base-chinese"
    max_length: int = 128
    dropout: float = 0.1

    # Training
    batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    gradient_clip: float = 1.0

    # Checkpoint
    patience: int = 3  # early stopping patience
    save_best: bool = True

    # Device
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────
# 2. Dataset
# ──────────────────────────────────────────────────────────────


class NERDataset(Dataset):
    """People's Daily NER dataset with BERT tokenisation & label alignment."""

    def __init__(self, json_path: str, tokenizer: AutoTokenizer, label2id: dict, max_length: int):
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.tokens_list = []
        self.labels_list = []
        self.lengths = []

        for example in tqdm(raw, desc=f"Loading {os.path.basename(json_path)}"):
            tokens = example["tokens"]  # list of Chinese characters
            ner_tags = example["ner_tags"]  # list of BIO tags (strings)

            # Convert tokens to a single text string for BERT tokenizer
            # For Chinese BERT, we can join characters (no spaces needed)
            text = "".join(tokens)

            # Tokenize — returns input_ids and offset_mapping
            encoding = tokenizer(
                text,
                return_offsets_mapping=True,
                max_length=max_length,
                truncation=True,
                padding="max_length",
            )

            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]
            offset_mapping = encoding["offset_mapping"]

            # Align labels with BERT sub-tokens using offset mapping.
            label_ids = []
            for offset in offset_mapping:
                if offset[0] == 0 and offset[1] == 0:
                    label_ids.append(label2id["O"])
                else:
                    orig_idx = offset[0]
                    if orig_idx < len(ner_tags):
                        label_ids.append(label2id[ner_tags[orig_idx]])
                    else:
                        label_ids.append(label2id["O"])

            self.tokens_list.append(torch.tensor(input_ids, dtype=torch.long))
            self.labels_list.append(torch.tensor(label_ids, dtype=torch.long))
            self.lengths.append(torch.tensor(attention_mask, dtype=torch.long))

    def __len__(self):
        return len(self.tokens_list)

    def __getitem__(self, idx):
        return {
            "input_ids": self.tokens_list[idx],
            "attention_mask": self.lengths[idx],
            "labels": self.labels_list[idx],
        }


# ──────────────────────────────────────────────────────────────
# 3. Model
# ──────────────────────────────────────────────────────────────


class BertForNER(nn.Module):
    """BERT + linear classification head for NER."""

    def __init__(self, bert_model_name: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (B, S, H)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # (B, S, num_labels)
        return logits


# ──────────────────────────────────────────────────────────────
# 4. Evaluation Metrics
# ──────────────────────────────────────────────────────────────


def extract_entities(labels, id2label, ignore_id=None):
    """Convert label ID sequence to list of (entity_type, start, end) tuples."""
    entities = []
    i = 0
    while i < len(labels):
        if ignore_id is not None and labels[i] == ignore_id:
            i += 1
            continue
        label = id2label[labels[i]]
        if label.startswith("B-"):
            entity_type = label[2:]
            start = i
            i += 1
            while i < len(labels):
                if ignore_id is not None and labels[i] == ignore_id:
                    i += 1
                    break
                next_label = id2label[labels[i]]
                if next_label == f"I-{entity_type}":
                    i += 1
                else:
                    break
            entities.append((entity_type, start, i - 1))
        else:
            i += 1
    return entities


def compute_f1(preds, targets, id2label, ignore_id):
    """Compute entity-level precision, recall, F1."""
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for pred_seq, target_seq in zip(preds, targets):
        pred_entities = extract_entities(pred_seq, id2label, ignore_id)
        target_entities = extract_entities(target_seq, id2label, ignore_id)

        pred_set = set(pred_entities)
        target_set = set(target_entities)

        true_positives += len(pred_set & target_set)
        false_positives += len(pred_set - target_set)
        false_negatives += len(target_set - pred_set)

    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    return precision, recall, f1


# ──────────────────────────────────────────────────────────────
# 5. Training helpers
# ──────────────────────────────────────────────────────────────


def train_epoch(model, dataloader, optimizer, scheduler, criterion, config):
    model.train()
    total_loss = 0
    steps = 0

    pbar = tqdm(dataloader, desc="Train")
    for batch in pbar:
        input_ids = batch["input_ids"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        labels = batch["labels"].to(config.device)

        optimizer.zero_grad()

        logits = model(input_ids, attention_mask)
        # (B, S, num_labels) -> (B*S, num_labels), labels -> (B*S)
        loss = criterion(logits.view(-1, model.num_labels), labels.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        steps += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / steps


@torch.no_grad()
def evaluate(model, dataloader, criterion, id2label, ignore_id, config):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Eval"):
        input_ids = batch["input_ids"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        labels = batch["labels"].to(config.device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits.view(-1, model.num_labels), labels.view(-1))
        total_loss += loss.item()

        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    # Compute entity-level F1 (ignoring padding / special tokens)
    precision, recall, f1 = compute_f1(all_preds, all_labels, id2label, ignore_id)

    return avg_loss, precision, recall, f1


# ──────────────────────────────────────────────────────────────
# 6. Main training loop
# ──────────────────────────────────────────────────────────────


def main():
    config = Config()
    os.makedirs(config.output_dir, exist_ok=True)
    print(f"Device: {config.device}")
    print(f"Config: {config}\n")

    # ── Labels ──
    with open(config.label_names_path, "r", encoding="utf-8") as f:
        label_names = json.load(f)
    label2id = {lbl: i for i, lbl in enumerate(label_names)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    num_labels = len(label_names)
    print(f"Labels ({num_labels}): {label_names}")

    # ── Tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(config.bert_model_name, use_fast=True)

    # ── Datasets ──
    train_dataset = NERDataset(
        os.path.join(config.data_dir, "train.json"),
        tokenizer, label2id, config.max_length,
    )
    val_dataset = NERDataset(
        os.path.join(config.data_dir, "validation.json"),
        tokenizer, label2id, config.max_length,
    )
    test_dataset = NERDataset(
        os.path.join(config.data_dir, "test.json"),
        tokenizer, label2id, config.max_length,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0,
    )

    print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"Test:  {len(test_dataset)} samples, {len(test_loader)} batches\n")

    # ── Model ──
    model = BertForNER(config.bert_model_name, num_labels, config.dropout).to(config.device)

    # ── Optimiser & scheduler ──
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_params, lr=config.learning_rate)

    total_steps = len(train_loader) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    # ── Loss ──
    ignore_id = label2id["O"]  # we include O in loss but not in entity evaluation
    criterion = nn.CrossEntropyLoss(ignore_index=-100)  # we won't set -100, so all count

    # ── Training ──
    best_f1 = 0.0
    patience_counter = 0
    train_times = []

    for epoch in range(1, config.num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config.num_epochs}")
        print(f"{'='*60}")

        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, config)
        epoch_time = time.time() - t0
        train_times.append(epoch_time)

        val_loss, precision, recall, f1 = evaluate(
            model, val_loader, criterion, id2label, ignore_id, config,
        )

        print(f"  Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}")
        print(f"  Precision: {precision:.4f}  |  Recall: {recall:.4f}  |  F1: {f1:.4f}")
        print(f"  Time: {epoch_time:.1f}s")

        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            if config.save_best:
                torch.save(model.state_dict(), os.path.join(config.output_dir, "best_model.pt"))
                print(f"  => New best model saved (F1={f1:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{config.patience})")
            if patience_counter >= config.patience:
                print(f"\nEarly stopping triggered after {epoch} epochs.")
                break

    print(f"\n{'='*60}")
    print(f"Training complete. Best val F1: {best_f1:.4f}")
    print(f"Average epoch time: {np.mean(train_times):.1f}s")

    # ── Test evaluation ──
    print(f"\n{'='*60}")
    print("Evaluating on test set...")
    if config.save_best:
        model.load_state_dict(torch.load(os.path.join(config.output_dir, "best_model.pt")))
        print("Loaded best model checkpoint.")

    test_loss, precision, recall, f1 = evaluate(
        model, test_loader, criterion, id2label, ignore_id, config,
    )
    print(f"\n{'='*60}")
    print("TEST SET RESULTS")
    print(f"{'='*60}")
    print(f"  Loss:      {test_loss:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")

    # ── Save final model ──
    torch.save(model.state_dict(), os.path.join(config.output_dir, "final_model.pt"))
    print(f"\nModel saved to {config.output_dir}")


if __name__ == "__main__":
    main()

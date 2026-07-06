import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader

from src.data_utils import build_vocab, TextClassificationDataset, load_csv_dataset
from src.model import RNNTextClassifier


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, dataloader, device, id2label):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["length"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, lengths)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    target_names = [id2label[i] for i in range(len(id2label))]
    report = classification_report(
        all_labels,
        all_preds,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    return acc, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/sample_data.csv")
    parser.add_argument("--model_type", type=str, default="lstm", choices=["rnn", "lstm", "gru"])
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--bidirectional", action="store_true", help="启用双向 RNN/LSTM/GRU")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test_size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    texts, labels = load_csv_dataset(args.data_path)

    unique_labels = sorted(set(labels))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts,
        labels,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels if len(set(labels)) > 1 else None,
    )

    vocab = build_vocab(train_texts)

    train_dataset = TextClassificationDataset(
        train_texts, train_labels, vocab, label2id, max_len=args.max_len
    )
    val_dataset = TextClassificationDataset(
        val_texts, val_labels, vocab, label2id, max_len=args.max_len
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = RNNTextClassifier(
        vocab_size=len(vocab),
        num_classes=len(label2id),
        model_type=args.model_type,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        dropout=args.dropout,
        pad_idx=vocab["<PAD>"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = -1.0
    best_path = output_dir / f"best_{args.model_type}.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            lengths = batch["length"].to(device)
            labels_tensor = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, lengths)
            loss = criterion(logits, labels_tensor)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)
        val_acc, val_report = evaluate(model, val_loader, device, id2label)

        print(f"Epoch [{epoch}/{args.epochs}] loss={avg_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)

    config = {
        "model_type": args.model_type,
        "vocab_size": len(vocab),
        "num_classes": len(label2id),
        "max_len": args.max_len,
        "embed_dim": args.embed_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "bidirectional": args.bidirectional,
        "dropout": args.dropout,
        "pad_idx": vocab["<PAD>"],
        "label2id": label2id,
        "id2label": id2label,
        "vocab": vocab,
        "best_model_path": str(best_path),
    }

    with open(output_dir / f"config_{args.model_type}.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("\nBest validation accuracy:", best_acc)
    print(f"Best model saved to: {best_path}")
    print(f"Config saved to: {output_dir / f'config_{args.model_type}.json'}")

    model.load_state_dict(torch.load(best_path, map_location=device))
    final_acc, final_report = evaluate(model, val_loader, device, id2label)
    print("\nFinal validation report:")
    print(final_report)


if __name__ == "__main__":
    main()

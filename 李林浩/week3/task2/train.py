import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# 类别定义：
# 文本长度固定为 5，只要包含“你”
# “你”在第 1 位 -> class 0
# “你”在第 2 位 -> class 1
# ...
# “你”在第 5 位 -> class 4


VOCAB_CHARS = list("你我他她它的是在有和不人中大上为个国学到说们生子时年得就那要下以可也会出发作地家用对小多好看")


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_vocab():
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for ch in VOCAB_CHARS:
        if ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab


def generate_sample():
    """
    随机生成一个长度为 5 且包含“你”的文本。
    label = “你”所在位置，范围 0~4。
    """
    label = random.randint(0, 4)
    chars = random.choices([c for c in VOCAB_CHARS if c != "你"], k=5)
    chars[label] = "你"
    text = "".join(chars)
    return text, label


def encode(text, vocab):
    return [vocab.get(ch, vocab["<UNK>"]) for ch in text]


class NiPositionDataset(Dataset):
    def __init__(self, size, vocab):
        self.samples = [generate_sample() for _ in range(size)]
        self.vocab = vocab

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        input_ids = encode(text, self.vocab)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


class SimpleTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64, num_classes=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5 * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        logits = self.classifier(x)
        return logits


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits = model(input_ids)
            preds = torch.argmax(logits, dim=-1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_size", type=int, default=5000)
    parser.add_argument("--val_size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vocab = build_vocab()

    train_dataset = NiPositionDataset(args.train_size, vocab)
    val_dataset = NiPositionDataset(args.val_size, vocab)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    model = SimpleTextClassifier(vocab_size=len(vocab)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    model_path = output_dir / "task2_model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for input_ids, labels in train_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader, device)

        print(f"Epoch [{epoch}/{args.epochs}] loss={avg_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab": vocab,
                    "embed_dim": 32,
                    "hidden_dim": 64,
                    "num_classes": 5,
                },
                model_path,
            )

    print(f"Best validation accuracy: {best_acc:.4f}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()

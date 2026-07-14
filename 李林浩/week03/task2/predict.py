import argparse

import torch
import torch.nn as nn


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


def encode(text, vocab):
    return [vocab.get(ch, vocab["<UNK>"]) for ch in text]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="outputs/task2_model.pt")
    parser.add_argument("--text", type=str, required=True)
    args = parser.parse_args()

    text = args.text.strip()

    if len(text) != 5:
        raise ValueError("输入文本必须正好是 5 个字。")

    if "你" not in text:
        raise ValueError("输入文本必须包含“你”字。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.model_path, map_location=device)
    vocab = checkpoint["vocab"]

    model = SimpleTextClassifier(
        vocab_size=len(vocab),
        embed_dim=checkpoint["embed_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_classes=checkpoint["num_classes"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    input_ids = torch.tensor([encode(text, vocab)], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(input_ids)
        probs = torch.softmax(logits, dim=-1)[0]
        pred_id = int(torch.argmax(probs).item())

    pred_class = pred_id + 1

    print(f"输入文本: {text}")
    print(f"预测类别: 第 {pred_class} 类")
    print(f"解释: 模型判断“你”在第 {pred_class} 位")
    print(f"各类别概率: {[round(float(p), 4) for p in probs]}")


if __name__ == "__main__":
    main()

"""
字符级单向 Transformer 语言模型：训练 + 文本生成。
用法:
    python transformer_language_model.py --epochs 20
    python transformer_language_model.py --generate --prompt "黄金" --max_new_tokens 200
"""

import math
import argparse
import glob
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────── 数据 ───────────────────────────

def load_corpus(pattern="*.txt"):
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text):
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    def __init__(self, text, char2idx, seq_len):
        self.seq_len = seq_len
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y


# ─────────────────────────── 模型 ───────────────────────────

def causal_mask(seq_len, device):
    """上三角为 True，表示该位置不可被关注（单向 / 因果注意力）。"""
    return torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
        diagonal=1,
    )


class TransformerLM(nn.Module):
    """单向（因果）Transformer 语言模型，结构类似 GPT 的 encoder-only 设计。"""

    def __init__(
        self,
        vocab_size,
        embed_dim,
        nhead,
        num_layers,
        dim_feedforward,
        dropout,
        max_len,
    ):
        super().__init__()
        self.max_len = max_len
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        self.drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        b, t = x.shape
        if t > self.max_len:
            raise ValueError(f"序列长度 {t} 超过 max_len={self.max_len}")

        pos = torch.arange(t, device=x.device).unsqueeze(0).expand(b, -1)
        h = self.drop(self.embed(x) + self.pos_embed(pos))
        mask = causal_mask(t, x.device)
        h = self.transformer(h, mask=mask)
        logits = self.fc(self.drop(h))  # (B, T, V)
        return logits


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


# ─────────────────────────── 文本生成 ───────────────────────────

@torch.no_grad()
def generate_text(
    model,
    char2idx,
    idx2char,
    prompt,
    max_new_tokens,
    temperature=1.0,
    top_k=None,
    device="cpu",
):
    model.eval()
    if not prompt:
        prompt = idx2char[0]

    unk = 0
    ids = [char2idx.get(c, unk) for c in prompt if c in char2idx]
    if not ids:
        ids = [0]

    for _ in range(max_new_tokens):
        x = torch.tensor([ids[-model.max_len:]], dtype=torch.long, device=device)
        logits = model(x)[:, -1, :] / max(temperature, 1e-8)

        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        ids.append(next_id)

        if next_id not in idx2char:
            break

    return "".join(idx2char[i] for i in ids)


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    return ckpt


def build_model_from_args(args, vocab_size, device):
    if args.embed_dim % args.nhead != 0:
        raise ValueError(f"embed_dim ({args.embed_dim}) 必须能被 nhead ({args.nhead}) 整除")

    model = TransformerLM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_len=args.seq_len,
    ).to(device)
    return model


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--corpus", default="*.txt")
    parser.add_argument("--save", default="best_transformer.pt")

    parser.add_argument("--generate", action="store_true", help="加载 checkpoint 并生成文本")
    parser.add_argument("--checkpoint", default="best_transformer.pt")
    parser.add_argument("--prompt", default="黄金")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.generate:
        ckpt = load_checkpoint(args.checkpoint, device)
        char2idx = ckpt["char2idx"]
        idx2char = ckpt["idx2char"]
        saved = ckpt.get("args", {})
        for k, v in saved.items():
            if hasattr(args, k):
                setattr(args, k, v)

        vocab_size = len(char2idx)
        model = build_model_from_args(args, vocab_size, device)
        model.load_state_dict(ckpt["model_state"])

        text = generate_text(
            model,
            char2idx,
            idx2char,
            args.prompt,
            args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
        )
        print(f"prompt: {args.prompt}")
        print("-" * 40)
        print(text)
        return

    print(f"device: {device}  model: Transformer (causal)")

    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 文件，请确认路径正确。")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text = "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, args.seq_len)
    val_ds = CharDataset(val_text, char2idx, args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True)

    model = build_model_from_args(args, vocab_size, device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "char2idx": char2idx,
                    "idx2char": idx2char,
                    "args": vars(args),
                },
                args.save,
            )

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}{marker}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")

    sample = generate_text(
        model,
        char2idx,
        idx2char,
        args.prompt,
        args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )
    print(f"\n生成示例 (prompt: {args.prompt!r}):")
    print("-" * 40)
    print(sample)


if __name__ == "__main__":
    main()

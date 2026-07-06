"""
基于 Transformer 的字符级单向语言模型：训练 + 文本生成。
使用 PyTorch 内置 nn.TransformerEncoder（因果掩码实现单向 LM）。

用法:
    python transformer_language_model.py --epochs 20
    python transformer_language_model.py --generate --prompt "黄金" --model_path best_transformer.pt
"""

import math
import argparse
import glob
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────── 数据 ───────────────────────────

def load_corpus(pattern="corpus.txt"):
    texts = []
    for path in sorted(glob.glob(pattern)):
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

class LM(nn.Module):
    """
    字符级单向语言模型。结构与 language_model.py 一致，
    将 RNN/LSTM 替换为 nn.TransformerEncoder + 因果掩码。
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, n_heads, dropout, max_pos):
        super().__init__()
        self.max_pos = max_pos
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_pos, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, vocab_size)
        # True = 不可 attend；预建因果掩码，forward 时按长度切片
        mask = torch.triu(torch.ones(max_pos, max_pos, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        # x: (B, T)
        b, t = x.shape
        if t > self.max_pos:
            raise ValueError(f"序列长度 {t} 超过 max_pos={self.max_pos}")

        pos = torch.arange(t, device=x.device).unsqueeze(0).expand(b, -1)
        e = self.drop(self.embed(x) + self.pos_emb(pos))          # (B, T, embed_dim)
        out = self.transformer(e, mask=self.causal_mask[:t, :t])
        return self.fc(self.drop(out))                            # (B, T, vocab_size)


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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 50.0))  # 防止 loss 过大时溢出
    return avg_loss, ppl


# ─────────────────────────── 文本生成 ───────────────────────────

def _filter_logits(logits, temperature, top_k):
    logits = logits / temperature
    if top_k is not None and top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[-1]] = float("-inf")
    return logits


@torch.no_grad()
def generate(model, char2idx, idx2char, prompt, max_new_tokens, device,
             temperature=0.8, top_k=50, greedy=False):
    model.eval()
    unk_id = 0
    ids = [char2idx[c] if c in char2idx else unk_id for c in prompt]
    if not ids:
        ids = [unk_id]

    for _ in range(max_new_tokens):
        ctx = ids[-model.max_pos:]
        x = torch.tensor([ctx], dtype=torch.long, device=device)
        logits = model(x)[0, -1, :]

        if greedy:
            next_id = int(logits.argmax().item())
        else:
            probs = F.softmax(_filter_logits(logits, temperature, top_k), dim=-1)
            next_id = int(torch.multinomial(probs, 1).item())

        ids.append(next_id)

    return "".join(idx2char[i] for i in ids)


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    char2idx = ckpt["char2idx"]
    idx2char = ckpt["idx2char"]
    cfg = ckpt["args"]

    model = LM(
        vocab_size=len(char2idx),
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        n_heads=cfg["n_heads"],
        dropout=0.0,
        max_pos=cfg["max_pos"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, char2idx, idx2char, cfg


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="transformer", help="固定为 transformer")
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--seq_len",    type=int,   default=64)
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--embed_dim",  type=int,   default=128)
    parser.add_argument("--hidden_dim", type=int,   default=512, help="Transformer FFN 维度")
    parser.add_argument("--n_heads",    type=int,   default=4)
    parser.add_argument("--num_layers", type=int,   default=4)
    parser.add_argument("--max_pos",    type=int,   default=256)
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--corpus",     default="corpus.txt", help="语料文件，支持 glob")
    parser.add_argument("--save",       default="best_transformer.pt")
    parser.add_argument("--generate",   action="store_true")
    parser.add_argument("--model_path", default="best_transformer.pt")
    parser.add_argument("--prompt",     default="黄金")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k",      type=int,   default=50)
    parser.add_argument("--greedy",     action="store_true")
    args = parser.parse_args()

    if args.max_pos < args.seq_len:
        args.max_pos = args.seq_len
    if args.embed_dim % args.n_heads != 0:
        raise ValueError(f"embed_dim ({args.embed_dim}) 必须能被 n_heads ({args.n_heads}) 整除")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.generate:
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"找不到模型文件: {args.model_path}")
        model, char2idx, idx2char, _ = load_checkpoint(args.model_path, device)
        text = generate(
            model, char2idx, idx2char,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            device=device,
            temperature=args.temperature,
            top_k=args.top_k,
            greedy=args.greedy,
        )
        print(f"模型: {args.model_path}  (Transformer, 词表 {len(char2idx)})")
        print(f"提示: {args.prompt!r}")
        print("生成:")
        print(text)
        return

    print(f"device: {device}  model: TRANSFORMER")

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
    val_text   = "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, args.seq_len)
    val_ds   = CharDataset(val_text,   char2idx, args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=True)

    model = LM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        max_pos=args.max_pos,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_ppl = float("inf")
    save_args = vars(args)

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": save_args,
            }, args.save)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}{marker}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")

    print("\n── 文本生成演示 ──")
    model, char2idx, idx2char, _ = load_checkpoint(args.save, device)
    for p in ("黄金", "避险", "美联储"):
        out = generate(model, char2idx, idx2char, p, 120, device)
        print(f"\n[{p!r}] →\n{out}")


if __name__ == "__main__":
    main()

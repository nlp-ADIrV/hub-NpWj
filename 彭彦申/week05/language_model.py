"""
字符级语言模型训练脚本，支持 RNN / LSTM / Transformer，含 PPL 计算与文本生成。
用法:
    python language_model.py --model lstm --epochs 20
    python language_model.py --model rnn  --epochs 20
    python language_model.py --model transformer --epochs 20
    python language_model.py --model transformer --generate --prompt "黄金"  (训练后生成)
    python language_model.py --load best_model.pt --prompt "黄金" --max_new 300  (仅生成)
"""

import math
import argparse
import glob
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
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


# ─────────────────────────── 位置编码 ───────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# ─────────────────────────── Transformer 模型 ───────────────────────────

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, nhead, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim)
        self.drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.ln = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def _causal_mask(self, sz, device):
        return torch.triu(torch.full((sz, sz), float("-inf"), device=device), diagonal=1)

    def forward(self, x):
        B, T = x.shape
        e = self.drop(self.embed(x))
        e = self.pos_enc(e)
        mask = self._causal_mask(T, x.device)
        out = self.transformer(e, mask=mask, is_causal=False)
        logits = self.fc(self.ln(out))
        return logits


# ─────────────────────────── RNN/LSTM 模型 ───────────────────────────

class LM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, model_type, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        rnn_cls = nn.LSTM if model_type == "lstm" else nn.RNN
        self.rnn = rnn_cls(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        e = self.drop(self.embed(x))
        out, _ = self.rnn(e)
        logits = self.fc(self.drop(out))
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
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


# ─────────────────────────── 文本生成 ───────────────────────────

@torch.no_grad()
def generate_text(model, char2idx, idx2char, prompt, max_new_tokens, device, temperature=1.0):
    model.eval()
    chars = list(prompt)
    ids = torch.tensor([[char2idx.get(c, 0) for c in chars]], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        logits = model(ids)[0, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        chars.append(idx2char[next_id])
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)

        if ids.size(1) > 512:
            ids = ids[:, -512:]

    return "".join(chars)


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="lstm", choices=["rnn", "lstm", "transformer"])
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--seq_len",    type=int,   default=64)
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--embed_dim",  type=int,   default=128)
    parser.add_argument("--hidden_dim", type=int,   default=256)
    parser.add_argument("--num_layers", type=int,   default=2)
    parser.add_argument("--nhead",      type=int,   default=4)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--corpus",     default="*.txt")
    parser.add_argument("--save",       default="best_model.pt")
    parser.add_argument("--generate",   action="store_true", help="训练后生成文本")
    parser.add_argument("--prompt",     default="黄金", help="生成文本的起始提示")
    parser.add_argument("--max_new",    type=int, default=200, help="生成的最大字符数")
    parser.add_argument("--temperature",type=float, default=0.8, help="采样温度")
    parser.add_argument("--load",       default="", help="加载已有模型进行生成")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  model: {args.model.upper()}")

    if args.load:
        checkpoint = torch.load(args.load, map_location=device)
        char2idx = checkpoint["char2idx"]
        idx2char = checkpoint["idx2char"]
        vocab_size = len(char2idx)
        saved_args = checkpoint["args"]
        print(f"从 {args.load} 加载模型，词表大小: {vocab_size}")
        # 使用保存的参数重建模型
        if saved_args["model"] == "transformer":
            model = TransformerLM(
                vocab_size=vocab_size,
                embed_dim=saved_args["embed_dim"],
                hidden_dim=saved_args["hidden_dim"],
                num_layers=saved_args["num_layers"],
                nhead=saved_args["nhead"],
                dropout=saved_args["dropout"],
            ).to(device)
        else:
            model = LM(
                vocab_size=vocab_size,
                embed_dim=saved_args["embed_dim"],
                hidden_dim=saved_args["hidden_dim"],
                num_layers=saved_args["num_layers"],
                model_type=saved_args["model"],
                dropout=saved_args["dropout"],
            ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        generated = generate_text(model, char2idx, idx2char, args.prompt, args.max_new, device, args.temperature)
        print(f"\n生成文本:\n{generated}")
        return

    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 文件，请确认路径正确。")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    if args.model == "transformer":
        model = TransformerLM(
            vocab_size=vocab_size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            nhead=args.nhead,
            dropout=args.dropout,
        ).to(device)
    else:
        model = LM(
            vocab_size=vocab_size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            model_type=args.model,
            dropout=args.dropout,
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text   = "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, args.seq_len)
    val_ds   = CharDataset(val_text,   char2idx, args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=True, drop_last=True)

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
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
            }, args.save)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}{marker}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")

    if args.generate:
        generated = generate_text(model, char2idx, idx2char, args.prompt, args.max_new, device, args.temperature)
        print(f"\n生成文本 (prompt: '{args.prompt}'):\n{generated}")


if __name__ == "__main__":
    main()

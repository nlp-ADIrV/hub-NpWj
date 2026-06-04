"""
字符级语言模型训练脚本，使用单向注意力TransformeDecoder，含 PPL 计算。
用法:
    python language_model.py --model TransformerDecoder --epochs 20
    python language_model.py --model rnn  --epochs 20
"""

import math
import argparse
import glob
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os

# ─────────────────────────── 数据 ───────────────────────────


def load_corpus(pattern="*.txt"):
    texts = []
    print(f"当前目录: {os.getcwd()}")
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
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y


# ─────────────────────────── 模型 ───────────────────────────


class DecoderOnlyLM(nn.Module):
    def __init__(
        self, vocab_size, embed_dim, num_layers, dropout, num_heads, max_seq_len=2048
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embed = nn.Embedding(vocab_size, embed_dim)

        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)  # 可学习位置编码

        # 使用 TransformerEncoder（更简洁，配合 mask）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,  # 输入输出维度
            nhead=num_heads,  # 注意力头数（新增参数）
            dim_feedforward=embed_dim * 4,  # FFN隐藏层维度
            batch_first=True,  # 使用 (B, T, E) 格式
            dropout=dropout,
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        """
        x: (batch, seq_len) 输入token索引
        返回: (batch, seq_len, vocab_size) logits
        """
        batch, seq_len = x.shape

        # Token embedding
        e = self.embed(x)  # (batch, seq_len, embed_dim)

        # 添加位置编码
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, seq_len)
        e = e + self.pos_embed(positions)  # (batch, seq_len, embed_dim)
        e = self.drop(e)

        # 生成因果掩码（使用PyTorch内置方法）
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=x.device
        )

        # Transformer 编码器（带因果掩码）
        out = self.encoder(e, mask=causal_mask)  # (batch, seq_len, embed_dim)

        # 输出层
        logits = self.fc(out)  # (batch, seq_len, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens, char2idx, idx2char, temperature=0.8):
        """自回归生成文本"""
        self.eval()
        device = next(self.parameters()).device

        # 编码 prompt
        idx = torch.tensor(
            [[char2idx.get(c, 0) for c in prompt]], dtype=torch.long, device=device
        )

        for _ in range(max_new_tokens):
            # 截断超长序列
            if idx.size(1) > self.max_seq_len:
                idx = idx[:, -self.max_seq_len :]

            # 前向传播
            logits = self(idx)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)

            # 采样
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

        # 解码
        return "".join([idx2char[i.item()] for i in idx[0]])


# ─────────────────────────── 训练 / 评估 ───────────────────────────


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    epoch=None,
    total_epochs=None,
    train=True,
):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0
    n_batches = len(loader)
    log_interval = max(1, n_batches // 10)

    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

        if batch_idx % log_interval == 0:
            ppl_now = (
                math.exp(total_loss / total_tokens)
                if total_loss / total_tokens < 100
                else float("inf")
            )
            mode = "Train" if train else "Val"
            ep = f"Epoch {epoch}/{total_epochs}" if epoch else ""
            print(
                f"  {mode} {ep} | batch {batch_idx}/{n_batches} | loss: {total_loss/total_tokens:.4f} | ppl: {ppl_now:.2f}"
            )

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss) if avg_loss < 100 else float("inf")
    return avg_loss, ppl


# ─────────────────────────── 主函数 ───────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)  # 训练轮数
    parser.add_argument("--seq_len", type=int, default=128)  # 序列长度
    parser.add_argument("--batch_size", type=int, default=64)  # 批量大小
    parser.add_argument("--embed_dim", type=int, default=128)  # 嵌入维度
    parser.add_argument("--num_layers", type=int, default=4)  # Transformer层数
    parser.add_argument("--dropout", type=float, default=0.3)  # Dropout概率
    parser.add_argument("--lr", type=float, default=1e-3)  # 学习率
    parser.add_argument("--num_heads", type=int, default=8)  # 注意力头数
    parser.add_argument("--val_ratio", type=float, default=0.05)  # 验证集比例
    parser.add_argument("--corpus", default="*.txt")  # 语料文件
    parser.add_argument("--generate", action="store_true")  # 是否生成文本
    parser.add_argument("--prompt", default="The")  # 生成文本的起始词
    parser.add_argument("--save", default="best_model.pt")  # 保存模型文件
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  model: transformer")

    # 数据准备
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

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    # 模型
    model = DecoderOnlyLM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_heads=args.num_heads,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_ppl = float("inf")

    print(
        f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}"
    )
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(
                model, val_loader, criterion, optimizer, device, train=False
            )

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

        print(
            f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}{marker}"
        )

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")

    # 生成示例
    if args.generate and best_val_ppl != float("inf"):
        print("\n" + "=" * 50)
        print("生成示例:")
        print("=" * 50)

        checkpoint = torch.load(args.save, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        char2idx = checkpoint["char2idx"]
        idx2char = checkpoint["idx2char"]

        for temp in [0.6, 0.8, 1.0]:
            print(f"\n温度={temp}:")
            generated = model.generate(
                args.prompt, 100, char2idx, idx2char, temperature=temp
            )
            print(generated)
            print("-" * 50)


if __name__ == "__main__":
    main()

"""
训练基于transformer的单向语言模型，并完成文本生成。
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
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y


# ─────────────────────────── 模型 ───────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 预计算位置编码矩阵 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, T, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class CausalTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, d_model)

        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, vocab_size)

        self._init_weights()
 
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def make_causal_mask(seq_len, device):
        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )
        return mask

    def forward(self, x):
        B, T = x.shape
        causal_mask = self.make_causal_mask(T, x.device)  # (T, T)

        h = self.pos_enc(self.embed(x))   # (B, T, d_model)

        h = self.transformer(
            tgt=h,
            memory=h,
            tgt_mask=causal_mask,
            memory_mask=causal_mask,
        )                              # (B, T, d_model)

        logits = self.fc_out(h)        # (B, T, vocab_size)
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
            # 梯度裁剪：防止梯度爆炸（Transformer 训练常用技巧）
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 20))   # 限制 exp 防止溢出
    return avg_loss, ppl


# ─────────────────────────── 文本生成 ───────────────────────────

@torch.no_grad()
def generate(model, prompt, char2idx, idx2char, max_new_tokens=200,
             temperature=1.0, top_k=40, device="cpu"):
    model.eval()
    ids = [char2idx[c] for c in prompt if c in char2idx]
    if not ids:
        print("[警告] prompt 中没有词表内的字符，使用随机起始 token。")
        ids = [random.randint(0, len(char2idx) - 1)]

    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

    for _ in range(max_new_tokens):
        # 只取最后 seq_len 个 token（防止超出位置编码范围）
        x_cond = x[:, -512:]
        logits = model(x_cond)            # (1, T, V)
        logits = logits[:, -1, :]         # 取最后一步的 logit (1, V)

        # 温度缩放
        logits = logits / temperature

        # Top-k 截断
        if top_k > 0:
            topk_vals, _ = torch.topk(logits, top_k)
            threshold = topk_vals[:, -1].unsqueeze(-1)
            logits = logits.masked_fill(logits < threshold, float("-inf"))

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)   # (1, 1)
        x = torch.cat([x, next_id], dim=1)

    generated_ids = x[0, len(ids):].tolist()
    return prompt + "".join(idx2char[i] for i in generated_ids)


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    # 模型结构
    parser.add_argument("--d_model",        type=int,   default=128)
    parser.add_argument("--nhead",          type=int,   default=4)
    parser.add_argument("--num_layers",     type=int,   default=2)
    parser.add_argument("--dim_feedforward",type=int,   default=512)
    parser.add_argument("--dropout",        type=float, default=0.1)
    # 训练
    parser.add_argument("--epochs",         type=int,   default=5)
    parser.add_argument("--seq_len",        type=int,   default=64)
    parser.add_argument("--batch_size",     type=int,   default=64)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--val_ratio",      type=float, default=0.05)
    # 数据 & 保存
    parser.add_argument("--corpus",         default="*.txt")
    parser.add_argument("--save",           default="best_transformer.pt")
    # 生成
    parser.add_argument("--generate_prompt", default="从前有座山")
    parser.add_argument("--gen_len",        type=int,   default=200)
    parser.add_argument("--temperature",    type=float, default=0.8)
    parser.add_argument("--top_p",          type=float,   default=0.9)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # ── 数据 ──
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
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=True)

    # ── 模型 ──
    model = CausalTransformerLM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    # 学习率预热 + 余弦退火（现代 Transformer 训练常用）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )

    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}  {'LR':>8}")
    print("─" * 66)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        marker = "  ✓" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
            }, args.save)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  "
              f"{va_loss:>10.4f}  {va_ppl:>10.2f}  {lr_now:>8.2e}{marker}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")

    # ── 文本生成 ──
    if args.generate_prompt:
        print(f"\n── 生成文本（prompt='{args.generate_prompt}'）──")
        # 加载最优权重
        ckpt = torch.load(args.save, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        result = generate(
            model, args.generate_prompt,
            char2idx, idx2char,
            max_new_tokens=args.gen_len,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
        )
        print(result)


if __name__ == "__main__":
    main()
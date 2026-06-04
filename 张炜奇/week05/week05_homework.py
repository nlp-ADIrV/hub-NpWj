"""
字符级语言模型训练脚本，支持 Transformer 模型，含 PPL 计算及文本生成。
用法:
    python language_model.py --epochs 20 --nhead 4
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

class PositionalEncoding(nn.Module):
    """标准Transformer位置编码"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)

class LM(nn.Module):
    """基于Transformer的单向语言模型"""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, nhead, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        
        # 使用TransformerEncoder，batch_first=True匹配(B, T, C)输入格式
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=nhead, 
            dim_feedforward=hidden_dim, # 对应原代码中的 hidden_dim，通常FFN维度
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        e = self.embed(x) * math.sqrt(self.embed_dim) # 常见的缩放
        e = self.pos_encoder(e)
        
        seq_len = x.size(1)
        device = x.device
        # 生成下三角因果掩码，防止关注未来信息
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
        
        out = self.transformer_encoder(e, mask=mask)
        logits = self.fc(out) # (B, T, V)
        return logits

def generate_text(model, start_text, char2idx, idx2char, max_len=100, device='cpu'):
    """使用训练好的模型进行文本生成"""
    model.eval()
    input_ids = [char2idx[c] for c in start_text if c in char2idx]
    if not input_ids:
        return "无法识别的起始文本"
    
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    generated_ids = input_ids.copy()
    
    for _ in range(max_len):
        with torch.no_grad():
            # 模型内部会自动生成对应长度的Causal Mask
            logits = model(input_tensor)
            
        # 取最后一个时间步的logits
        next_logits = logits[0, -1, :] 
        # 选择概率最大的token
        next_token = torch.argmax(next_logits, dim=-1).item()
        generated_ids.append(next_token)
        
        # 将新生成的token拼接到输入序列末尾，供下一次预测
        input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]], dtype=torch.long).to(device)], dim=1)
        
    return "".join([idx2char[id] for id in generated_ids])

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

# ─────────────────────────── 主函数 ───────────────────────────
def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=5) #快速训练版
    # parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=32) #快速训练版
    # parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64) #快速训练版
    # parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=64) #快速训练版
    # parser.add_argument("--hidden_dim", type=int, default=256) 
    parser.add_argument("--hidden_dim", type=int, default=128) #快速训练版
    # parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=1) #快速训练版
    # parser.add_argument("--nhead", type=int, default=4, help="Transformer 多头注意力头数")
    parser.add_argument("--nhead", type=int, default=2, help="Transformer 多头注意力头数") #快速训练版
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--corpus", default="*.txt")
    parser.add_argument("--save", default="best_model.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} model: TRANSFORMER")

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
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # 模型
    model = LM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        nhead=args.nhead,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6} {'Train Loss':>10} {'Train PPL':>10} {'Val Loss':>10} {'Val PPL':>10}")
    print("-" * 56)
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        marker = " *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
            }, args.save)

        print(f"{epoch:>6} {tr_loss:>10.4f} {tr_ppl:>10.2f} {va_loss:>10.4f} {va_ppl:>10.2f}{marker}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f} 已保存至 {args.save}")

    # ─────────────────────────── 文本生成测试 ───────────────────────────
    print("\n--- 文本生成测试 ---")
    # 加载模型进行生成
    checkpoint = torch.load(args.save, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # 测试续写一个简短的文本
    start_text = "中国"
    generated_text = generate_text(model, start_text, char2idx, idx2char, max_len=10, device=device)
    print(f"起始文本: '{start_text}'")
    print(f"生成结果: '{generated_text}'")

if __name__ == "__main__":
    main()

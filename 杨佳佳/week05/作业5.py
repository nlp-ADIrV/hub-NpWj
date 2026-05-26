import math
import argparse
import glob
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# -------------------------- 1. 参数解析 --------------------------
parser = argparse.ArgumentParser(description="Transformer单向语言模型作业")
parser.add_argument("--model", default="transformer", choices=["rnn", "lstm", "transformer"])
parser.add_argument("--epochs", type=int, default=25)
parser.add_argument("--seq_len", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--embed_dim", type=int, default=128)
parser.add_argument("--hidden_dim", type=int, default=256)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--num_heads", type=int, default=4)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--corpus", default="*.txt")
parser.add_argument("--save", default="lm_best.pt")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"运行设备: {device} | 模型: {args.model}")

# -------------------------- 2. 读取语料 --------------------------
def load_corpus(pattern="*.txt"):
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(text)

corpus = load_corpus(args.corpus)
print(f"语料总长度: {len(corpus):,} 字符")

# -------------------------- 3. 构建字符词表 --------------------------
def build_vocab(text):
    chars = sorted(list(set(text)))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for i, c in enumerate(chars)}
    return char2idx, idx2char

char2idx, idx2char = build_vocab(corpus)
VOCAB_SIZE = len(char2idx)
print(f"词表大小: {VOCAB_SIZE}")

# -------------------------- 4. 数据集类 --------------------------
class CharDataset(Dataset):
    def __init__(text, char2idx, seq_len):
        self.seq_len = seq_len
        self.data = [char2idx[c] for c in text if c in char2idx]

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+1:idx+1+self.seq_len]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

split_idx = int(len(corpus) * 0.9)
train_text = corpus[:split_idx]
val_text = corpus[split_idx:]

train_ds = CharDataset(train_text, char2idx, args.seq_len)
val_ds = CharDataset(val_text, char2idx, args.seq_len)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True)

# -------------------------- 5. 模型定义 --------------------------
class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers=num_layers,
                           batch_first=True, dropout=dropout if num_layers>1 else 0)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.emb(x)
        out, _ = self.rnn(x)
        return self.fc(out)

class LSTMLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                             batch_first=True, dropout=dropout if num_layers>1 else 0)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.emb(x)
        out, _ = self.lstm(x)
        return self.fc(out)

# 5.2 Transformer Decoder-only
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.num_heads, self.d_k).transpose(1,2)
        k = k.view(B, T, self.num_heads, self.d_k).transpose(1,2)
        v = v.view(B, T, self.num_heads, self.d_k).transpose(1,2)

        attn_score = q @ k.transpose(-2,-1) / math.sqrt(self.d_k)
        if mask is not None:
            attn_score = attn.masked_fill(mask == 0, -1e9)
        attn_weight = F.softmax(attn_score, dim=-1)
        out = attn_weight @ v
        out = out.transpose(1,2).contiguous().view(B,T,C)
        return self.out(out)

class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4),
            nn.GELU(),
            nn.Linear(embed_dim*4, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.norm1(x + self.drop(self.attn(x, mask)))
        x = self.norm2(x + self.ffn(x))
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, seq_len, dropout):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(seq_len, embed_dim)
        self.layers = nn.ModuleList([
            DecoderLayer(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embed_dim, vocab_size)

        self.register_buffer("mask", torch.tril(torch.ones(seq_len, seq_len)))

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.emb(x) + self.pos_emb(pos)
        for layer in self.layers:
            x = layer(x, self.mask[:T, :T])
        return self.fc(x)

# -------------------------- 6. 训练函数 --------------------------
def run_epoch(model, loader, criterion, optimizer, train=True):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl

# -------------------------- 7. Top-K+Top-P生成 --------------------------
def top_k_top_p_filtering(logits, top_k=50, top_p=0.9, filter_val=-1e9):
    if top_k > 0:
        top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < top_k_vals[..., -1, None]] = filter_val
    if top_p > 0.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cum_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = 0
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(1, sorted_idx, sorted_mask)
        logits[mask] = filter_val
    return logits

def generate(model, start_text, max_len=150, top_k=50, top_p=0.9, temp=0.7):
    model.eval()
    ids = [char2idx.get(c, 0) for c in start_text]
    with torch.no_grad():
        for _ in range(max_len - len(ids)):
            x = torch.tensor([ids[-args.seq_len:]], device=device)
            logits = model(x)[:, -1, :] / temp
            logits = top_k_top_p_filtering(logits, top_k, top_p)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            ids.append(next_id)
    return "".join([idx2char[i] for i in ids])

# -------------------------- 8. 主训练流程 --------------------------
if __name__ == "__main__":
    # 初始化模型
    if args.model == "rnn":
        model = RNNLM(VOCAB_SIZE, args.embed_dim, args.hidden_dim, args.num_layers, args.dropout).to(device)
    elif args.model == "lstm":
        model = LSTMLM(VOCAB_SIZE, args.embed_dim, args.hidden_dim, args.num_layers, args.dropout).to(device)
    else:
        model = TransformerLM(VOCAB_SIZE, args.embed_dim, args.num_layers, args.num_heads, args.seq_len, args.dropout).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_ppl = float("inf")

    print(f"\n===== 开始训练 {args.model} 语言模型 =====")
    print(f"{'Epoch':<6} {'Train Loss':<12} {'Train PPL':<10} {'Val Loss':<12} {'Val PPL':<10}")
    print("-" * 58)

    for epoch in range(1, args.epochs+1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, train=True)
        val_loss, val_ppl = run_epoch(model, val_loader, criterion, optimizer, train=False)

        mark = " *" if val_pp < best_ppl else ""
        if val_ppl < best_ppl:
            best_ppl = val_ppl
            torch.save(model.state_dict(), args.save)

        print(f"{epoch:<6} {tr_loss:<12.4f} {tr_ppl:<10.2f} {val_loss:<12.4f} {val_ppl:<10.2f}{mark}")

    print(f"\n训练完成！最佳验证PPL: {best_ppl:.2f}")
    print("加载最优模型，开始文本生成...\n")

    # 生成测试
    model.load_state_dict(torch.load(args.save))
    prompts = ["人类文明", "大宗商品", "黄金牛市", "科技发展", "金融市场"]
    for p in prompts:
        print(f"输入：{p}\n生成：{generate(model, p)}\n")

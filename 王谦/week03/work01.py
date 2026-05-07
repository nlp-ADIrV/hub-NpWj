"""
王谦
2026-05-01
第三周作业
中文句子分类 —— 简单 RNN 版本

任务：对一个任意包含“中”字的五个字的文本，“中”在第几位，就属于第几类。
模型：Embedding → RNN → 取最后隐藏状态 → Linear
优化：Adam (lr=1e-3)   损失：cross_entropy
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random

# ─── 超参数 ────────────────────────────────────────────────
SEED        = 42
N_SAMPLES   = 4000
MAXLEN      = 32
EMBED_DIM   = 64
HIDDEN_DIM  = 64
LR          = 1e-3
BATCH_SIZE  = 64
EPOCHS      = 20
TRAIN_RATIO = 0.8

random.seed(SEED)
torch.manual_seed(SEED)

# ─── 1. 数据生成 ────────────────────────────────────────────
POS_KEYS = ['好', '棒', '赞', '喜', '满','差','职','业','生','涯','规','划']

#随机从pos_keys中取5个字组成一句话，并随机将其中一个字替换为“中”
def make_positive():
    sent = [random.choice(POS_KEYS) for _ in range(5)]
    idx = random.randint(0, len(sent)-1)
    sent[idx] = '中'
    return ''.join(sent), idx

# print(make_positive())
#构建数据集
def build_dataset(n=N_SAMPLES):
    data = [make_positive() for _ in range(n)]
    random.shuffle(data)
    return data

# print(build_dataset(10))

# ─── 2. 词表构建与编码 ──────────────────────────────────────
#构建词表
def build_vocab(data):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab
def encode(sent, vocab, maxlen=MAXLEN):
    ids  = [vocab.get(ch, 1) for ch in sent]
    ids  = ids[:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids

# data = build_dataset(10)
# print(data)
# vocab = build_vocab(data)
# print(vocab)
# ─── 3. Dataset / DataLoader ────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(s, vocab) for s, _ in data]
        self.y = [lb for _, lb in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.float),
        )
# ─── 4. 模型定义 ────────────────────────────────────────────
class KeywordLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 5)

    def forward(self, x):
        # x: (batch, seq_len)
        e, _ = self.rnn(self.embedding(x))  # (B, L, hidden_dim)
        pooled = e.max(dim=1)[0]            # (B, hidden_dim)  对序列做 max pooling
        pooled = self.dropout(self.bn(pooled))
        out = torch.sigmoid(self.fc(pooled).squeeze(1))  # (B,)
        return out

# ─── 5. 训练与评估 ──────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            prob = model(X)
            pred = torch.argmax(prob, dim=1)
            correct += (pred == y.long()).sum().item()
            total += len(y)
    return correct / total

def train():
    print("开始训练...")
    data = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)
    print(f"  样本数：{len(data)}，词表大小：{len(vocab)}")

    split = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    test_data  = data[split:]
    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(TextDataset(test_data, vocab), batch_size=BATCH_SIZE)

    model = KeywordLSTM(vocab_size=len(vocab))  
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR) 
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量：{total_params:,}\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            optimizer.zero_grad()
            prob = model(X)
            loss = criterion(prob, y.long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, test_loader)
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  准确率val_acc={val_acc:.4f}")
    
    print("\n--- 推理示例 ---")
    model.eval()
    test_sents = build_dataset(10)
    with torch.no_grad():
        for sent, _ in test_sents:
            sent_str = ''.join(sent)
            id = sent_str.find('中')
            ids = torch.tensor([encode(sent, vocab)], dtype=torch.long)
            prob = torch.argmax(model(ids), dim=1)

            print(f"{id}   第{prob}类：  {sent}")


if __name__ == '__main__':
    train()
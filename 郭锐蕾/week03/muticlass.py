# train_you_position_cls.py
"""
5字中文文本“你”位置多分类（5类）：
- “你”在第1位 -> 类别0
- “你”在第2位 -> 类别1
- ...
- “你”在第5位 -> 类别4

模型：Embedding -> (RNN/LSTM) -> 取最后时刻隐藏状态 -> Linear(5类)
损失：CrossEntropyLoss
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────
SEED = 42
N_SAMPLES = 5000
TRAIN_RATIO = 0.8
BATCH_SIZE = 64
EPOCHS = 12
LR = 1e-3

EMBED_DIM = 32
HIDDEN_DIM = 64

random.seed(SEED)
torch.manual_seed(SEED)

# ─── 1. 数据构造 ────────────────────────────────────────────
# 仅用字符级；每条样本长度固定为5，且“你”只出现一次
CHAR_POOL = list("天地人和风雨花月山海云星光春秋冬夏晨夜红蓝绿白黑金木水火土")

def make_one_sample():
    # pos: 0~4 表示“你”的位置
    pos = random.randint(0, 4)
    chars = random.choices(CHAR_POOL, k=5)
    chars[pos] = "你"
    sent = "".join(chars)  # 长度固定5
    label = pos            # 0~4
    return sent, label

def build_dataset(n=N_SAMPLES):
    data = [make_one_sample() for _ in range(n)]
    random.shuffle(data)
    return data

# ─── 2. 词表与编码 ─────────────────────────────────────────
def build_vocab(data):
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab

def encode(sent, vocab, maxlen=5):
    ids = [vocab.get(ch, vocab["<UNK>"]) for ch in sent]
    return ids[:maxlen] + [vocab["<PAD>"]] * (maxlen - len(ids))

# ─── 3. Dataset / DataLoader ───────────────────────────────
class YouPosDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(s, vocab, 5) for s, _ in data]
        self.y = [lb for _, lb in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long),
        )

# ─── 4. 模型定义 ────────────────────────────────────────────
class PosRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_classes=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (B, 5)
        emb = self.embedding(x)          # (B, 5, E)
        out, _ = self.rnn(emb)           # (B, 5, H)
        feat = out[:, -1, :]             # 取最后时刻
        logits = self.fc(feat)           # (B, 5)
        return logits

class PosLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_classes=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)          # (B, 5, E)
        out, _ = self.lstm(emb)          # (B, 5, H)
        feat = out[:, -1, :]             # 取最后时刻
        logits = self.fc(feat)           # (B, 5)
        return logits

# ─── 5. 训练与评估 ──────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in loader:
            logits = model(X)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

def train_one_model(model_name, model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"\n===== 训练 {model_name} =====")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            logits = model(X)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    final_acc = evaluate(model, val_loader)
    print(f"{model_name} 最终验证准确率：{final_acc:.4f}")
    return model

def predict_demo(model, vocab, sents):
    model.eval()
    print("\n--- 推理示例 ---")
    with torch.no_grad():
        for sent in sents:
            x = torch.tensor([encode(sent, vocab, 5)], dtype=torch.long)
            pred = model(x).argmax(dim=1).item()  # 0~4
            print(f"文本: {sent} -> 预测类别: {pred} (即“你”在第{pred+1}位)")

def main():
    print("生成数据...")
    data = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)
    print(f"样本数: {len(data)}, 词表大小: {len(vocab)}")

    split = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data = data[split:]

    train_loader = DataLoader(YouPosDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(YouPosDataset(val_data, vocab), batch_size=BATCH_SIZE)

    rnn_model = PosRNN(vocab_size=len(vocab))
    lstm_model = PosLSTM(vocab_size=len(vocab))

    train_one_model("RNN", rnn_model, train_loader, val_loader)
    train_one_model("LSTM", lstm_model, train_loader, val_loader)

    test_sents = [
        "你春风花月",  # 你在第1位 -> 类别0
        "春你风花月",  # 第2位 -> 类别1
        "春风你花月",  # 第3位 -> 类别2
        "春风花你月",  # 第4位 -> 类别3
        "春风花月你",  # 第5位 -> 类别4
    ]

    print("\n[RNN结果]")
    predict_demo(rnn_model, vocab, test_sents)

    print("\n[LSTM结果]")
    predict_demo(lstm_model, vocab, test_sents)

if __name__ == "__main__":
    main()

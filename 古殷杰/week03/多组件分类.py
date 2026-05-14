"""
train_pos_cls_lstm.py

任务：
输入：长度为5的句子（包含“你”）
输出：“你”在第几个位置（0~4）

模型：
Embedding → LSTM → 取最后一步 → BN → Dropout → Linear → CrossEntropy
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── 参数 ─────────────────────────────────────
SEED = 42
N_SAMPLES = 4000
SEQ_LEN = 5
EMBED_DIM = 64
HIDDEN_DIM = 64
LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 20
TRAIN_RATIO = 0.8

random.seed(SEED)
torch.manual_seed(SEED)

# ─── 1. 生成数据 ──────────────────────────────
CHARS = list("啊吧的了吗好坏是有在这那很不就人都一个上也说要去会着没有看好自己")

def make_sample():
    pos = random.randint(0, 4)  # “你”的位置

    sent = []
    for i in range(SEQ_LEN):
        if i == pos:
            sent.append("你")
        else:
            sent.append(random.choice(CHARS))

    return "".join(sent), pos


def build_dataset(n=N_SAMPLES):
    return [make_sample() for _ in range(n)]


# ─── 2. 词表 ─────────────────────────────────
def build_vocab(data):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab


def encode(sent, vocab):
    return [vocab.get(ch, 1) for ch in sent]


# ─── 3. 数据集 ───────────────────────────────
class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(s, vocab) for s, _ in data]
        self.y = [lb for _, lb in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long),  # 分类必须是long
        )


# ─── 4. 模型 ─────────────────────────────────
class PositionModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)

        # 用 LSTM（比 RNN 稳定）
        self.lstm = nn.LSTM(EMBED_DIM, HIDDEN_DIM, batch_first=True)

        self.bn = nn.BatchNorm1d(HIDDEN_DIM)
        self.dropout = nn.Dropout(0.3)

        # 输出5类（位置0~4）
        self.fc = nn.Linear(HIDDEN_DIM, 5)

    def forward(self, x):
        # x: (B, L)

        e = self.embedding(x)  # (B, L, D)

        out, _ = self.lstm(e)  # (B, L, H)

        # 取最后一个时间步
        last = out[:, -1, :]  # (B, H)

        last = self.dropout(self.bn(last))

        out = self.fc(last)   # (B, 5)

        return out  # 不要 sigmoid


# ─── 5. 评估 ─────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for X, y in loader:
            logits = model(X)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            total += len(y)

    return correct / total


# ─── 6. 训练 ─────────────────────────────────
def train():
    print("生成数据...")
    data = build_dataset()
    vocab = build_vocab(data)

    split = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data = data[split:]

    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TextDataset(val_data, vocab), batch_size=BATCH_SIZE)

    model = PositionModel(len(vocab))

    # 用交叉熵
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for X, y in train_loader:
            logits = model(X)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        acc = evaluate(model, val_loader)
        print(f"Epoch {epoch+1}  loss={total_loss:.4f}  val_acc={acc:.4f}")

    print("\n测试一下：")
    test_sents = [
        "你好啊啊啊",
        "啊你啊啊啊",
        "啊啊你啊啊",
        "啊啊啊你啊",
        "啊啊啊啊你",
    ]

    for s in test_sents:
        ids = torch.tensor([encode(s, vocab)])
        pred = torch.argmax(model(ids), dim=1).item()
        print(f"{s} → 位置={pred}")


if __name__ == "__main__":
    train()

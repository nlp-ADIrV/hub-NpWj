"""
homework.py
中文句子关键词分类

任务：对一个任意包含“你”字的五个字的文本，“你”在第几位，就属于第几类
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────
SEED        = 42
N_SAMPLES   = 4000
MAXLEN      = 5
EMBED_DIM   = 64
HIDDEN_DIM  = 64
LR          = 1e-3
BATCH_SIZE  = 64
EPOCHS      = 20
TRAIN_RATIO = 0.8

random.seed(SEED)
torch.manual_seed(SEED)

# ─── 1. 数据生成 ────────────────────────────────────────────

# 构造简单词表
vocabulary = {"<PAD>": 0, "<UNK>": 1, "你": 2, "我": 3, "他": 4, "她": 5, "它": 6, "的": 7, "了": 8, "是": 9, "在": 10, "有": 11, "和": 12, "不": 13, "就": 14}


# 生成一个样本
# 随机生成一个5维向量，包含“你”字，根据“你”字所在位置构建Y
def build_sample():
    # 随机生成一个5维向量
    keys = list(vocabulary.values())
    keys.remove(2)  # 移除"你"字的ID
    x = random.choices(keys, k=MAXLEN-1)  # 随机选择4个词
    # 随机插入一个位置放入“你”字
    you_pos = random.randint(0, MAXLEN-1)
    x.insert(you_pos, 2)  # 在随机位置插入"你"字的ID
    return x, you_pos



def build_dataset(n=N_SAMPLES):
    data = []
    for _ in range(n):
        x, y = build_sample()
        data.append((x, y))
    return data



# ─── 2. 编码 ──────────────────────────────────────
def encode(sent, vocab, maxlen=MAXLEN):
    ids  = [vocab.get(ch, 1) for ch in sent]
    ids  = ids[:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids


# ─── 3. Dataset / DataLoader ────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, data):
        self.X = [s for s, _ in data]
        self.y = [lb for _, lb in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long),
        )


# ─── 4. 模型定义 ────────────────────────────────────────────
class KeywordLSTM(nn.Module):
    """
    中文关键词分类器（LSTM + MaxPooling 版）
    架构：Embedding → LSTM → MaxPool → BN → Dropout → Linear → Sigmoid → (MSELoss)
    """
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.LSTM       = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.bn        = nn.BatchNorm1d(hidden_dim)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim, MAXLEN)


    def forward(self, x, y=None):
        # x: (batch, seq_len)
        e, _ = self.LSTM(self.embedding(x))  # (B, L, hidden_dim)
        pooled = e.max(dim=1)[0]            # (B, hidden_dim)  对序列做 max pooling
        pooled = self.dropout(self.bn(pooled))
        out = self.fc(pooled)                # (B, 5)  输出每个类别的得分
        if y is not None:
            return nn.functional.cross_entropy(out, y)  # 预测值和真实值计算损失
        else:
            return torch.softmax(out, dim=1)  # 输出预测结果



# ─── 5. 训练与评估 ──────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            prob    = model(X)
            pred    = torch.argmax(prob, dim=1)  # 取概率最高的类别作为预测结果
            correct += (pred == y).sum().item()
            total   += len(y)
    return correct / total



def train():
    print("生成数据集...")
    data  = build_dataset(N_SAMPLES)
    vocab = vocabulary
    print(f"  样本数：{len(data)}，词表大小：{len(vocab)}")

    split      = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data   = data[split:]

    train_loader = DataLoader(TextDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TextDataset(val_data), batch_size=BATCH_SIZE)

    model     = KeywordLSTM(vocab_size=len(vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量：{total_params:,}\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            loss = model(X, y) # 前向传播计算损失
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc  = evaluate(model, val_loader)
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    print(f"\n最终验证准确率：{evaluate(model, val_loader):.4f}")

    print("\n--- 推理示例 ---")
    model.eval()
    test_sents = [
        '你我他她它',
        '我你他她它',
        '他我你她它',
        '她我他你它',
        '它我他她你',
        '你这款产品真的很棒，非常满意',
        '1231你',
    ]
    with torch.no_grad():
        for sent in test_sents:
            # 这里外层再套一个中括号，是因为模型输入需要是一个batch，即使只有一个样本，也要保持维度一致
            ids   = torch.tensor([encode(sent, vocab)], dtype=torch.long)
            res = torch.argmax(model(ids), dim=1).item()  # 取概率最高的类别作为预测结果
            print(f'{sent} 预测类别：{res}')


if __name__ == '__main__':
    train()

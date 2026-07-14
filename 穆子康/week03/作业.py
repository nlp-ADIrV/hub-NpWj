import random
import torch
import torch.nn as nn

# ===================== 超参数 =====================
SEED = 42
N_SAMPLES = 4000
MAXLEN = 5
EMBED_DIM = 16
HIDDEN_DIM = 32
LR = 5e-4
BATCH_SIZE = 128
EPOCHS = 50
TRAIN_RATIO = 0.8

random.seed(SEED)
torch.manual_seed(SEED)

# 生成单条样本
def build_one_sample():
    sentence = [""] * 5
    target_pos = random.randint(0, 4)
    sentence[target_pos] = "你"
    others = ["我", "他", "她", "它"]
    for i in range(5):
        if sentence[i] == "":
            sentence[i] = random.choice(others)
    return sentence, target_pos

# 生成全部数据
def build_dataset(n):
    data = []
    for _ in range(n):
        sent, label = build_one_sample()
        data.append((sent, label))
    random.shuffle(data)
    return data

# 构建词表
def build_vocab(data):
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab

# 文字转数字编码
def encode(sent, vocab, maxlen=MAXLEN):
    ids = [vocab.get(ch, 1) for ch in sent]
    ids = ids[:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids

# ===================== 模型：只去掉CNN，其余全部保留 =====================
class FinalModel(nn.Module):
    def __init__(self, vocab_size, num_classes=5):
        super().__init__()

        # 1. Embedding 层
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)

        # 2. LSTM 层
        self.lstm = nn.LSTM(EMBED_DIM, HIDDEN_DIM, batch_first=True)

        # 3. 池化层
        self.pool = nn.AdaptiveMaxPool1d(1)

        # 4. 归一化层
        self.norm = nn.LayerNorm(HIDDEN_DIM)

        # 5. Dropout
        self.dropout = nn.Dropout(0.2)

        # 6. 全连接层
        self.fc = nn.Linear(HIDDEN_DIM, num_classes)

    def forward(self, x):
        # 词嵌入
        x = self.embedding(x)  # [B, 5, 16]

        # LSTM 时序特征
        out, _ = self.lstm(x)  # [B, 5, 32]

        # 池化
        out = out.transpose(1, 2)  # [B, 32, 5]
        feat = self.pool(out).squeeze(-1)  # [B, 32]

        # 归一化 + Dropout
        feat = self.norm(feat)
        feat = self.dropout(feat)

        # 分类
        out = self.fc(feat)
        return out

# 评估
def evaluate(model, test_x, test_y):
    model.eval()
    with torch.no_grad():
        pred = model(test_x)
        pred_class = torch.argmax(pred, dim=1)
        acc = (pred_class == test_y).sum() / len(test_y)
    return acc.item()

# 训练
def train():
    data = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)

    all_x = []
    all_y = []
    for sent, label in data:
        all_x.append(encode(sent, vocab))
        all_y.append(label)

    all_x = torch.tensor(all_x, dtype=torch.long)
    all_y = torch.tensor(all_y, dtype=torch.long)

    split = int(len(all_x) * TRAIN_RATIO)
    train_x, test_x = all_x[:split], all_x[split:]
    train_y, test_y = all_y[:split], all_y[split:]

    model = FinalModel(len(vocab))
    criterion = nn.CrossEntropyLoss()

    # Adam 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for start in range(0, len(train_x), BATCH_SIZE):
            end = start + BATCH_SIZE
            xb = train_x[start:end]
            yb = train_y[start:end]

            pred = model(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        acc = evaluate(model, test_x, test_y)
        print(f"第 {epoch+1}轮 | loss={total_loss:.4f} | acc={acc:.4f}")

    # 测试
    print("\n=== 测试结果 ===")
    model.eval()
    test_sents = [
        ["我", "你", "他", "她", "它"],
        ["我", "他", "你", "她", "它"],
        ["我", "他", "她", "你", "它"],
        ["我", "他", "她", "它", "你"]
    ]
    with torch.no_grad():
        for s in test_sents:
            ids = encode(s, vocab)
            x = torch.tensor([ids])
            pos = torch.argmax(model(x)).item()
            print(f"{s} -> 你在第 {pos} 位")

if __name__ == "__main__":
    train()

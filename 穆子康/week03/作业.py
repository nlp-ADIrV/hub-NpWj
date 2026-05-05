# 导入需要的工具包
import random
import torch
import torch.nn as nn

# ===================== 超参数 =====================
SEED = 42
N_SAMPLES = 4000
MAXLEN = 5
EMBED_DIM = 16
HIDDEN_DIM = 32
LR = 1e-3
BATCH_SIZE = 128
EPOCHS = 15
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

# ===================== LSTM 模型=====================
class TextLSTM(nn.Module):
    def __init__(self, vocab_size, num_classes=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        self.lstm = nn.LSTM(EMBED_DIM, HIDDEN_DIM, batch_first=True)
        self.fc = nn.Linear(HIDDEN_DIM, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.lstm(x)
        feat = output[:, -1, :]
        out = self.fc(feat)
        return out

# 计算准确率
def evaluate(model, test_x, test_y):
    model.eval()
    with torch.no_grad():
        pred = model(test_x)
        pred_class = torch.argmax(pred, dim=1)
        acc = (pred_class == test_y).sum() / len(test_y)
    return acc.item()

def train():
    # 1. 造数据
    data = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)

    # 2. 全部提前编码成 数字张量
    all_x = []
    all_y = []
    for sent, label in data:
        all_x.append(encode(sent, vocab))
        all_y.append(label)

    # 转成张量
    all_x = torch.tensor(all_x, dtype=torch.long)
    all_y = torch.tensor(all_y, dtype=torch.long)

    # 划分训练/测试
    split = int(len(all_x) * TRAIN_RATIO)
    train_x, test_x = all_x[:split], all_x[split:]
    train_y, test_y = all_y[:split], all_y[split:]

    # 初始化模型、损失、优化器
    model = TextLSTM(len(vocab))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 训练
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for start in range(0, len(train_x), BATCH_SIZE):
            end = start + BATCH_SIZE
            x_batch = train_x[start:end]
            y_batch = train_y[start:end]

            # 前向传播 + 损失
            pred = model(x_batch)
            loss = criterion(pred, y_batch)

            # 梯度+更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 测试准确率
        acc = evaluate(model, test_x, test_y)
        print(f"第 {epoch+1}轮 | loss={total_loss:.4f} | acc={acc:.4f}")

    # 推理演示
    print("\n=== 推理演示 ===")
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
            tensor_x = torch.tensor([ids], dtype=torch.long)
            out = model(tensor_x)
            pos = torch.argmax(out).item()
            print(f"{s}  → 你在第 {pos} 位")

if __name__ == "__main__":
    train()

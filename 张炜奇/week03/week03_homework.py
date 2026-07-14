"""
中文文本多分类任务 —— RNN、LSTM 模型实验
任务：输入5字文本，"你"在第几位就属于第几类（0/1/2/3/4）
模型：Embedding → RNN/LSTM → 展平保留位置信息 → BN → Dropout → Linear
损失：CrossEntropyLoss
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────
SEED = 42
N_SAMPLES = 4000
SEQ_LEN = 5            # 固定5个字
EMBED_DIM = 64
HIDDEN_DIM = 64
LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 20
TRAIN_RATIO = 0.8
NUM_CLASSES = 5        # 5个类别（"你"在位置0~4）

random.seed(SEED)
torch.manual_seed(SEED)

# ─── 1. 数据生成 ────────────────────────────────────────────
# 常用填充汉字（不含"你"）
FILLER_CHARS = list("我今天很开心学东西吃好看书跑步写代码打游戏去上班吃饭睡觉看电影听音乐逛街买衣服做运动锻炼身体旅游散心")
FILLER_CHARS = [c for c in FILLER_CHARS if c != '你']


def make_sample():
    """生成一个5字样本：'你'在随机位置，标签 = 位置索引"""
    pos = random.randint(0, 4)
    chars = []
    for i in range(SEQ_LEN):
        if i == pos:
            chars.append('你')
        else:
            chars.append(random.choice(FILLER_CHARS))
    return ''.join(chars), pos


def build_dataset(n=N_SAMPLES):
    data = [make_sample() for _ in range(n)]
    random.shuffle(data)
    return data


# ─── 2. 词表构建与编码 ──────────────────────────────────────
def build_vocab(data):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab


def encode(sent, vocab, maxlen=SEQ_LEN):
    ids = [vocab.get(ch, 1) for ch in sent][:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids


# ─── 3. Dataset / DataLoader ────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(s, vocab) for s, _ in data]
        self.y = [lb for _, lb in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),      # (SEQ_LEN,)
            torch.tensor(self.y[i], dtype=torch.long),       # 标签用 LongTensor
        )


# ─── 4. 模型定义 ────────────────────────────────────────────
class KeywordRNN(nn.Module):
    """RNN 多分类模型
    注意：这里把 RNN 所有时间步的输出展平拼接，而不是 MaxPool。
    原因：MaxPool 只保留每个特征维度的最大值，会丢失"最大值来自哪个位置"的信息。
         而核心是判断"你"在哪个位置，位置信息不能丢。
         展平后 Linear 层可以学到"第20~83个特征对应位置2有特殊模式 → 类别2"。
    """
    def __init__(self, vocab_size, num_classes=NUM_CLASSES,
                 seq_len=SEQ_LEN, embed_dim=EMBED_DIM,
                 hidden_dim=HIDDEN_DIM, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.bn = nn.BatchNorm1d(seq_len * hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(seq_len * hidden_dim, num_classes)

    def forward(self, x):
        e, _ = self.rnn(self.embedding(x))        # (B, L, H)  L=5, H=64
        flat = e.reshape(e.size(0), -1)           # (B, L*H) = (B, 320)
        flat = self.dropout(self.bn(flat))
        return self.fc(flat)                       # (B, num_classes) 原始logits


class KeywordLSTM(nn.Module):
    """LSTM 多分类模型（结构与 RNN 版完全对称，仅替换循环层）"""
    def __init__(self, vocab_size, num_classes=NUM_CLASSES,
                 seq_len=SEQ_LEN, embed_dim=EMBED_DIM,
                 hidden_dim=HIDDEN_DIM, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.bn = nn.BatchNorm1d(seq_len * hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(seq_len * hidden_dim, num_classes)

    def forward(self, x):
        e, _ = self.lstm(self.embedding(x))       # LSTM 返回 (output, (h_n, c_n))
        flat = e.reshape(e.size(0), -1)           # (B, L*H)
        flat = self.dropout(self.bn(flat))
        return self.fc(flat)                       # (B, num_classes) 原始logits


# ─── 5. 训练与评估 ──────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            logits = model(X)                       # (B, num_classes)
            pred = logits.argmax(dim=1)             # (B,)  取最大值索引作为预测类别
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / total


def train(model, model_name, train_loader, val_loader, epochs=EPOCHS):
    print(f"\n{'=' * 55}")
    print(f"  训练模型：{model_name}")
    print(f"{'=' * 55}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量：{total_params:,}")

    criterion = nn.CrossEntropyLoss()               # 多分类损失（内部自带 softmax）
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            logits = model(X)                        # (B, num_classes)
            loss = criterion(logits, y)              # CrossEntropy 直接吃 logits
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader)
        print(f"  Epoch {epoch:2d}/{epochs}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    final_acc = evaluate(model, val_loader)
    print(f"  >>> 最终验证准确率：{final_acc:.4f}")
    return final_acc


def predict(model, model_name, vocab, sentences):
    print(f"\n--- {model_name} 推理示例 ---")
    model.eval()
    with torch.no_grad():
        for sent in sentences:
            ids = torch.tensor([encode(sent, vocab)], dtype=torch.long)
            logits = model(ids)                       # (1, num_classes)
            probs = torch.softmax(logits, dim=1)      # 转成概率分布
            pred_class = logits.argmax(dim=1).item()
            conf = probs[0, pred_class].item()
            prob_str = [f"{p:.3f}" for p in probs[0].tolist()]
            print(f"  输入：{sent}")
            print(f"    '你'在位置{sent.index('你')}  |  预测：第{pred_class}类"
                  f"（置信度 {conf:.2f}）")
            print(f"    概率分布：[{', '.join(prob_str)}]")
            mark = "正确" if pred_class == sent.index('你') else "错误"
            print(f"    判断：{mark}\n")


# ─── 6. 主流程 ──────────────────────────────────────────────
def main():
    # 生成数据
    print("生成数据集...")
    data = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)
    print(f"  样本数：{len(data)}")
    print(f"  词表大小：{len(vocab)}")
    print(f"  示例（文本, 标签）：{data[:5]}")

    # 划分训练/验证集
    split = int(len(data) * TRAIN_RATIO)
    train_data, val_data = data[:split], data[split:]
    train_loader = DataLoader(TextDataset(train_data, vocab),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TextDataset(val_data, vocab),
                            batch_size=BATCH_SIZE)

    # 训练 RNN
    rnn_model = KeywordRNN(vocab_size=len(vocab))
    rnn_acc = train(rnn_model, "RNN", train_loader, val_loader)

    # 训练 LSTM
    lstm_model = KeywordLSTM(vocab_size=len(vocab))
    lstm_acc = train(lstm_model, "LSTM", train_loader, val_loader)

    # 对比结果
    print(f"\n{'=' * 55}")
    print(f"  模型对比结果")
    print(f"{'=' * 55}")
    print(f"  RNN  准确率：{rnn_acc:.4f}")
    print(f"  LSTM 准确率：{lstm_acc:.4f}")
    diff = abs(lstm_acc - rnn_acc)
    if diff < 0.005:
        print(f"  >>> 两者表现持平（差距 {diff:.4f}）")
    elif lstm_acc > rnn_acc:
        print(f"  >>> LSTM 更好（高出 {diff:.4f}）")
    else:
        print(f"  >>> RNN 更好（高出 {diff:.4f}）")

    # 推理示例
    test_sents = [
        '我好喜欢你啊',      # '你'在位置2 → 类别2
        '你今天真开心',      # '你'在位置0 → 类别0
        '今天天气你好',      # '你'在位置4 → 类别4
        '我今天你看书',      # '你'在位置3 → 类别3
        '我你很高心啊',      # '你'在位置1 → 类别1
    ]
    predict(rnn_model, "RNN", vocab, test_sents)
    predict(lstm_model, "LSTM", vocab, test_sents)


if __name__ == '__main__':
    main()

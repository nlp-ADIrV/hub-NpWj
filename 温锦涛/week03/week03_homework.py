"""
中文文本多分类 —— RNN/LSTM 模型实验

任务： 输入5个字的中文文本，判断"你"字在第几位（0-4类）
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SEED        = 42
N_SAMPLES   = 3000
TEXT_LEN    = 5          # 固定5个字
EMBED_DIM   = 64
HIDDEN_DIM  = 64
LR          = 1e-3
BATCH_SIZE  = 64
EPOCHS      = 20
TRAIN_RATIO = 0.8

random.seed(SEED)
torch.manual_seed(SEED)

BASE_CHARS = ['我', '爱', '中', '国', '你', '是', '在', '有', '和', '人', '大', '来', '上', '学', '生']

TEMPLATES = [
    '你{}好{}',
    '我{}你{}',
    '{}爱你{}',
    '{}你很{}',
    '爱你{}好{}',
    '喜欢{}你{}',
    '{}喜欢{}你',
    '你好{}爱{}',
]


def make_sample():
    """生成5个字文本，"你"字在随机位置"""
    pos = random.randint(0, 4)  # "你"的位置
    # 在"你"的位置前后填充其他字
    prefix_len = pos
    suffix_len = 4 - pos
    prefix = ''.join(random.choice(BASE_CHARS) for _ in range(prefix_len))
    suffix = ''.join(random.choice(BASE_CHARS) for _ in range(suffix_len))
    text = prefix + '你' + suffix
    return text, pos


def build_dataset(n=N_SAMPLES):
    return [make_sample() for _ in range(n)]

def build_vocab(data):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for text, _ in data:
        for ch in text:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab


def encode(text, vocab, maxlen=TEXT_LEN):
    ids = [vocab.get(ch, 1) for ch in text]
    ids = ids[:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids

class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(text, vocab) for text, _ in data]
        self.y = [label for _, label in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long),
        )

class LSTMModel(nn.Module):
    """LSTM 版本：Embedding → LSTM → Linear → Softmax"""
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_classes=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc        = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len=5)
        _, (hidden, _) = self.lstm(self.embedding(x))  # hidden: (1, B, hidden)
        hidden         = hidden.squeeze(0)             # (B, hidden)
        out            = self.fc(hidden)               # (B, 5)
        return out

def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, name="Model"):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\n{'='*50}")
    print(f"训练 {name}，参数量：{sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*50}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            out   = model(X)
            loss  = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in val_loader:
                out    = model(X)
                pred   = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total   += len(y)
        val_acc = correct / total

        print(f"Epoch {epoch:2d}/{epochs}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    return model


def predict(model, text, vocab):
    """预测"你"字位置"""
    model.eval()
    ids = [vocab.get(ch, 1) for ch in text]
    ids = ids[:TEXT_LEN]
    ids += [0] * (TEXT_LEN - len(ids))
    ids_tensor = torch.tensor([ids], dtype=torch.long)

    with torch.no_grad():
        logits = model(ids_tensor)
        pred   = logits.argmax(dim=1).item()
        prob   = torch.softmax(logits, dim=1)[0].tolist()

    return pred, prob


# ─── 6. 主程序 ──────────────────────────────────────────────
def main():
    print("=" * 55)
    print("多分类任务：判断'你'字在文本中的位置（0-4）")
    print("=" * 55)

    # 生成数据
    print("\n生成数据集...")
    data  = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)
    print(f"  样本数：{len(data)}，词表大小：{len(vocab)}")

    # 划分数据集
    split = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data   = data[split:]

    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TextDataset(val_data, vocab), batch_size=BATCH_SIZE)

    # ── 训练 LSTM 模型 ──
    lstm_model = LSTMModel(vocab_size=len(vocab))
    lstm_model = train_model(lstm_model, train_loader, val_loader, name="LSTM")

    # ── 对比测试 ──
    print("\n" + "=" * 55)
    print("模型对比测试")
    print("=" * 55)

    test_cases = [
        "你好世界",   # 你在位置0
        "我爱你哦",   # 你在位置2
        "喜欢我你",   # 你在位置3
        "爱我喜欢你", # 你在位置4
        "我们爱你",   # 你在位置3
    ]

    print(f"\n{'文本':<12} | {'LSTM预测':<8} | {'实际位置':<8}")
    print("-" * 50)
    for text in test_cases:
        lstm_pred, lstm_prob = predict(lstm_model, text, vocab)
        actual = list(text).index('你')
        lstm_ok = "✓" if lstm_pred == actual else "✗"
        print(f"{text:<12} | {lstm_pred}{lstm_ok} ({lstm_prob[lstm_pred]:.2f}) | {actual}")


if __name__ == '__main__':
    main()
# coding: utf-8
import torch
import torch.nn as nn
import numpy as np
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ===================== 配置 =====================
MAX_LEN = 5
NUM_CLASSES = 5
VOCAB_SIZE = 100
EMBED_DIM = 16
HIDDEN_DIM = 32
BATCH_SIZE = 32
EPOCHS = 60
LR = 0.001

# ===================== 词表 =====================
vocab = {"[PAD]": 0, "[UNK]": 1, "你": 2}
common_chars = "我他她它好呀哈嘿哎哇哦嘛呢啦嘻"
for char in common_chars:
    if char not in vocab:
        vocab[char] = len(vocab)
while len(vocab) < VOCAB_SIZE:
    vocab[f"char{len(vocab)}"] = len(vocab)

# 文本编码
def encode_text(text):
    ids = [vocab.get(char, vocab["[UNK]"]) for char in text]
    ids = ids[:MAX_LEN] + [0] * (MAX_LEN - len(ids))
    return torch.tensor(ids, dtype=torch.long)

# ===================== 数据集生成 =====================
def build_sample():
    other_chars = [c for c in vocab if c != "你" and c != "[PAD]"]
    target_pos = random.randint(0, 4)
    text = [""]*5
    for i in range(5):
        if i == target_pos:
            text[i] = "你"
        else:
            text[i] = random.choice(other_chars)
    return "".join(text), target_pos

def build_dataset(n):
    x, y = [], []
    for _ in range(n):
        txt, label = build_sample()
        x.append(encode_text(txt))
        y.append(torch.tensor(label))
    return torch.stack(x), torch.stack(y)

train_x, train_y = build_dataset(8000)
test_x, test_y = build_dataset(2000)

# ===================== LSTM模型 =====================
class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=0)
        self.lstm = nn.LSTM(EMBED_DIM, HIDDEN_DIM, batch_first=True)
        self.fc = nn.Linear(HIDDEN_DIM * MAX_LEN, NUM_CLASSES)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.lstm(x)
        output = output.flatten(1)
        return self.fc(output)

# ===================== RNN模型 =====================
class RNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=0)
        self.rnn = nn.RNN(EMBED_DIM, HIDDEN_DIM, batch_first=True)
        self.fc = nn.Linear(HIDDEN_DIM * MAX_LEN, NUM_CLASSES)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.rnn(x)
        output = output.flatten(1)
        return self.fc(output)

# ===================== 训练 =====================
def train(model, model_name):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for i in range(0, len(train_x), BATCH_SIZE):
            bx, by = train_x[i:i+BATCH_SIZE], train_y[i:i+BATCH_SIZE]
            pred = model(bx)
            loss = loss_fn(pred, by)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        model.eval()
        acc = (model(test_x).argmax(1) == test_y).float().mean().item()
        if acc == 1.0:
            print(f"{model_name} 准确率100%，学会啦！")
            break
        print(f"{model_name} Epoch {epoch+1:2d} | 准确率: {acc:.4f}")
    return model

# 训练模型
rnn_model = train(RNNClassifier(), "RNN模型")
lstm_model = train(LSTMClassifier(), "LSTM模型")

# ===================== 预测 =====================
def predict(model, text):
    model.eval()
    x = encode_text(text).unsqueeze(0)
    idx = model(x).argmax(1).item()
    return f"文本：{text} → 「你」在第{idx+1}位（类别{idx}）"

# 测试句子
test_texts = [
    "你哈呀哈哈",
    "我哈哈你它",
    "他她你它哈",
    "哈嘿你嘿哦",
    "嘻嘻啦呀你"
]
print("\n=================== RNN模型预测结果 ====================")
for t in test_texts:
    print(predict(rnn_model, t))

print("\n=================== LSTM模型预测结果 ====================")
for t in test_texts:
    print(predict(lstm_model, t))

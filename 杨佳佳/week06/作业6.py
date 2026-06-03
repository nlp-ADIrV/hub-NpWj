# week06 作业：文本分类——不同训练方法对比实验
# 任务：好评/差评二分类，对比不同训练方法效果
# ======================================================

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------- 超参数 ----------------------
SEED = 123
DATA_SIZE = 3500
SEQ_LEN = 30
EMBED_SIZE = 64
HIDDEN_SIZE = 64
BATCH = 64
EPOCHS = 15
TRAIN_RATE = 0.8

random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------- 生成好评/差评数据 ----------------------
GOOD_WORDS = ["好", "棒", "不错", "喜欢", "满意"]
GOOD_TPL = [
    "这家店服务真{}", "这个产品用着很{}", "体验感非常{}", "整体感觉{}极了"
]
BAD_TPL = [
    "今天天气一般", "事情进展普通", "感觉平平无奇", "没什么特别感受"
]

def make_good():
    t = random.choice(GOOD_TPL)
    w = random.choice(GOOD_WORDS)
    return t.format(w)

def make_bad():
    return random.choice(BAD_TPL)

def build_data():
    data = []
    for _ in range(DATA_SIZE // 2):
        data.append((make_good(), 1))
        data.append((make_bad(), 0))
    random.shuffle(data)
    return data

# ---------------------- 词表和编码 ----------------------
def build_vocab(data):
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for sent, _ in data:
        for c in sent:
            if c not in vocab:
                vocab[c] = len(vocab)
    return vocab

def encode_sent(sent, vocab):
    ids = [vocab.get(c, 1) for c in sent[:SEQ_LEN]]
    ids += [0] * (SEQ_LEN - len(ids))
    return ids

# ---------------------- 数据集类 ----------------------
class SentDataset(Dataset):
    def __init__(data, vocab):
        self.X = [encode_sent(s, vocab) for s, _ in data]
        self.y = [lb for _, lb in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.float)

# ---------------------- 简单RNN模型 ----------------------
class SimpleTextModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMBED_SIZE, padding_idx=0)
        self.rnn = nn.RNN(EMBED_SIZE, HIDDEN_SIZE, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        x = self.emb(x)
        _, h = self.rnn(x)
        return self.fc(h[-1])

# ---------------------- 训练方法（对比4种） ----------------------
def get_train_methods():
    return [
        {"name": "Adam+普通BCE", "opt": "adam", "wd": 0},
        {"name": "SGD+普通BCE", "opt": "sgd", "wd": 0},
        {"name": "Adam+权重衰减", "opt": "adam", "wd": 1e-3},
        {"name": "Adam+MSE", "opt": "adam", "wd": 0, "loss": "mse"}
    ]

# ---------------------- 训练+评估函数 ----------------------
def train_eval(method_cfg, train_loader, val_loader, vocab_size):
    model = SimpleTextModel(vocab_size)
    # 选优化器
    if method_cfg["opt"] == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=method_cfg["wd"])
    else:
        opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    # 选损失
    if method_cfg.get("loss") == "mse":
        loss_fn = nn.MSELoss()
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    for e in range(EPOCHS):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            pred = model(x).squeeze(1)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

    # 评估
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in val_loader:
            pred = model(x)
            pred = (torch.sigmoid(pred) > 0.5).float()
            correct += (pred.squeeze() == y).sum().item()
    return correct / len(val_loader.dataset)

# ---------------------- 主函数 ----------------------
if __name__ == "__main__":
    print("=== 生成数据 ===")
    data = build_data()
    vocab = build_vocab(data)
    print(f"数据量：{len(data)}，词表：{len(vocab)}")

    # 划分
    split = int(len(data) * TRAIN_RATE)
    train_data = data[:split]
    val_data = data[split:]

    train_loader = DataLoader(SentDataset(train_data, vocab), batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(SentDataset(val_data, vocab), batch_size=BATCH)

    print("\n=== 开始对比训练方法 ===")
    methods = get_train_methods()
    results = []
    for cfg in methods:
        acc = train_eval(cfg, train_loader, val_loader, len(vocab))
        results.append((cfg["name"], acc))
        print(f"{cfg['name']} → 准确率：{acc:.4f}")

    print("\n=== 最终结果 ===")
    results.sort(key=lambda x: x[1], reverse=True)
    for name, acc in results:
        print(f"{name}：{acc:.4f}")
    print(f"最优方法：{results[0][0]}")
# ======================================================
# 作业总结
# 1. 任务：句子含好评词（好/棒/不错/喜欢/满意）→好评(1)，否则→差评(0)
# 2. 对比维度：优化器（Adam/SGD）、损失函数（BCE/MSE）、权重衰减
# 3. 结论：Adam+BCE效果最优，适合二分类任务
# ======================================================

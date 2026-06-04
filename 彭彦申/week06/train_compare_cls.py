"""
train_compare_cls.py
中文文本分类 —— 不同训练方法对比实验

任务：句子中含有关键字（好/棒/赞/喜欢/满意）→ 正样本(1)，否则 → 负样本(0)
模型：固定 KeywordRNN 架构，对比不同训练方法的效果
对比维度：优化器 / 损失函数 / 正则化 / 学习率策略

用法：
    python train_compare_cls.py

依赖：torch >= 2.0   (pip install torch)
"""

import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────
SEED        = 42
N_SAMPLES   = 4000
MAXLEN      = 32
EMBED_DIM   = 64
HIDDEN_DIM  = 64
BATCH_SIZE  = 64
EPOCHS      = 20
TRAIN_RATIO = 0.8

random.seed(SEED)
torch.manual_seed(SEED)

# ─── 1. 数据生成（同 week02）────────────────────────────────
POS_KEYS = ['好', '棒', '赞', '喜欢', '满意']

TEMPLATES_POS = [
    '这家{}真的很{}，下次还来',
    '这款{}设计让我{}',
    '{}的服务态度让我感到{}',
    '{}体验非常{}',
    '这次购物感觉{}极了',
]

TEMPLATES_NEG = [
    '今天天气阴沉，出门忘带雨伞',
    '这部电影情节比较平淡',
    '下午开了三个小时的会议',
    '路上堵车耽误了不少时间',
    '这道题做了很久还没解出来',
    '最近工作任务比较繁重',
    '超市里人很多，排队结账',
    '这个季节换季容易感冒',
    '今天作业布置得有点多',
    '公交车又晚点了十分钟',
]

OBJ_WORDS = ['店铺', '餐厅', '产品', '服务', '环境', '系统', '设计', '课程']
ADJ_WORDS = ['方便', '简洁', '独特', '舒适', '高效']


def make_positive():
    kw   = random.choice(POS_KEYS)
    tmpl = random.choice(TEMPLATES_POS)
    obj  = random.choice(OBJ_WORDS)
    try:
        sent = tmpl.format(obj, kw)
    except Exception:
        sent = obj + kw + random.choice(ADJ_WORDS)
    if random.random() < 0.3:
        extra = random.choice(POS_KEYS)
        pos   = random.randint(0, len(sent))
        sent  = sent[:pos] + extra + sent[pos:]
    return sent


def make_negative():
    base = random.choice(TEMPLATES_NEG)
    if random.random() < 0.4:
        base += random.choice(TEMPLATES_NEG)
    return base


def build_dataset(n=N_SAMPLES):
    data = []
    for _ in range(n // 2):
        data.append((make_positive(), 1))
        data.append((make_negative(), 0))
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


def encode(sent, vocab, maxlen=MAXLEN):
    ids  = [vocab.get(ch, 1) for ch in sent]
    ids  = ids[:maxlen]
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
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.float),
        )


# ─── 4. 模型定义 ────────────────────────────────────────────
class KeywordRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn       = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.bn        = nn.BatchNorm1d(hidden_dim)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        e, _ = self.rnn(self.embedding(x))
        pooled = e.max(dim=1)[0]
        pooled = self.dropout(self.bn(pooled))
        return self.fc(pooled)  # 输出 logits


# ─── 5. 标签平滑损失 ────────────────────────────────────────
class LabelSmoothingBCE(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        smooth_pos = 1.0 - self.smoothing
        smooth_neg = self.smoothing
        smooth_targets = targets * smooth_pos + (1 - targets) * smooth_neg
        return F.binary_cross_entropy_with_logits(logits, smooth_targets)


# ─── 6. 训练配置定义 ────────────────────────────────────────
TRAINING_METHODS = [
    {
        "name": "Adam + BCE",
        "optimizer": "adam",
        "loss_fn": "bce",
        "weight_decay": 0.0,
        "lr_scheduler": None,
        "label_smoothing": 0.0,
        "optimizer_kwargs": {},
    },
    {
        "name": "SGD + BCE",
        "optimizer": "sgd",
        "loss_fn": "bce",
        "weight_decay": 0.0,
        "lr_scheduler": None,
        "label_smoothing": 0.0,
        "optimizer_kwargs": {"momentum": 0.9},
    },
    {
        "name": "Adam + MSE",
        "optimizer": "adam",
        "loss_fn": "mse",
        "weight_decay": 0.0,
        "lr_scheduler": None,
        "label_smoothing": 0.0,
        "optimizer_kwargs": {},
    },
    {
        "name": "AdamW + weight decay",
        "optimizer": "adamw",
        "loss_fn": "bce",
        "weight_decay": 1e-2,
        "lr_scheduler": None,
        "label_smoothing": 0.0,
        "optimizer_kwargs": {},
    },
    {
        "name": "Adam + LabelSmooth",
        "optimizer": "adam",
        "loss_fn": "bce",
        "weight_decay": 0.0,
        "lr_scheduler": None,
        "label_smoothing": 0.1,
        "optimizer_kwargs": {},
    },
    {
        "name": "Adam + CosineLR",
        "optimizer": "adam",
        "loss_fn": "bce",
        "weight_decay": 0.0,
        "lr_scheduler": "cosine",
        "label_smoothing": 0.0,
        "optimizer_kwargs": {},
    },
]


def make_criterion(cfg):
    if cfg["loss_fn"] == "mse":
        return nn.MSELoss()
    if cfg["label_smoothing"] > 0:
        return LabelSmoothingBCE(smoothing=cfg["label_smoothing"])
    return nn.BCEWithLogitsLoss()


def make_optimizer(model, cfg, lr=1e-3):
    opt_name = cfg["optimizer"]
    wd = cfg["weight_decay"]
    kwargs = cfg.get("optimizer_kwargs", {})
    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd, **kwargs)
    elif opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, **kwargs)
    elif opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, **kwargs)
    raise ValueError(f"Unknown optimizer: {opt_name}")


def make_scheduler(optimizer, cfg):
    if cfg["lr_scheduler"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    return None


# ─── 7. 训练与评估 ──────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            logits = model(X)
            pred   = (torch.sigmoid(logits).squeeze(1) > 0.5).long()
            correct += (pred == y.long()).sum().item()
            total   += len(y)
    return correct / total


def train_method(cfg, train_loader, val_loader, vocab_size, verbose=True):
    model = KeywordRNN(vocab_size=vocab_size)
    criterion = make_criterion(cfg)
    optimizer = make_optimizer(model, cfg)
    scheduler = make_scheduler(optimizer, cfg)

    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"\n{'='*60}")
        print(f"  方法: {cfg['name']}")
        print(f"  模型参数量: {total_params:,}")
        print(f"{'='*60}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            logits = model(X).squeeze(1)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if scheduler is not None:
            scheduler.step()

        avg_loss = total_loss / len(train_loader)
        val_acc  = evaluate(model, val_loader)

        if verbose:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}  lr={lr_now:.2e}")

    final_acc = evaluate(model, val_loader)
    if verbose:
        print(f"  最终验证准确率: {final_acc:.4f}")
    return final_acc


# ─── 8. 主流程 ──────────────────────────────────────────────
def main():
    print("生成数据集...")
    data  = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)
    print(f"  样本数: {len(data)}，词表大小: {len(vocab)}")

    split      = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data   = data[split:]

    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TextDataset(val_data,   vocab), batch_size=BATCH_SIZE)

    print(f"\n{'='*60}")
    print(f"  开始对比 {len(TRAINING_METHODS)} 种训练方法")
    print(f"{'='*60}")

    results = []
    for cfg in TRAINING_METHODS:
        acc = train_method(cfg, train_loader, val_loader, len(vocab))
        results.append((cfg["name"], acc))

    print(f"\n{'='*60}")
    print("  实验结果汇总")
    print(f"{'='*60}")
    print(f"  {'训练方法':<25}  {'验证准确率':>10}")
    print(f"  {'-'*25}  {'-'*10}")
    for name, acc in sorted(results, key=lambda x: x[1], reverse=True):
        print(f"  {name:<25}  {acc:>10.4f}")

    best = max(results, key=lambda x: x[1])
    print(f"\n  最佳方法: {best[0]}  (val_acc = {best[1]:.4f})")


if __name__ == '__main__':
    main()

"""
my_train_chinese_cls_rnn.py
中文句子关键词分类 —— 简单 RNN 版本

任务：文本主题分类任务 → 体育(0)娱乐(1)科技(2)美食(3)
模型：Embedding → RNN → 取最后隐藏状态 → Linear → Sigmoid
优化：Adam (lr=1e-3)   损失：MSELoss   无需 GPU，CPU 即可运行

依赖：torch >= 2.0   (pip install torch)
"""

import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ─── 超参数 ────────────────────────────────────────────────
SEED = 42
N_SAMPLES = 4000
MAXLEN = 32
EMBED_DIM = 64
HIDDEN_DIM = 64
LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 20
TRAIN_RATIO = 0.8

random.seed(SEED)
torch.manual_seed(SEED)

# ─── 1. 数据生成 ────────────────────────────────────────────
# 任务：文本主题分类 → 体育(0) / 娱乐(1) / 科技(2) / 美食(3)
N_CLASSES = 4
CLASS_NAMES = ["体育", "娱乐", "科技", "美食"]

# 每个类别的关键词
KEYWORDS = {
    0: [
        "足球",
        "篮球",
        "比赛",
        "冠军",
        "运动员",
        "教练",
        "世界杯",
        "奥运会",
        "联赛",
        "进球",
    ],
    1: [
        "电影",
        "电视剧",
        "明星",
        "导演",
        "票房",
        "音乐",
        "演唱会",
        "综艺",
        "演员",
        "歌手",
    ],
    2: [
        "手机",
        "电脑",
        "人工智能",
        "机器人",
        "芯片",
        "互联网",
        "算法",
        "自动驾驶",
        "编程",
        "数据",
    ],
    3: ["火锅", "烧烤", "蛋糕", "咖啡", "奶茶", "餐厅", "菜品", "厨师", "甜点", "外卖"],
}

# 每个类别的句子模板
TEMPLATES = {
    0: [
        "今天的{}比赛非常精彩",
        "这场{}看得很过瘾",
        "他是最优秀的{}运动员",
        "今年的{}联赛太激烈了",
        "那个{}进球太漂亮了",
        "{}世界杯快开始了",
        "我昨晚看了{}比赛",
        "这个{}教练很有经验",
        "{}奥运会马上要开幕了",
        "这场{}冠军赛很紧张",
    ],
    1: [
        "这部{}真的很好看",
        "最近的{}很火",
        "那个{}演得不错",
        "今年的{}票房很高",
        "我昨天去看了{}演唱会",
        "这个{}节目很有趣",
        "{}导演的新作品上映了",
        "我追的{}终于更新了",
        "那个{}歌手唱得很好听",
        "最近{}综艺很受欢迎",
    ],
    2: [
        "这款{}性能很强",
        "最近{}技术发展很快",
        "那个{}设计得很巧妙",
        "未来的{}会改变生活",
        "我在学{}编程",
        "{}算法越来越智能了",
        "这台{}运行速度很快",
        "{}互联网应用很广泛",
        "最新的{}芯片发布了",
        "{}自动驾驶技术很先进",
    ],
    3: [
        "这家{}很好吃",
        "今天的{}味道不错",
        "我刚点了{}外卖",
        "周末去吃了{}大餐",
        "这个{}做法很简单",
        "我很喜欢喝{}奶茶",
        "那家{}餐厅很有名",
        "这个{}甜点很精致",
        "今天的{}咖啡很好喝",
        "我想吃{}烧烤",
    ],
}

# 辅助修饰词，用于丰富句子
FILL_WORDS = {
    0: ["职业", "业余", "国际", "国内", "青少年", "大学", "城市", "乡村"],
    1: ["国产", "好莱坞", "经典", "最新", "热门", "小众", "年度", "网络"],
    2: ["最新", "智能", "高端", "入门", "专业", "便携", "家用", "商用"],
    3: ["四川", "广东", "日式", "法式", "传统", "创意", "街边", "高级"],
}


def make_sample(category_id):
    key_word = random.choice(KEYWORDS[category_id])
    template = random.choice(TEMPLATES[category_id])

    # 50% 的概率添加辅助修饰词
    if random.random() < 0.5:
        fill_word = random.choice(FILL_WORDS[category_id])
        key_word = fill_word + key_word

    sent = template.format(key_word)
    return sent, category_id


def build_dataset(n=N_SAMPLES):
    data = []
    samples_per_class = n // N_CLASSES

    for category_Id in range(N_CLASSES):
        for _ in range(samples_per_class):
            sent, label = make_sample(category_Id)
            data.append((sent, label))

    random.shuffle(data)
    return data


# ─── 2. 词表构建与编码 ──────────────────────────────────────
def build_vocab(data):
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab


def encode(sent, vocab, maxlen=MAXLEN):
    ids = [vocab.get(ch, 1) for ch in sent]
    ids = ids[:maxlen]
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
            torch.tensor(self.y[i], dtype=torch.long),
        )


# ─── 4. 模型定义 ────────────────────────────────────────────
class KeywordRNN(nn.Module):
    """
    中文关键词分类器（RNN + MaxPooling 版）
    架构：Embedding → RNN → MaxPool → BN → Dropout → Linear → Sigmoid → (MSELoss)
    """

    def __init__(
        self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=0.3
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, N_CLASSES)

    def forward(self, x):
        # x: (batch, seq_len)
        e, _ = self.rnn(self.embedding(x))  # (B, L, hidden_dim)
        pooled = e.max(dim=1)[0]  # (B, hidden_dim)  对序列做 max pooling
        pooled = self.dropout(self.bn(pooled))
        out = self.fc(pooled)  # sigmoid
        return out


# ─── 5. 训练与评估 ──────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            logits = model(X)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / total


def train():
    print("生成数据集...")
    data = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)
    print(f"  样本数：{len(data)}，词表大小：{len(vocab)}")

    split = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data = data[split:]

    train_loader = DataLoader(
        TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(TextDataset(val_data, vocab), batch_size=BATCH_SIZE)

    model = KeywordRNN(vocab_size=len(vocab))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量：{total_params:,}\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            pred = model(X)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    print(f"\n最终验证准确率：{evaluate(model, val_loader):.4f}")

    print("\n--- 推理示例 ---")
    model.eval()
    test_sents = [
        "今天的足球比赛非常精彩",
        "那部电影真的很好看",
        "这款手机性能很强",
        "这家火锅很好吃",
    ]
    with torch.no_grad():
        for sent in test_sents:
            ids = torch.tensor([encode(sent, vocab)], dtype=torch.long)
            logits = model(ids)
            pred = logits.argmax(dim=1).item()
            probs = torch.softmax(logits, dim=1)
            conf = probs[0, pred].item()
            label = CLASS_NAMES[pred]
            print(f"  [{label}({conf:.2f})]  {sent}")


# --- 临时测试数据生成 ---
if __name__ == "__main__":
    train()

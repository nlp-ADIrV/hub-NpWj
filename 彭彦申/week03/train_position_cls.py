"""
train_position_cls.py
中文文本位置分类 —— RNN / LSTM 多分类实验

任务：随机生成含"你"的5字文本，"你"在第几位(0~4)就属于第几类（5分类）
输入：5个中文字符
标签：0, 1, 2, 3, 4 （"你"所在位置）
模型：Embedding → RNN/LSTM → MaxPool → Linear → Softmax
损失：CrossEntropyLoss   优化：Adam

依赖：torch >= 2.0   (pip install torch)
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────
SEED        = 42
N_SAMPLES   = 8000
MAXLEN      = 5
EMBED_DIM   = 64
HIDDEN_DIM  = 64
LR          = 1e-3
BATCH_SIZE  = 64
EPOCHS      = 20
TRAIN_RATIO = 0.8

random.seed(SEED)
torch.manual_seed(SEED)

# ─── 1. 数据生成 ───────────────────────────────────────────

CHAR_POOL = [
    '我', '们', '他', '她', '它', '这', '那', '什', '么',
    '大', '小', '多', '少', '高', '低', '长', '短', '新', '旧',
    '好', '坏', '美', '丑', '快', '慢', '早', '晚', '冷', '热',
    '天', '地', '人', '山', '水', '火', '风', '云', '花', '草',
    '书', '笔', '桌', '窗', '门', '路', '车', '房', '灯', '树',
    '吃', '喝', '走', '跑', '看', '听', '说', '写', '读', '想',
    '日', '月', '星', '光', '明', '暗', '白', '黑', '红', '绿',
    '手', '心', '口', '头', '眼', '脚', '身', '声', '气', '力',
    '学', '生', '师', '校', '课', '考', '题', '答', '文', '字',
    '电', '影', '音', '乐', '唱', '歌', '舞', '画', '图', '游',
    '家', '妈', '爸', '哥', '弟', '妹', '姐', '友', '爱', '情',
    '东', '南', '西', '北', '上', '下', '左', '右', '前', '后',
    '中', '里', '外', '边', '旁', '远', '近', '内', '间', '处',
    '春', '夏', '秋', '冬', '年', '月', '日', '时', '分', '秒',
    '一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
    '金', '木', '水', '火', '土', '石', '铁', '铜', '银', '纸',
    '鱼', '鸟', '虫', '马', '牛', '羊', '狗', '猫', '虎', '龙',
    '开', '关', '进', '出', '来', '去', '到', '过', '回', '起',
    '忙', '闲', '富', '贫', '厚', '薄', '深', '浅', '宽', '窄',
    '安', '全', '危', '险', '容', '易', '难', '简', '复', '杂',
    '飞', '落', '升', '降', '流', '停', '动', '静', '通', '断',
    '平', '安', '喜', '欢', '怒', '哀', '乐', '悲', '苦', '甜',
    '正', '反', '真', '假', '对', '错', '是', '否', '有', '无',
    '工', '商', '农', '业', '城', '市', '村', '庄', '街', '道',
    '衣', '服', '鞋', '帽', '米', '面', '菜', '肉', '茶', '酒',
    '洗', '睡', '起', '坐', '站', '拿', '放', '给', '送', '收',
    '信', '话', '词', '句', '名', '号', '码', '数', '量', '倍',
]


def make_sample():
    pos = random.randint(0, MAXLEN - 1)
    chars = [random.choice(CHAR_POOL) for _ in range(MAXLEN)]
    chars[pos] = '你'
    return ''.join(chars), pos


def build_dataset(n_samples=N_SAMPLES):
    data = [make_sample() for _ in range(n_samples)]
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
            torch.tensor(self.y[i], dtype=torch.long),
        )


# ─── 4. 模型定义 ────────────────────────────────────────────
class PositionRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_classes=MAXLEN, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn       = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.bn        = nn.BatchNorm1d(hidden_dim)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        e, _   = self.rnn(self.embedding(x))
        pooled = e.max(dim=1)[0]
        pooled = self.dropout(self.bn(pooled))
        return self.fc(pooled)


class PositionLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_classes=MAXLEN, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.bn        = nn.BatchNorm1d(hidden_dim)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        e, _   = self.lstm(self.embedding(x))
        pooled = e.max(dim=1)[0]
        pooled = self.dropout(self.bn(pooled))
        return self.fc(pooled)

# ─── 5. 训练与评估 ──────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            logits  = model(X)
            pred    = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total   += len(y)
    return correct / total


def train_model(model_cls, model_name):
    print(f"\n{'='*60}")
    print(f"  模型: {model_name}")
    print(f"{'='*60}")

    print("生成数据集...")
    data  = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)
    print(f"  样本数：{len(data)}，词表大小：{len(vocab)}")

    split      = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data   = data[split:]

    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TextDataset(val_data,   vocab), batch_size=BATCH_SIZE)

    model     = model_cls(vocab_size=len(vocab))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量：{total_params:,}\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            logits = model(X)
            loss   = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc  = evaluate(model, val_loader)
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    final_acc = evaluate(model, val_loader)
    print(f"\n最终验证准确率：{final_acc:.4f}")

    print("\n--- 推理示例 ---")
    model.eval()
    test_samples = [make_sample() for _ in range(8)]
    print(f"    {'预测':>6} {'实际':>6}  文本")
    with torch.no_grad():
        for sent, label in test_samples:
            ids    = torch.tensor([encode(sent, vocab)], dtype=torch.long)
            logits = model(ids)
            pred   = logits.argmax(dim=1).item()
            mark   = "OK" if pred == label else "XX"
            print(f"  [{mark}] {pred:>4}   {label:>4}     {sent}")

    return final_acc


if __name__ == '__main__':
    results = {}
    for cls, name in [
        (PositionRNN,   "RNN"),
        (PositionLSTM,  "LSTM"),
    ]:
        results[name] = train_model(cls, name)

    print(f"\n{'='*60}")
    print("  实验结果汇总")
    print(f"{'='*60}")
    for name, acc in results.items():
        print(f"  {name:<10}  val_acc = {acc:.4f}")

"""
train_chinese_cls_lstm.py
中文句子关键词分类 —— 简单 LSTM 版本（按关键字位置进行多分类）

任务：句子中含有关键字（你）→ 你在第几位 就属于第几类；未含关键字的句子属于0类
模型：Embedding → LSTM → 取最后隐藏状态 → Linear → CrossEntropyLoss
优化：Adam (lr=1e-3)   损失：CrossEntropyLoss

依赖：torch >= 2.0   (pip install torch)
"""

import random
import torch
import torch.nn as nn
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────
SEED = 42
N_SAMPLES = 4000
MAXLEN = 20
EMBED_DIM = 64
HIDDEN_DIM = 128
LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 20
TRAIN_RATIO = 0.8
NUM_CLASSES = MAXLEN + 1

random.seed(SEED)
torch.manual_seed(SEED)


# 1) 数据生成与词表
def build_vocab_with_you(max_len: int) -> Tuple[dict, int]:
    pool = [
        # 核心关键字
        "你",
        "我",
        "他",
        "她",
        "它",
        "您",
        "我们",
        "你们",
        "他们",
        # 自然/生活
        "天",
        "地",
        "日",
        "月",
        "风",
        "云",
        "雨",
        "雪",
        "雷",
        "电",
        "山",
        "水",
        "火",
        "木",
        "金",
        "土",
        "石",
        "田",
        "河",
        "海",
        "春",
        "夏",
        "秋",
        "冬",
        "白",
        "黑",
        "蓝",
        "绿",
        "红",
        "黄",
        # 人物/身体
        "人",
        "口",
        "手",
        "头",
        "眼",
        "耳",
        "鼻",
        "心",
        "身",
        "家",
        "爸",
        "妈",
        "哥",
        "姐",
        "弟",
        "妹",
        "爷",
        "奶",
        "亲",
        "友",
        # 动作
        "看",
        "听",
        "说",
        "讲",
        "问",
        "答",
        "读",
        "写",
        "学",
        "教",
        "吃",
        "喝",
        "走",
        "跑",
        "跳",
        "坐",
        "站",
        "躺",
        "睡",
        "醒",
        "开",
        "关",
        "拿",
        "放",
        "找",
        "见",
        "用",
        "做",
        "玩",
        "帮",
        # 状态/形容词
        "好",
        "坏",
        "大",
        "小",
        "多",
        "少",
        "高",
        "低",
        "快",
        "慢",
        "新",
        "旧",
        "美",
        "丑",
        "冷",
        "热",
        "暖",
        "凉",
        "软",
        "硬",
        "真",
        "假",
        "对",
        "错",
        "满",
        "空",
        "静",
        "闹",
        "强",
        "弱",
        # 时间/方位
        "今",
        "明",
        "昨",
        "早",
        "晚",
        "午",
        "夜",
        "时",
        "分",
        "秒",
        "前",
        "后",
        "左",
        "右",
        "上",
        "下",
        "里",
        "外",
        "中",
        "间",
        # 代词/副词/介词
        "这",
        "那",
        "哪",
        "谁",
        "什",
        "么",
        "哪",
        "里",
        "怎",
        "么",
        "很",
        "最",
        "太",
        "都",
        "也",
        "还",
        "正",
        "刚",
        "就",
        "才",
        "在",
        "和",
        "与",
        "跟",
        "同",
        "为",
        "由",
        "从",
        "到",
        "被",
        # 名词/物品
        "书",
        "笔",
        "纸",
        "本",
        "包",
        "杯",
        "碗",
        "盘",
        "桌",
        "椅",
        "床",
        "灯",
        "门",
        "窗",
        "墙",
        "路",
        "车",
        "船",
        "票",
        "钱",
        "饭",
        "菜",
        "水",
        "茶",
        "酒",
        "花",
        "草",
        "树",
        "果",
        "鸟",
        # 助词/语气
        "的",
        "了",
        "着",
        "过",
        "吧",
        "吗",
        "呢",
        "啊",
        "呀",
        "哇",
        "啦",
        "哦",
        "嗯",
        "矣",
        "乎",
        "者",
        "也",
        "已",
        "及",
        "之",
        # 常用动词扩展
        "爱",
        "恨",
        "想",
        "念",
        "思",
        "考",
        "记",
        "忘",
        "懂",
        "会",
        "能",
        "要",
        "愿",
        "敢",
        "需",
        "求",
        "允",
        "许",
        "认",
        "为",
        # 情绪/感受
        "喜",
        "怒",
        "哀",
        "乐",
        "愁",
        "苦",
        "痛",
        "累",
        "困",
        "饿",
        "烦",
        "闷",
        "惊",
        "怕",
        "羞",
        "愧",
        "勇",
        "气",
        "安",
        "宁",
    ]
    unique_chars = sorted(set(pool))
    char_to_idx = {ch: i + 1 for i, ch in enumerate(unique_chars)}
    vocab_size = len(unique_chars)

    if "你" not in char_to_idx:
        char_to_idx["你"] = vocab_size + 1
        vocab_size += 1

    return char_to_idx, vocab_size


class ChineseLSTMDataset(Dataset):
    """
    固定长度的中文句子数据集。
    每条样本包含一个位置 pos (0-based) 的字符 '你'，其余位置为随机中文字符。
    标签为你在句子中的位置 pos，对应的类别编号为 pos（0-based）。
    """

    def __init__(
        self, num_samples: int, max_len: int, char_to_idx: dict, seed: int = 42
    ):
        self.num_samples = num_samples
        self.max_len = max_len
        self.char_to_idx = char_to_idx
        self.vocab_chars = [c for c in char_to_idx.keys()]
        self.seed = seed
        random.seed(seed)

        # 预生成数据以加速迭代
        self.samples: List[Tuple[List[int], int]] = []
        self._generate_all_samples()

    def _rand_char_excluding(self, exclude: str) -> str:
        choices = [c for c in self.vocab_chars if c != exclude]
        return random.choice(choices)

    def _generate_all_samples(self):
        for _ in range(self.num_samples):
            pos = random.randint(0, self.max_len - 1)  # 0-based 位置
            sentence_indices = []
            for i in range(self.max_len):
                if i == pos:
                    ch = "你"
                else:
                    ch = self._rand_char_excluding("你")
                idx = self.char_to_idx.get(ch, 0)  # 应该总是存在，保持为 0 作为兜底
                sentence_indices.append(idx)
            label = pos  # 0-based
            self.samples.append((sentence_indices, label))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        seq, label = self.samples[idx]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(
            label, dtype=torch.long
        )


# ─── 4. 模型定义 ────────────────────────────────────────────
class SimpleLSTMClassifier(nn.Module):
    """
    中文关键词分类器（LSTM + MaxPooling 版，按位置进行多分类）
    架构：Embedding → LSTM → MaxPool → BN → Dropout → Linear → (Logits) → CrossEntropyLoss
    """

    def __init__(
        self,
        vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        dropout=0.3,
        num_classes=NUM_CLASSES,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        e, (h_n, c_n) = self.lstm(self.embedding(x))  # (B, L, hidden_dim)
        pooled = e.max(dim=1)[0]  # (B, hidden_dim)  对序列做 max pooling
        pooled = self.dropout(self.bn(pooled))
        logits = self.fc(pooled)  # (B, num_classes)
        return logits


# ==================== 测试函数 ====================
def test_model(model, char_to_idx, max_len):
    """训练完后测试句子，预测“你”在第几个位置（0-based）"""
    model.eval()
    test_sentences = [
        "我你在",
        "你好呀",
        "今天你开心",
        "我真的很喜欢你呀",
        "我在等你出现",
        "你在哪里呀",
    ]

    print("\n" + "=" * 50)
    print(" 模型测试（预测“你”的位置）")
    print("=" * 50)

    with torch.no_grad():
        for sent in test_sentences:
            # 把句子转成索引
            ids = []
            for c in sent:
                ids.append(char_to_idx.get(c, 0))
            # 截断 + padding
            ids = ids[:max_len]
            ids += [0] * (max_len - len(ids))

            # 转成 tensor
            input_tensor = torch.tensor([ids], dtype=torch.long)

            # 预测
            logits = model(input_tensor)
            pred_pos = logits.argmax(dim=1).item()
            prob = torch.softmax(logits, dim=1)[0, pred_pos].item()
            # 输出
            print(f"句子：{sent}")
            print(f"→ 预测“你”在第【{pred_pos}】位 (置信度: {prob:.3f})")
            print("-" * 50)


# ─── 5. 训练与评估 ──────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            logits = model(X)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def train():
    print("生成数据集...")
    char_to_idx, vocab_size = build_vocab_with_you(MAXLEN)
    dataset = ChineseLSTMDataset(
        num_samples=N_SAMPLES,
        max_len=MAXLEN,
        char_to_idx=char_to_idx,
        seed=SEED,
    )  # 生成数据集
    # 划分训练/验证
    train_size = int(TRAIN_RATIO * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
    )  # 生成数据集
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )  # 生成数据集

    print(f"样本数：{len(dataset)}，词汇表：{vocab_size}")

    print("开始训练...")

    model = SimpleLSTMClassifier(vocab_size=vocab_size, num_classes=NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量：{total_params:,}\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            logits = model(X)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    print(f"\n最终验证准确率：{evaluate(model, val_loader):.4f}")

    print("\n--- 推理示例 ---")
    # ==================== 调用测试 ====================
    test_model(model, char_to_idx, NUM_CLASSES)


if __name__ == "__main__":
    train()

"""
作者：深衷浅貌
日期：2026年04月27日--22:19
项目：NLP
文件名：按字的位置分类
"""

"""
中文句子分类 —— 简单 RNN 版本

任务：对一个任意包含“你”字的五个字的文本，“你”在第几位，就属于第几类。
模型：Embedding → RNN → 取最后隐藏状态 → Linear
优化：Adam (lr=1e-3)   损失：cross_entropy
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────
SEED = 42
N_SAMPLES = 1000  # 每轮样本总数
MAXLEN = 10  # 句子长度
EMBED_DIM = 10
HIDDEN_DIM = 64
LR = 1e-3
BATCH_SIZE = 64  # 每批次样本数量
EPOCHS = 50
TRAIN_RATIO = 0.8

random.seed(SEED)
torch.manual_seed(SEED)

# ─── 1. 数据生成 ────────────────────────────────────────────
POS_KEYS = [
    '我',
    '走',
    '他',
    '好',
    '爱',
    '恨',
    '天',
    '南',
    '山',
    '河',
    '江',
    '湖',
    '海',
    '日',
    '月'
            ]




def make_positive():
    """
    构造含关键字的句子
    :return:
    """
    sent = [random.choice(POS_KEYS) for _ in range(5)]
    idx = random.randint(0, len(sent)-1)
    sent[idx] = '你'
    return (sent, idx)


def build_dataset(n=N_SAMPLES):
    """
    构造每轮数据集
    :param n:
    :return:
    """
    data = []
    for _ in range(n):
        sent_idx = make_positive()
        data.append(sent_idx)
    return data


# ─── 2. 词表构建与编码 ──────────────────────────────────────
def build_vocab(data):
    """
    把数据集包含的字符都加入词表，并且去重，生成词表
    :param data:
    :return:
    """
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab


def encode(sent, vocab, maxlen=MAXLEN):
    """
    句子转化为索引向量
    :param sent: 句子
    :param vocab: 已经构建好的词表
    :param maxlen:
    :return: 索引向量
    """
    ids = [vocab.get(ch, 1) for ch in sent]
    return ids


# ─── 3. Dataset / DataLoader ────────────────────────────────
class TextDataset(Dataset):
    """
    定义数据加载类，方便获取任意位置的tensor类型的训练数据和对应标签
    """

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

    batch_first=True 代表输出结果时，把batch_size放在第一维
    每个批次样本数，B
    句子长度，L
    词嵌入维度，E
    RNN隐藏层维度，H
    """

    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)  # 输出形状(B,L,E)，索引张量(B,L)转为词嵌入张量(B,L,E)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)  # 输出形状(B,L,H)   (1,B,H)
        """
        output, h_n = rnn(input)
        RNN定义2个输出，output是每个时间步的隐藏状态，h_n是最后一个时间步的隐藏状态
        output形状(B,L,H)
        h_n形状(num_layers×num_dir,B,H)       默认参数：num_layers=1（单层 RNN）   num_dir单向双向标准，默认1单向
        """
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)  # 随机屏蔽比例默认是0.3
        self.fc = nn.Linear(hidden_dim, 5)  # Linear只对输入对最后一维做线性转换，任意数量的前置维度（比如批次维度、甚至多批次维度），Linear 层会自动保留这些维度，只处理最后一维

    def forward(self, x):
        # x: (batch, seq_len)   索引张量(B,L)
        e, _ = self.rnn(self.embedding(x))  # (B, L, hidden_dim)
        # 池化（降维）
        pooled = e.max(dim=1)[0]  # (B, hidden_dim)  对序列做 max pooling
        """
        dim（也叫 axis）是 PyTorch 张量维度的索引，dim=k 表示沿着第 k 个维度执行最大值计算。
        PyTorch 中 tensor.max(dim=k) 并非只返回最大值，而是返回一个二元元组 (values, indices)  dim=1代表沿着e的L维度寻找最大值
        0-沿 dim=k 计算的最大值本身
        1-沿 dim=k 计算的最大值对应的位置索引（比如最大值出现在第几个字符位置）
        """
        # 1-归一化bn
        # 2-随机屏蔽dropout
        pooled = self.dropout(self.bn(pooled))  # 不变，形状还是(B,H)
        """
        bn
        无论前层输出分布怎么变，BatchNorm （归一化层）都会把输入拉回稳定分布
        实现原理：把该维度的所有值减去均值、除以标准差 → 让这个维度的均值 = 0、方差 = 1；再通过可学习的缩放（γ）和偏移（β）参数，保留特征的 “个性化信息”
        效果：无论前层输出分布怎么变，BatchNorm 都会把输入拉回稳定分布 → 后层（全连接层）不用再 “适配漂移”，训练方向更稳定。

        1-缓解梯度消失/爆炸：深度学习的激活函数（比如 Sigmoid、Tanh）有 “饱和区”：输入值过大 / 过小，梯度会趋近于 0（梯度消失）；输入值波动过大，梯度会急剧放大（梯度爆炸）。
        归一化后，输入值被限制在均值 0、方差 1 的区间 → 激活函数的输入落在 “非饱和区”，梯度能有效传递。
        2-降低参数初始化的敏感度：没有归一化时，参数初始化的值如果偏大 / 偏小，会直接导致输入分布偏离 → 训练起步就 “跑偏”；
        """
        out = self.fc(pooled) # (B,)
        """
        squeeze(1):删除张量中 “长度为 1 的维度” 的函数
        核心规则：只删除 dim=k 这个维度，且仅当该维度的长度为 1 时才会删除；如果该维度长度≠1，调用 squeeze(k) 不会有任何变化。
        通俗类比：就像给张量 “瘦身”，只去掉那些 “占位置但没实际内容” 的维度（长度为 1 的维度）。
        [[0.8], [0.2], [0.9], [0.1]] 变为：
        [0.8000, 0.2000, 0.9000, 0.1000]

        代码中损失函数是 MSELoss，而标签 y 的形状是 (64,)（一维），这里的输出是pred，需要和y保持一致
        """
        return out


# ─── 5. 训练与评估 ──────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            prob = model(X)
            pred = torch.argmax(prob, dim=1)
            correct += (pred == y.long()).sum().item()
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

    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TextDataset(val_data, vocab), batch_size=BATCH_SIZE)

    model = KeywordRNN(vocab_size=len(vocab))
    criterion = nn.functional.cross_entropy  # 损失函数设置为：交叉熵函数
    """
    代码是二分类任务（正 / 负样本），通常二分类优先用 BCELoss（二元交叉熵），但 MSELoss 也能适配

    MSELoss 的适用场景
    1-回归任务（核心场景）：比如预测房价、温度、销量等连续值；
    2-二分类任务（替代方案）：如代码中，标签是 0/1 且输出经过 Sigmoid 时可用；
    3-不适用场景：多分类任务（优先用 CrossEntropyLoss）。
    """
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
            loss.backward()     # 反向传播
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  准确率val_acc={val_acc:.4f}")

    print(f"\n最终验证准确率：{evaluate(model, val_loader):.4f}")

    print("\n--- 推理示例 ---")
    model.eval()
    test_sents = build_dataset(10)
    with torch.no_grad():
        for sent, _ in test_sents:
            sent_str = ''.join(sent)
            id = sent_str.find('你')
            ids = torch.tensor([encode(sent, vocab)], dtype=torch.long)
            prob = torch.argmax(model(ids), dim=1)

            print(f"{id}   第{prob}类：  {sent}")


if __name__ == '__main__':
    train()



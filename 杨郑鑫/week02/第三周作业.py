import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random

# ================= 1. 数据准备  =================

torch.manual_seed(42)
random.seed(42)

TARGET_CHAR = "你"
MAX_LEN = 50  # 允许的最大句子长度

# 构建字符集
COMMON_CHARS = [chr(i) for i in range(0x4e00, 0x9fa5)]
if TARGET_CHAR not in COMMON_CHARS:
    COMMON_CHARS.append(TARGET_CHAR)

char_to_idx = {char: i for i, char in enumerate(COMMON_CHARS)}
idx_to_char = {i: char for i, char in enumerate(COMMON_CHARS)}
VOCAB_SIZE = len(COMMON_CHARS)


class LongTextDataset(Dataset):
    def __init__(self, num_samples=5000):
        self.data = []
        self.labels = []

        for _ in range(num_samples):
            # 1. 随机决定句子长度 (10 到 50 之间)
            seq_len = random.randint(10, MAX_LEN)

            # 2. 生成随机句子
            text = []
            # 确保句子里至少有 1-3 个 "你" 字，增加难度
            num_targets = random.randint(1, 3)
            target_positions = random.sample(range(seq_len), num_targets)

            for i in range(seq_len):
                if i in target_positions:
                    text.append(TARGET_CHAR)
                else:
                    text.append(random.choice(COMMON_CHARS))

            # 3. 转换为索引
            seq = [char_to_idx[c] for c in text]
            # 标签：1 表示是"你"，0 表示不是
            label = [1 if c == TARGET_CHAR else 0 for c in text]

            self.data.append(seq)
            self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 返回张量
        return (torch.tensor(self.data[idx], dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.float))


# ================= 2. 模型定义 (序列标注版) =================

class SequenceLabelingLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(SequenceLabelingLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # batch_first=True 方便处理变长序列
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        # 输出层：每个字对应一个概率 (0~1)，所以输出维度是 1
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embed = self.embedding(x)
        # lstm_out: (Batch, Seq_Len, Hidden_Dim)
        # 注意：这里我们不需要 h_n，因为我们要对每个时间步都做预测
        lstm_out, _ = self.lstm(embed)

        # 对每个时间步的输出做分类
        logits = self.fc(lstm_out)
        probs = self.sigmoid(logits)  # 输出 0~1 之间的概率
        return probs


# ================= 3. 训练 =================

def collate_fn(batch):
    """
    自定义 collate_fn 用于处理变长序列
    将 batch 中的序列填充到当前 batch 的最大长度
    """
    # 找到 batch 中最长的句子长度
    max_len = max(len(x[0]) for x in batch)

    padded_inputs = []
    padded_labels = []
    lengths = []  # 记录每个句子的真实长度，用于后续计算 loss 时忽略填充部分

    for seq, label in batch:
        seq_len = len(seq)
        lengths.append(seq_len)

        # 填充 input
        padding_input = torch.zeros(max_len, dtype=torch.long)
        padding_input[:seq_len] = seq
        padded_inputs.append(padding_input)

        # 填充 label
        padding_label = torch.zeros(max_len, dtype=torch.float)
        padding_label[:seq_len] = label
        padded_labels.append(padding_label)

    return torch.stack(padded_inputs), torch.stack(padded_labels), lengths


def train_model():
    print("正在生成长文本数据集...")
    dataset = LongTextDataset(num_samples=5000)
    # 使用自定义 collate_fn 处理变长数据
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    model = SequenceLabelingLSTM(VOCAB_SIZE, embed_dim=64, hidden_dim=128)
    criterion = nn.BCELoss()  # 二分类损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"字符集大小: {VOCAB_SIZE}, 开始训练...")

    model.train()
    for epoch in range(5):
        total_loss = 0
        correct = 0
        total_chars = 0

        for inputs, labels, lengths in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)  # (Batch, Seq_Len, 1)

            # 计算 Loss
            # 需要把 outputs 和 labels 拉平才能计算
            loss = criterion(outputs.squeeze(-1), labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 计算准确率
            preds = (outputs.squeeze(-1) > 0.5).float()
            # 只计算有效长度内的准确率（忽略 padding）
            for i, length in enumerate(lengths):
                correct += (preds[i, :length] == labels[i, :length]).sum().item()
                total_chars += length

        acc = 100 * correct / total_chars
        print(f"Epoch [{epoch + 1}/5], Loss: {total_loss / len(train_loader):.4f}, Accuracy: {acc:.2f}%")

    return model


# ================= 4. 推理与测试 =================

def predict_sentence(model, text):
    model.eval()
    print(f"\n--- 测试: {text} ---")

    # 1. 预处理
    seq = [char_to_idx.get(c, 0) for c in text]  # 转索引
    input_tensor = torch.tensor([seq], dtype=torch.long)  # (1, Seq_Len)

    # 2. 预测
    with torch.no_grad():
        outputs = model(input_tensor)  # (1, Seq_Len, 1)
        probs = outputs.squeeze(-1).squeeze(0)  # 变成 (Seq_Len,)

        # 3. 结果解析
        # 找出概率大于 0.5 的位置
        found_positions = []
        for i, prob in enumerate(probs):
            if prob > 0.5:
                found_positions.append(i + 1)  # 位置从 1 开始计数

        # 4. 可视化输出
        # 生成一个标记行，例如： 我 正 在 看 着 你
        #                       0  0  0  0  0  1
        markers = []
        for i, char in enumerate(text):
            if (i + 1) in found_positions:
                markers.append("↑")
            else:
                markers.append(" ")

        print(f"文本: {' '.join(text)}")
        print(f"标记: {' '.join(markers)}")
        if found_positions:
            print(f"结果: 找到 '你' 在第 {found_positions} 位")
        else:
            print("结果: 未找到 '你'")


# ================= 5. 主程序入口 =================

if __name__ == "__main__":
    # 1. 训练模型
    trained_model = train_model()

    # 2. 测试长句子
    test_cases = [
        "我正在看着你，看着你，目不转睛",
        "你若安好，便是晴天",
        "爱你一万年",
        "你你你你你",
        "他好像不认识你"
    ]

    for text in test_cases:
        predict_sentence(trained_model, text)

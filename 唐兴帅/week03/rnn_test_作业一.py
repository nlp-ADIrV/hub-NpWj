import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ─── 1. 超参数设置 ────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# 任务相关
VOCAB_CHARS = "你我他她它abcdefghij"  # 字符集：包含“你”和其他干扰字符
SENTENCE_LEN = 5                      # 句子长度固定为 5
NUM_CLASSES = SENTENCE_LEN + 1        # 分类数：5个位置 + 1个“不存在”类 = 6类

# 模型相关
EMBED_DIM = 32        # 词向量维度
HIDDEN_DIM = 64       # RNN 隐藏层维度
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.001

# ─── 2. 数据生成 ────────────────────────────────────────────────
def build_sample():
    """
    生成单个样本：
    返回 (文本字符串, 标签类别)
    """
    # 1. 随机生成 5 个字符
    text_list = [random.choice(VOCAB_CHARS) for _ in range(SENTENCE_LEN)]
    text_str = "".join(text_list)
    
    # 2. 确定标签
    # 查找“你”字的位置
    if "你" in text_list:
        label = text_list.index("你")  # 找到索引 0-4
    else:
        label = SENTENCE_LEN           # 如果没找到，标签为 5 (第6类)
        
    return text_str, label

def build_dataset(n_samples=2000):
    """生成数据集列表"""
    data = []
    for _ in range(n_samples):
        data.append(build_sample())
    return data

# ─── 3. 词表与编码 ────────────────────────────────────────────────
# 构建简单的字到索引的映射
# {'你': 1, '我': 2, ...}
vocab = {"pad": 0}
for i, char in enumerate(VOCAB_CHARS):
    vocab[char] = i + 1

def encode(text_str):
    """将字符串转换为数字索引列表"""
    return [vocab.get(char, 0) for char in text_str]

# ─── 4. PyTorch Dataset ─────────────────────────────────────────
class PositionDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text_str, label = self.data[idx]
        # 转换文本为索引
        x = torch.tensor(encode(text_str), dtype=torch.long)
        # 标签直接转 tensor
        y = torch.tensor(label, dtype=torch.long)
        return x, y

# ─── 5. 模型定义 (RNN) ──────────────────────────────────────────
class PositionRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(PositionRNN, self).__init__()
        # 1. 嵌入层：将字索引转为向量
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 2. RNN层：处理序列信息
        # batch_first=True 表示输入形状为 (Batch, Seq, Feature)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        
        # 3. 全连接层：将 RNN 的输出映射到分类数
        # 这里我们取 RNN 最后一个时间步的输出，或者对所有时间步做池化
        # 为了简单，我们取最后一个时间步的隐藏状态作为整个句子的特征
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: (Batch, Seq_Len)
        embed = self.embedding(x)  # (Batch, Seq_Len, Embed_Dim)
        
        # RNN 输出
        # output: (Batch, Seq_Len, Hidden_Dim)
        # hidden: (1, Batch, Hidden_Dim)
        output, hidden = self.rnn(embed)
        
        # 策略：使用最后一个时间步的输出作为特征
        # output[:, -1, :] 取出每句话最后一个字的 RNN 输出
        # 对于 RNN 来说，最后一个时间步的输出包含了前面所有字的信息
        last_output = output[:, -1, :]
        
        # 分类
        logits = self.fc(last_output) # (Batch, Num_Classes)
        return logits

# ─── 6. 训练与评估 ──────────────────────────────────────────────
def train():
    print(f"{'='*30}\n开始训练：定位'你'字的位置\n{'='*30}")
    
    # 1. 准备数据
    train_data = build_dataset(3000) # 训练集
    test_data = build_dataset(500)   # 测试集
    
    train_loader = DataLoader(PositionDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(PositionDataset(test_data), batch_size=BATCH_SIZE)
    
    # 2. 初始化模型、损失函数、优化器
    model = PositionRNN(
        vocab_size=len(vocab), 
        embed_dim=EMBED_DIM, 
        hidden_dim=HIDDEN_DIM, 
        num_classes=NUM_CLASSES
    )
    
    # 多分类任务通常使用交叉熵损失
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 记录损失用于画图
    loss_history = []
    
    # 3. 训练循环
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for x_batch, y_batch in train_loader:
            # 前向传播
            logits = model(x_batch)
            
            # 计算损失
            # CrossEntropyLoss 会自动处理 Softmax，输入是 logits，目标是类别索引
            loss = criterion(logits, y_batch)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        
        # 每 3 轮打印一次状态
        if (epoch + 1) % 3 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

    # 4. 测试与推理演示
    print(f"\n{'='*30}\n开始测试与推理\n{'='*30}")
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            logits = model(x_batch)
            # 获取概率最大的类别索引
            _, predicted = torch.max(logits, 1)
            
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            
    print(f"测试集准确率: {100 * correct / total:.2f}%")
    
    # 5. 手动推理几个例子
    print(f"\n--- 随机推理示例 ---")
    test_samples = [
        "你我他ab", # "你"在位置0 -> 类别0
        "ab你cd",   # "你"在位置2 -> 类别2
        "abcde",    # 没"你" -> 类别5
        "hh你kk"    # "你"在位置2 -> 类别2
    ]
    
    for text in test_samples:
        # 编码
        x = torch.tensor([encode(text)], dtype=torch.long)
        # 预测
        with torch.no_grad():
            logit = model(x)
            prob = torch.softmax(logit, dim=1) # 转为概率
            pred_class = torch.argmax(prob, dim=1).item()
            
        # 解释结果
        if pred_class < SENTENCE_LEN:
            pos_str = f"第 {pred_class + 1} 位"
        else:
            pos_str = "不存在"
            
        print(f"文本: '{text}' -> 预测类别: {pred_class} ({pos_str}) | 真实位置: {'存在' if '你' in text else '无'}")

    # 画损失图
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

if __name__ == '__main__':
    train()

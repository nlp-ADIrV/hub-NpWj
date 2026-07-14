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
VOCAB_CHARS = "你我他她它abcdefghij"
SENTENCE_LEN = 5
NUM_CLASSES = SENTENCE_LEN + 1        # 6类：位置0-4 + 不存在(5)

# 模型相关
EMBED_DIM = 32
HIDDEN_DIM = 64
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 0.001

# ─── 2. 数据生成 ────────────────────────────────────────────────
def build_sample():
    text_list = [random.choice(VOCAB_CHARS) for _ in range(SENTENCE_LEN)]
    text_str = "".join(text_list)
    
    if "你" in text_list:
        label = text_list.index("你")
    else:
        label = SENTENCE_LEN
    return text_str, label

def build_dataset(n_samples=2000):
    data = []
    for _ in range(n_samples):
        data.append(build_sample())
    return data

# ─── 3. 词表与编码 ────────────────────────────────────────────────
vocab = {"pad": 0}
for i, char in enumerate(VOCAB_CHARS):
    vocab[char] = i + 1

def encode(text_str):
    return [vocab.get(char, 0) for char in text_str]

# ─── 4. PyTorch Dataset ─────────────────────────────────────────
class PositionDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text_str, label = self.data[idx]
        x = torch.tensor(encode(text_str), dtype=torch.long)
        y = torch.tensor(label, dtype=torch.long)
        return x, y

# ─── 5. 模型定义 (LSTM) ──────────────────────────────────────────
class PositionLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(PositionLSTM, self).__init__()
        # 1. 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 2. LSTM层：核心变化点
        # LSTM 的输入输出维度逻辑与 RNN 基本一致
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        
        # 3. 全连接层
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: (Batch, Seq_Len)
        embed = self.embedding(x)  # (Batch, Seq_Len, Embed_Dim)
        
        # LSTM 前向传播
        # output: (Batch, Seq_Len, Hidden_Dim)
        # (h_n, c_n): 分别是最后的隐藏状态和细胞状态
        output, (h_n, c_n) = self.lstm(embed)
        
        # 策略：使用最后一个时间步的输出
        # 对于 LSTM，output[:, -1, :] 等同于 h_n[-1] (如果是单向)
        last_output = output[:, -1, :]
        
        # 分类
        logits = self.fc(last_output)
        return logits

# ─── 6. 训练与评估 ──────────────────────────────────────────────
def train():
    print(f"{'='*30}\n开始训练：LSTM 定位'你'字的位置\n{'='*30}")
    
    # 1. 准备数据
    train_data = build_dataset(3000)
    test_data = build_dataset(500)
    
    train_loader = DataLoader(PositionDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(PositionDataset(test_data), batch_size=BATCH_SIZE)
    
    # 2. 初始化模型
    model = PositionLSTM(
        vocab_size=len(vocab), 
        embed_dim=EMBED_DIM, 
        hidden_dim=HIDDEN_DIM, 
        num_classes=NUM_CLASSES
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    loss_history = []
    
    # 3. 训练循环
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for x_batch, y_batch in train_loader:
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 3 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

    # 4. 测试与推理
    print(f"\n{'='*30}\n开始测试与推理\n{'='*30}")
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            logits = model(x_batch)
            _, predicted = torch.max(logits, 1)
            
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            
    print(f"测试集准确率: {100 * correct / total:.2f}%")
    
    # 5. 手动推理示例
    print(f"\n--- 随机推理示例 ---")
    test_samples = [
        "你我他ab", 
        "ab你cd",   
        "abcde",    
        "hh你kk"    
    ]
    
    for text in test_samples:
        x = torch.tensor([encode(text)], dtype=torch.long)
        with torch.no_grad():
            logit = model(x)
            pred_class = torch.argmax(logit, dim=1).item()
            
        if pred_class < SENTENCE_LEN:
            pos_str = f"第 {pred_class + 1} 位"
        else:
            pos_str = "不存在"
            
        print(f"文本: '{text}' -> 预测类别: {pred_class} ({pos_str})")

if __name__ == '__main__':
    train()

# 尝试完成一个多分类任务的训练:一个随机向量，哪一维数字最大就属于第几类。

'''
环境准备：安装 torch
数据生成：生成随机向量（比如 10 维向量，10 分类任务），标签是 argmax (向量)
构建数据集和数据加载器
构建简单的神经网络（因为任务简单，全连接层就行）
定义损失函数（交叉熵损失，多分类标配）、优化器
训练循环
模型评估
推理测试
'''

# 1. 导入依赖库
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ===================== 超参数 =====================
VECTOR_DIM = 10    # 随机向量的维度 = 分类类别数（10维=10分类）
TRAIN_SAMPLES = 10000  # 训练样本数量
TEST_SAMPLES = 2000    # 测试样本数量
BATCH_SIZE = 64        # 批次大小
EPOCHS = 10            # 训练轮数
LR = 0.001             # 学习率
# =============================================================

# 2. 生成数据集（核心：随机向量 + argmax标签）
def generate_data(sample_num, dim):
    """生成随机向量和对应标签（最大值索引）"""
    # 生成随机浮点数向量 (样本数, 向量维度)
    data = torch.randn(sample_num, dim)
    # 标签：每一行最大值所在的维度索引 → 就是我们的分类目标
    labels = torch.argmax(data, dim=1)
    return data, labels

# 生成训练集、测试集
train_data, train_labels = generate_data(TRAIN_SAMPLES, VECTOR_DIM)
test_data, test_labels = generate_data(TEST_SAMPLES, VECTOR_DIM)

# 封装为PyTorch数据集 + 数据加载器
train_dataset = TensorDataset(train_data, train_labels)
test_dataset = TensorDataset(test_data, test_labels)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. 构建模型（简单全连接网络，适配任务）
class MaxIndexClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        # 任务极简单：单层线性层即可完成拟合
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # 前向传播：输出每个类别的预测分数
        return self.fc(x)

# 初始化模型、损失函数、优化器
model = MaxIndexClassifier(VECTOR_DIM, VECTOR_DIM)
criterion = nn.CrossEntropyLoss()  # 多分类必备损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# 4. 训练循环
print("========== 开始训练 ==========")
for epoch in range(EPOCHS):
    model.train()  # 训练模式
    total_loss = 0
    correct = 0
    total = 0

    for batch_data, batch_labels in train_loader:
        # 前向传播
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)

        # 反向传播 + 优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计损失和准确率
        total_loss += loss.item()
        pred = torch.argmax(outputs, dim=1)  # 模型预测的类别
        correct += (pred == batch_labels).sum().item()
        total += batch_labels.size(0)

    # 打印训练结果
    train_acc = 100 * correct / total
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] | 损失: {avg_loss:.4f} | 训练准确率: {train_acc:.2f}%")

# 5. 测试集评估
print("\n========== 测试集评估 ==========")
model.eval()  # 评估模式
with torch.no_grad():  # 关闭梯度计算
    correct = 0
    total = 0
    for data, labels in test_loader:
        outputs = model(data)
        pred = torch.argmax(outputs, dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

test_acc = 100 * correct / total
print(f"测试集准确率: {test_acc:.2f}%")

# 6. 单样本推理（验证效果）
print("\n========== 单样本测试 ==========")
# 随机生成1个10维向量
test_vector = torch.randn(1, VECTOR_DIM)
# 真实标签
true_label = torch.argmax(test_vector, dim=1).item()
# 模型预测
model.eval()
with torch.no_grad():
    output = model(test_vector)
    pred_label = torch.argmax(output, dim=1).item()

print(f"随机向量: {test_vector[0].numpy()}")
print(f"真实类别(最大值索引): {true_label}")
print(f"模型预测类别: {pred_label}")



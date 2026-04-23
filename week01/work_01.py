import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ===================== 1. 任务核心：生成数据集 =====================
# 规律：5维随机向量 → 最大值在第几维 → 标签就是几（0/1/2/3/4，5分类任务）
def generate_data(batch_size=32):
    # 生成 batch_size 个 5维随机向量（输入数据）
    x = torch.randn(batch_size, 5)  # 形状：[32,5]
    # 生成标签：找每一行最大值的**索引**（核心规律！）
    y = torch.argmax(x, dim=1)     # 形状：[32]  值：0/1/2/3/4
    return x, y

# ===================== 2. 构建PyTorch模型 =====================
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        # 单层全连接层：输入5维 → 输出5维（5分类）
        self.linear = nn.Linear(5, 5)

    # 前向传播（预测）
    def forward(self, x):
        output = self.linear(x)  # 输出原始分数logits，不用手动softmax
        return output

# ===================== 3. 初始化模型、损失、优化器 =====================
model = Classifier()
# 损失函数：交叉熵损失（多分类标配，**自带softmax**）
criterion = nn.CrossEntropyLoss()
# 优化器：梯度下降更新参数
optimizer = optim.SGD(model.parameters(), lr=0.1)

# ===================== 4. 开始训练 =====================
print("======== 开始训练 ========")
for epoch in range(100):  # 训练100轮
    # 生成数据
    x, y_true = generate_data(5)
    # 1. 前向传播：算预测值
    y_pred = model(x)
    print("原始分数:", y_pred)
    # 2. 计算交叉熵损失
    loss = criterion(y_pred, y_true)
    # 3. 反向传播：算梯度
    optimizer.zero_grad()  # 梯度清零
    loss.backward()        # 梯度计算（核心！）
    # 4. 更新参数
    optimizer.step()

    # 每10轮打印一次损失
    if epoch % 10 == 0:
        print(f"轮次:{epoch} | 交叉熵损失:{loss.item():.4f}")

# ===================== 5. 测试模型 =====================
print("\n======== 测试模型 ========")
# 生成1个测试用的5维向量
test_x, test_y = generate_data(batch_size=1)
print("测试5维向量:", test_x.numpy())
print("真实标签(最大值所在维度):", test_y.item())

# 模型预测
with torch.no_grad():  # 不计算梯度
    pred = model(test_x)
    pred_label = torch.argmax(pred, dim=1).item()  # 取预测最大值索引
print("模型预测标签:", pred_label)
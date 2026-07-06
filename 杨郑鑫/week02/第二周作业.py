import torch
import torch.nn as nn
import torch.optim as optim

INPUT_DIM = 10  # 向量维度（也是分类类别数）
NUM_SAMPLES = 50000 # 训练数据总量
BATCH_SIZE = 64  # 批次大小
EPOCHS = 100  # 训练轮数
LR = 0.01  # 学习率

# 生成随机向量 [5000, 10]
X = torch.rand(NUM_SAMPLES, INPUT_DIM)
# 生成标签：找出每一行最大值的索引 (0-9)
y = torch.argmax(X, dim=1)

print(f"数据形状: {X.shape}, 标签形状: {y.shape}")


class DeepMaxFinder(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(DeepMaxFinder, self).__init__()
        # 定义一个多层感知机
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),  # 输入层 -> 隐藏层1
            nn.ReLU(),  # 激活函数：引入非线性
            nn.Linear(32, 64),  # 隐藏层1 -> 隐藏层2
            nn.ReLU(),
            nn.Linear(64, num_classes)  # 输出层：输出每个类别的得分（Logits）
        )

    def forward(self, x):
        return self.net(x)


model = DeepMaxFinder(INPUT_DIM, INPUT_DIM)

# 交叉熵损失：专门用于多分类，内部会自动处理 Softmax
criterion = nn.CrossEntropyLoss()
# 随机梯度下降优化器
optimizer = optim.SGD(model.parameters(), lr=LR)

print("\n🚀 开始训练...")
for epoch in range(EPOCHS):
    model.train()  # 设置为训练模式
    total_loss = 0

    # 打乱数据顺序（每个epoch都打乱，防止模型死记硬背顺序）
    perm = torch.randperm(NUM_SAMPLES)
    X_shuffled = X[perm]
    y_shuffled = y[perm]

    # 批处理
    for i in range(0, NUM_SAMPLES, BATCH_SIZE):
        batch_X = X_shuffled[i: i + BATCH_SIZE]
        batch_y = y_shuffled[i: i + BATCH_SIZE]

        # 1. 前向传播：计算预测值
        outputs = model(batch_X)

        # 2. 计算损失
        loss = criterion(outputs, batch_y)

        # 3. 反向传播：计算梯度
        optimizer.zero_grad()  # 清空旧梯度
        loss.backward()  # 计算新梯度
        optimizer.step()  # 更新权重

        total_loss += loss.item()

    avg_loss = total_loss * BATCH_SIZE / NUM_SAMPLES
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.4f}")

print("\n🔍 开始测试...")
model.eval()  # 设置为评估模式
with torch.no_grad():  # 测试时不需要计算梯度
    # 生成全新的测试数据
    test_X = torch.rand(100, INPUT_DIM)
    test_y_true = torch.argmax(test_X, dim=1)  # 真实标签

    # 模型预测
    test_outputs = model(test_X)
    # 获取预测概率最大的类别索引
    _, test_y_pred = torch.max(test_outputs, 1)

    # 计算准确率
    accuracy = (test_y_pred == test_y_true).sum().item() / test_X.size(0)

    print(f"测试集准确率: {accuracy * 100:.2f}%")

    # 打印几个具体例子
    print("\n--- 预测详情 ---")
    for i in range(5):
        print(f"输入: {test_X[i].numpy()}")
        print(f"真实类别: {test_y_true[i].item()}, 预测类别: {test_y_pred[i].item()}")
        print("-" * 30)

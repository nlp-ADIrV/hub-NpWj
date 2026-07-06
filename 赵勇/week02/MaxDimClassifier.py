import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 输入向量维度：每个样本是5维随机向量
input_dim = 5
# 分类任务的总类别数
num_classes = 5
# 每批次训练的样本数量
batch_size = 16
# 模型训练的总轮数
epochs = 1000
# 优化器的学习率
lr = 0.05
# 测试阶段使用的样本数量
test_batch = 200


# 模型定义
class MaxDimClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        # 调用父类初始化方法
        super().__init__()
        # 定义线性层：将输入向量映射为类别输出
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, features):
        # 前向传播：输入特征经过线性层得到预测分数
        return self.linear(features)


# 实例化分类模型
model = MaxDimClassifier(input_dim, num_classes)
# 多分类任务标准损失函数：交叉熵损失
criterion = nn.CrossEntropyLoss()
# 随机梯度下降优化器
optimizer = optim.SGD(model.parameters(), lr=lr)


# 数据生成函数
def generate_data(batch_size, input_dim):
    """
    生成训练/测试数据：随机5维向量，标签为最大值所在维度索引
    """
    # 生成标准正态分布随机特征向量 (batch_size, input_dim)
    features = torch.randn(batch_size, input_dim)
    # 标签->向量中最大值所在的维度索引
    labels = torch.argmax(features, dim=1)
    return features, labels


# 记录每轮训练的损失值
loss_history = []
# 记录每轮训练的准确率
acc_history = []

print("============ 开始训练 ============\n")
for epoch in range(epochs):
    # 切换模型为训练模式
    model.train()
    # 生成一批训练数据（特征 + 标签）
    train_features, train_labels = generate_data(batch_size, input_dim)
    # 清空上一步梯度，防止梯度累积
    optimizer.zero_grad()
    # 前向传播，得到模型输出
    outputs = model(train_features)
    # 计算损失
    loss = criterion(outputs, train_labels)
    # 反向传播计算梯度
    loss.backward()
    # 更新模型参数
    optimizer.step()

    # 记录损失
    loss_history.append(loss.item())
    # 获取预测类别
    predictions = torch.argmax(outputs, dim=1)
    # 计算当前批次准确率
    acc = (predictions == train_labels).sum().item() / batch_size
    acc_history.append(acc)

    # 每100轮打印训练信息
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}] | Loss: {loss:.4f} | Acc: {acc:.2%}")

print("\n============ 批量测试 ============")


def batch_test(model, input_dim, test_num):
    """
    模型测试函数
    param model: 训练好的模型
    param input_dim: 数据维度
    test_num: 测试样本数量
    return: 真实标签、预测标签
    """
    # 将模型设置为评估模式
    model.eval()
    # 测试阶段不使用梯度计算，节省内存和计算资源
    with torch.no_grad():
        # 生成测试数据
        test_features, test_labels = generate_data(test_num, input_dim)
        # 模型推理得到输出
        outputs = model(test_features)
        # 获取预测类别
        test_predictions = torch.argmax(outputs, dim=1)
        # 统计正确预测的样本数
        correct = (test_predictions == test_labels).sum().item()
        # 计算测试准确率
        acc = correct / test_num

    print(f"测试样本数: {test_num}")
    print(f"测试准确率: {acc:.2%}")
    return test_labels.numpy(), test_predictions.numpy()


# 执行测试
y_true, y_pred = batch_test(model, input_dim, test_batch)

print(f"\n============ 分类结果统计 ============")
print(f"总测试样本数: {len(y_true)}")
print(f"正确预测数: {np.sum(y_true == y_pred)}")
print(f"错误预测数: {np.sum(y_true != y_pred)}")
print(f"整体准确率: {np.mean(y_true == y_pred):.2%}")

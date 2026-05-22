# coding:utf8

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 解决OpenMP库冲突问题

import torch  # 导入PyTorch
import torch.nn as nn  # 导入神经网络模块
import numpy as np  # 导入NumPy
import matplotlib.pyplot as plt  # 导入绘图库

"""
任务：5维向量分类问题
输入：5个随机数
输出：最大值所在的位置（0~4）
本质：多分类问题（5分类）
"""

# =========================
# 1️⃣ 模型定义
# =========================
class TorchFiveClassModel(nn.Module):
    def __init__(self, input_size):
        super(TorchFiveClassModel, self).__init__()  # 初始化父类
        self.fc = nn.Linear(input_size, 5)  # 全连接层，输入5维，输出5类
        self.loss_fn = nn.CrossEntropyLoss()  # 分类损失函数，内部自带softmax

    def forward(self, x, y=None):
        logits = self.fc(x)  # 前向计算，得到每类的logits
        if y is not None:
            # 如果传入y，说明是训练模式，返回loss
            return self.loss_fn(logits, y)
        else:
            # 如果没有y，说明是预测模式，返回预测类别索引
            return torch.argmax(logits, dim=1)  # 每行取最大值索引作为类别


# =========================
# 2️⃣ 数据生成
# =========================
def build_sample():
    x = np.random.random(5)  # 生成5个0~1随机数
    y = np.argmax(x)  # 最大值所在位置作为类别标签
    return x, y  # 返回输入和标签

def build_dataset(total_sample_num):
    X, Y = [], []  # 初始化列表存储样本和标签
    for _ in range(total_sample_num):
        x, y = build_sample()  # 生成单个样本
        X.append(x)  # 添加到输入列表
        Y.append(y)  # 添加到标签列表
    return torch.FloatTensor(X), torch.LongTensor(Y)  # 转成Tensor，X为float，Y为long


# =========================
# 3️⃣ 评估函数
# =========================
def evaluate(model, test_sample_num=100):
    model.eval()  # 切换到评估模式，不计算梯度
    x, y_true = build_dataset(test_sample_num)  # 构造测试集
    correct = 0  # 统计预测正确数量

    with torch.no_grad():  # 不计算梯度
        y_pred = model(x)  # 得到预测类别索引
        for y_p, y_t in zip(y_pred, y_true):  # 遍历预测值和真实值
            if y_p.item() == y_t.item():  # 如果预测等于真实
                correct += 1  # 正确数加1

    accuracy = correct / test_sample_num  # 计算准确率
    print(f"正确预测个数: {correct}, 正确率: {accuracy:.6f}")  # 打印结果
    return accuracy  # 返回准确率


# =========================
# 4️⃣ 训练函数
# =========================
def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每个batch大小
    train_sample = 5000  # 总训练样本数
    input_size = 5  # 输入维度
    learning_rate = 0.01  # 学习率

    # 建立模型
    model = TorchFiveClassModel(input_size)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam优化器

    # 构造训练集
    train_x, train_y = build_dataset(train_sample)
    log = []  # 用于记录每轮的准确率和loss

    # 训练循环
    for epoch in range(epoch_num):
        model.train()  # 切换到训练模式
        loss_list = []  # 记录每个batch的loss
        for i in range(train_sample // batch_size):
            x = train_x[i*batch_size:(i+1)*batch_size]  # 取batch输入
            y = train_y[i*batch_size:(i+1)*batch_size]  # 取batch标签
            loss = model(x, y)  # 前向+计算loss
            loss.backward()  # 反向传播，计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度清零
            loss_list.append(loss.item())  # 保存loss

        avg_loss = np.mean(loss_list)  # 平均loss
        print(f"第{epoch+1}轮 平均loss: {avg_loss:.6f}")

        # 每轮训练后评估
        acc = evaluate(model)  # 测试当前模型准确率
        log.append([acc, avg_loss])  # 记录

    # 保存模型权重
    torch.save(model.state_dict(), "modelfiveclass.bin")

    # 绘制训练曲线
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 准确率曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # loss曲线
    plt.legend()
    plt.show()


# =========================
# 5️⃣ 预测函数
# =========================
def predict(model_path, input_vec):
    input_size = 5
    model = TorchFiveClassModel(input_size)  # 重新创建模型结构
    model.load_state_dict(torch.load(model_path, map_location="cpu"))  # 加载权重
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model(torch.FloatTensor(input_vec))  # 得到预测类别
    for vec, res in zip(input_vec, result):  # 遍历输出
        print(f"输入: {vec}, 预测类别: {res}")  # 打印结果


# =========================
# 6️⃣ 程序入口
# =========================
if __name__ == "__main__":
    main()  # 训练模型

    # 测试数据预测
    test_vec = [
        [0.888, 0.152, 0.310, 0.035, 0.889],
        [0.949, 0.552, 0.957, 0.955, 0.848],
        [0.207, 0.674, 0.136, 0.346, 0.198],
        [0.993, 0.594, 0.925, 0.415, 0.135]
    ]
    predict("modelfiveclass.bin", test_vec)  # 调用预测函数

# coding:utf8

# 解决 OpenMP 库冲突问题
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个n维向量，哪一维数字最大就属于第几类（多分类任务）

"""


class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  # 线性层，输出维度等于类别数
        self.loss = nn.functional.cross_entropy  # loss函数采用交叉熵损失（适合多分类）

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, num_classes)
        y_pred = x  # 直接输出logits，不需要softmax（cross_entropy内部会做softmax）
        if y is not None:
            # y需要是长整型，形状为(batch_size,)
            return self.loss(y_pred, y.squeeze().long())
        else:
            # 返回预测的类别和概率
            return torch.softmax(y_pred, dim=1)


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个n维向量，最大值所在的索引即为类别标签
def build_sample(input_size):
    x = np.random.random(input_size)
    # 找出最大值的索引
    y = np.argmax(x)
    return x, y


# 随机生成一批样本
def build_dataset(total_sample_num, input_size):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample(input_size)
        X.append(x)
        Y.append([y])  # 保持为列向量形式
    return torch.FloatTensor(X), torch.FloatTensor(Y)


# 测试每轮模型的准确率
def evaluate(model, input_size):
    model.eval()
    test_sample_num = 200
    x, y = build_dataset(test_sample_num, input_size)

    # 统计每个类别的样本数
    y_list = y.squeeze().long().numpy()
    unique, counts = np.unique(y_list, return_counts=True)
    class_dist = dict(zip(unique, counts))
    print("本次预测集样本分布：", class_dist)

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测，返回概率分布
        pred_classes = torch.argmax(y_pred, dim=1)  # 获取预测的类别
        y_true = y.squeeze().long()

        correct = (pred_classes == y_true).sum().item()
        wrong = len(y_true) - correct

    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 30  # 训练轮数
    batch_size = 32  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    num_classes = input_size  # 类别数等于向量维度
    learning_rate = 0.01  # 学习率

    # 建立模型
    model = TorchModel(input_size, num_classes)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    # 创建训练集
    train_x, train_y = build_dataset(train_sample, input_size)

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []

        # 打乱数据
        shuffle_indices = torch.randperm(train_sample)
        train_x = train_x[shuffle_indices]
        train_y = train_y[shuffle_indices]

        for batch_index in range(train_sample // batch_size):
            # 取出一个batch数据作为输入
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())

        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, input_size)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])

    # 保存模型
    torch.save(model.state_dict(), "model_multiclass.bin")

    # 画图
    print(log)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc", color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(len(log)), [l[1] for l in log], label="loss", color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    plt.tight_layout()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = len(input_vec[0])
    num_classes = input_size
    model = TorchModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测，返回概率分布
        pred_classes = torch.argmax(result, dim=1)  # 获取预测类别
        max_probs = torch.max(result, dim=1)[0]  # 获取最大概率值

    for vec, pred_class, max_prob in zip(input_vec, pred_classes, max_probs):
        print("输入：%s, 预测类别：%d, 置信度：%f" % (vec, pred_class.item(), max_prob.item()))
        # 打印每个类别的概率
        # probs = torch.softmax(torch.FloatTensor(vec), dim=0)
        # print("各类别概率：", probs.numpy())


if __name__ == "__main__":
    main()

    # 测试预测
    test_vec = [
        [0.1, 0.2, 0.3, 0.4, 0.5],  # 最大值在第4维（索引4），应预测为4类
        [0.9, 0.1, 0.2, 0.3, 0.4],  # 最大值在第0维，应预测为0类
        [0.2, 0.8, 0.1, 0.3, 0.4],  # 最大值在第1维，应预测为1类
        [0.3, 0.2, 0.7, 0.1, 0.4],  # 最大值在第2维，应预测为2类
        [0.4, 0.3, 0.2, 0.6, 0.1],  # 最大值在第3维，应预测为3类
        [0.5, 0.5, 0.5, 0.5, 0.5]  # 有多个最大值，取第一个（索引0）
    ]
    predict("model_multiclass.bin", test_vec)

# coding:utf8

# 解决 OpenMP 库冲突问题
import os
import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
在一个自行构造的多类规律任务上进行学习。
规律：x是一个5维向量，若 x 中最大值所在的维度为 i（0-based，i ∈ [0,4]），
则样本类别为 i。

"""


class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes=5):
        super(TorchModel, self).__init__()
        # 只有一层线性映射到所有类别
        self.linear = nn.Linear(input_size, num_classes)
        # 使用交叉熵损失，直接对未归一化的logits计算多分类损失
        self.loss = nn.CrossEntropyLoss()

    # 当输入真实标签，返回损失值；无真实标签，返回预测的logits
    def forward(self, x, y=None):
        logits = self.linear(x)  # (batch_size, num_classes)
        if y is not None:
            return self.loss(logits, y)  # 预测值与真实标签计算损失
        else:
            return logits  # 输出预测的logits


# 生成一个样本。规则：x 是 5 维向量，x 中最大值所在的维度的下标即为样本类别（0-4）
def build_sample():
    x = np.random.random(5)
    # 找到最大值的下标，作为类别
    cls = int(np.argmax(x))
    return x, cls


# 随机生成一批样本
# 多类别，每个样本的类别由上述规则决定
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    # print(X)
    # print(Y)
    return torch.FloatTensor(X), torch.LongTensor(Y)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, num_classes=5):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    counts = torch.bincount(y, minlength=num_classes).tolist()
    print("本次预测集中各类别分布：%s" % counts)
    correct, wrong = 0, 0
    with torch.no_grad():
        logits = model(x)  # (N, num_classes)
        preds = torch.argmax(logits, dim=1)
        correct = int((preds == y).sum().item())
        wrong = test_sample_num - correct
        acc = correct / (correct + wrong)
    print("正确预测个数：%d, 正确率：%f" % (correct, acc))
    return acc


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    num_classes = 5  # 类别数
    learning_rate = 0.01  # 学习率
    # 建立模型
    model = TorchModel(input_size, num_classes)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            # 取出一个batch数据作为输入
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())  # 记录loss
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, num_classes)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    num_classes = 5
    model = TorchModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        logits = model(torch.FloatTensor(input_vec))  # 模型预测
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        for vec, pred, prob in zip(input_vec, preds, probs):
            print(
                "输入：%s, 预测类别：%d, 概率值：%f"
                % (vec, int(pred), float(prob[pred]))
            )


if __name__ == "__main__":
    main()
    test_vec = [
        [0.56, 0.21, 0.89, 0.14, 0.37],
        [0.12, 0.95, 0.33, 0.28, 0.41],
        [0.78, 0.15, 0.22, 0.59, 0.09],
        [0.31, 0.45, 0.18, 0.92, 0.25],
        [0.05, 0.27, 0.39, 0.11, 0.83],
    ]
    predict("model.bin", test_vec)

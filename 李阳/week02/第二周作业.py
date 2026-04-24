"""
作者：深衷浅貌
日期：2026年04月19日--16:38
项目：NLP
文件名：自己训练一个任务
"""

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
import matplotlib
matplotlib.use('TkAgg')  # 强制换画图引擎

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，第几个数据最大，数据就是第几类，0～4类
    笔记：
        因为任务是 5 分类任务（0、1、2、3、4 五类）
        输出的 5 个数字 → 分别对应 类别 0 ~ 类别 4 的得分
        哪个数字最大 → 模型就判断为哪一类
        交叉熵损失函数 要求输出维度 = 分类数量
        所以：
            10 分类 → 模型输出必须 10 维
"""

import torch
import torch.nn as nn


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        # 改这里：输出 5 个类别
        # 网络层/线性层
        self.linear = nn.Linear(input_size, 5)
        # 交叉熵损失
        self.loss = nn.functional.cross_entropy
        # 交叉熵损失函数会自动完成两件事：1、通过softmax函数做归一化，（变成 0-1 之间的概率）；2、计算交叉熵损失（模型输出的是原始logits）
        """
        只有这两种情况需要手动加：
        1、二分类 + binary_cross_entropy → 需要加 sigmoid
        2、纯推理输出概率（不计算 loss）→ 可以手动加 softmax
        """

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        # 输出原始 logits，【绝对不要加激活函数】
        logits = self.linear(x)  # shape: (batch_size, 5)

        if y is not None:
            # 直接把 logits 丢进去，不要做任何激活
            return self.loss(logits, y)
        else:
            # 推理时返回 logits，或取最大索引
            return logits


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，如果第一个值大于第五个值，认为是正样本，反之为负样本
def build_sample():
    # 随机5维向量
    x = np.random.random(5)

    # 真实标签：最大值索引
    y = np.argmax(x)
    """
    模型输出 ≠ 真实标签
    它们本来就不需要长得一样！
    交叉熵损失函数的设计，就是用「5 维模型输出」去匹配「1 维真实标签」，这是深度学习分类任务的标准规则。
    
    你训练时是一批数据一起输入，比如 100 个样本：
    模型输出形状：(100, 5) → 100 个样本，每个样本 5 个猜测得分
    真实标签形状：(100,) → 100 个样本，每个样本 1 个标准答案
    ✅ 这就是 PyTorch 交叉熵要求的完美匹配格式！
    
    模型输出是「猜测列表」（5 个选项），标签是「正确答案」（1 个数字）
    """

    # 转成 PyTorch 张量（模型训练必须用）
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)  # 分类标签必须是 long 类型

    return x, y


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    # 关键：堆叠成张量
    return torch.stack(X), torch.stack(Y)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    """
    用来测试每轮模型的准确率
    :param model:
    :return: 成功率
    """
    model.eval()  # 切换到评估模式
    total_sample_num = 100  # 测试数据量
    x, y = build_dataset(total_sample_num)  # 生成测试数据

    correct, wrong = 0, 0

    with torch.no_grad():  # 关闭梯度计算
        y_pred = model(x)  # 模型输出 logits，shape: [100,5]

        # 核心改动：取每一行最大值的索引 = 预测类别
        pred_classes = torch.argmax(y_pred, dim=1)

        # 对比预测类别 和 真实类别
        for y_p, y_t in zip(pred_classes, y):
            if y_p == y_t:
                correct += 1
            else:
                wrong += 1

    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本数
    train_sample = 5000  # 每轮训练样本总数，相当于每轮要训练（train_sample/batch_size）次
    input_size = 5  # 输入向量维度
    learning_rate = 0.01  # 学习率
    # 建立模型
    model = TorchModel(input_size)
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
            # 取出一个batch数据作为输入   train_x[0:20]  train_y[0:20] train_x[20:40]  train_y[20:40]
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model.forward(x,y)  -->  必须和你自己定义的 forward 参数、返回值对应
            """
            当你写 model(x, y) 时，PyTorch 会自动调用你写的 forward 函数
            同时还会帮你处理：训练 / 评估模式、梯度计算、网络钩子等底层操作
            
            PyTorch 的 cross_entropy 默认行为：
            对这 20 个样本分别算 loss → 然后自动求平均值 → 返回 1 个数字！
            """
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())  # loss是每个批次（20个样本）的平均损失
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))  # np.mean(watch_loss)是每一轮（5000个样本）的损失平均值。把这一整轮 250 个 batch 的 loss 全部加起来求平均值
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    # print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model(torch.FloatTensor(input_vec))  # 模型预测
        result = torch.argmax(result, dim=1)
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))  # 打印结果


if __name__ == "__main__":
    main()
    test_vec = [[0.88889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.99349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    predict("model.bin", test_vec)


# coding:utf8

# 解决 OpenMP 库冲突问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "model.bin")

"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
【作业】规律：x是一个5维向量，哪一维数字最大就属于第几类（共5类：0/1/2/3/4）
"""


class TorchModel(nn.Module):

    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  # 改为输出维度从1变为num_classes（5类）
        self.loss = nn.CrossEntropyLoss()  # 多分类用CrossEntropyLoss（内部自带softmax+log）

    def forward(self, x, y=None):
        y_pred = self.linear(x)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失
        else:
            return F.softmax(y_pred, dim=1)  # 改为推理时用softmax转为概率


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量
def build_sample():
    x = np.random.random(5)
    return x, int(np.argmax(x))  # 改为直接找最大值的索引作为类别


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)  # 改为直接存类别索引0~4，不再是嵌套列表[y]

    return torch.FloatTensor(X), torch.LongTensor(Y)  #Y改为用LongTensor（CrossEntropyLoss要求）


# 评估函数
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)

    for c in range(5):
        print("  类别%d的样本数：%d" % (c, sum(1 for yi in y if yi == c)))

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred_prob = model(x)  # (batch, 5) softmax概率分布
        y_pred_class = torch.argmax(y_pred_prob, dim=1)  # (batch,) 预测类别
        for y_p, y_t in zip(y_pred_class, y):
            if int(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1

    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20         # 训练轮数
    batch_size = 20        # 每次训练样本个数
    train_sample = 5000    # 每轮训练总共训练的样本总数
    input_size = 5         # 输入向量维度
    num_classes = 5        # 分类数
    learning_rate = 0.01   # 学习率

    # 建立模型，改为传入num_classes
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
            #取出一个batch数据作为输入   train_x[0:20]  train_y[0:20] train_x[20:40]  train_y[20:40]
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]

            loss = model(x, y)    # 计算loss  model.forward(x,y)
            loss.backward()       # 计算梯度
            optim.step()          # 更新权重
            optim.zero_grad()     # 梯度归零
            watch_loss.append(loss.item())

        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])

    # 保存模型
    torch.save(model.state_dict(), model_path)

    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")   # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    num_classes = 5
    model = TorchModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path, weights_only=True))  # 加载训练好的权重
    
    model.eval()  # 测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # (batch, 5) 概率分布

    for vec, res in zip(input_vec, result):
        pred_class = int(torch.argmax(res))
        print("输入：%s, 预测类别：%d, 概率值：%s" % (
            vec, pred_class, [f"{r:.4f}" for r in res.tolist()]
        ))


if __name__ == "__main__":
    # 训练
    # main()

    test_vec = [
        [0.81889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],  # 第5维最大->类别4
        [0.94963533, 0.55242560, 0.95758807, 0.95520434, 0.84890681],  # 第3维最大->类别2
        [0.90797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],  # 第1维最大->类别0
        [0.39349776, 0.59416669, 0.92579291, 0.41567412, 0.83588940],  # 第3维最大->类别2
    ]
    predict(model_path, test_vec)

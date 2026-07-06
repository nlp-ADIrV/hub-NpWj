import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""

尝试完成一个多分类任务的训练:一个随机向量，哪一维数字最大就属于第几类。

"""

class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  # num_classes维
        self.activation = nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss()  # 交叉熵损失

    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, num_classes)
        if y is not None:
            return self.loss(x, y)  # 预测值和真实值计算损失
        else:
            # 预测时：手动加Softmax，得到每个类别的概率
            y_pred = self.activation(x)
            return y_pred   # (batch_size, num_classes)


# 哪一维最大，标签就是几
def build_sample():
    x = np.random.random(5)
    y = int(np.argmax(x))
    return x, y


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, num_classes):
    model.eval()
    test_sample_num = 200
    x, y = build_dataset(test_sample_num)

    class_counts = [(y == i).sum().item() for i in range(num_classes)]
    print("各类别样本数：", {f"第{i}类": class_counts[i] for i in range(num_classes)})

    correct = 0
    with torch.no_grad():
        y_pred = model(x) # shape: (test_sample_num, num_classes)
        predicted_classes = torch.argmax(y_pred, dim=1)  # shape: (test_sample_num,)
        correct = (predicted_classes == y).sum().item()
 
    acc = correct / test_sample_num
    print("正确预测个数：%d, 正确率：%f" % (correct, acc))
    return acc


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    num_classes = 5 # 类别数量（5维向量对应5个类）
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
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]

            loss = model(x, y)  # 前向传播, 计算loss  model.forward(x,y)
            loss.backward()  # 反向传播, 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())

        avg_loss = np.mean(watch_loss)
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, avg_loss))
        acc = evaluate(model, num_classes)  # 测试本轮模型结果
        log.append([acc, float(avg_loss)])

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
    print("模型权重：", model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))  # 打印结果


if __name__ == "__main__":
    main()
    # test_vec = [[0.88889086,0.15229675,0.31082123,0.03504317,0.88920843],
    #             [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
    #             [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392],
    #             [0.99349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    # predict("model.bin", test_vec)

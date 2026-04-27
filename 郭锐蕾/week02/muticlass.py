
# 解决 OpenMP 库冲突问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
基于pytorch框架编写模型训练
完成一个多分类任务的训练:一个随机向量，哪一维数字最大就属于第几类

"""

class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  # 输出每个类别的logits，如：[2.1, -0.3, 0.7]
        self.loss = nn.CrossEntropyLoss()  # 多分类常用损失

    # 传入y则返回loss；不传y则返回logits
    def forward(self, x, y=None):
        logits = self.linear(x)  # (batch_size, input_size) -> (batch_size, num_classes)
        if y is not None:
            return self.loss(logits, y)
        return logits


# 生成单个样本：最大值索引即类别
def build_sample(input_size=5):
    x = np.random.random(input_size)
    y = int(np.argmax(x))
    return x, y


# 生成数据集
def build_dataset(total_sample_num, input_size=5):
    X, Y = [], []
    for i in range(total_sample_num):
        x, y = build_sample(input_size)
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, input_size=5, num_classes=5):
    model.eval()
    test_sample_num = 200
    x, y = build_dataset(test_sample_num, input_size)

    class_count = [0] * num_classes
    for label in y.tolist():
        class_count[label] += 1
    print("测试集各类别数量:", class_count)

    with torch.no_grad():
        logits = model(x)
        y_pred = torch.argmax(logits, dim=1)  # 取最大logit对应类别
        correct = (y_pred == y).sum().item()
        acc = correct / test_sample_num

    print("正确预测个数：%d, 正确率：%f" % (correct, acc))
    return acc
    
def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练的样本数
    train_sample = 5000  # 每轮训练的样本数
    input_size = 5  # 输入向量的维度
    num_classes = 5  # 类别数
    learning_rate = 0.01  # 学习率

    # 建立模型
    model = TorchModel(input_size, num_classes)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 随机生成训练样本
    log = []
    train_x, train_y = build_dataset(train_sample, input_size)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []

        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]

            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())

        avg_loss = float(np.mean(watch_loss))
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, avg_loss))
        acc = evaluate(model, input_size, num_classes)
        log.append([acc, avg_loss])

    torch.save(model.state_dict(), "model_multiclass.bin")
    print("训练完成，模型已保存为 model_multiclass.bin")
    print(log)

    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    num_classes = 5
    model = TorchModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(input_vec)
        logits = model(x)
        pred = torch.argmax(logits, dim=1)

    for vec, cls in zip(input_vec, pred.tolist()):
        print("输入：%s, 预测类别：%d" % (vec, cls))

if __name__ == "__main__":
    main()
    test_vec = [
        [0.1, 0.2, 0.9, 0.3, 0.4],   # 应该是2类
        [0.8, 0.2, 0.1, 0.3, 0.4],   # 应该是0类
        [0.1, 0.95, 0.9, 0.3, 0.4],  # 应该是1类
    ]
    predict("model_multiclass.bin", test_vec)



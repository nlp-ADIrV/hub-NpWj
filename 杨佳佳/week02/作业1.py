import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
多分类任务规则：
随机生成一个5维向量，哪一维数字最大，就属于第几类（类别0~4，对应第1~5维）
"""

class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  # 线性层输出5维（5分类）
        self.loss = nn.CrossEntropyLoss()  # 多分类专用损失函数：交叉熵

    # 当输入真实标签，返回loss值；无真实标签，返回预测概率
    def forward(self, x, y=None):
        logits = self.linear(x)  # (batch_size, input_size) -> (batch_size, 5)
        if y is not None:
            return self.loss(logits, y)  # 交叉熵损失直接接收原始输出+整数标签
        else:
            return torch.softmax(logits, dim=1)  # 归一化为概率，输出5个类别的概率

# 生成一个样本：5维向量，最大值所在的索引为标签（0-4）
def build_sample():
    x = np.random.random(5)
    label = np.argmax(x)  # 找最大值的维度索引，作为分类标签
    return x, label

# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    # 标签转为长整型（交叉熵损失要求标签为整数）
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 测试代码：测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测：输出5个类别的概率
        pred_labels = torch.argmax(y_pred, dim=1)  # 取概率最大的索引为预测标签
        # 对比预测值与真实标签
        for y_p, y_t in zip(pred_labels, y):
            if y_p == y_t:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    # 配置参数（与原代码一致，仅新增分类数）
    epoch_num = 50        # 训练轮数
    batch_size = 20       # 每次训练样本个数
    train_sample = 5000   # 训练样本总数
    input_size = 5        # 输入向量维度
    num_classes = 5       # 分类类别数（5分类）
    learning_rate = 0.01  # 学习率
    # 建立模型
    model = TorchModel(input_size, num_classes)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)      # 计算loss
            loss.backward()         # 计算梯度
            optim.step()            # 更新权重
            optim.zero_grad()       # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)       # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "multi_model.bin")
    # 画图（准确率+损失曲线）
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()
    return

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    num_classes = 5
    model = TorchModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    model.eval()  # 测试模式
    with torch.no_grad():
        result = model(torch.FloatTensor(input_vec))  # 模型预测
    # 输出结果：输入向量、预测类别、各类别概率
    for vec, res in zip(input_vec, result):
        pred_class = torch.argmax(res).item()  # 预测类别
        print(f"输入：{vec}, 预测类别：{pred_class+1}（第{pred_class+1}维最大）, 概率值：{res.numpy()}")

if __name__ == "__main__":
    main()
    # 测试预测
    test_vec = [
        [0.1, 0.2, 0.8, 0.3, 0.2],   # 第3维最大 → 类别3
        [0.9, 0.1, 0.1, 0.1, 0.1],   # 第1维最大 → 类别1
        [0.2, 0.2, 0.2, 0.2, 0.9],   # 第5维最大 → 类别5
        [0.3, 0.7, 0.2, 0.1, 0.1]    # 第2维最大 → 类别2
    ]
    predict("multi_model.bin", test_vec)

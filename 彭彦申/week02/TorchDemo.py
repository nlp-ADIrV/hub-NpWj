import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
规律：x是一个5维向量，哪个位置的数最大，标签就是该位置的索引(0,1,2,3,4)
"""

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        # 1. 输出维度为5（五分类）
        self.linear = nn.Linear(input_size, 5)
        # 2. 多分类损失函数=交叉熵
        self.loss = nn.CrossEntropyLoss()

    # 前向传播
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, 5) -> 输出5个原始分数(logits)
        if y is not None:
            return self.loss(x, y)  # 训练：计算交叉熵损失
        else:
            return x  # 预测：返回原始分数，后续取最大值索引

# 生成单个样本：5维向量，最大值索引为标签
def build_sample():
    x = np.random.random(5)
    # 找最大值的索引作为标签 0/1/2/3/4
    label = np.argmax(x)
    return x, label

# 生成批量样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 测试模型准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型输出 [batch,5] 的分数
        pred_labels = torch.argmax(y_pred, dim=1)
        # 对比真实标签和预测标签
        for y_p, y_t in zip(pred_labels, y):
            if y_p == y_t:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    # 配置参数
    epoch_num = 20
    batch_size = 20
    train_sample = 5000
    input_size = 5
    learning_rate = 0.001
    # 建立模型
    model = TorchModel(input_size)
    # 优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 生成训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练
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
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model_5class.bin")
    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()

# 预测函数
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        result = model(torch.FloatTensor(input_vec))
        # 取最大值索引为预测类别
        pred_labels = torch.argmax(result, dim=1)
    for vec, res in zip(input_vec, pred_labels):
        print(f"输入：{vec}, 预测类别：{res.item()+1}")

if __name__ == "__main__":
    main()
    # 测试预测（取消注释即可运行）
    test_vec = [
        [0.1, 0.8, 0.2, 0.3, 0.4],   # 最大值索引1 → 类别2
        [0.9, 0.1, 0.1, 0.1, 0.1],   # 最大值索引0 → 类别1
        [0.2, 0.2, 0.9, 0.2, 0.2],   # 最大值索引2 → 类别3
        [0.3, 0.3, 0.3, 0.9, 0.3],   # 最大值索引3 → 类别4
        [0.5, 0.5, 0.5, 0.5, 0.9]    # 最大值索引4 → 类别5
    ]
    predict("model_5class.bin", test_vec)

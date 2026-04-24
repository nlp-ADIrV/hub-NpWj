import os
import torch
import torch.nn as nn
import numpy as np

"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
五维判断：x是一个5维向量，向量中哪个标量最大就输出哪一维下标

2分类：输出0-1之间的一个数 大于0.5正类，小于0.5负类
多分类：输出概率分布  要分几类就输出几维的向量【0.1,0.1,0.1,0.5,0.2】   x
2分类也可以输出2维向量 用交叉熵损失函数 【0.1,0.9】
"""

class MultiClassficationModel(nn.Module):
    def __init__(self, input_size):
        super(MultiClassficationModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        y_pred = self.linear(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return torch.softmax(y_pred, dim=1)

# 最大值        
def build_sample():
    x = np.random.random(5)
    # 获取最大值的索引
    max_index = np.argmax(x)
    return x, max_index

# 正负样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 预测准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if torch.argmax(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1

    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

# 测试模型
def main():
    epoch_num = 20
    batch_size = 20
    train_sample = 5000
    input_size = 5
    learning_rate = 0.005
    model = MultiClassficationModel(input_size)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    train_x, train_y = build_dataset(train_sample)
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])

    torch.save(model.state_dict(), "model.pt")
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()
    return

# 进行预测
def predict(model_path, input_vec):
    input_size = 5
    model = MultiClassficationModel(input_size)
    model.load_state_dict(torch.load(model_path))
    print(model.state_dict())

    model.eval()
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vec))
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%s, 概率值：%s" % (vec, torch.argmax(res), res))

if __name__ == "__main__":
    main()
    test_vec = [[0.47889086, 0.15229675, 0.31082123, 0.03504317, 0.18920843],
                [0.4963533, 0.5524256, 0.95758807, 0.65520434, 0.84890681],
                [0.48797868, 0.67482528, 0.13625847, 0.34675372, 0.09871392],
                [0.49349776, 0.59416669, 0.92579291, 0.41567412, 0.7358894]]
    predict("model.pt", test_vec)

#-------------------------------------------------------------------------------------
'''
  return torch.FloatTensor(X), torch.LongTensor(Y)
=========
第1轮平均loss:1.427676
正确预测个数：82, 正确率：0.820000
=========
第2轮平均loss:1.145298
正确预测个数：90, 正确率：0.900000
=========
第3轮平均loss:0.961141
正确预测个数：87, 正确率：0.870000
=========
第4轮平均loss:0.835542
正确预测个数：95, 正确率：0.950000
=========
第5轮平均loss:0.745496
正确预测个数：96, 正确率：0.960000
=========
第6轮平均loss:0.677922
正确预测个数：92, 正确率：0.920000
=========
第7轮平均loss:0.625236
正确预测个数：93, 正确率：0.930000
=========
第8轮平均loss:0.582865
正确预测个数：92, 正确率：0.920000
=========
第9轮平均loss:0.547921
正确预测个数：94, 正确率：0.940000
=========
第10轮平均loss:0.518510
正确预测个数：93, 正确率：0.930000
=========
第11轮平均loss:0.493336
正确预测个数：94, 正确率：0.940000
=========
第12轮平均loss:0.471487
正确预测个数：91, 正确率：0.910000
=========
第13轮平均loss:0.452303
正确预测个数：97, 正确率：0.970000
=========
第14轮平均loss:0.435291
正确预测个数：93, 正确率：0.930000
=========
第15轮平均loss:0.420077
正确预测个数：94, 正确率：0.940000
=========
第16轮平均loss:0.406371
正确预测个数：94, 正确率：0.940000
=========
第17轮平均loss:0.393946
正确预测个数：90, 正确率：0.900000
=========
第18轮平均loss:0.382617
正确预测个数：96, 正确率：0.960000
=========
第19轮平均loss:0.372237
正确预测个数：93, 正确率：0.930000
=========
第20轮平均loss:0.362683
正确预测个数：92, 正确率：0.920000
[[0.82, 1.4276761407852172], [0.9, 1.145297780752182], [0.87, 0.9611405346393586], [0.95, 0.8355420320034027], [0.96, 0.7454959025382996], [0.92, 0.6779215338230133], [0.93, 0.6252363214492798], [0.92, 0.5828647013902665], [0.94, 0.5479214072227478], [0.93, 0.518509810090065], [0.94, 0.4933358598947525], [0.91, 0.47148746502399447], [0.97, 0.4523032463788986], [0.93, 0.43529101979732515], [0.94, 0.42007698690891265], [0.94, 0.40637135863304136], [0.9, 0.39394557762146], [0.96, 0.3826168448925018], [0.93, 0.3722367832660675], [0.92, 0.3626834152936935]]
'''

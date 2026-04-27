import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
任务修改：多分类任务
规律：x是一个5维向量，哪一维的数字最大，就属于第几类（类别0-4）
"""

class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        # 【修改点1】输出层维度改为 num_classes (5)，不再是 1
        self.linear = nn.Linear(input_size, num_classes)  
        
        # 【修改点2】多分类通常不需要在这里加 Sigmoid。
        # CrossEntropyLoss 期望接收原始分数 (Logits)，它内部会处理 Softmax。
        # 如果只是为了看概率，我们在预测时再单独加 Softmax。
        
        # 【修改点3】损失函数改为交叉熵，这是多分类的标准损失函数
        # self.loss = nn.CrossEntropyLoss()
        self.loss = nn.functional.cross_entropy

    # 当输入真实标签，返回loss值；无真实标签，返回预测概率
    def forward(self, x, y=None):
        y_pred = self.linear(x)  # (batch_size, 5) -> 输出5个类别的分数
        if y is not None:
            # CrossEntropyLoss 要求 y 是 LongTensor 类型的类别索引，而不是 FloatTensor
            return self.loss(y_pred, y)  
        else:
            # 预测时，使用 Softmax 将分数转换为概率分布
            return torch.softmax(y_pred, dim=1)

# 生成一个样本
# 规律：哪一维数字最大，标签就是几
def build_sample():
    x = np.random.random(5)
    # 【修改点4】使用 np.argmax 找到最大值的索引
    y = np.argmax(x) 
    return x, y

# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        # 【修改点5】标签 Y 必须是 Long 类型 (整数)，不能是列表 [y]
        Y.append(y) 
    
    # X 转为 FloatTensor, Y 转为 LongTensor (CrossEntropyLoss 的要求)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 测试代码
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    
    # 统计各类别数量（可选，仅用于展示）
    # print("本次预测集样本分布:", np.bincount(y.numpy()))
    
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 得到的是概率分布 (100, 5)
        
        # 【修改点6】获取预测概率最大的索引
        # torch.max(tensor, dim=1) 返回 (最大值, 最大值索引)
        _, predicted_class = torch.max(y_pred, 1)
        
        for p_c, t_c in zip(predicted_class, y):
            if p_c == t_c:
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
    num_classes = 5  # 【修改点7】定义类别数量
    learning_rate = 0.01
    
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
            
            loss = model(x, y)  # 计算 loss
            loss.backward()     # 计算梯度
            optim.step()        # 更新权重
            optim.zero_grad()   # 梯度归零
            watch_loss.append(loss.item())
            
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])
        
    # 保存模型
    torch.save(model.state_dict(), "model_multi.bin")
    
    # 画图
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
    model.load_state_dict(torch.load(model_path))   # 加载训练好的权重
    print(model.state_dict())
    
    model.eval()
    with torch.no_grad():
        # 输入转为 Tensor
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
        # x = torch.FloatTensor(input_vec)
        # result = model(x)  # 得到概率分布
        
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%s, 概率值：%s" % (vec, torch.argmax(res), res))  # 打印结果

if __name__ == "__main__":
    main()
    # 测试预测代码
    test_vec = [[0.1, 0.2, 0.9, 0.1, 0.1], # 显然属于第2类
                [0.5, 0.5, 0.5, 0.5, 0.51]] # 显然属于第4类
    predict("model_multi.bin", test_vec)

"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，交叉熵实现一个多分类任务，五维随机向量最大的数字在哪维就属于哪一类。
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 定义模型
class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  # 线性层

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, num_classes)
        if y is not None:
            # 直接用线性层输出计算交叉熵（自带softmax，删除手动softmax）
            return nn.functional.cross_entropy(x, y)
        else:
            # 预测时才需要softmax
            return torch.softmax(x, dim=1)

# 生成样本数据
def build_dataset(total_sample_num):
    x = np.random.randint(0, 10, (total_sample_num, 5))
    y = np.argmax(x, axis=1)
    x = torch.FloatTensor(x)
    y = torch.LongTensor(y)
    return x, y   

# 测试代码：用来测试每轮模型的准确率
def evaluate(model):
    model.eval() # 模型评估模式
    X, Y = build_dataset(100) # 生成测试数据
    with torch.no_grad(): # 评估模式不计算梯度
        y_pred = model(X) # 模型预测
        predicted_classes = torch.argmax(y_pred, dim=1) # 取概率最大的类别
        accuracy = (predicted_classes == Y).float().mean() # 计算准确率
    # print(f"Evaluation Accuracy: {accuracy:.4f}")
    return accuracy  # 修复：返回准确率！

# 主训练函数
def main():
    # 配置参数
    epoch_num = 20    # 训练轮数
    batch_size = 20   # 每次训练样本个数
    train_sample = 5000  # 训练样本总数
    input_size = 5    # 输入特征维度
    num_classes = 5   # 分类类别数    
    model = TorchModel(input_size, num_classes)
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    
    log = []

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        X, Y = build_dataset(train_sample)
        watch_loss = []   # 存储每轮平均loss
        # 按批次遍历数据
        for i in range(0, train_sample, batch_size):
            x_batch = X[i:i+batch_size]
            y_batch = Y[i:i+batch_size]
            loss = model(x_batch, y_batch)
            optim.zero_grad()
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        
        print(f"Epoch {epoch+1}/{epoch_num}, Average Loss: {np.mean(watch_loss)}")
        
        # 测试准确率
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])

    # 保存模型
    torch.save(model.state_dict(), "model.pth")  # 建议用.pth后缀
    
    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    model = TorchModel(5, 5)  # 输入维度5，类别数5
    model.load_state_dict(torch.load(model_path))
    print(model.state_dict())
    model.eval()
    with torch.no_grad():
        y_pred = model(input_vec)
        print("y_pred:=====",y_pred)
        predicted_class = torch.argmax(y_pred, dim=1)
    for vec, res in zip(input_vec, predicted_class):
        print("输入：%s, 预测类别：%d" % (vec, round(float(res))))  # 打印结果
if __name__ == "__main__":
    main()
    # test_vec = [[10,11,12,13,14],
    #             [22222,333333,444444,1,44222],
    #             [0.1,0.2,0.3,0.4,0.5],
    #             [100,200,300,400,500],
    #             [1,3,8,4,5]]
    # test_vec = torch.FloatTensor(test_vec)
    # predict("model.pth", test_vec)
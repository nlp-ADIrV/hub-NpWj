import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes) 
        self.loss = nn.CrossEntropyLoss()                  

    def forward(self, x, y=None):
        value = self.linear(x)    
        if y is not None:
            #计算损失值
            return self.loss(value, y)
        else:
            return torch.softmax(value, dim=-1)

def build_sample():
    x = np.random.random(5)
    label = np.argmax(x)      
    return x, label

# 生成样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for _ in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)  


def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y_true = build_dataset(test_sample_num)

    unique, counts = torch.unique(y_true, return_counts=True)
    for cla, num in zip(unique.tolist(), counts.tolist()):
        print(f"类别 {cla} 样本数：{num}")

    correct = 0
    with torch.no_grad():
        logits = model.linear(x)               
        y_pred = torch.argmax(logits, dim=1)   
        correct = (y_pred == y_true).sum().item()
    accuracy = correct / len(y_true)
    return accuracy

def main():
    epoch_num = 20           
    batch_size = 20          
    train_sample = 5000      
    input_size = 5           
    num_classes = 5          
    learning_rate = 0.01

    model = TorchModel(input_size, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_x, train_y = build_dataset(train_sample)

    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_idx in range(train_sample // batch_size):
            x = train_x[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            y = train_y[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            loss = model(x, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            watch_loss.append(loss.item())

        acc = evaluate(model)
        log.append([acc, np.mean(watch_loss)])

    torch.save(model.state_dict(), "model_class.bin")

    plt.plot(range(len(log)), [l[0] for l in log], label="accuracy")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()


def predict(model_path, input_vec):
    input_size = 5
    num_classes = 5
    model = TorchModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        probs = model(torch.FloatTensor(input_vec))  
        pred_classes = torch.argmax(probs, dim=1)
    for vec, cls, prob in zip(input_vec, pred_classes, probs):
        print(f"输入：{vec}，预测类别：{cls.item()}，概率分布：{prob.numpy()}")

if __name__ == "__main__":
    main()
    test_vec = [
        [10, 0.5, 0.2, 5, 6],
        [0.8, 7, 15, 0.1, 5],
        [0.2, 2, 0.9, 17, 0.1]
    ]
    predict("model_class.bin", test_vec)

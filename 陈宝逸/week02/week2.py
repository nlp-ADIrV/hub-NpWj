import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# ------------------------------
# 1. 生成训练数据
# ------------------------------
def generate_data(num_samples, vec_dim):
    """
    生成随机向量及对应的标签（最大值所在索引）
    """
    X = torch.randn(num_samples, vec_dim)  # 随机向量
    y = torch.argmax(X, dim=1)             # 标签：最大值的下标
    return X, y

vec_dim = 10        # 向量维度（也是类别数）
train_size = 5000
test_size = 1000

train_X, train_y = generate_data(train_size, vec_dim)
test_X, test_y = generate_data(test_size, vec_dim)

# 包装为 DataLoader
train_dataset = TensorDataset(train_X, train_y)
test_dataset = TensorDataset(test_X, test_y)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ------------------------------
# 2. 定义神经网络模型
# ------------------------------
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

model = SimpleClassifier(input_dim=vec_dim, hidden_dim=64, output_dim=vec_dim)

# ------------------------------
# 3. 损失函数与优化器
# ------------------------------
criterion = nn.CrossEntropyLoss()   # 多分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ------------------------------
# 4. 训练循环
# ------------------------------
num_epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("开始训练...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # 每个 epoch 结束后在测试集上评估
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, Test Accuracy: {accuracy:.2f}%")

# ------------------------------
# 5. 验证一个随机样本
# ------------------------------
sample = torch.randn(1, vec_dim).to(device)
pred = model(sample)
pred_class = torch.argmax(pred, dim=1).item()
true_class = torch.argmax(sample, dim=1).item()
print(f"\n随机样本预测类别：{pred_class}，真实类别：{true_class}")
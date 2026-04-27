import torch
import torch.nn as nn

"""
作业要求：
完成一个多分类任务的训练：一个随机向量，哪一维数字最大就属于第几类。

"""

class TorchModel(nn.Module):
    """简单线性分类模型，输出 logits。"""

    def __init__(self, input_dim):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        return self.linear(x)


def evaluate(model, x, y):
    """在给定数据集上计算准确率。"""
    model.eval()
    with torch.no_grad():
        logits = model(x)
        predictions = torch.argmax(logits, dim=1)
        return (predictions == y).float().mean().item()


def main():
    torch.manual_seed(0)

    input_dim = 5
    batch_size = 4
    learning_rate = 0.01
    epochs = 20

    train_X = torch.tensor([
        [0.1, 0.2, 0.9, 0.3, 0.4],
        [0.8, 0.2, 0.1, 0.3, 0.4],
        [0.2, 0.9, 0.1, 0.3, 0.4],
        [0.4, 0.3, 0.2, 0.9, 0.1],
        [0.2, 0.3, 0.4, 0.1, 0.95],
        [0.9, 0.8, 0.3, 0.2, 0.1],
        [0.1, 0.8, 0.3, 0.5, 0.4],
        [0.3, 0.2, 0.1, 0.4, 0.9],
        [0.55, 0.54, 0.50, 0.49, 0.48],
        [0.45, 0.46, 0.47, 0.44, 0.43],
        [0.31, 0.33, 0.32, 0.30, 0.29],
        [0.27, 0.26, 0.25, 0.28, 0.24],
        [0.15, 0.14, 0.16, 0.13, 0.12],
        [0.19, 0.18, 0.17, 0.20, 0.16],
        [0.42, 0.41, 0.40, 0.43, 0.39],
    ], dtype=torch.float32)
    train_y = torch.argmax(train_X, dim=1)

    test_X = torch.tensor([
        [0.12, 0.18, 0.17, 0.16, 0.15],
        [0.52, 0.51, 0.50, 0.49, 0.53],
        [0.28, 0.29, 0.27, 0.30, 0.26],
        [0.05, 0.07, 0.06, 0.08, 0.04],
        [0.34, 0.33, 0.35, 0.32, 0.31],
        [0.23, 0.25, 0.24, 0.22, 0.26],
    ], dtype=torch.float32)
    test_y = torch.argmax(test_X, dim=1)

    train_samples = train_X.size(0)
    test_samples = test_X.size(0)

    model = TorchModel(input_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("开始训练：随机向量的最大值所在维度作为类别")
    print(f"训练样本: {train_samples}, 测试样本: {test_samples}\n")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for start in range(0, train_samples, batch_size):
            end = start + batch_size
            x_batch = train_X[start:end]
            y_batch = train_y[start:end]

            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x_batch.size(0)

        epoch_loss /= train_samples
        train_acc = evaluate(model, train_X, train_y)
        test_acc = evaluate(model, test_X, test_y)

        print(f"Epoch {epoch:02d}: loss={epoch_loss:.4f}, train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")

    final_acc = evaluate(model, test_X, test_y)
    print(f"\n训练结束。最终测试准确率：{final_acc:.4f}\n")

    print("=== 示例预测 ===")
    sample_X = torch.tensor([
        [0.1, 0.2, 0.9, 0.3, 0.4],
        [0.8, 0.2, 0.1, 0.3, 0.4],
        [0.1, 0.95, 0.9, 0.3, 0.4],
    ], dtype=torch.float32)
    sample_y = torch.argmax(sample_X, dim=1)
    sample_logits = model(sample_X)
    sample_preds = torch.argmax(sample_logits, dim=1)

    for i in range(sample_X.size(0)):
        x_list = [f"{v:.2f}" for v in sample_X[i].tolist()]
        print(
            f"输入: {x_list}, 真实标签: {sample_y[i].item()}, 预测标签: {sample_preds[i].item()}"
        )


if __name__ == '__main__':
    main()

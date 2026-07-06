'''1、设计一个以文本为输入的多分类任务，实验一下用RNN，LSTM等模型的跑通训练。
2、对一个任意包含“你”字的五个字的文本，“你”在第几位，就属于第几类。'''

import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    """自定义 PyTorch 文本数据集

    输入的数据为固定长度为 5 的中文字符文本，每个文本中的一个字符是“你”。
    标签是“你”字符出现的位置，范围为 0 到 4。
    """

    def __init__(self, texts, labels, vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        # 将每个字符映射为词表 id
        ids = [self.vocab[ch] for ch in text]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


class RNNClassifier(nn.Module):
    """基于简单 RNN 的分类器"""

    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: [batch_size, seq_len]
        x = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        out, _ = self.rnn(x)  # [batch_size, seq_len, hidden_size]
        # 仅使用最后一个时间步作为分类特征
        out = out[:, -1, :]
        return self.fc(out)


class LSTMClassifier(nn.Module):
    """基于 LSTM 的分类器"""

    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: [batch_size, seq_len]
        x = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_size]
        out = out[:, -1, :]
        return self.fc(out)


def build_data(num_samples=1000):
    """生成训练/验证/测试所需的合成文本数据"""
    fillers = ['我', '是', '的', '了', '在', '他', '她', '们', '好', '不']
    texts = []
    labels = []

    for _ in range(num_samples):
        target_pos = random.randrange(5)
        sample = []
        for pos in range(5):
            if pos == target_pos:
                sample.append('你')
            else:
                sample.append(random.choice(fillers))
        texts.append(''.join(sample))
        labels.append(target_pos)

    # 生成词表，将字符映射成整数 id，保留一个 padding id
    vocab_chars = sorted(set(''.join(texts)))
    vocab = {ch: idx + 1 for idx, ch in enumerate(vocab_chars)}
    vocab['<pad>'] = 0
    return texts, labels, vocab


def collate_fn(batch):
    """DataLoader 的 collate_fn，用于将 batch 中样本拼接成 tensor"""
    texts, labels = zip(*batch)
    texts = torch.stack(texts)
    labels = torch.stack(labels)
    return texts, labels


def evaluate(model, dataloader, device):
    """计算模型在数据集上的准确率"""
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for texts, labels in dataloader:
            texts = texts.to(device)
            labels = labels.to(device)
            outputs = model(texts)
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return correct / total


def train_model(model, train_loader, val_loader, device, epochs=10, lr=0.001):
    """训练模型并在每个 epoch 后打印训练损失和验证准确率"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for texts, labels in train_loader:
            texts = texts.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        val_acc = evaluate(model, val_loader, device)
        print(f'Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}')


def main():
    # 固定随机种子，确保实验可复现
    random.seed(42)
    torch.manual_seed(42)

    # 构造数据集：1200 个样本
    texts, labels, vocab = build_data(num_samples=1200)
    train_texts = texts[:1000]
    train_labels = labels[:1000]
    val_texts = texts[1000:1100]
    val_labels = labels[1000:1100]
    test_texts = texts[1100:1200]
    test_labels = labels[1100:1200]

    # 创建 Dataset 对象
    train_dataset = TextDataset(train_texts, train_labels, vocab)
    val_dataset = TextDataset(val_texts, val_labels, vocab)
    test_dataset = TextDataset(test_texts, test_labels, vocab)

    # 创建 DataLoader，用于批量加载数据
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 这里可以切换模型：'RNN' 或 'LSTM'
    model_type = 'LSTM'
    if model_type == 'RNN':
        model = RNNClassifier(vocab_size=len(vocab), embed_dim=32, hidden_size=64, num_classes=5)
    else:
        model = LSTMClassifier(vocab_size=len(vocab), embed_dim=32, hidden_size=64, num_classes=5)

    print(f'Using model: {model_type}, vocab size: {len(vocab)}, device: {device}')

    # 模型训练
    train_model(model, train_loader, val_loader, device, epochs=10, lr=0.001)

    # 测试集评估
    test_acc = evaluate(model, test_loader, device)
    print(f'Test Accuracy: {test_acc:.4f}')

    # 使用几个样本文本进行推理演示
    sample_texts = ['你好吗你你', '我你是在你', '不知道你在吗', '今天你很好', '你你你你你']
    model.eval()
    with torch.no_grad():
        for text in sample_texts:
            ids = torch.tensor([[vocab.get(ch, 0) for ch in text]], dtype=torch.long).to(device)
            pred = model(ids).argmax(dim=1).item()
            print(f'文本: {text} -> 你的位置类: {pred}')


if __name__ == '__main__':
    main()


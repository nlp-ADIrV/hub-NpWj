import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
import random

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ==================== 1. 数据生成 ====================
def generate_data(num_samples=10000):
    """生成训练数据：5字文本，'你'在指定位置"""
    # 常用中文字符集（去掉'你'）
    common_chars = '天地人和中国山水年月日时分秒上下左右大小多少金银财宝福禄寿喜'
    
    data = []
    labels = []
    
    for _ in range(num_samples):
        # 随机选择'你'的位置（1-5）
        pos = random.randint(1, 5)
        
        # 构建5个字的文本
        chars = []
        for i in range(5):
            if i == pos - 1:  # 索引从0开始
                chars.append('你')
            else:
                chars.append(random.choice(common_chars))
        
        text = ''.join(chars)
        data.append(text)
        labels.append(pos - 1)  # 类别0-4
    
    return data, labels

# ==================== 2. 构建词表 ====================
class Vocab:
    """简单的字符级词表"""
    def __init__(self, texts):
        self.char2idx = {}
        self.idx2char = {}
        
        # 收集所有字符
        all_chars = set()
        for text in texts:
            all_chars.update(text)
        
        # 添加特殊token
        special_tokens = ['<PAD>', '<UNK>']
        for token in special_tokens:
            self.char2idx[token] = len(self.char2idx)
            self.idx2char[self.char2idx[token]] = token
        
        # 添加普通字符
        for char in sorted(all_chars):
            if char not in self.char2idx:
                self.char2idx[char] = len(self.char2idx)
                self.idx2char[self.char2idx[char]] = char
        
        self.vocab_size = len(self.char2idx)
        self.pad_idx = self.char2idx['<PAD>']
        self.unk_idx = self.char2idx['<UNK>']
    
    def encode(self, text, max_len=5):
        """将文本编码为索引序列"""
        indices = [self.char2idx.get(char, self.unk_idx) for char in text]
        # 截断或填充
        if len(indices) < max_len:
            indices += [self.pad_idx] * (max_len - len(indices))
        else:
            indices = indices[:max_len]
        return indices
    
    def decode(self, indices):
        """将索引序列解码为文本"""
        return ''.join([self.idx2char.get(idx, '<UNK>') for idx in indices])

# ==================== 3. 数据集类 ====================
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=5):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.vocab.encode(text, self.max_len)
        label = self.labels[idx]
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# ==================== 4. RNN模型 ====================
class RNNClassifier(nn.Module):
    """简单RNN分类器"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=2, dropout=0.5):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, 
                         batch_first=True, dropout=dropout, bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        output, hidden = self.rnn(embedded)  # output: (batch_size, seq_len, hidden_dim)
        # 使用最后一个时间步的隐藏状态
        last_hidden = output[:, -1, :]  # (batch_size, hidden_dim)
        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden)
        return logits

class LSTMClassifier(nn.Module):
    """LSTM分类器"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=2, dropout=0.5):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=dropout, bidirectional=False)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        output, (hidden, cell) = self.lstm(embedded)  # output: (batch_size, seq_len, hidden_dim)
        # 使用最后一个时间步的隐藏状态
        last_output = output[:, -1, :]  # (batch_size, hidden_dim)
        last_output = self.dropout(last_output)
        logits = self.fc(last_output)
        return logits

# ==================== 5. 训练函数 ====================
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device, model_name="Model"):
    """训练模型"""
    model = model.to(device)
    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        
        train_acc = 100. * correct / total
        train_losses.append(total_loss / len(train_loader))
        
        # 验证阶段
        val_acc = evaluate_model(model, val_loader, device)
        val_accuracies.append(val_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f'{model_name} - Epoch {epoch+1}/{epochs}: Train Loss: {total_loss/len(train_loader):.4f}, '
                  f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
    
    return train_losses, val_accuracies

def evaluate_model(model, data_loader, device):
    """评估模型"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return 100. * correct / total

def test_model(model, test_loader, device):
    """测试模型并显示详细结果"""
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * 5
    class_total = [0] * 5
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # 每个类别的准确率
            for i in range(len(target)):
                label = target[i].item()
                pred_label = pred[i].item()
                class_total[label] += 1
                if label == pred_label:
                    class_correct[label] += 1
    
    print(f'\n总体准确率: {100.*correct/total:.2f}%')
    print('各类别准确率:')
    for i in range(5):
        if class_total[i] > 0:
            print(f'  类别{i+1} ("你"在第{i+1}位): {100.*class_correct[i]/class_total[i]:.2f}%')
    
    return 100.*correct/total

def predict(model, text, vocab, device):
    """预测单个文本"""
    model.eval()
    encoded = torch.tensor([vocab.encode(text, max_len=5)], dtype=torch.long).to(device)
    with torch.no_grad():
        output = model(encoded)
        pred = output.argmax(dim=1).item()
    return pred + 1  # 转换回1-5

# ==================== 6. 主程序 ====================
def main():
    # 参数设置
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 256
    BATCH_SIZE = 64
    EPOCHS = 50
    LEARNING_RATE = 0.001
    NUM_CLASSES = 5
    NUM_LAYERS = 2
    DROPOUT = 0.5
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 生成数据
    print('生成训练数据...')
    train_texts, train_labels = generate_data(8000)
    val_texts, val_labels = generate_data(1000)
    test_texts, test_labels = generate_data(1000)
    
    # 构建词表
    vocab = Vocab(train_texts + val_texts + test_texts)
    print(f'词表大小: {vocab.vocab_size}')
    
    # 创建数据加载器
    train_dataset = TextDataset(train_texts, train_labels, vocab)
    val_dataset = TextDataset(val_texts, val_labels, vocab)
    test_dataset = TextDataset(test_texts, test_labels, vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 创建模型
    rnn_model = RNNClassifier(vocab.vocab_size, EMBEDDING_DIM, HIDDEN_DIM, 
                              NUM_CLASSES, NUM_LAYERS, DROPOUT)
    lstm_model = LSTMClassifier(vocab.vocab_size, EMBEDDING_DIM, HIDDEN_DIM, 
                                NUM_CLASSES, NUM_LAYERS, DROPOUT)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=LEARNING_RATE)
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE)
    
    # 训练RNN
    print('\n' + '='*50)
    print('训练RNN模型...')
    train_model(rnn_model, train_loader, val_loader, criterion, 
                rnn_optimizer, EPOCHS, device, "RNN")
    
    # 训练LSTM
    print('\n' + '='*50)
    print('训练LSTM模型...')
    train_model(lstm_model, train_loader, val_loader, criterion, 
                lstm_optimizer, EPOCHS, device, "LSTM")
    
    # 测试模型
    print('\n' + '='*50)
    print('RNN模型测试结果:')
    test_model(rnn_model, test_loader, device)
    
    print('\nLSTM模型测试结果:')
    test_model(lstm_model, test_loader, device)
    
    # 演示预测
    print('\n' + '='*50)
    print('预测演示:')
    test_texts_example = ['你天地人', '天你地人', '天地你人', '天地人你', '天地人你吗']
    for text in test_texts_example:
        rnn_pred = predict(rnn_model, text, vocab, device)
        lstm_pred = predict(lstm_model, text, vocab, device)
        print(f'文本: "{text}" -> RNN预测: 第{rnn_pred}位, LSTM预测: 第{lstm_pred}位')
    
    # 保存模型
    torch.save(lstm_model.state_dict(), 'lstm_model.pth')
    print('\n模型已保存到 lstm_model.pth')

if __name__ == '__main__':
    main()
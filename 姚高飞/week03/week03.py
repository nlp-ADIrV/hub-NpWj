import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


SEED = 42
N_SAMPLES = 8000       
MAXLEN = 5          
EMBED_DIM = 32
HIDDEN_DIM = 64
LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 25
TRAIN_RATIO = 0.8

random.seed(SEED)
torch.manual_seed(SEED)

#生成数据
TARGET_CHAR = '你'
OTHER_CHARS = ['我', '是', '一', '哈', '人', '好', '的', '了', '不', '在',
               '有', '和', '这', '哈', '也', '对', '可', '以', '没', '有']


def make_sample():
    index = random.randint(0, 4)   
    chars = []
    for i in range(5):
        if i == index:
          chars.append(TARGET_CHAR)
        else:
          chars.append(random.choice(OTHER_CHARS))
    text = ''.join(chars)
    return text, index

#生成样本
def build_dataset(n=N_SAMPLES):
    data = []
    for _ in range(n):
        text, label = make_sample()
        data.append((text, label))
    random.shuffle(data)
    return data


def build_vocab(data):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sent, _ in data:
        for ch in sent:
          if ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab


def encode(sent, vocab, maxlen=MAXLEN):
    ids = [vocab.get(ch, 1) for ch in sent]
    ids = ids[:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids


class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(s, vocab) for s, _ in data]
        self.y = [lb for _, lb in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long),   
        )

class PositionRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 5) 

    def forward(self, x):
        e, _ = self.rnn(self.embedding(x))   
        pooled = e.max(dim=1)[0]                 
        pooled = self.dropout(self.bn(pooled))
        logits = self.fc(pooled)           
        return logits


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            logits = model(X)       
            pred = logits.argmax(dim=1) 
            correct += (pred == y).sum().item()
            total   += len(y)
    return correct / total


def train():
    data  = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)

    split      = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data   = data[split:]

    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TextDataset(val_data,   vocab), batch_size=BATCH_SIZE)

    model     = PositionRNN(vocab_size=len(vocab))
    criterion = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            logits = model(X)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc  = evaluate(model, val_loader)
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    print(f"\n最终验证准确率：{evaluate(model, val_loader):.4f}")

    model.eval()
    for _ in range(6):
        text, true_pos = make_sample()
        ids = torch.tensor([encode(text, vocab)], dtype=torch.long)
        with torch.no_grad():
            logits = model(ids)
            pred_pos = logits.argmax(dim=1).item()
        print(f"文本:'{text}'的真实位置为{true_pos} 预测的位置为 {pred_pos} {'正确' if true_pos==pred_pos else '错误'}")


if __name__ == '__main__':
    train()

import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader, Dataset

# 1、超参
EPOCH = 20
SAMPE_NUM = 1000
SENT_LEN = 5
UNK_WORD = '[unk]'
PAD_WORD = '[pad]'
LR = 0.01
TRAIN_RATIO = 0.8
BATCH_SIZE = 32
EMBD_SIZE = 16
HIDDEN_SIZE = 25
DROPOUT_RATIO = 0.2


# 2、创建数据集（很多的句子，其中每个句子只有一个“你”）
def create_sent(vocab_dict):
    # 创建句子时先去除pad和unk和“你”字符
    vocab_copy = vocab_dict.copy()
    del vocab_copy[PAD_WORD]
    del vocab_copy[UNK_WORD]
    del vocab_copy['你']
    # 此处不使用list得到的是字典的一个内容，不是list
    all_sample = list(vocab_copy.keys()) 
    selected = random.sample(all_sample, k=SENT_LEN)
    idx = random.randint(0, SENT_LEN - 1)
    selected[idx] = '你'
    sent = ''.join(selected)
    return sent, idx

def build_dataset(vocabs):
    data = []
    for _ in range(SAMPE_NUM):
        data.append(create_sent(vocabs))
    return data

class SentDataSet(Dataset):
    def __init__(self, data, vocabs):
        # print(f'dataset vocabs len before:{len(vocabs)}')
        self.x = [encode(sent, vocabs, SENT_LEN) for sent, _ in data]
        self.y = [idx for _, idx in data]
        # print(f'dataset vocabs len before:{len(vocabs)}')
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.x[idx], dtype=torch.long),
            # 因为要用交叉熵的损失函数，所以y的类型必须是long，否则会导致计算交叉熵的时候会报错
            torch.tensor(self.y[idx], dtype=torch.long)
        )
    
    def __len__(self):
        return len(self.y)

# 3、创建字典及句子编码
def build_vocabs():
    # 之前没有加绝对路径，导致读到了运行python的路径下的文件
    with open('C:/code/pre/week03/vocabs.txt', 'r', encoding='utf-8') as f:
        vocabs = [line.strip() for line in f]
    vocab_dict = {v:i for i, v in enumerate(vocabs)}
    return vocab_dict

def encode(sent, vocabs, max_len):
    unk_idx = vocabs.get(UNK_WORD)
    ids = [vocabs.get(c, unk_idx) for c in sent]
    ids = ids[:max_len]
    ids += [0] * (max_len - len(ids))
    return ids

# 5、创建模型
class WordClassification(nn.Module):
    def __init__(self, vocab_size, embd_size, hidden_size, sent_len):
        super().__init__()
        # print(f'vocab_size:{vocab_size}')
        self.embd = nn.Embedding(vocab_size, embd_size, padding_idx=0)
        self.rnn = nn.RNN(embd_size, hidden_size, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(DROPOUT_RATIO)
        self.fc = nn.Linear(hidden_size, sent_len)

    def forward(self, x):
        # print(f'x:{x},x shape:{x.shape}')
        y = self.embd(x)
        y, _ = self.rnn(y)
        y = y.max(dim=1)[0]
        y = self.dropout(self.norm(y))
        y = self.fc(y)
        return y

class BagModel(nn.Module):
    def __init__(self, vocab_size, embd_size, sent_len):
        super().__init__()
        # print(f'vocab_size:{vocab_size}')
        self.embd = nn.Embedding(vocab_size, embd_size, padding_idx=0)
        self.norm = nn.LayerNorm(embd_size)
        self.dropout = nn.Dropout(DROPOUT_RATIO)
        self.fc = nn.Linear(embd_size, sent_len)

    def forward(self, x):
        y = self.embd(x)
        y = y.max(dim=1)[0]
        y = self.dropout(self.norm(y))
        y = self.fc(y)
        return y
    
# 6、训练及评估
def train(model, vocabs):
    data = build_dataset(vocabs)   
    split = int(SAMPE_NUM * TRAIN_RATIO)
    train_data = data[:split]
    eval_data = data[split:]   
    train_loader = DataLoader(SentDataSet(train_data, vocabs), BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(SentDataSet(eval_data, vocabs), BATCH_SIZE)   
    # print(eval_data)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), LR)

    for _ in range(EPOCH):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            y_pred = model(x)
            optimizer.zero_grad()
            # print(f'y_pred:{y_pred}, y:{y}')
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        acc = evaluate(model, eval_loader)
        print(f'平均损失值为：{avg_loss}，精确度为:{acc}')

def evaluate(model, eval_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in eval_loader:
            y_pred = model(x)
            y_class = y_pred.max(dim=1)[1]
            # print(f'y_class:{y_class}, y:{y}')
            correct += (y_class == y).sum().item()
    return correct / len(eval_loader.dataset)


if __name__ == '__main__':
    vocabs = build_vocabs()
    rnn_model = WordClassification(len(vocabs), EMBD_SIZE, HIDDEN_SIZE, SENT_LEN) 
    bag_model = BagModel(len(vocabs), EMBD_SIZE, SENT_LEN)
    train(rnn_model, vocabs)
    train(bag_model, vocabs)

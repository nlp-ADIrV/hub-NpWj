import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math

# 数据构建
# 数据处理
def load_corpus(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

# 和RNN之前的用例不一样，不用加入pad和unk
def build_vocabs(text):
    chars = sorted(set(text))
    char2idx = {c:i for i, c in enumerate(chars)}
    idx2char = {i:c for i, c in enumerate(chars)}
    return char2idx, idx2char

class SentDataSet(Dataset):
    def __init__(self, text, vocabs, seq_len):
        self.seq_len = seq_len
        ids = [vocabs[c] for c in text if c in vocabs]
        self.data = torch.tensor(ids, dtype=torch.long)
    
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y

# 模型
class MultiHeadAttention(nn.Module):
    def __init__(self, head_num, embd_size):
        super().__init__()
        self.head_num = head_num
        self.d_k = embd_size // head_num
        self.qkv = nn.Linear(embd_size, embd_size * 3)
        self.out = nn.Linear(embd_size, embd_size)
        self.mask = torch.zeros(embd_size, embd_size)
        for i in range(embd_size):
            for j in range(embd_size):
                if j <= i:
                    self.mask[i, j] = 1
    
    def forward(self, x, mask=None):
        # x(B, seq_len, embd_size)
        B, L, T = x.shape
        # (B, seq_len, embd_size * 3) -> 3 * (B, seq_len, embd_size)
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # 对qkv做多头切分 -> (B, head_num, seq_len, dk)
        # 错误写法：q.view(B, L, self.head_num, self.d_k).transpose(1, 2)
        q = q.view(B, L, self.head_num, self.d_k).transpose(1, 2)
        k = k.view(B, L, self.head_num, self.d_k).transpose(1, 2)
        v = v.view(B, L, self.head_num, self.d_k).transpose(1, 2)

        l = q @ k.transpose(-1, -2) / math.sqrt(self.d_k)
        if mask is not None:
            # 错误写法l.fill_mask(mask == 0, 1e-9)
            l = l * self.mask
            l = l.masked_fill(mask == 0, 1e-9)
        
        # 错误写法torch.softmax(l, dim=-1)
        l = torch.softmax(l, dim=-1)
        output = l @ v

        # 漏写了转化形状
        output = output.transpose(1, 2).contiguous().view(B, L, T)
        return self.out(output)

class EncodeLayer(nn.Module):
    def __init__(self, head_num, embd_size, ff):
        super().__init__()
        self.multi_head = MultiHeadAttention(head_num, embd_size)
        # 错误写法：写了一个layer norm，用了两次，但是会影响梯度训练
        self.layer_norm1 = nn.LayerNorm(embd_size)
        self.ffn = nn.Sequential(
            nn.Linear(embd_size, ff),
            nn.GELU(),
            nn.Linear(ff, embd_size)
        )
        self.layer_norm2 = nn.LayerNorm(embd_size)
    
    def forward(self, x, mask):
        y = self.layer_norm1(x + self.multi_head(x, mask))
        # 错误写法：self.layer_norm2(x + self.ffn(x))，此时的输入应该是上一步的输出y
        y = self.layer_norm2(y + self.ffn(y))
        return y
    
class TransformerEncoder(nn.Module):
    def __init__(self, head_num=12, embd_size=768, ff=3072, layer_num=12):
        super().__init__()
        # 不会使用ModuleList生成模型列表
        self.encoder = nn.ModuleList([EncodeLayer(head_num, embd_size, ff) for _ in range(layer_num)])
    
    def forward(self, x, mask=None):
        for encoder in self.encoder:
            x = encoder(x, mask)
        return x

class LLMModel(nn.Module):
    def __init__(self, vocabs_size, head_num, embd_size, ff, layer_num):
        super().__init__()
        self.embd = nn.Embedding(vocabs_size, embd_size)
        self.bert = TransformerEncoder(head_num, embd_size, ff, layer_num)
        # 这一步是一开始没有想到的，直到在计算loss的时候报错了
        self.fc = nn.Linear(embd_size, vocabs_size)
    
    def forward(self, x):
        # x[batch_size, seq_len]
        # embd[batch_size, seq_len, embd_size]
        embd = self.embd(x)
        # output[batch_size, seq_len, embd_size]
        output = self.bert(embd)
        # [batch_size, seq_len, vocabs_size]
        return self.fc(output)

# 训练/推理
def run_epoch(dataloader, model, optimizer, cretirion, train):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0
    for x, y in dataloader:
        y_pred = model(x)
        # print(f'y_pred shape:{y_pred.shape},y shape:{y.shape}')
        loss = cretirion(y_pred.reshape(-1, y_pred.size(-1)), y.reshape(-1))
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    ppl = -math.exp(avg_loss)
    return avg_loss, ppl

# 训练超参
EPOC = 10
LR = 0.001
EMBD_SIZE = 512
TRANSFORMER_LAYER = 6
HEAD_NUM = 8
FF = 2048
SEQ_LEN = 64
TRAIN_RATIO = 0.8
BATCH_SIZE = 64
PATH = "C:/code/pre/week05/corpus.txt"
MODEL_FILE = 'llm.pth'

def train(vocabs, corpus):
    lines = corpus.splitlines()
    print(f'语料库行数：{len(lines)}')
    
    llm = LLMModel(len(vocabs), HEAD_NUM, EMBD_SIZE, FF, TRANSFORMER_LAYER)
    total_params = sum([p.numel() for p in llm.parameters()])
    print(f'模型参数量：{total_params}')

    split = int(len(lines) * TRAIN_RATIO)
    train_data = '\n'.join(lines[:split])
    eval_data = '\n'.join(lines[split:])
    train_ds = SentDataSet(train_data, vocabs, SEQ_LEN)
    eval_ds = SentDataSet(eval_data, vocabs, SEQ_LEN)

    train_dataloader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, drop_last=True)
    eval_dataloader = DataLoader(eval_ds, BATCH_SIZE, shuffle=True, drop_last=True)

    optimizer = torch.optim.Adam(llm.parameters(), LR) 
    cretision = nn.CrossEntropyLoss()

    best_ppl = float('inf')
    print(f'{'Epoch':>5} {'Train loss':>12} {'Train PPL':>12} {'Eval loss':>12} {'Eval PPL':>12}')
    print('-' * 80)
    for i in range(EPOC):
        save = ""
        train_avg_loss, train_ppl = run_epoch(train_dataloader, llm, optimizer, cretision, train=True)
        with torch.no_grad():
            eval_avg_loss, eval_ppl = run_epoch(eval_dataloader, llm, optimizer, cretision, train=False)
        
        if eval_ppl < best_ppl:
            torch.save(llm.state_dict(), MODEL_FILE)
            save = "已保存模型"
        print(f'{i:>5} {train_avg_loss:>12.2f} {train_ppl:>12.2f} {eval_avg_loss:>12.2f} {eval_ppl:>12.2f} ({save})')

# 应用大模型
TOP_P = 0.9
TEMPERATURE = 1.0

def use_model(vocabs, ids2char):
    model = LLMModel(len(vocabs), HEAD_NUM, EMBD_SIZE, FF, TRANSFORMER_LAYER)
    model.load_state_dict(torch.load(MODEL_FILE))
    sent = "我想知道为什么"
    sent_ids = [vocabs[c] for c in sent]
    sent_torch = torch.tensor([sent_ids], dtype=torch.long)
    print(f'{sent}', end='')
    
    for _ in range(SEQ_LEN):
        out = model(sent_torch)
        out = torch.softmax(out[0,-1,:]/TEMPERATURE, dim=-1)
        # print(f'out :{out}, out shape:{out.shape}')
        last_choose = out.max(dim=-1)[1]
        sent_ids.append(last_choose.item())
        ch = ids2char[last_choose.item()]
        last_choose = last_choose.view(1, 1)
        sent_torch = torch.cat([sent_torch, last_choose], dim=-1)
        print(f'{ch}', end='')

if __name__ == '__main__':
    content = load_corpus(PATH)
    vocabs, ids2char =  build_vocabs(content)
    print(f'词表大小:{len(vocabs)}')
    train(vocabs, content)
    use_model(vocabs, ids2char)    

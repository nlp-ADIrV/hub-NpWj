# 训练基于 Transformer 的单向（自回归）字符级语言模型，并完成简单文本生成
# 关键点：单向模型在注意力计算中必须屏蔽未来信息（因果遮罩），防止信息泄露
import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader, TensorDataset

# 模型超参数
d_model = 64      # 嵌入/特征维度
n_head = 2        # 注意力头数
n_layers = 2      # Transformer层数
seq_len = 16      # 最大句子长度
batch_size = 2    # 批次大小
lr = 1e-3         # 学习率
epochs = 100      # 训练轮数
# 选择设备（优先 CUDA，否则使用 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 固定随机种子以便结果可复现（影响 torch、numpy、random）
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# 构建数据集 + 词汇表
# 训练语料
# 训练语料（字符级示例），可以根据需要替换为更大语料
corpus = (
    "hello world! this is a simple transformer language model. "
    "it can generate text based on the training data. "
    "the model learns to predict the next character in a sequence."
)
# 构建字符级词汇表
# 构建字符表：去重并排序以保证映射稳定
chars = sorted(list(set(corpus)))

# 建立字符到索引的映射（用于 embedding 输入）
char2idx = {ch: idx for idx, ch in enumerate(chars)}

# 建立索引到字符的映射（用于从模型输出索引恢复字符）
idx2char = {index: ch for ch, index in char2idx.items()}

# 词表大小
vocab_size = len(chars)
# 构建训练数据（输入序列和目标序列）
# 构造训练样本：对每个位置构造长度为 seq_len 的输入序列，目标为下一个时间步的字符序列
input_seqs = []
target_seqs = []
for i in range(len(corpus) - seq_len):
    # 取连续的 seq_len 字符作为输入
    input_seq = corpus[i : i + seq_len]
    # 目标为相同窗口但向右移动一个字符，用于监督下一步预测
    target_seq = corpus[i + 1 : i + seq_len + 1]
    # 将字符序列转为索引序列，便于 embedding 查表
    input_seqs.append([char2idx[ch] for ch in input_seq])
    target_seqs.append([char2idx[ch] for ch in target_seq])

# 转为 LongTensor 以供 PyTorch 模型训练使用
input_seqs = torch.LongTensor(input_seqs)
target_seqs = torch.LongTensor(target_seqs)

# 使用 TensorDataset 封装输入与目标，DataLoader 负责批次化与打乱
dataset = TensorDataset(input_seqs, target_seqs)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 数据预处理完成，构建模型
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 初始化位置编码矩阵 shape: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # 位置索引 0,1,...,max_len-1，形状 (max_len, 1)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 按照 Vaswani 等人的公式计算频率项：每两个维度使用相同的频率
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        # 偶数索引维度使用 sin，奇数索引维度使用 cos
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)

        # 在 batch 维度前增加一维，便于与输入相加
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)

        # 将位置编码注册为 buffer（不作为参数更新，但随模型保存）
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        # 将前 seq_len 个位置编码加到输入嵌入上返回
        return x + self.pe[:, : x.size(1)]
    
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layers):
        super().__init__()
        # 保存模型的隐藏维度
        self.d_model = d_model

        # 字符索引到向量的查表层
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 位置编码模块（固定，不随训练更新）
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer 的单层 encoder 配置（注意这里使用 encoder 作为基础构建块）
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head)

        # 多层 Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 输出线性层：将 Transformer 的输出映射回词表大小，用于预测每个位置的下一个字符分布
        self.fc_out = nn.Linear(d_model, vocab_size)

    def _generate_causal_mask(self, sz, device):
        """生成上三角遮罩，屏蔽当前位置后面的时间步（因果/自回归遮罩）。
        返回形状为 (sz, sz) 的张量，适用于 Transformer 的 `mask` 参数。
        """
        # 生成上三角矩阵（不含主对角线），将未来位置设置为 -inf 以屏蔽它们
        mask = torch.triu(torch.ones(sz, sz, device=device) * float("-inf"), diagonal=1)
        return mask

    def forward(self, src):
        # src: [batch_size, seq_len]
        # 将输入索引映射到连续向量空间，并进行尺度缩放（与论文实现一致，稳定训练）
        embedded = self.embedding(src) * math.sqrt(self.d_model)  # shape: [batch_size, seq_len, d_model]

        # 加上位置编码，使模型能区分不同时间步的位置
        embedded = self.pos_encoder(embedded)

        # PyTorch 的 Transformer 模块期望输入格式为 [seq_len, batch_size, d_model]
        embedded = embedded.permute(1, 0, 2)

        # 生成因果遮罩以屏蔽当前位置之后的时间步（保证模型不能看到未来信息）
        seq_len = embedded.size(0)
        src_mask = self._generate_causal_mask(seq_len, embedded.device)

        # 将遮罩传入 transformer encoder（遮罩形状为 [seq_len, seq_len]）
        transformer_out = self.transformer_encoder(embedded, mask=src_mask)  # shape: [seq_len, batch_size, d_model]

        # 恢复为 [batch_size, seq_len, d_model] 以便后续线性映射
        transformer_out = transformer_out.permute(1, 0, 2)

        # 将每个时间步的表示映射为对词表上每个字符的 logits
        output = self.fc_out(transformer_out)  # shape: [batch_size, seq_len, vocab_size]

        # 返回 logits，不经过 softmax，交叉熵损失函数内部会处理 softmax
        return output

# 实例化模型并移动到指定设备
model = TransformerLM(vocab_size, d_model, n_head, n_layers).to(device)

# 优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# 训练模式（启用 dropout 等训练特性，如果有的话）
model.train()

# 训练循环：按 epoch 遍历数据集
for epoch in range(epochs):
    total_loss = 0.0  # 累计当 epoch 的损失

    # 按批次遍历训练数据
    for batch_inputs, batch_targets in dataloader:
        # 将 batch 数据移动到计算设备（GPU/CPU）
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

        # 清零之前的梯度
        optimizer.zero_grad()

        # 前向计算：得到每个位置的 logits，shape: [batch_size, seq_len, vocab_size]
        outputs = model(batch_inputs)

        # 交叉熵期望的输入 shape 为 [N, C] 与目标 shape 为 [N]
        # 因此将 logits 和 targets 拉平到 (batch_size*seq_len, ...)
        loss = criterion(outputs.view(-1, vocab_size), batch_targets.view(-1))

        # 反向传播计算梯度
        loss.backward()

        # 参数更新一步
        optimizer.step()

        # 累计损失供日志使用
        total_loss += loss.item()

    # 计算并打印平均损失用于监控训练进展
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# 文本生成
# 评估/生成阶段：关闭梯度计算以节省内存
model.eval()
with torch.no_grad():
    # 以指定的起始字符开始生成文本
    start_char = "h"

    # 将起始字符转为索引并构造成 batch_size=1 的输入，shape: [1, 1]
    input_seq = torch.LongTensor([[char2idx[start_char]]]).to(device)

    # 存放生成的文本（从 start_char 开始）
    generated_text = start_char

    # 迭代生成多个字符
    for _ in range(100):  # 生成 100 个后续字符
        # 前向得到 logits，shape: [1, seq_len, vocab_size]
        output = model(input_seq)

        # 取最后一个时间步的 logits 并选择概率最高的索引作为输出字符
        next_char_idx = output.argmax(dim=-1)[0, -1].item()

        # 将索引映射回字符并追加到生成文本
        next_char = idx2char[next_char_idx]
        generated_text += next_char

        # 将新字符索引拼接到输入序列末尾，保持长度不超过 seq_len
        input_seq = torch.cat([input_seq, torch.LongTensor([[next_char_idx]]).to(device)], dim=1)[:, -seq_len:]

    # 打印生成结果
    print("Generated Text:\n", generated_text)

    

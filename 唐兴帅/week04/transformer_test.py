"""
面试版 Transformer Encoder
核心：Multi-Head Self-Attention / FFN / 残差 + LN / 堆叠
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 定义多头自注意力机制模块
class MultiHeadAttention(nn.Module):
    # 初始化函数：hidden为隐藏层维度，n_head为注意力头的数量
    def __init__(self, hidden, n_head):
        super().__init__()
        # 确保隐藏层维度能被头数整除，保证每个头能均匀分配维度
        assert hidden % n_head == 0
        self.n_head = n_head
        # 计算每个注意力头的维度（例如 hidden=512, n_head=8, 则 d_k=64）
        self.d_k = hidden // n_head
        # 定义一个大的线性层，一次性将输入映射为 Q, K, V 三个矩阵的拼接（维度扩大3倍）
        self.qkv = nn.Linear(hidden, hidden * 3)   
        # 定义多头注意力计算后的输出线性层，用于特征融合
        self.out = nn.Linear(hidden, hidden)

    def forward(self, x, mask=None):
        # 获取输入张量的形状：B=批次大小, T=序列长度, H=隐藏层维度
        B, T, H = x.shape
        # 将 QKV 大矩阵在最后一个维度切分成均等的3份，分别作为 Q, K, V
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # 将 Q 变形为 [B, T, n_head, d_k]，并交换维度变为 [B, n_head, T, d_k] 以便并行计算
        q = q.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        # 将 K 变形为 [B, n_head, T, d_k]
        k = k.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        # 将 V 变形为 [B, n_head, T, d_k]
        v = v.view(B, T, self.n_head, self.d_k).transpose(1, 2)

        # 计算缩放点积注意力分数：Q 乘以 K 的转置，再除以根号d_k防止梯度消失
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        # 如果传入了掩码（如处理Padding或防止信息泄露），将对应位置的分数替换为极大的负数
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        # 在最后一个维度进行 Softmax 归一化，将分数转化为概率分布（注意力权重）
        attn = F.softmax(scores, dim=-1)

        # 用注意力权重对 V 进行加权求和，得到注意力输出 [B, n_head, T, d_k]
        out = attn @ v                              
        # 交换维度回 [B, T, n_head, d_k]，并将多头结果拼接（view）回原始的隐藏层维度 H
        out = out.transpose(1, 2).contiguous().view(B, T, H)
        # 经过最后的线性层进行特征融合并返回
        return self.out(out)

# 定义单个 Transformer 编码器层
class EncoderLayer(nn.Module):
    # 初始化函数：hidden为隐藏层维度，n_head为头数，ff为前馈网络中间层的扩展维度
    def __init__(self, hidden, n_head, ff):
        super().__init__()
        # 实例化多头自注意力模块
        self.attn = MultiHeadAttention(hidden, n_head)
        # 实例化第一个层归一化（LayerNorm）模块
        self.ln1 = nn.LayerNorm(hidden)
        # 实例化前馈神经网络（FFN）：升维 -> GELU激活 -> 降维
        self.ffn = nn.Sequential(
            nn.Linear(hidden, ff),
            nn.GELU(),
            nn.Linear(ff, hidden),
        )
        # 实例化第二个层归一化（LayerNorm）模块
        self.ln2 = nn.LayerNorm(hidden)

    def forward(self, x, mask=None):
        # 采用 Pre-LN 架构：先对输入做 LN，再进入注意力计算，最后加上原始输入 x 实现残差连接
        x = self.ln1(x + self.attn(x, mask))        
        # 同样采用 Pre-LN 架构：先对输入做 LN，再进入前馈网络，最后加上原始输入 x 实现残差连接
        x = self.ln2(x + self.ffn(x))
        return x

# 定义完整的 Transformer 编码器（由多个 EncoderLayer 堆叠而成）
class TransformerEncoder(nn.Module):
    # 初始化函数：hidden=768, n_layer=12, n_head=12, ff=3072 是类似 BERT-Base 的经典配置
    def __init__(self, hidden=768, n_layer=12, n_head=12, ff=3072):
        super().__init__()
        # 使用 ModuleList 循环创建并堆叠 n_layer 个 EncoderLayer
        self.layers = nn.ModuleList([EncoderLayer(hidden, n_head, ff) for _ in range(n_layer)])

    def forward(self, x, mask=None):
        # 遍历每一个编码器层，将输入 x 依次传入
        for layer in self.layers:
            x = layer(x, mask)
        # 返回经过所有层深度提取特征后的最终输出
        return x

# 主程序入口，用于测试模型的前向传播和维度变化
if __name__ == "__main__":
    # 实例化模型：隐藏层512维，堆叠6层，8个注意力头，前馈网络扩展维度1024
    model = TransformerEncoder(hidden=512, n_layer=6, n_head=8, ff=1024)
    # 随机生成一个模拟输入张量：批次大小为2，序列长度为16，特征维度为512
    x = torch.randn(2, 16, 512)        
    # 将输入传入模型，并打印输出的形状（预期输出维度应与输入保持一致）
    print(model(x).shape)              

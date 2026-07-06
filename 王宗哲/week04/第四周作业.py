# 本周作业：PyTorch实现Transformer层 
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# 1. 实现：多头注意力 (MultiHeadAttention)
# ---------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim          # 词向量维度 例如 512
        self.num_heads = num_heads          # 头数 例如 8
        self.head_dim = embed_dim // num_heads  # 每个头的维度

        # 三个线性层：生成 Q K V
        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)

        # 输出线性层
        self.w_o = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, k, v):
        B = q.shape[0]  # batch大小

        # 1. 线性变换得到 Q K V
        Q = self.w_q(q)
        K = self.w_k(k)
        V = self.w_v(v)

        # 2. 拆分成多头：[B, S, E] → [B, 头数, S, 每头维度]
        Q = Q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. 计算注意力分数：Q * K^T / sqrt(head_dim)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attn_weights = F.softmax(attn_scores, dim=-1)  # 转成概率

        # 4. 注意力加权 V
        out = torch.matmul(attn_weights, V)

        # 5. 拼接多头
        out = out.transpose(1, 2).contiguous().view(B, -1, self.embed_dim)

        # 6. 最后线性层
        out = self.w_o(out)
        return out


# ---------------------------
# 2. 实现：Transformer Encoder 层
# ---------------------------
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):
        super().__init__()

        # 多头注意力
        self.attn = MultiHeadAttention(embed_dim, num_heads)

        # 前馈网络 FFN
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)

        # 归一化层
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # -------------------
        # 第一步：多头注意力 + 残差 + 归一化
        # -------------------
        attn_out = self.attn(x, x, x)  # self-attention：Q=K=V
        x = self.norm1(x + self.dropout(attn_out))

        # -------------------
        # 第二步：前馈网络 + 残差 + 归一化
        # -------------------
        ff_out = self.linear2(F.relu(self.linear1(x)))
        x = self.norm2(x + self.dropout(ff_out))

        return x


# ---------------------------
# 3. 随机数据测试
# ---------------------------
if __name__ == "__main__":
    # 超参数
    BATCH_SIZE = 2
    SEQ_LEN = 10     # 句子长度
    EMBED_DIM = 512  # 词向量维度
    NUM_HEADS = 8    # 多头注意力头数
    HIDDEN_DIM = 1024

    # 生成随机输入张量（模拟文本embedding）
    x = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)

    # 初始化Transformer层
    encoder_layer = TransformerEncoderLayer(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        hidden_dim=HIDDEN_DIM
    )

    # 前向传播
    out = encoder_layer(x)

    # 打印形状
    print("输入形状:", x.shape)
    print("输出形状:", out.shape)
    print("Transformer 层实现 & 随机数据测试 运行成功！")


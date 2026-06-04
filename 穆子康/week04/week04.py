import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim          # 向量长度
        self.num_heads = num_heads          # 区分出几个头，就是矩阵切分成几个小矩阵？
        self.head_dim = embed_dim // num_heads # 每个头的长度，需要整除，不能是小数

        # 3行线性层，算出 Q、K、V
        self.w_q = nn.Linear(embed_dim, embed_dim) #输入长度 = embed_dim（词向量长度）输出长度 = embed_dim（词向量长度）
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)

        # 输出层
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.shape  # 批次、序列长度、维度

        # 1. 算出 Q K V
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        # 2. 拆成多头
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. 注意力公式：softmax(Q*K^T / √d) * V
        attn = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)

        # 4. 加权求和
        out = attn @ V

        # 5. 拼接多头
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out(out)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # 注意力 + 残差 + 归一化
        x = self.norm1(x + self.attn(x))

        # 前馈 + 残差 + 归一化
        x = self.norm2(x + self.ffn(x))

        return x

# ------------------- 测试 -------------------
if __name__ == "__main__":
    # 输入：2句话，每句10个词，每个词64维
    x = torch.randn(2, 10, 64)

    # 建一层 Transformer
    transformer = TransformerBlock(embed_dim=64, num_heads=2)

    # 计算
    out = transformer(x)

    print("输入形状:", x.shape)
    print("输出形状:", out.shape)

import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# 1. Multi-Head Attention
# =========================
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # 每个头的维度
        self.head_dim = embed_dim // num_heads

        # Q K V 线性层
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)

        # 输出层
        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape

        # 生成 Q K V
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)

        # shape:
        # [batch, seq_len, embed_dim]
        # -> [batch, seq_len, num_heads, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # 调整维度
        # -> [batch, num_heads, seq_len, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Attention 分数
        scores = torch.matmul(Q, K.transpose(-2, -1))

        # 缩放
        scores = scores / (self.head_dim ** 0.5)

        # softmax
        attention = F.softmax(scores, dim=-1)

        # Attention 输出
        out = torch.matmul(attention, V)

        # 拼接多个头
        out = out.transpose(1, 2)

        # -> [batch, seq_len, embed_dim]
        out = out.contiguous().view(batch_size, seq_len, embed_dim)

        # 最终输出
        out = self.out_linear(out)

        return out


# =========================
# 2. Feed Forward
# =========================
class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(FeedForward, self).__init__()

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x


# =========================
# 3. Transformer Encoder Layer
# =========================
class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super(TransformerLayer, self).__init__()

        # Multi-head Attention
        self.attention = MultiHeadAttention(embed_dim, num_heads)

        # LayerNorm
        self.norm1 = nn.LayerNorm(embed_dim)

        # FeedForward
        self.ffn = FeedForward(embed_dim, hidden_dim)

        # LayerNorm
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):

        # Attention
        attn_out = self.attention(x)

        # 残差连接 + LayerNorm
        x = self.norm1(x + attn_out)

        # FeedForward
        ffn_out = self.ffn(x)

        # 残差连接 + LayerNorm
        x = self.norm2(x + ffn_out)

        return x


# =========================
# 4. 测试代码
# =========================
if __name__ == "__main__":

    # 参数
    batch_size = 2
    seq_len = 5
    embed_dim = 16
    num_heads = 4
    hidden_dim = 64

    # 随机输入
    x = torch.randn(batch_size, seq_len, embed_dim)

    # Transformer Layer
    model = TransformerLayer(
        embed_dim,
        num_heads,
        hidden_dim
    )

    # 前向传播
    output = model(x)

    print("输入 shape:", x.shape)
    print("输出 shape:", output.shape)

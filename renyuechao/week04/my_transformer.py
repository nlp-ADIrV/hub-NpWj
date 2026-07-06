import torch
import torch.nn as nn
import math

class MySelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MySelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        # dim of head
        self.head_dim = hidden_size // num_heads

        #Q,K,V linear layer
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)

        # output
        self.out_linear = nn.Linear(hidden_size, hidden_size)


    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # 经过线性层得到 Q, K, V
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)

        # Multi-Head
        # view() 用来改变形状，transpose(1, 2) 是把 num_heads 换到前面去方便计算
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        # K.transpose(-1, -2) 是把最后两个维度反转，为了能乘起来
        # 公式: score = (Q @ K^T) / sqrt(d_k)
        scores = Q @ K.transpose(-1, -2) / math.sqrt(self.head_dim)

        # 把分数变成概率 (Softmax)
        # softmax 会让每一行的分数加起来等于 1
        attn_weights = torch.softmax(scores, dim=-1)

        # 用概率给 V 乘上权重 (加权平均)
        # 公式: output = softmax(score) @ V
        attn_output = attn_weights @ V

        # 把多个头拼接回原来的形状
        # 形状变回 [batch_size, seq_len, hidden_size]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        # 最后经过一个线性层输出
        output = self.out_linear(attn_output)

        return output


class MyFeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(MyFeedForward, self).__init__()
        # 两层全连接，中间通常用 GELU 或 ReLU 激活函数
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x

class MyTransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, dropout=0.1):
        super(MyTransformerEncoderLayer, self).__init__()
        # 自注意力机制
        self.attention = MySelfAttention(hidden_size, num_heads)
        # 层归一化
        self.norm1 = nn.LayerNorm(hidden_size)
        # 前馈神经网络
        self.fnn = MyFeedForward(hidden_size, intermediate_size)
        # 层归一化
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # Attention 部分 + 残差连接 + 归一化
        attn_out = self.attention(x)
        # 残差连接: x + attn_out
        x = self.norm1(x + attn_out)

        # FFN 部分 + 残差连接 + 归一化
        fnn_out = self.fnn(x)
        # 残差连接: x + ffn_out
        x = self.norm2(x + fnn_out)

        return x



if __name__ == '__main__':
    # 模拟输入: [batch_size=2, seq_len=5, hidden_size=768]
    dummy_input = torch.randn(2, 5, 768)

    # 实例化一个完整的 Transformer 层 (参数参考 BERT-base)
    layer = MyTransformerEncoderLayer(hidden_size=768, num_heads=12, intermediate_size=3072)
    # 运行
    output = layer(dummy_input)
    print("Transformer 层输入:", dummy_input.shape)
    print("Transformer 层输出:", output.shape)
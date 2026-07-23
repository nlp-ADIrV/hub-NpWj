"""
使用python实现transformer模型 — 单层
包含：多头注意力、前馈网络、单层Transformer层
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─── 1. 缩放点积注意力 ──────────────────────────────────────
def scaled_dot_product_attention(q, k, v, mask=None):
    """
    q: [batch_size, n_heads, len_q, d_k]
    k: [batch_size, n_heads, len_k, d_k]
    v: [batch_size, n_heads, len_v, d_v]
    mask: [batch_size, 1, len_q, len_k] 或广播兼容形状
    """
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn = F.softmax(scores, dim=-1)
    context = torch.matmul(attn, v)
    return context, attn


# ─── 2. 多头注意力 ─────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.depth = d_model // n_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.n_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, v, k, q, mask=None):
        batch_size = q.size(0)
        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)

        context, attn_weights = scaled_dot_product_attention(q, k, v, mask)
        context = context.permute(0, 2, 1, 3).reshape(batch_size, -1, self.d_model)
        output = self.dense(context)
        return output, attn_weights


# ─── 3. 前馈网络 ───────────────────────────────────────────
class FeedForward(nn.Module):
    def __init__(self, d_model, dff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dff)
        self.linear2 = nn.Linear(dff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


# ─── 4. 单层Transformer（编码器层） ─────────────────────────
class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, dff, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, dff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.mha(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x


# ─── 5. 使用示例 ─────────────────────────────────────────
if __name__ == "__main__":
    batch_size, seq_len = 2, 10
    d_model, n_heads, dff = 64, 8, 256

    x = torch.randn(batch_size, seq_len, d_model)
    layer = TransformerLayer(d_model, n_heads, dff)
    out = layer(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    print(f"参数量: {sum(p.numel() for p in layer.parameters()):,}")

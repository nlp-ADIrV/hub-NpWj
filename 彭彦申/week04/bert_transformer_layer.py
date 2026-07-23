import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BertSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, L, D = x.shape

        Q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = attn_weights @ V
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)
        return out


class BertFeedForward(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, intermediate_dim)
        self.linear2 = nn.Linear(intermediate_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class BertTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, intermediate_dim=None, dropout=0.1):
        super().__init__()
        if intermediate_dim is None:
            intermediate_dim = hidden_dim * 4

        self.self_attn = BertSelfAttention(hidden_dim, num_heads, dropout)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)

        self.ffn = BertFeedForward(hidden_dim, intermediate_dim, dropout)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        residual = x
        x = self.self_attn(x, mask)
        x = self.attn_dropout(x)
        x = self.attn_norm(residual + x)

        residual = x
        x = self.ffn(x)
        x = self.ffn_dropout(x)
        x = self.ffn_norm(residual + x)
        return x


if __name__ == '__main__':
    hidden_dim = 768
    num_heads = 12
    batch_size = 2
    seq_len = 16

    layer = BertTransformerLayer(hidden_dim, num_heads)
    x = torch.randn(batch_size, seq_len, hidden_dim)
    out = layer(x)

    total_params = sum(p.numel() for p in layer.parameters())
    print(f"BertTransformerLayer params: {total_params:,}")
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")

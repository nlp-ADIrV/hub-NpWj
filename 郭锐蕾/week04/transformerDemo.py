"""
PyTorch 实现单层 Transformer（多头自注意力 + 位置前馈网络 + 残差 + LayerNorm）。
采用 Norm-First（先 LayerNorm 再子层）结构，训练更稳定，与许多现代实现一致。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TransformerLayer(nn.Module):
    """
    单层 Transformer 编码器风格块。

    参数
    ----
    d_model : 模型维度（与词向量维度一致）
    n_heads : 多头注意力头数（d_model 需能被 n_heads 整除）
    d_ff : 前馈网络隐藏层维度（通常取 4 * d_model）
    dropout : Dropout 比例
    causal : True 时使用因果掩码，禁止位置 attend 未来 token（适合自回归语言模型）
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        causal: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.causal = causal

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """上三角为 True 表示禁止关注（与 MultiheadAttention 的 attn_mask 语义一致）。"""
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        attn_mask: 可选，(seq_len, seq_len) 或 (batch*heads, seq, seq)，bool/float
        key_padding_mask: 可选，(batch, seq_len)，True 表示该位置为 padding 需忽略
        返回: 与 x 同形状
        """
        if self.causal and attn_mask is None:
            attn_mask = self._causal_mask(x.size(1), x.device, torch.bool)

        h = self.norm1(x)
        attn_out, _ = self.attn(
            h, h, h,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout1(attn_out)

        h = self.norm2(x)
        x = x + self.ff(h)
        return x


if __name__ == "__main__":
    B, L, D, H = 2, 16, 64, 4
    layer = TransformerLayer(d_model=D, n_heads=H, d_ff=D * 4, dropout=0.1, causal=True)
    inp = torch.randn(B, L, D)
    out = layer(inp)
    assert out.shape == inp.shape
    print("TransformerLayer 前向形状检查通过:", out.shape)

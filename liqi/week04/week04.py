import torch
import torch.nn as nn

'''

尝试用pytorch实现一个transformer层

'''

class TransformerLayer(nn.Module):
    def __init__(self, d_model=768, nhead=12):
        super().__init__()
        # 多头注意力
        self.self_attn = nn.MultiheadAttention(d_model, nhead)

        # 归一化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 前馈网络
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        
    def forward(self, src):
        # src: (S, N, E)  S是序列长度，N是batch_size，E是embedding维度
        # 对应 z = LLayerNorm(x + MultiHead(x, x, x))
        attn_output, _ = self.self_attn(src, src, src)
        src = src + attn_output
        src = self.norm1(src)
        src = nn.GELU()(self.linear1(src))
        src = self.linear2(src)
        # 对应 z = LLayerNorm(x + FFN(x))
        src = self.norm2(src)
        return src


if __name__ == '__main__':
    transformer_model = TransformerLayer(d_model=768, nhead=12)
    src = torch.rand((10, 32, 768))  # (S, N, E)  S是序列长度，N是batch_size，E是embedding维度

    out = transformer_model(src)
    print(out.shape)  # (S, N, E)

# 本周作业：PyTorch实现Transformer层
import torch
import torch.nn as nn

# 1. 定义标准Transformer层
class TransformerBlock(nn.Module):
    def __init__(self, dim=768, num_heads=12, hidden_dim=3072):
        super().__init__()
        # 官方多头注意力
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        # 层归一化
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        # 第一步：多头注意力 + 残差连接 + 层归一化
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)

        # 第二步：前馈网络 + 残差连接 + 层归一化
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


# 2. 测试代码
if __name__ == '__main__':
    # 模拟输入：2个批次，每句10个词，词向量768维
    x = torch.randn(2, 10, 768)
    # 初始化Transformer层
    block = TransformerBlock()
    # 前向传播
    out = block(x)
    print("输入形状:", x.shape)
    print("输出形状:", out.shape)
    print("✅ Transformer层实现成功！")

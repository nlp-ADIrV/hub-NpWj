# 用pytorch实现一个transformer层
import torch
import torch.nn as nn
class Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super(Transformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
    def forward(self, src):
        output = self.transformer_encoder(src)
        return output
# 测试代码
if __name__ == "__main__":    
    d_model = 512
    nhead = 8
    num_layers = 6
    transformer = Transformer(d_model, nhead, num_layers)
    src = torch.rand(32, 10, d_model)  # (batch_size, sequence_length, d_model)
    output = transformer(src)
    print(output.shape)  # (batch_size, sequence_length, d_model)



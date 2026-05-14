import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleTransformerLayer(nn.Module):
    
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.d_model = d_model
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        

        
        Q = self.W_q(x)  
        K = self.W_k(x)
        V = self.W_v(x)

    
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_model ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)  
        attn_out = torch.matmul(attn_weights, V) 

    
        ff_out = self.fc2(F.relu(self.fc1(attn_out)))

        return ff_out


batch, seq_len, d_model, d_ff = 2, 5, 8, 32
x = torch.randn(batch, seq_len, d_model)
layer = SimpleTransformerLayer(d_model, d_ff)
out = layer(x)
print("输入形状:", x.shape)  
print("输出形状:", out.shape) 

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 1. 缩放点积注意力
# ==========================================
def scaled_dot_product_attention(Q, K, V, attn_mask=None):
    """
    Args:
        Q: (batch_size, num_heads, seq_len, d_k)
        K: (batch_size, num_heads, seq_len, d_k)
        V: (batch_size, num_heads, seq_len, d_v)
        attn_mask: (batch_size, 1, 1, seq_len) 填充掩码，1为有效词，0为Padding
    """
    d_k = Q.size(-1)
    # 计算注意力分数
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # BERT只需要Padding Mask，不需要Causal Mask
    if attn_mask is not None:
        # 将Padding位(0)的分数设为极小值，softmax后趋近于0
        scores = scores.masked_fill(attn_mask == 0, -1e9)
        
    attn_weights = F.softmax(scores, dim=-1)
    
    # 可选：Dropout (BERT原论文在Attention权重上加了Dropout)
    # attn_weights = F.dropout(attn_weights, p=0.1, training=self.training)
    
    output = torch.matmul(attn_weights, V)
    return output, attn_weights

# ==========================================
# 2. 多头自注意力机制
# ==========================================
class BertMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(BertMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.size(0)
        
        # 1. 线性映射
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # 2. 拆分为多头: (B, S, D) -> (B, S, H, d_k) -> (B, H, S, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 3. 注意力计算
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, attn_mask)
        
        # 4. 合并多头: (B, H, S, d_k) -> (B, S, H, d_k) -> (B, S, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 5. 输出映射
        output = self.W_o(attn_output)
        return output, attn_weights

# ==========================================
# 3. BERT 前馈神经网络 (使用 GELU 激活函数)
# ==========================================
class BertFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(BertFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        # BERT的标配是GELU
        self.activation = F.gelu 
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# ==========================================
# 4. BERT Encoder Layer (核心！)
# ==========================================
class BertEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(BertEncoderLayer, self).__init__()
        self.self_attn = BertMultiHeadAttention(d_model, num_heads)
        self.ffn = BertFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, hidden_states, attn_mask=None):
        # 1. Multi-Head Self-Attention + Add + Norm
        attn_output, attn_weights = self.self_attn(
            hidden_states, hidden_states, hidden_states, attn_mask
        )
        attn_output = self.dropout1(attn_output)
        # Post-LN: 先做残差连接，再做LayerNorm
        hidden_states = self.norm1(hidden_states + attn_output)
        
        # 2. Feed Forward + Add + Norm
        ffn_output = self.ffn(hidden_states)
        ffn_output = self.dropout2(ffn_output)
        hidden_states = self.norm2(hidden_states + ffn_output)
        
        return hidden_states, attn_weights

# ==========================================
# 5. 完整 BERT 模型外壳 
# ==========================================
class BertModel(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_heads=12, d_ff=3072, num_layers=12, dropout=0.1):
        super(BertModel, self).__init__()
        # 词嵌入 + 位置嵌入 + 段落嵌入
        self.word_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(512, d_model) # 假设最大长度512
        self.token_type_embeddings = nn.Embedding(2, d_model) # 0: Sentence A, 1: Sentence B
        
        self.embedding_layer_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # 堆叠多个 BertEncoderLayer
        self.encoder_layers = nn.ModuleList([
            BertEncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        batch_size, seq_len = input_ids.size()
        
        # 1. 获取位置id和段落id
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
            
        # 2. 构建嵌入层
        word_emb = self.word_embeddings(input_ids)
        pos_emb = self.position_embeddings(position_ids)
        seg_emb = self.token_type_embeddings(token_type_ids)
        
        embeddings = word_emb + pos_emb + seg_emb
        embeddings = self.embedding_layer_norm(embeddings)
        embeddings = self.embedding_dropout(embeddings)
        
        # 3. 处理 Attention Mask
        # 输入的 attention_mask 形状: (B, S)，1代表有效，0代表Padding
        # 需要扩展维度以适应多头注意力: (B, 1, 1, S)
        if attention_mask is not None:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        else:
            extended_attention_mask = None
            
        # 4. 逐层过 Encoder
        hidden_states = embeddings
        all_attentions = []
        for layer in self.encoder_layers:
            hidden_states, attn_weights = layer(hidden_states, extended_attention_mask)
            all_attentions.append(attn_weights)
            
        return hidden_states, all_attentions

# ==========================================
# 测试运行代码
# ==========================================
if __name__ == "__main__":
    # 模拟 BERT-Base 的超参数
    vocab_size = 30000
    batch_size = 2
    seq_len = 10
    d_model = 768
    num_heads = 12
    d_ff = 3072
    
    model = BertModel(vocab_size, d_model, num_heads, d_ff, num_layers=1)
    
    # 模拟输入: 2句话，每句10个词
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    token_type_ids = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1], 
                                   [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 0], # 最后一个是Padding
                                   [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.long) # 最后三个是Padding
    
    # 前向传播
    with torch.no_grad():
        last_hidden_state, all_attentions = model(input_ids, token_type_ids, attention_mask)
        
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Attention Mask shape: {attention_mask.shape}")
    print(f"Output Hidden State shape: {last_hidden_state.shape}") # 期待: (2, 10, 768)

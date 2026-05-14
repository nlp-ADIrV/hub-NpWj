# 第四周作业：使用 PyTorch 实现一个 Transformer 层

本周作业尝试使用 PyTorch 从基础模块实现一个 Transformer Encoder Layer。

实现内容包括：

- 多头自注意力机制（Multi-Head Self-Attention）
- Scaled Dot-Product Attention
- 残差连接（Residual Connection）
- Layer Normalization
- Position-wise Feed Forward Network
- Dropout
- Padding Mask

代码没有直接调用 `nn.TransformerEncoderLayer`，而是用 `nn.Linear`、`nn.LayerNorm`、`nn.Dropout` 等基础组件手动组合出一个 Transformer 层。

---

## 目录结构

```text
week04/
├── transformer_layer.py
└── README.md
```

---

## 环境依赖

推荐 Python 3.9+。

安装 PyTorch：

```bash
pip install torch
```

如果已经配置好 PyTorch 环境，可以直接运行代码。

---

## 运行方式

在 `week04` 目录下运行：

```bash
python transformer_layer.py
```

或者在项目根目录运行：

```bash
python week04/transformer_layer.py
```

---

## 输出示例

```text
Input shape: torch.Size([2, 5, 64])
Output shape: torch.Size([2, 5, 64])
Attention weights shape: torch.Size([2, 4, 5, 5])
First token output preview: tensor([...], grad_fn=<SliceBackward0>)
```

其中：

- `2` 表示 batch size
- `5` 表示序列长度
- `64` 表示每个 token 的向量维度
- `4` 表示 attention head 的数量

输出张量形状仍然是 `[batch_size, seq_len, d_model]`，说明该层可以继续堆叠成多层 Transformer Encoder。

---

## 核心结构说明

### 1. 多头自注意力

输入 `x` 的形状为：

```text
[batch_size, seq_len, d_model]
```

先经过三个线性层得到：

```text
Q = Linear(x)
K = Linear(x)
V = Linear(x)
```

然后将 `d_model` 拆成多个 head：

```text
[batch_size, seq_len, d_model]
-> [batch_size, num_heads, seq_len, head_dim]
```

注意力分数计算公式：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(head_dim))V
```

### 2. 残差连接和 LayerNorm

Transformer 层中包含两次残差连接：

```text
x = LayerNorm(x + SelfAttention(x))
x = LayerNorm(x + FeedForward(x))
```

这样可以缓解深层网络训练中的梯度消失问题，也能让模型保留原始输入信息。

### 3. 前馈网络

每个位置上的 token 都会独立经过同一个两层 MLP：

```text
Linear(d_model, d_ff)
ReLU
Dropout
Linear(d_ff, d_model)
```

它不会改变序列长度和最终特征维度。

---

## 文件说明

`transformer_layer.py` 中主要包含三个类：

- `MultiHeadSelfAttention`：实现多头自注意力
- `PositionWiseFeedForward`：实现前馈网络
- `TransformerLayer`：组合注意力、前馈网络、残差连接和归一化

`demo()` 函数构造了一个随机输入，并使用 padding mask 模拟第二条样本后两个位置是无效 token 的情况。

---

## 学习总结

通过本次实现可以看到，一个 Transformer 层的核心不是单个复杂函数，而是由几个比较清晰的模块组合而成：

1. 用 Q、K、V 计算 token 之间的相关性。
2. 使用多头机制从不同子空间学习关系。
3. 使用残差连接保留原始信息。
4. 使用 LayerNorm 稳定训练。
5. 使用前馈网络增强每个 token 的非线性表达能力。

后续如果要实现完整 Transformer，还需要继续加入词向量 Embedding、位置编码、多层堆叠，以及根据任务选择分类头或生成头。

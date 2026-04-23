import torch
import torch.nn as nn
import numpy as np
'''
手动实现交叉熵的计算
'''
criterion = nn.CrossEntropyLoss()

logits =  torch.FloatTensor([[-0.5, 0.2, 0.3], [-0.1, -0.2, 0.3]])  # 模拟模型输出的原始分数logits

labels = torch.LongTensor([0, 2])  # 真实标签（类别索引）

loss = criterion(logits, labels)

print(logits)
print(labels)
print(loss)
import torch
import torch.nn as nn


# 模型1：基础版本 BERT+Linear（不使用CRF）
class BertNER(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.embedding = nn.Embedding(21128, 768)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        embed_out = self.embedding(input_ids)
        drop_out = self.dropout(embed_out)
        logits = self.classifier(drop_out)

        # 训练阶段计算损失
        if labels is not None:
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(logits.view(-1, logits.size(-1)), labels.view(-1))
            return loss, logits
        # 推理阶段返回预测标签
        return torch.argmax(logits, dim=-1)


# 模型2：优化版本 BERT+CRF（本项目简化实现，用于对比）
class BertCRFNER(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.embedding = nn.Embedding(21128, 768)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        embed_out = self.embedding(input_ids)
        drop_out = self.dropout(embed_out)
        logits = self.classifier(drop_out)

        if labels is not None:
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(logits.view(-1, logits.size(-1)), labels.view(-1))
            return loss, logits
        return torch.argmax(logits, dim=-1)

"""BERT 序列标注模型：Linear baseline 与 CRF 版本。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class BertForTokenClassification(nn.Module):
    """BERT + Linear，每个 token 独立预测 BIO 标签。"""

    def __init__(self, bert_path: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        logits = self.classifier(self.dropout(outputs.last_hidden_state))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.num_labels),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss


class BertCRFForTokenClassification(nn.Module):
    """BERT + CRF，用 Viterbi 解码整条 BIO 序列。"""

    def __init__(self, bert_path: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        from torchcrf import CRF

        self.bert = BertModel.from_pretrained(bert_path)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.num_labels = num_labels

    def _emissions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        return self.classifier(self.dropout(outputs.last_hidden_state))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        emissions = self._emissions(input_ids, attention_mask, token_type_ids)
        mask = attention_mask.bool()

        loss = None
        if labels is not None:
            labels_for_crf = labels.clone()
            labels_for_crf[labels_for_crf == -100] = 0
            loss = -self.crf(emissions, labels_for_crf, mask=mask, reduction="mean")
        return emissions, loss

    def decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> list[list[int]]:
        emissions = self._emissions(input_ids, attention_mask, token_type_ids)
        return self.crf.decode(emissions, mask=attention_mask.bool())


def build_model(
    model_type: str,
    bert_path: str,
    num_labels: int,
    dropout: float = 0.1,
) -> nn.Module:
    """按 model_type 创建模型。"""
    if model_type == "linear":
        model = BertForTokenClassification(bert_path, num_labels, dropout)
    elif model_type == "crf":
        model = BertCRFForTokenClassification(bert_path, num_labels, dropout)
    else:
        raise ValueError(f"未知模型类型: {model_type}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型类型：BERT + {model_type.upper()}")
    print(f"标签数量：{num_labels}")
    print(f"参数量：{total_params / 1e6:.1f}M")
    return model

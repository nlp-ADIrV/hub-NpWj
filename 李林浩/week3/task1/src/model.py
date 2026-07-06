from typing import Literal

import torch
import torch.nn as nn


class RNNTextClassifier(nn.Module):
    """
    支持 rnn / lstm / gru 三类循环神经网络的文本分类模型。

    输入:
        input_ids: [batch_size, seq_len]
        lengths:   [batch_size]

    输出:
        logits: [batch_size, num_classes]
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        model_type: Literal["rnn", "lstm", "gru"] = "lstm",
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.2,
        pad_idx: int = 0,
    ):
        super().__init__()

        model_type = model_type.lower()
        if model_type not in {"rnn", "lstm", "gru"}:
            raise ValueError("model_type 必须是 rnn、lstm 或 gru。")

        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,
        )

        rnn_cls = {
            "rnn": nn.RNN,
            "lstm": nn.LSTM,
            "gru": nn.GRU,
        }[model_type]

        self.encoder = rnn_cls(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        direction_factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * direction_factor, num_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)

        # lengths 放 CPU，避免部分环境 pack 报错
        lengths_cpu = lengths.detach().cpu()

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths_cpu,
            batch_first=True,
            enforce_sorted=False,
        )

        packed_output, hidden = self.encoder(packed)

        if self.model_type == "lstm":
            hidden = hidden[0]  # LSTM 返回 (h_n, c_n)

        # hidden shape:
        # [num_layers * num_directions, batch_size, hidden_dim]
        if self.bidirectional:
            # 取最后一层的正向和反向 hidden
            forward_hidden = hidden[-2]
            backward_hidden = hidden[-1]
            final_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        else:
            final_hidden = hidden[-1]

        final_hidden = self.dropout(final_hidden)
        logits = self.classifier(final_hidden)
        return logits

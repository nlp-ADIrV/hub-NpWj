import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerLanguageModel(nn.Module):
    """
    基于 Transformer Encoder 实现的单向自回归语言模型。
    关键点：
    1. 输入序列从左到右建模；
    2. 通过 causal mask 遮蔽未来 token；
    3. 训练目标为预测下一个 token。
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int = 128,
        n_embd: int = 128,
        n_head: int = 4,
        n_layer: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=4 * n_embd,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layer,
        )

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _causal_mask(self, size: int, device: torch.device):
        """
        mask 上三角部分，使当前位置不能看到未来位置。
        True / -inf 表示被遮蔽。
        """
        mask = torch.triu(
            torch.ones(size, size, device=device, dtype=torch.bool),
            diagonal=1,
        )
        return mask

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """
        idx: [batch_size, seq_len]
        targets: [batch_size, seq_len]
        """
        batch_size, seq_len = idx.shape
        if seq_len > self.block_size:
            raise ValueError(
                f"输入长度 {seq_len} 超过 block_size={self.block_size}"
            )

        token_emb = self.token_embedding(idx)
        pos = torch.arange(0, seq_len, device=idx.device).unsqueeze(0)
        pos_emb = self.position_embedding(pos)

        x = token_emb + pos_emb
        mask = self._causal_mask(seq_len, idx.device)

        x = self.transformer(x, mask=mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ):
        """
        自回归生成：
        每次取当前上下文，预测下一个 token，并拼接到序列末尾。
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            if top_k is not None:
                values, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)

        return idx

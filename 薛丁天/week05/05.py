"""
Transformer-based Unidirectional Language Model
A minimal implementation for text generation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math


class TransformerLM(nn.Module):
    """Unidirectional Transformer Language Model"""

    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=3, d_ff=512, dropout=0.1, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

        # Unidirectional mask: causal mask prevents attending to future tokens
        self.causal_mask = self._create_causal_mask(max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

        self.vocab_size = vocab_size

    def _create_causal_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask

    def forward(self, x):
        batch_size, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)

        x = self.token_embedding(x) + self.pos_embedding(positions)
        x = self.dropout(x)

        # Apply causal mask
        mask = self.causal_mask[:seq_len, :seq_len].to(x.device)
        x = self.transformer(x, mask=mask)

        return self.fc(x)


class TextDataset(Dataset):
    def __init__(self, text, tokenizer, max_len=64):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tokens = tokenizer.encode(text)
        # Pre-compute all sequences
        self.sequences = []
        for i in range(0, len(self.tokens) - max_len):
            self.sequences.append(self.tokens[i:i + max_len])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y


class SimpleTokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.stoi = {c: i + 3 for i, c in enumerate(chars)}  # 0=PAD, 1=BOS, 2=UNK
        self.itos = {i + 3: c for i, c in enumerate(chars)}  # 0=PAD, 1=BOS, 2=UNK
        self.stoi['<PAD>'] = 0
        self.itos[0] = '<PAD>'
        self.vocab_size = len(chars) + 4

    def encode(self, text):
        return [self.stoi.get(c, 2) for c in text]

    def decode(self, tokens):
        return ''.join([self.itos.get(t, '<UNK>') for t in tokens if t > 2])


def train(model, dataloader, epochs=50, lr=1e-3, device='cpu'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, model.vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")


def generate(model, tokenizer, prompt, max_new_tokens=50, temperature=1.0, device='cpu'):
    model.eval()
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    generated = tokens.copy()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids)
            next_token_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            generated.append(next_token)
            input_ids = torch.tensor(generated, dtype=torch.long).unsqueeze(0).to(device)

    return tokenizer.decode(generated)


if __name__ == '__main__':
    # Sample text corpus
    text = """
    The quick brown fox jumps over the lazy dog.
    A journey of a thousand miles begins with a single step.
    To be or not to be that is the question.
    """ * 10

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize
    tokenizer = SimpleTokenizer(text)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    dataset = TextDataset(text, tokenizer, max_len=32)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = TransformerLM(vocab_size=tokenizer.vocab_size, d_model=128, nhead=4, num_layers=3)
    model = model.to(device)

    # Train
    print("\nTraining...")
    train(model, dataloader, epochs=5)

    # Generate
    print("\nGeneration samples:")
    prompts = ["The quick", "A journey", "To be or"]
    for p in prompts:
        result = generate(model, tokenizer, p, max_new_tokens=30, temperature=0.8)
        print(f"  Prompt: '{p}' -> {result}")

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class BiEncoder(nn.Module):
    def __init__(self, model_name="bert-base-chinese"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

    def encode(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids,
                           attention_mask=attention_mask)
        return out.last_hidden_state[:, 0]

    def forward(self, input_ids, attention_mask):
        return self.encode(input_ids, attention_mask)


def predict(model, tokenizer, data):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for s1, s2, label in data:

            enc1 = tokenizer(s1, return_tensors="pt", truncation=True, padding=True)
            enc2 = tokenizer(s2, return_tensors="pt", truncation=True, padding=True)

            v1 = model(enc1["input_ids"], enc1["attention_mask"])
            v2 = model(enc2["input_ids"], enc2["attention_mask"])

            score = F.cosine_similarity(v1, v2).item()
            pred = 1 if score > 0.5 else 0

            y_true.append(label)
            y_pred.append(pred)

    return y_true, y_pred

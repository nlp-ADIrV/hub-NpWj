import torch
import torch.nn as nn
from transformers import AutoModel


class CrossEncoder(nn.Module):
    def __init__(self, model_name="bert-base-chinese"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids,
                           attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        return self.classifier(cls)


def predict(model, tokenizer, data):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for s1, s2, label in data:

            enc = tokenizer(
                s1,
                s2,
                return_tensors="pt",
                truncation=True,
                padding=True
            )

            logits = model(enc["input_ids"], enc["attention_mask"])
            pred = logits.argmax(dim=-1).item()

            y_true.append(label)
            y_pred.append(pred)

    return y_true, y_pred

import argparse
import json
from pathlib import Path

import torch

from src.data_utils import encode_text
from src.model import RNNTextClassifier


def load_model(config_path: str, model_path: str = None):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RNNTextClassifier(
        vocab_size=config["vocab_size"],
        num_classes=config["num_classes"],
        model_type=config["model_type"],
        embed_dim=config["embed_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        bidirectional=config["bidirectional"],
        dropout=config["dropout"],
        pad_idx=config["pad_idx"],
    ).to(device)

    ckpt_path = model_path or config["best_model_path"]
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    return model, config, device


def predict_one(text: str, model, config, device):
    token_ids, length = encode_text(text, config["vocab"], config["max_len"])

    input_ids = torch.tensor([token_ids], dtype=torch.long).to(device)
    lengths = torch.tensor([length], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(input_ids, lengths)
        probs = torch.softmax(logits, dim=-1)[0]
        pred_id = int(torch.argmax(probs).item())

    id2label = {int(k): v for k, v in config["id2label"].items()}
    pred_label = id2label[pred_id]

    return pred_label, probs.cpu().tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="outputs/config_lstm.json")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--text", type=str, required=True)
    args = parser.parse_args()

    model, config, device = load_model(args.config_path, args.model_path)
    pred_label, probs = predict_one(args.text, model, config, device)

    print("Input:", args.text)
    print("Predicted label:", pred_label)
    print("Probabilities:", probs)


if __name__ == "__main__":
    main()

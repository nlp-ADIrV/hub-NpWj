import argparse

import torch
import torch.nn as nn


class MaxIndexClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_dataset(num_samples: int, input_dim: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(num_samples, input_dim, generator=g)
    y = torch.argmax(x, dim=1)
    return x, y


def load_model(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    input_dim = checkpoint["input_dim"]
    hidden_dim = checkpoint["hidden_dim"]

    model = MaxIndexClassifier(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, input_dim


def test_single_sample(model: nn.Module, vector, device: torch.device):
    x = torch.tensor([vector], dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1).item()
    true_label = torch.argmax(x, dim=1).item()
    print("Single sample test")
    print(f"Input vector: {vector}")
    print(f"Expected class: {true_label}")
    print(f"Predicted class: {pred}")


def test_batch(model: nn.Module, input_dim: int, num_samples: int, seed: int, device: torch.device):
    x_test, y_test = build_dataset(num_samples, input_dim, seed)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    with torch.no_grad():
        logits = model(x_test)
        preds = torch.argmax(logits, dim=1)

    acc = (preds == y_test).float().mean().item()
    print("\nBatch test")
    print(f"Number of samples: {num_samples}")
    print(f"Accuracy: {acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Test a trained max-index classifier.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model.pth", help="Checkpoint path.")
    parser.add_argument(
        "--vector",
        type=float,
        nargs="*",
        default=None,
        help="Optional custom input vector for single-sample inference, e.g. --vector 0.1 0.8 0.3 0.2 0.4",
    )
    parser.add_argument("--num-test-samples", type=int, default=1000, help="Number of synthetic samples for batch evaluation.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed used for batch evaluation.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, input_dim = load_model(args.checkpoint, device)

    if args.vector is not None:
        if len(args.vector) != input_dim:
            raise ValueError(f"Custom vector length must equal input_dim={input_dim}, got {len(args.vector)}")
        vector = args.vector
    else:
        vector = [0.1, 0.8, 0.3, 0.2, 0.4] if input_dim == 5 else [0.0] * input_dim
        if input_dim != 5:
            vector[0] = 0.1
            vector[1] = 0.9

    test_single_sample(model, vector, device)
    test_batch(model, input_dim, args.num_test_samples, args.seed, device)


if __name__ == "__main__":
    main()

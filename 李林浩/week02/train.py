import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


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


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            preds = torch.argmax(logits, dim=1)
            total_loss += loss.item() * batch_x.size(0)
            total_correct += (preds == batch_y).sum().item()
            total_count += batch_x.size(0)

    avg_loss = total_loss / total_count
    avg_acc = total_correct / total_count
    return avg_loss, avg_acc


def main():
    parser = argparse.ArgumentParser(description="Train a max-index multi-class classifier.")
    parser.add_argument("--input-dim", type=int, default=5, help="Dimension of the input vector.")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Hidden layer size.")
    parser.add_argument("--num-samples", type=int, default=10000, help="Total number of synthetic samples.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--save-path", type=str, default="checkpoints/model.pth", help="Path to save checkpoint.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x, y = build_dataset(args.num_samples, args.input_dim, args.seed)

    train_size = int(args.num_samples * args.train_ratio)
    x_train, x_val = x[:train_size], x[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=args.batch_size, shuffle=False)

    model = MaxIndexClassifier(input_dim=args.input_dim, hidden_dim=args.hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            total_loss += loss.item() * batch_x.size(0)
            total_correct += (preds == batch_y).sum().item()
            total_count += batch_x.size(0)

        train_loss = total_loss / total_count
        train_acc = total_correct / total_count
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch [{epoch:02d}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": args.input_dim,
            "hidden_dim": args.hidden_dim,
            "seed": args.seed,
        },
        args.save_path,
    )
    print(f"\nModel saved to: {args.save_path}")


if __name__ == "__main__":
    main()

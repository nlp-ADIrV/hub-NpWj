import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.dataset import CharTokenizer, TextDataset
from src.model import TransformerLanguageModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="训练基于 Transformer 的单向语言模型"
    )
    parser.add_argument("--data_path", type=str, default="data/sample.txt")
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def get_device(device_arg: str):
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    data_path = Path(args.data_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    text = data_path.read_text(encoding="utf-8")
    if len(text.strip()) == 0:
        raise ValueError("训练文本为空，请检查 data_path。")

    tokenizer = CharTokenizer()
    tokenizer.fit(text)

    dataset = TextDataset(
        text=text,
        tokenizer=tokenizer,
        block_size=args.block_size,
    )

    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    device = get_device(args.device)

    model = TransformerLanguageModel(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    config = vars(args)
    config["vocab_size"] = tokenizer.vocab_size
    config["device"] = device

    tokenizer.save(out_dir / "vocab.json")
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]")
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            logits, loss = model(x, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in tqdm(
                val_loader,
                desc=f"Epoch {epoch}/{args.epochs} [valid]",
            ):
                x = x.to(device)
                y = y.to(device)
                _, loss = model(x, y)
                val_losses.append(loss.item())

        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}"
        )

        latest_ckpt = {
            "model_state_dict": model.state_dict(),
            "config": config,
        }
        torch.save(latest_ckpt, out_dir / "latest.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(latest_ckpt, out_dir / "best.pt")
            print(f"保存最佳模型：{out_dir / 'best.pt'}")

    print("训练完成。")
    print(f"模型目录：{out_dir}")


if __name__ == "__main__":
    main()

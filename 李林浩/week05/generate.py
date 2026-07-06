import argparse
import json
from pathlib import Path

import torch

from src.dataset import CharTokenizer
from src.model import TransformerLanguageModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="使用训练好的 Transformer 单向语言模型生成文本"
    )
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--vocab_path", type=str, default="checkpoints/vocab.json")
    parser.add_argument("--prompt", type=str, default="人工智能")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def get_device(device_arg: str):
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def main():
    args = parse_args()
    device = get_device(args.device)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"未找到模型文件：{checkpoint_path}。请先运行 train.py 完成训练。"
        )

    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]

    tokenizer = CharTokenizer.load(args.vocab_path)

    model = TransformerLanguageModel(
        vocab_size=config["vocab_size"],
        block_size=config["block_size"],
        n_embd=config["n_embd"],
        n_head=config["n_head"],
        n_layer=config["n_layer"],
        dropout=config["dropout"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    try:
        prompt_ids = tokenizer.encode(args.prompt)
    except ValueError as e:
        print("生成失败：prompt 中包含训练语料中没有出现过的字符。")
        print(e)
        print("建议：换一个 prompt，或把相关字符加入训练语料后重新训练。")
        return

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model.generate(
            idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    result = tokenizer.decode(out[0].tolist())
    print("=" * 60)
    print(result)
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
基于预训练模型的交互式文本续写脚本。
用法:
    python generate.py
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pre_trained import LM

# ──────────────────────── 默认配置 ────────────────────────

DEFAULT_MAX_LEN = 200
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.9
STOP_CHARS = '\n。！？…'  # 句号 ！ ？ ； … " "
CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.pt")

# ──────────────────────── 生成逻辑 ────────────────────────

@torch.no_grad()
def generate(model, prompt, char2idx, idx2char, device,
             max_len=DEFAULT_MAX_LEN, temperature=DEFAULT_TEMPERATURE,
             top_p=DEFAULT_TOP_P, seq_len=64):
    model.eval()

    ids = [char2idx[c] for c in prompt if c in char2idx]
    if not ids:
        return prompt
    input_ids = torch.tensor([ids], dtype=torch.long).to(device)

    generated = list(prompt)

    for _ in range(max_len):
        if input_ids.size(1) > seq_len:
            input_ids = input_ids[:, -seq_len:]

        logits = model(input_ids)
        next_logits = logits[0, -1, :] / temperature

        # Top-p (nucleus) filtering
        sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumsum = torch.cumsum(probs, dim=-1)
        cutoff = cumsum > top_p
        cutoff[1:] = cutoff[:-1].clone()
        cutoff[0] = False
        sorted_logits[cutoff] = -float("inf")

        filtered_probs = torch.softmax(sorted_logits, dim=-1)
        chosen = torch.multinomial(filtered_probs, 1).item()
        next_id = sorted_indices[chosen].item()

        next_char = idx2char[next_id]
        generated.append(next_char)

        if next_char in STOP_CHARS:
            break

        input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)

    return "".join(generated)


# ──────────────────────── 入口 ────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载检查点
    ckpt_path = CHECKPOINT
    if not os.path.exists(ckpt_path):
        print(f"未找到模型文件: {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    char2idx = ckpt["char2idx"]
    idx2char = ckpt["idx2char"]
    vocab_size = len(char2idx)

    print(f"词表大小: {vocab_size}  embed_dim={args['embed_dim']}  "
          f"nhead={args['nhead']}  num_layers={args['num_layers']}")

    model = LM(
        vocab_size=vocab_size,
        embed_dim=args["embed_dim"],
        nhead=args["nhead"],
        num_layers=args["num_layers"],
        dim_feedforward=args["dim_feedforward"],
        dropout=args["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}  已加载权重: {ckpt_path}")

    seq_len = args.get("seq_len", 64)

    print(f"\n{'='*50}")
    print("输入文字开头，模型将自动续写。")
    print(f"参数: temperature={DEFAULT_TEMPERATURE}  top_p={DEFAULT_TOP_P}  "
          f"max_len={DEFAULT_MAX_LEN}  stop_chars={repr(STOP_CHARS)}")
    print("输入 'quit' 或 'exit' 退出。")
    print(f"{'='*50}")

    while True:
        try:
            prompt = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if prompt.lower() in ("quit", "exit"):
            print("再见！")
            break
        if not prompt:
            continue

        result = generate(
            model, prompt, char2idx, idx2char, device,
            max_len=DEFAULT_MAX_LEN,
            temperature=DEFAULT_TEMPERATURE,
            top_p=DEFAULT_TOP_P,
            seq_len=seq_len,
        )
        print(f"\n{result}")


if __name__ == "__main__":
    main()

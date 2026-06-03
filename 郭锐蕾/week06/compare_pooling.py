"""
三种池化策略（cls / mean / max）效果对比

教学重点：
  1. 控制变量：除 --pool 外，epochs、lr、batch_size 等参数保持一致
  2. val_acc 看整体，macro F1 更能反映类别不均衡下的真实表现
  3. 同一图中对比训练曲线与最终指标，直观看出策略差异

使用方式：
  # 先分别训练三种池化（见下方 main 中的提示）
  python compare_pooling.py

  # 仅根据已有 train_log_*.json 画曲线（无 checkpoint 也可）
  python compare_pooling.py --logs_only
"""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score

matplotlib.rcParams["axes.unicode_minus"] = False

POOLS = ["cls", "mean", "max"]
POOL_LABELS = {"cls": "CLS", "mean": "Mean", "max": "Max"}
POOL_COLORS = {"cls": "#4C72B0", "mean": "#55A868", "max": "#C44E52"}


def _find_chinese_font():
    candidates = ["SimHei", "Microsoft YaHei", "PingFang SC",
                  "Noto Sans CJK SC", "WenQuanYi Micro Hei"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return None


_CN_FONT = _find_chinese_font()
if _CN_FONT:
    plt.rcParams["font.family"] = _CN_FONT

ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data"
BERT_PATH = ROOT.parent.parent / "pretrain_models" / "bert-base-chinese"
OUTPUT_DIR = ROOT / "outputs"
CKPT_DIR  = OUTPUT_DIR / "checkpoints"
FIG_DIR   = OUTPUT_DIR / "figures"


def load_train_logs(output_dir: Path) -> dict:
    """读取 train_log_{pool}.json，返回 {pool: [epoch_records]}。"""
    logs = {}
    for pool in POOLS:
        path = output_dir / f"train_log_{pool}.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                logs[pool] = json.load(f)
    return logs


def get_predictions(ckpt_path, data_dir, bert_path, device):
    from model import build_model
    from dataset import build_dataloaders
    from transformers import BertTokenizer

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    pool = ckpt["pool"]
    with open(data_dir / "label_map.json", encoding="utf-8") as f:
        num_labels = json.load(f)["num_labels"]

    model = build_model(bert_path, num_labels=num_labels, pool=pool)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()

    tokenizer = BertTokenizer.from_pretrained(bert_path)
    _, val_loader, _ = build_dataloaders(
        data_dir, tokenizer, max_length=128, batch_size=64
    )

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["token_type_ids"].to(device),
            )
            all_preds.extend(logits.argmax(-1).cpu().numpy())
            all_labels.extend(batch["label"].numpy())

    return np.array(all_preds), np.array(all_labels), ckpt


def compare_and_plot(
    logs: dict,
    ckpt_metrics: dict | None,
    preds_by_pool: dict | None,
    labels,
    id2name,
    output_dir: Path = FIG_DIR,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    has_logs = bool(logs)
    has_ckpt = bool(ckpt_metrics)
    n_panels = (2 if has_logs else 0) + (2 if has_ckpt and preds_by_pool else 0)
    if n_panels == 0:
        print("没有可用的日志或 checkpoint，无法绘图。")
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]
    ax_idx = 0

    # ── 训练曲线：val_acc / val_macro_f1 ─────────────────────────────────────
    if has_logs:
        ax_acc = axes[ax_idx]
        ax_idx += 1
        ax_f1 = axes[ax_idx]
        ax_idx += 1

        for pool in POOLS:
            if pool not in logs:
                continue
            epochs = [r["epoch"] for r in logs[pool]]
            val_acc = [r["val_acc"] for r in logs[pool]]
            val_f1 = [r["val_macro_f1"] for r in logs[pool]]
            label = POOL_LABELS[pool]
            color = POOL_COLORS[pool]
            ax_acc.plot(epochs, val_acc, "o-", label=label, color=color, linewidth=2)
            ax_f1.plot(epochs, val_f1, "o-", label=label, color=color, linewidth=2)

        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Val Accuracy")
        ax_acc.set_title("验证集 Accuracy（按 epoch）")
        ax_acc.legend()
        ax_acc.grid(True, alpha=0.3)

        ax_f1.set_xlabel("Epoch")
        ax_f1.set_ylabel("Val Macro F1")
        ax_f1.set_title("验证集 Macro F1（按 epoch）")
        ax_f1.legend()
        ax_f1.grid(True, alpha=0.3)

    # ── 最优 checkpoint 指标柱状图 + 各类 Recall ─────────────────────────────
    if has_ckpt and preds_by_pool and labels is not None:
        ax_bar = axes[ax_idx]
        ax_idx += 1

        pools_avail = [p for p in POOLS if p in ckpt_metrics]
        x = np.arange(len(pools_avail))
        w = 0.35
        accs = [ckpt_metrics[p]["accuracy"] for p in pools_avail]
        mf1s = [ckpt_metrics[p]["macro_f1"] for p in pools_avail]
        names = [POOL_LABELS[p] for p in pools_avail]

        ax_bar.bar(x - w / 2, accs, w, label="Accuracy", color="#4C72B0", alpha=0.85)
        ax_bar.bar(x + w / 2, mf1s, w, label="Macro F1", color="#DD8452", alpha=0.85)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(names)
        ax_bar.set_ylabel("Score")
        ax_bar.set_title("最优 checkpoint 整体指标")
        ax_bar.legend()
        ax_bar.set_ylim(0, max(max(accs), max(mf1s)) * 1.08 + 0.02)
        for i, (a, f) in enumerate(zip(accs, mf1s)):
            ax_bar.text(i - w / 2, a + 0.005, f"{a:.3f}", ha="center", va="bottom", fontsize=8)
            ax_bar.text(i + w / 2, f + 0.005, f"{f:.3f}", ha="center", va="bottom", fontsize=8)

        # 各类别 Recall（三种策略并排）
        ax_recall = axes[ax_idx]
        label_ids = sorted(id2name.keys())
        class_names = [id2name[i] for i in label_ids]
        n_cls = len(class_names)
        x_cls = np.arange(n_cls)
        bar_w = 0.25

        for j, pool in enumerate(pools_avail):
            recall = recall_score(
                labels, preds_by_pool[pool],
                labels=label_ids, average=None, zero_division=0,
            )
            offset = (j - 1) * bar_w
            ax_recall.bar(
                x_cls + offset, recall, bar_w,
                label=POOL_LABELS[pool], color=POOL_COLORS[pool], alpha=0.85,
            )

        ax_recall.set_xticks(x_cls)
        ax_recall.set_xticklabels(class_names, rotation=40, ha="right")
        ax_recall.set_ylabel("Recall")
        ax_recall.set_title("各类别 Recall 对比（val set）")
        ax_recall.legend()
        ax_recall.axhline(0.5, color="gray", linestyle="--", alpha=0.4)

    plt.suptitle("BERT 三种池化策略对比（cls / mean / max）", fontsize=13)
    plt.tight_layout()
    save_path = output_dir / "compare_pooling.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n对比图已保存 → {save_path}")


def print_summary(logs: dict, ckpt_metrics: dict):
    print(f"\n{'='*60}")
    print(f"{'三种池化策略对比':^60}")
    print(f"{'='*60}")

    if logs:
        print(f"\n{'训练日志（各策略最优 epoch）':^60}")
        print(f"{'策略':<8} {'最优 val_acc':>14} {'对应 macro_f1':>16}")
        print("-" * 45)
        for pool in POOLS:
            if pool not in logs:
                continue
            best = max(logs[pool], key=lambda r: r["val_acc"])
            print(f"  {POOL_LABELS[pool]:<6} {best['val_acc']:>14.4f} {best['val_macro_f1']:>16.4f}  (epoch {best['epoch']})")

    if ckpt_metrics:
        print(f"\n{'Checkpoint 评估（val set）':^60}")
        print(f"{'策略':<8} {'Accuracy':>12} {'Macro F1':>12}")
        print("-" * 35)
        for pool in POOLS:
            if pool not in ckpt_metrics:
                continue
            m = ckpt_metrics[pool]
            print(f"  {POOL_LABELS[pool]:<6} {m['accuracy']:>12.4f} {m['macro_f1']:>12.4f}")


def main():
    parser = argparse.ArgumentParser(description="对比 cls / mean / max 池化策略")
    parser.add_argument("--bert_path", default=str(BERT_PATH))
    parser.add_argument("--data_dir",  default=str(DATA_DIR))
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--logs_only", action="store_true",
                        help="仅根据 train_log_*.json 绘图，不加载 checkpoint")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    data_dir   = Path(args.data_dir)
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logs = load_train_logs(output_dir)
    missing_logs = [p for p in POOLS if p not in logs]

    ckpt_metrics = {}
    preds_by_pool = {}
    labels = None
    id2name = None

    if not args.logs_only:
        with open(data_dir / "label_map.json", encoding="utf-8") as f:
            id2name = {int(k): v for k, v in json.load(f)["id2name"].items()}

        for pool in POOLS:
            ckpt_path = CKPT_DIR / f"best_{pool}.pt"
            if not ckpt_path.exists():
                continue
            print(f"加载 {pool} 模型：{ckpt_path.name}")
            preds, labels, ckpt = get_predictions(
                ckpt_path, data_dir, args.bert_path, device
            )
            preds_by_pool[pool] = preds
            acc = accuracy_score(labels, preds)
            mf1 = f1_score(labels, preds, average="macro", zero_division=0)
            ckpt_metrics[pool] = {
                "accuracy": acc,
                "macro_f1": mf1,
                "ckpt_val_acc": ckpt["val_acc"],
            }

    missing_ckpt = [p for p in POOLS if p not in ckpt_metrics and not args.logs_only]

    if not logs and not ckpt_metrics:
        print("未找到任何 train_log_*.json 或 best_{cls,mean,max}.pt。")
        print("\n请先在 src/ 目录下依次训练（除 --pool 外参数保持一致）：")
        print("  python train.py --pool cls  --epochs 3")
        print("  python train.py --pool mean --epochs 3")
        print("  python train.py --pool max  --epochs 3")
        print("\n然后运行：")
        print("  python compare_pooling.py")
        return

    if missing_logs or missing_ckpt:
        print("\n[提示] 尚未齐全：")
        if missing_logs:
            print(f"  缺少训练日志: {', '.join(missing_logs)}")
        if missing_ckpt and not args.logs_only:
            print(f"  缺少 checkpoint: {', '.join(missing_ckpt)}")
        print("  补齐后重新运行 compare_pooling.py 可获得完整对比图。")

    print_summary(logs, ckpt_metrics)
    with open(data_dir / "label_map.json", encoding="utf-8") as f:
        id2name = {int(k): v for k, v in json.load(f)["id2name"].items()}
    compare_and_plot(logs, ckpt_metrics, preds_by_pool or None, labels, id2name)


if __name__ == "__main__":
    main()

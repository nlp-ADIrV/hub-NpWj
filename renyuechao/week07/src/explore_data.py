"""人民日报 NER 数据集探索与可视化。

输出：
  outputs/figures/entity_distribution.png          实体类别分布
  outputs/figures/text_length_distribution.png     句子长度分布
  outputs/figures/entity_length_distribution.png   实体长度分布
  outputs/figures/training_curve_linear.png        BERT+Linear 训练曲线（若日志存在）
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

matplotlib.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "Arial Unicode MS",
    "PingFang SC",
]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "peoples_daily"
FIG_DIR = ROOT / "outputs" / "figures"
LOG_DIR = ROOT / "outputs" / "logs"

ENTITY_TYPE_ZH = {
    "PER": "人名",
    "ORG": "机构",
    "LOC": "地名",
}


def load_split(split: str, data_dir: Path = DATA_DIR) -> list[dict]:
    path = data_dir / f"{split}.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_entities(tokens: list[str], tags: list[str]) -> list[dict]:
    """从 BIO 标签中抽取实体片段。

    若遇到不规范的 I-X 开头，按新实体处理，便于统计时不丢数据。
    """
    entities: list[dict] = []
    current_type: str | None = None
    current_tokens: list[str] = []

    def flush_current() -> None:
        nonlocal current_type, current_tokens
        if current_type and current_tokens:
            text = "".join(current_tokens)
            entities.append(
                {
                    "type": current_type,
                    "text": text,
                    "length": len(current_tokens),
                }
            )
        current_type = None
        current_tokens = []

    for token, tag in zip(tokens, tags):
        if tag == "O":
            flush_current()
            continue

        prefix, _, entity_type = tag.partition("-")
        if prefix == "B":
            flush_current()
            current_type = entity_type
            current_tokens = [token]
        elif prefix == "I":
            if current_type == entity_type:
                current_tokens.append(token)
            else:
                flush_current()
                current_type = entity_type
                current_tokens = [token]
        else:
            flush_current()

    flush_current()
    return entities


def collect_stats(records: list[dict]) -> dict:
    entity_type_counts = Counter()
    entity_lengths: list[int] = []
    text_lengths: list[int] = []
    entity_per_sentence: list[int] = []
    entities_by_type: dict[str, list[str]] = {}

    for row in records:
        tokens = row["tokens"]
        tags = row["ner_tags"]
        text_lengths.append(len(tokens))
        entities = extract_entities(tokens, tags)
        entity_per_sentence.append(len(entities))

        for entity in entities:
            entity_type = entity["type"]
            entity_type_counts[entity_type] += 1
            entity_lengths.append(entity["length"])
            entities_by_type.setdefault(entity_type, []).append(entity["text"])

    return {
        "entity_type_counts": entity_type_counts,
        "entity_lengths": entity_lengths,
        "text_lengths": text_lengths,
        "entity_per_sentence": entity_per_sentence,
        "entities_by_type": entities_by_type,
    }


def percentile(values: list[int], ratio: float) -> int:
    sorted_values = sorted(values)
    index = int((len(sorted_values) - 1) * ratio)
    return sorted_values[index]


def print_summary(stats_train: dict, stats_val: dict, stats_test: dict) -> None:
    print("=" * 70)
    print("人民日报 NER 数据集统计摘要")
    print("=" * 70)

    for split_name, stats in [
        ("训练集", stats_train),
        ("验证集", stats_val),
        ("测试集", stats_test),
    ]:
        lengths = stats["text_lengths"]
        entity_lengths = stats["entity_lengths"]
        print(f"\n【{split_name}】")
        print(f"  样本数：{len(lengths)}")
        print(f"  句子平均长度：{sum(lengths) / len(lengths):.1f}")
        print(f"  句子 P95/P99：{percentile(lengths, 0.95)} / {percentile(lengths, 0.99)}")
        print(f"  句子最大长度：{max(lengths)}")
        print(f"  实体总数：{sum(stats['entity_type_counts'].values())}")
        print(f"  平均实体数/句：{sum(stats['entity_per_sentence']) / len(stats['entity_per_sentence']):.2f}")
        print(f"  平均实体长度：{sum(entity_lengths) / max(len(entity_lengths), 1):.1f}")

    print("\n【训练集实体频次】")
    for entity_type, count in sorted(stats_train["entity_type_counts"].items()):
        print(f"  {entity_type:4s} ({ENTITY_TYPE_ZH.get(entity_type, entity_type):2s}) : {count:5d}")

    print("\n【训练集实体示例】")
    for entity_type in sorted(stats_train["entities_by_type"]):
        examples = list(dict.fromkeys(stats_train["entities_by_type"][entity_type]))[:5]
        print(f"  {entity_type:4s} : {' | '.join(examples)}")


def plot_entity_distribution(stats_train: dict) -> None:
    counts = stats_train["entity_type_counts"]
    keys = sorted(counts)
    labels = [f"{key}\n({ENTITY_TYPE_ZH.get(key, key)})" for key in keys]
    values = [counts[key] for key in keys]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color="#4C72B0", alpha=0.86, edgecolor="white")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 80,
            str(value),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_title("人民日报 NER 各类实体频次分布（训练集）", fontsize=14)
    ax.set_xlabel("实体类型")
    ax.set_ylabel("实体数量")
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / "entity_distribution.png"
    fig.savefig(out_path, dpi=140)
    plt.close()
    print(f"  已保存 -> {out_path}")


def plot_text_length_distribution(stats_train: dict) -> None:
    lengths = stats_train["text_lengths"]
    p95 = percentile(lengths, 0.95)
    p99 = percentile(lengths, 0.99)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(lengths, bins=50, color="#4C72B0", alpha=0.82, edgecolor="white")
    ax.axvline(x=128, color="orange", linestyle="--", linewidth=1.5, label="max_length=128")
    ax.axvline(x=160, color="red", linestyle="--", linewidth=1.5, label="max_length=160")
    ax.axvline(x=p95, color="green", linestyle="--", linewidth=1.5, label=f"P95={p95}")
    ax.axvline(x=p99, color="purple", linestyle="--", linewidth=1.5, label=f"P99={p99}")
    ax.set_title("人民日报 NER 句子长度分布（训练集）", fontsize=14)
    ax.set_xlabel("句子字符数")
    ax.set_ylabel("样本数")
    ax.legend()
    plt.tight_layout()
    out_path = FIG_DIR / "text_length_distribution.png"
    fig.savefig(out_path, dpi=140)
    plt.close()
    print(f"  已保存 -> {out_path}")
    print(f"  训练集 P95={p95}, P99={p99}，max_length=160 能覆盖绝大多数样本")


def plot_entity_length_distribution(stats_train: dict) -> None:
    lengths = Counter(stats_train["entity_lengths"])
    xs = sorted(lengths)
    ys = [lengths[x] for x in xs]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar([str(x) for x in xs[:20]], ys[:20], color="#55A868", alpha=0.86, edgecolor="white")
    ax.set_title("人民日报 NER 实体长度分布（训练集，前20）", fontsize=14)
    ax.set_xlabel("实体字符数")
    ax.set_ylabel("出现次数")
    plt.tight_layout()
    out_path = FIG_DIR / "entity_length_distribution.png"
    fig.savefig(out_path, dpi=140)
    plt.close()
    print(f"  已保存 -> {out_path}")


def plot_training_curve(model_type: str) -> None:
    log_path = LOG_DIR / f"train_peoples_daily_{model_type}.json"
    if not log_path.exists():
        print(f"  未找到训练日志，跳过训练曲线：{log_path}")
        return

    with log_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    epochs = [row["epoch"] for row in records]
    train_loss = [row["train_loss"] for row in records]
    val_loss = [row["val_loss"] for row in records]
    val_f1 = [row["val_entity_f1"] for row in records]

    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.plot(epochs, train_loss, marker="o", color="#4C72B0", label="train_loss")
    ax1.plot(epochs, val_loss, marker="o", color="#55A868", label="val_loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_xticks(epochs)

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_f1, marker="s", color="#C44E52", label="val_f1")
    ax2.set_ylabel("Entity F1")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax1.set_title(f"BERT + {model_type.upper()} 训练曲线（人民日报 NER）", fontsize=14)
    plt.tight_layout()

    out_path = FIG_DIR / f"training_curve_{model_type}.png"
    fig.savefig(out_path, dpi=140)
    plt.close()
    print(f"  已保存 -> {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="探索人民日报 NER 数据集并生成图表")
    parser.add_argument("--model_type", choices=["linear", "crf"], default="linear")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_records = load_split("train")
    val_records = load_split("validation")
    test_records = load_split("test")

    stats_train = collect_stats(train_records)
    stats_val = collect_stats(val_records)
    stats_test = collect_stats(test_records)

    print_summary(stats_train, stats_val, stats_test)

    print("\n正在生成可视化图表...")
    plot_entity_distribution(stats_train)
    plot_text_length_distribution(stats_train)
    plot_entity_length_distribution(stats_train)
    plot_training_curve(args.model_type)

    print("\n探索完成！图表已保存到 outputs/figures/")


if __name__ == "__main__":
    main()

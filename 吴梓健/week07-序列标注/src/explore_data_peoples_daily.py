"""
peoples_daily 数据集探索与可视化

教学重点：
    1. BIO 标注格式的统计方法
  2. 各实体类型的分布差异（为什么类别不均衡是NER的难点）
  3. 文本长度分布（影响 BERT max_length 的选择）
  4. 实体长度分布（短实体 vs 长实体的识别难度差异）

使用方式：
    python explore_data_peoples_daily.py

输出：
  outputs/figures/entity_distribution.png   各类实体频次分布
  outputs/figures/text_length_distribution.png  文本长度分布
  outputs/figures/entity_length_distribution.png 实体长度分布
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import argparse
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "peoples_daily"
FIG_DIR = ROOT / "outputs" / "figures"
LABEL_PATH = DATA_DIR / "label_names.json"


def load_split(split: str) -> list:
    path = DATA_DIR / f"{split}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_label_names() -> list:
    with open(LABEL_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def decode_bio(tokens: list, tags: list) -> list:
    entities = []
    current_type = None
    current_start = None

    def flush(end_idx: int):
        nonlocal current_type, current_start
        if current_type is None or current_start is None:
            return
        surface = "".join(tokens[current_start:end_idx])
        entities.append((current_type, current_start, end_idx - 1, surface))
        current_type = None
        current_start = None

    for idx, tag in enumerate(tags + ["O"]):
        if tag.startswith("B-"):
            flush(idx)
            current_type = tag[2:]
            current_start = idx
        elif tag.startswith("I-"):
            tag_type = tag[2:]
            if current_type is None or tag_type != current_type:
                flush(idx)
                current_type = tag_type
                current_start = idx
        else:
            flush(idx)

    return entities


def collect_stats(records: list) -> dict:
    entity_type_counts = Counter()
    tag_counts = Counter()
    entity_lengths = []
    text_lengths = []
    entity_per_sentence = []
    entities_by_type = {}

    for row in records:
        tokens = row["tokens"]
        tags = row["ner_tags"]
        text_lengths.append(len(tokens))
        tag_counts.update(tags)

        total_entities = 0
        for etype, start, end, surface in decode_bio(tokens, tags):
            entity_type_counts[etype] += 1
            entity_lengths.append(end - start + 1)
            total_entities += 1
            if etype not in entities_by_type:
                entities_by_type[etype] = []
            entities_by_type[etype].append(surface)

        entity_per_sentence.append(total_entities)

    return {
        "entity_type_counts": entity_type_counts,
        "tag_counts": tag_counts,
        "entity_lengths": entity_lengths,
        "text_lengths": text_lengths,
        "entity_per_sentence": entity_per_sentence,
        "entities_by_type": entities_by_type,
    }


def print_summary(stats_train: dict, stats_val: dict):
    print("=" * 70)
    print("peoples_daily 数据集统计摘要")
    print("=" * 70)

    print("\n【训练集】")
    print(f"  样本数：{len(stats_train['text_lengths'])} 条")
    print(f"  平均句长：{sum(stats_train['text_lengths']) / len(stats_train['text_lengths']):.1f} 字")
    print(f"  最大句长：{max(stats_train['text_lengths'])} 字")
    print(f"  句长中位数：{sorted(stats_train['text_lengths'])[len(stats_train['text_lengths'])//2]} 字")
    print(f"  平均实体数/句：{sum(stats_train['entity_per_sentence']) / len(stats_train['entity_per_sentence']):.2f}")
    print(f"  实体总数：{sum(stats_train['entity_type_counts'].values())}")
    print(f"  平均实体长度：{sum(stats_train['entity_lengths']) / len(stats_train['entity_lengths']):.1f} 字")

    print("\n【BIO 标签频次（训练集）】")
    for tag, cnt in sorted(stats_train["tag_counts"].items(), key=lambda x: (-x[1], x[0])):
        print(f"  {tag:8s} : {cnt:6d} 次")

    print("\n【各类实体频次（训练集）】")
    et_label = {
        "PER": "人名",
        "LOC": "地名",
        "ORG": "机构",
    }
    for etype, cnt in sorted(stats_train["entity_type_counts"].items(), key=lambda x: -x[1]):
        cn = et_label.get(etype, etype)
        print(f"  {etype:15s} ({cn:8s}) : {cnt:5d} 条")

    print("\n【各类实体示例（训练集，取前5个）】")
    for etype in sorted(stats_train["entities_by_type"]):
        cn = et_label.get(etype, etype)
        examples = list(dict.fromkeys(stats_train["entities_by_type"][etype]))[:5]
        print(f"  {etype:15s} ({cn}) : {' | '.join(examples)}")

    print()


def plot_entity_distribution(stats_train: dict):
    et_label = {
        "PER": "人名",
        "LOC": "地名",
        "ORG": "机构",
    }
    counts = stats_train["entity_type_counts"]
    labels = [f"{k}\n({et_label.get(k,k)})" for k in sorted(counts)]
    values = [counts[k] for k in sorted(counts)]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(labels, values, color="#4C72B0", alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20, str(v),
                ha="center", va="bottom", fontsize=9)
    ax.set_title("peoples_daily 各类实体频次分布（训练集）", fontsize=14)
    ax.set_ylabel("实体数量")
    ax.set_xlabel("实体类型")
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "entity_distribution.png", dpi=120)
    print(f"  已保存 → {FIG_DIR / 'entity_distribution.png'}")
    plt.close()


def plot_text_length_distribution(stats_train: dict):
    lengths = stats_train["text_lengths"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(lengths, bins=40, color="#4C72B0", alpha=0.8, edgecolor="white")
    ax.axvline(x=64, color="red", linestyle="--", linewidth=1.5, label="max_length=64")
    ax.axvline(x=128, color="orange", linestyle="--", linewidth=1.5, label="max_length=128")
    ax.axvline(x=256, color="purple", linestyle="--", linewidth=1.5, label="max_length=256")
    p95 = sorted(lengths)[int(len(lengths) * 0.95)]
    ax.axvline(x=p95, color="green", linestyle="--", linewidth=1.5, label=f"P95={p95}")
    ax.set_title("文本长度分布（训练集）", fontsize=14)
    ax.set_xlabel("文本字符数")
    ax.set_ylabel("样本数")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_DIR / "text_length_distribution.png", dpi=120)
    print(f"  已保存 → {FIG_DIR / 'text_length_distribution.png'}")
    plt.close()
    suggested_max_length = 128 if p95 <= 128 else 256
    print(f"  P95 文本长度={p95}，建议 max_length={suggested_max_length}")


def plot_entity_length_distribution(stats_train: dict):
    from collections import Counter
    lengths = Counter(stats_train["entity_lengths"])
    xs = sorted(lengths.keys())
    ys = [lengths[x] for x in xs]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar([str(x) for x in xs[:20]], ys[:20], color="#55A868", alpha=0.85, edgecolor="white")
    ax.set_title("实体长度分布（训练集，前20）", fontsize=14)
    ax.set_xlabel("实体字符数")
    ax.set_ylabel("出现次数")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "entity_length_distribution.png", dpi=120)
    print(f"  已保存 → {FIG_DIR / 'entity_length_distribution.png'}")
    plt.close()

    avg_len = sum(stats_train["entity_lengths"]) / len(stats_train["entity_lengths"])
    print(f"  实体平均长度={avg_len:.1f}字，BIO 任务里 CRF 对边界识别通常更稳")


def main():
    parse_args()

    train_records = load_split("train")
    val_records = load_split("validation")
    label_names = load_label_names()

    stats_train = collect_stats(train_records)
    stats_val = collect_stats(val_records)

    print_summary(stats_train, stats_val)
    print("\n【标签集合】")
    print("  " + " | ".join(label_names))

    print("正在生成可视化图表...")
    plot_entity_distribution(stats_train)
    plot_text_length_distribution(stats_train)
    plot_entity_length_distribution(stats_train)

    print("\n探索完成！图表已保存到 outputs/figures/")
    print("下一步：python train.py               # 训练 BERT+Linear")
    print("         python train.py --use_crf    # 训练 BERT+CRF")


def parse_args():
    parser = argparse.ArgumentParser(description="探索 peoples_daily 数据集")
    return parser.parse_args()


if __name__ == "__main__":
    main()

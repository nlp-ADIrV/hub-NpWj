"""人民日报 NER 数据集：BIO 标签读取 + BERT 子词对齐。

教学重点：
  1. 人民日报 NER 已经是 BIO 格式，不需要 span -> BIO 转换
  2. tokens 与 ner_tags 一一对应，标签体系为 PER / ORG / LOC 三类实体
  3. BERT tokenizer 后用 word_ids() 对齐标签，特殊 token 和非首子词设为 -100
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import json

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "peoples_daily"

PEOPLES_DAILY_LABELS = [
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
]


def build_label_schema() -> tuple[list[str], dict[str, int], dict[int, str]]:
    """构建人民日报 NER 的 BIO 标签体系。"""
    labels = list(PEOPLES_DAILY_LABELS)
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return labels, label2id, id2label


def load_records(split: str, data_dir: Path | str = DATA_DIR) -> list[dict]:
    """读取 train / validation / test 数据，并做基本格式校验。"""
    path = Path(data_dir) / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件: {path}")

    with path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    for index, row in enumerate(records):
        tokens = row.get("tokens")
        tags = row.get("ner_tags")
        if not isinstance(tokens, list) or not isinstance(tags, list):
            raise ValueError(f"{path} 第 {index} 条缺少 tokens 或 ner_tags 列表")
        if len(tokens) != len(tags):
            raise ValueError(
                f"{path} 第 {index} 条 tokens 和 ner_tags 长度不一致: "
                f"{len(tokens)} != {len(tags)}"
            )
    return records


def align_bio_tags(
    tags: list[str],
    word_ids: Iterable[int | None],
    label2id: dict[str, int],
) -> list[int]:
    """把字符级 BIO 标签对齐到 BERT token。"""
    aligned: list[int] = []
    previous_word_id: int | None = None

    for word_id in word_ids:
        if word_id is None:
            aligned.append(-100)
        elif word_id != previous_word_id:
            if word_id >= len(tags):
                aligned.append(-100)
            else:
                aligned.append(label2id[tags[word_id]])
            previous_word_id = word_id
        else:
            aligned.append(-100)

    return aligned


def resolve_default_bert_path() -> Path:
    """返回当前机器上最可能存在的 bert-base-chinese 路径。"""
    candidates = [
        ROOT / "pretrain_models" / "bert-base-chinese",
        Path("/Users/skywalker124/workspace/nlp_learn/视频和课件/week6文本分类问题/pretrain_models/bert-base-chinese"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


class PeopleDailyDataset(Dataset):
    """人民日报 NER 数据集封装。"""

    def __init__(
        self,
        records: list[dict],
        tokenizer: BertTokenizer,
        label2id: dict[str, int],
        max_length: int = 160,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.records[index]
        tokens = row["tokens"]
        tags = row["ner_tags"]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        aligned_labels = align_bio_tags(tags, encoding.word_ids(batch_index=0), self.label2id)

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(aligned_labels, dtype=torch.long),
        }
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)
        else:
            item["token_type_ids"] = torch.zeros_like(item["input_ids"])
        return item


def _limit(records: list[dict], max_samples: int | None) -> list[dict]:
    if max_samples is None or max_samples <= 0:
        return records
    return records[:max_samples]


def build_dataloaders(
    tokenizer: BertTokenizer,
    label2id: dict[str, int],
    batch_size: int = 32,
    max_length: int = 160,
    data_dir: Optional[Path] = None,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """构建 train / validation / test 三个 DataLoader。"""
    data_root = data_dir or DATA_DIR
    train_records = _limit(load_records("train", data_root), max_train_samples)
    val_records = _limit(load_records("validation", data_root), max_eval_samples)
    test_records = _limit(load_records("test", data_root), max_eval_samples)

    train_dataset = PeopleDailyDataset(train_records, tokenizer, label2id, max_length)
    val_dataset = PeopleDailyDataset(val_records, tokenizer, label2id, max_length)
    test_dataset = PeopleDailyDataset(test_records, tokenizer, label2id, max_length)

    print(
        "数据集规模："
        f"train={len(train_dataset)}, validation={len(val_dataset)}, test={len(test_dataset)}"
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader

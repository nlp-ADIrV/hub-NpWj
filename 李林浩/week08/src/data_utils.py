"""数据读取与通用工具。支持 JSONL 格式：sentence1、sentence2、label。"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List


def load_jsonl(path: str | Path) -> List[Dict]:
    path = Path(path)
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            rows.append({
                "sentence1": str(item["sentence1"]),
                "sentence2": str(item["sentence2"]),
                "label": int(item["label"]),
            })
    return rows


def load_dataset(data_dir: str | Path, split: str) -> List[Dict]:
    path = Path(data_dir) / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件：{path}")
    return load_jsonl(path)


def balanced_sample(rows: List[Dict], n: int, seed: int = 42) -> List[Dict]:
    """按正负样本尽量均衡采样；n<=0 时返回原数据。"""
    if n <= 0 or n >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    pos = [r for r in rows if r["label"] == 1]
    neg = [r for r in rows if r["label"] == 0]
    n_pos = min(len(pos), n // 2)
    n_neg = min(len(neg), n - n_pos)
    sample = rng.sample(pos, n_pos) + rng.sample(neg, n_neg)
    rng.shuffle(sample)
    return sample


def dump_json(path: str | Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

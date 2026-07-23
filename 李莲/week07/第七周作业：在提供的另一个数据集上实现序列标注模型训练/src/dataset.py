"""
NER 数据集类：span 标注→BIO 转换 + BERT 子词对齐

教学重点：
  1. 医疗数据集的 span 格式转为 BIO 格式
     - 医疗格式: [{"text": "张三", "label": "人名", "start": 3, "end": 5}]
     - BIO:  ['O','O',...,'B-人名','I-人名',...]
  2. BERT 子词对齐（word_ids 策略）
     - 中文字符通常一字一token，但 [UNK] 和特殊字符可能例外
     - 非首子词标记为 -100，在 loss 计算中被忽略
  3. DataLoader 工厂函数统一封装

使用方式：
  from dataset import build_label_schema, build_dataloaders
"""

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

ROOT = Path(__file__).parent.parent
# ====================== 修改1：默认数据集路径改为医疗数据集 ======================
DATA_DIR = ROOT / "data" / "medical_ner"

# ====================== 修改2：实体类型改为医疗领域5类 ======================
ENTITY_TYPES = [
    "人名", "症状", "时间", "疾病", "药物"
]


def build_label_schema() -> tuple[list[str], dict[str, int], dict[int, str]]:
    """构建 BIO 标签体系，返回 (labels, label2id, id2label)。
    自动根据 ENTITY_TYPES 生成 B-xxx 和 I-xxx 标签
    """
    labels = ["O"]
    for etype in ENTITY_TYPES:
        labels.append(f"B-{etype}")
        labels.append(f"I-{etype}")

    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    return labels, label2id, id2label


# ====================== 修改3：重写 span_to_bio 函数适配医疗数据集格式 ======================
def span_to_bio(text: str, entities: list, label2id: dict) -> list[int]:
    """将医疗数据集的 entities 格式转为逐字符 BIO 标签 id 列表。

    医疗数据集格式:
    [{"text": "张三", "label": "人名", "start": 3, "end": 5}, ...]
    注意: end 是开区间（Python 切片风格），即 text[start:end] = 实体文本

    教学要点：先全部初始化为 O，再按 span 位置填入 B/I。
    若存在嵌套实体，后处理的实体会覆盖先处理的实体。
    """
    n = len(text)
    bio = ["O"] * n

    if not entities:
        return [label2id[t] for t in bio]

    for entity in entities:
        etype = entity["label"]
        start = entity["start"]
        end = entity["end"]
        
        # 边界安全检查，防止标注错误导致越界
        if start < 0 or end > n or start >= end:
            continue
            
        b_tag = f"B-{etype}"
        i_tag = f"I-{etype}"
        
        # 实体第一个字符标记为 B-xxx
        bio[start] = b_tag
        # 实体后续字符标记为 I-xxx（end是开区间，所以循环到end）
        for idx in range(start + 1, end):
            bio[idx] = i_tag

    return [label2id.get(t, 0) for t in bio]


# ====================== 修改4：类名改为通用名，适配所有BIO格式数据集 ======================
class BioNerDataset(Dataset):
    """通用 BIO 格式 NER 数据集的 PyTorch Dataset。

    教学流程：
      text → span_to_bio → 字符级 BIO ids
           → BertTokenizer (is_split_into_words=True)
           → 用 word_ids() 对齐子词标签（非首子词设为 -100）
           → 返回 input_ids / attention_mask / token_type_ids / labels
    """

    def __init__(
        self,
        records: list,
        tokenizer: BertTokenizer,
        label2id: dict,
        max_length: int = 128,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row = self.records[idx]
        text: str = row["text"]
        # ====================== 修改5：从 entities 字段获取标注，而非 label 字段 ======================
        entities: list = row.get("entities") or []

        # 1. entities → 字符级 BIO id 列表
        char_labels = span_to_bio(text, entities, self.label2id)

        # 2. 将文本拆为字符列表，传入 tokenizer
        #    is_split_into_words=True：把 word_ids() 与字符索引精确对齐
        chars = list(text)
        encoding = self.tokenizer(
            chars,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # 3. 子词对齐：取每个 token 对应的字符索引
        #    - word_ids() 返回 [None, 0, 0, 1, 2, 2, ..., None]
        #      None 对应 [CLS]/[SEP]/[PAD] 特殊token
        #    - 一个中文字符通常只有 1 个子词，但 ##xx 子词是非首子词
        #    - 非首子词、特殊token 标记为 -100，cross_entropy 的 ignore_index
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        prev_word_id = None
        for wid in word_ids:
            if wid is None:
                aligned_labels.append(-100)
            elif wid != prev_word_id:
                # 首次出现这个字符索引：使用对应的 BIO 标签
                if wid < len(char_labels):
                    aligned_labels.append(char_labels[wid])
                else:
                    aligned_labels.append(-100)
                prev_word_id = wid
            else:
                # 同一字符的后续子词（中文极少出现，但保留正确处理逻辑）
                aligned_labels.append(-100)

        labels_tensor = torch.tensor(aligned_labels, dtype=torch.long)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "labels": labels_tensor,
        }


def load_records(split: str, data_dir: Optional[Path] = None) -> list:
    """加载指定划分的数据集文件（train/validation/test）"""
    d = data_dir or DATA_DIR
    with open(d / f"{split}.json", "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataloaders(
    tokenizer: BertTokenizer,
    label2id: dict,
    batch_size: int = 32,
    max_length: int = 128,
    data_dir: Optional[Path] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """构建训练/验证/测试 DataLoader，返回 (train_loader, val_loader, test_loader)。
    
    可通过 data_dir 参数动态指定不同数据集路径，例如：
    build_dataloaders(tokenizer, label2id, data_dir=ROOT/"data"/"cluener")
    """
    train_records = load_records("train", data_dir)
    val_records = load_records("validation", data_dir)
    test_records = load_records("test", data_dir)

    train_ds = BioNerDataset(train_records, tokenizer, label2id, max_length)
    val_ds = BioNerDataset(val_records, tokenizer, label2id, max_length)
    test_ds = BioNerDataset(test_records, tokenizer, label2id, max_length)

    print(f"数据集加载成功：训练={len(train_ds)}条，验证={len(val_ds)}条，测试={len(test_ds)}条")
    print(f"标签体系：{list(label2id.keys())}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader
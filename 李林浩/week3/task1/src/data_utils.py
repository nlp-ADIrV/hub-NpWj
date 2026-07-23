import re
from collections import Counter
from typing import List, Tuple, Dict

import pandas as pd
import torch
from torch.utils.data import Dataset


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def simple_tokenize(text: str) -> List[str]:
    """
    简单 tokenizer：
    - 如果文本中包含空格，则按空格切分，适合已经分好词的中文或英文。
    - 如果没有空格，则按字符切分，适合中文 demo。
    真实项目中可替换为 jieba、BPE、SentencePiece 或 HuggingFace tokenizer。
    """
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    if " " in text:
        return text.split(" ")
    return list(text)


def build_vocab(texts: List[str], min_freq: int = 1, max_vocab_size: int = 30000) -> Dict[str, int]:
    counter = Counter()
    for text in texts:
        counter.update(simple_tokenize(text))

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, freq in counter.most_common(max_vocab_size - len(vocab)):
        if freq >= min_freq:
            vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: Dict[str, int], max_len: int) -> Tuple[List[int], int]:
    tokens = simple_tokenize(text)
    token_ids = [vocab.get(tok, vocab[UNK_TOKEN]) for tok in tokens[:max_len]]
    length = len(token_ids)

    if length < max_len:
        token_ids += [vocab[PAD_TOKEN]] * (max_len - length)

    # pack_padded_sequence 要求 length 至少为 1
    length = max(length, 1)
    return token_ids, length


class TextClassificationDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: List[str],
        vocab: Dict[str, int],
        label2id: Dict[str, int],
        max_len: int = 64,
    ):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        token_ids, length = encode_text(self.texts[idx], self.vocab, self.max_len)
        label_id = self.label2id[self.labels[idx]]
        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
            "label": torch.tensor(label_id, dtype=torch.long),
        }


def load_csv_dataset(path: str):
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV 文件必须包含 text 和 label 两列。")

    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(str).tolist()
    return texts, labels

import json
from pathlib import Path
import torch
from torch.utils.data import Dataset


class CharTokenizer:
    """
    字符级 tokenizer。
    优点：无需额外分词工具，中文、英文、符号都可以直接跑通。
    """

    def __init__(self):
        self.stoi = {}
        self.itos = {}

    def fit(self, text: str):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @property
    def vocab_size(self):
        return len(self.stoi)

    def encode(self, text: str):
        unknown = [ch for ch in text if ch not in self.stoi]
        if unknown:
            unique_unknown = sorted(set(unknown))
            raise ValueError(
                f"输入中存在训练词表没有见过的字符：{unique_unknown[:20]}"
            )
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        return "".join([self.itos[int(i)] for i in ids])

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "stoi": self.stoi,
                    "itos": {str(k): v for k, v in self.itos.items()},
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, path: str | Path):
        tok = cls()
        with Path(path).open("r", encoding="utf-8") as f:
            obj = json.load(f)
        tok.stoi = {k: int(v) for k, v in obj["stoi"].items()}
        tok.itos = {int(k): v for k, v in obj["itos"].items()}
        return tok


class TextDataset(Dataset):
    """
    语言模型训练样本：
    x = text[i : i + block_size]
    y = text[i + 1 : i + block_size + 1]
    即每个位置预测下一个字符。
    """

    def __init__(self, text: str, tokenizer: CharTokenizer, block_size: int):
        super().__init__()
        self.block_size = block_size
        self.ids = tokenizer.encode(text)

        if len(self.ids) <= block_size:
            raise ValueError(
                f"文本长度必须大于 block_size。当前文本长度={len(self.ids)}，"
                f"block_size={block_size}"
            )

    def __len__(self):
        return len(self.ids) - self.block_size

    def __getitem__(self, idx):
        x = torch.tensor(
            self.ids[idx : idx + self.block_size],
            dtype=torch.long,
        )
        y = torch.tensor(
            self.ids[idx + 1 : idx + self.block_size + 1],
            dtype=torch.long,
        )
        return x, y

import torch
import json
from torch.utils.data import Dataset

# 标签映射表（BIO标注体系）
LABEL_LIST = ["O", "B-person", "I-person", "B-location", "I-location", "B-organization", "I-organization"]
LABEL2ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL = {idx: label for idx, label in enumerate(LABEL_LIST)}
MAX_SEQ_LEN = 128


# NER数据集类
class NERDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        print(f"加载数据集，样本总数：{len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = sample["text"]
        # 初始化标签为非实体O
        label_list = ["O"] * len(text)

        # 根据实体位置构建BIO标签
        for ent_type, span_dict in sample["label"].items():
            for span in span_dict.values():
                start, end = span[0]
                label_list[start] = f"B-{ent_type}"
                for i in range(start + 1, end + 1):
                    label_list[i] = f"I-{ent_type}"

        # 字符转数字ID，截断/补全到固定长度
        input_ids = [min(ord(c), 21127) for c in text]
        input_ids = input_ids[:MAX_SEQ_LEN]
        input_ids += [0] * (MAX_SEQ_LEN - len(input_ids))

        # 注意力掩码
        attention_mask = [1] * len(text) + [0] * (MAX_SEQ_LEN - len(text))
        attention_mask = attention_mask[:MAX_SEQ_LEN]

        # 标签转ID
        label_ids = [LABEL2ID[label] for label in label_list]
        label_ids = label_ids[:MAX_SEQ_LEN]
        label_ids += [0] * (MAX_SEQ_LEN - len(label_ids))

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(label_ids)
        }

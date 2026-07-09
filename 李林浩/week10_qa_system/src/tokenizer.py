import re
from typing import List


STOPWORDS = {
    "的", "了", "是", "在", "和", "与", "及", "或", "一个", "一种",
    "什么", "哪些", "如何", "为什么", "可以", "进行", "主要", "通常",
    "the", "a", "an", "is", "are", "of", "to", "and", "or", "in",
}


def tokenize(text: str) -> List[str]:
    """
    对中英文混合文本进行轻量分词。

    英文保留单词和数字；
    中文使用单字和相邻双字组合，避免依赖额外分词库。
    """
    text = text.lower().strip()

    english_tokens = re.findall(r"[a-z0-9_]+", text)
    chinese_sequences = re.findall(r"[\u4e00-\u9fff]+", text)

    chinese_tokens = []
    for seq in chinese_sequences:
        chinese_tokens.extend(list(seq))
        if len(seq) >= 2:
            chinese_tokens.extend(seq[i:i + 2] for i in range(len(seq) - 1))

    tokens = english_tokens + chinese_tokens
    return [token for token in tokens if token not in STOPWORDS]

import math
from collections import Counter
from typing import Dict, List

from .tokenizer import tokenize


class BM25:
    def __init__(self, documents: List[str], k1: float = 1.5, b: float = 0.75):
        if not documents:
            raise ValueError("BM25 需要至少一个文档。")

        self.k1 = k1
        self.b = b
        self.tokenized_docs = [tokenize(text) for text in documents]
        self.doc_lengths = [len(tokens) for tokens in self.tokenized_docs]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)

        self.term_frequencies = [Counter(tokens) for tokens in self.tokenized_docs]
        self.document_frequencies = self._build_document_frequencies()
        self.idf = self._build_idf()

    def _build_document_frequencies(self) -> Dict[str, int]:
        df = Counter()

        for tokens in self.tokenized_docs:
            for token in set(tokens):
                df[token] += 1

        return dict(df)

    def _build_idf(self) -> Dict[str, float]:
        total_docs = len(self.tokenized_docs)
        idf = {}

        for token, df in self.document_frequencies.items():
            idf[token] = math.log(
                1 + (total_docs - df + 0.5) / (df + 0.5)
            )

        return idf

    def score(self, query: str, doc_index: int) -> float:
        query_tokens = tokenize(query)
        frequencies = self.term_frequencies[doc_index]
        doc_length = self.doc_lengths[doc_index]

        score = 0.0

        for token in query_tokens:
            if token not in frequencies:
                continue

            tf = frequencies[token]
            denominator = tf + self.k1 * (
                1 - self.b + self.b * doc_length / max(self.avg_doc_length, 1)
            )

            score += self.idf.get(token, 0.0) * (
                tf * (self.k1 + 1) / denominator
            )

        return score

    def search(self, query: str, top_k: int = 3):
        scored = [
            (index, self.score(query, index))
            for index in range(len(self.tokenized_docs))
        ]

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]
